"""Gymnasium-compatible causal Version 1 HAPS-ISAC environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from haps_isac.config import ExperimentConfig, load_config
from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.action_transform import action_from_mapping, clip_high_level_action
from haps_isac.control.physics_completion import complete_action
from haps_isac.control.safety_repair import repair_action
from haps_isac.control.virtual_queues import (
    constraint_increments,
    queue_slices,
    update_virtual_queues,
)
from haps_isac.envs.observation_builder import (
    GLOBAL_FEATURE_DIM,
    PAIR_TOKEN_DIM,
    PREVIOUS_ACTION_DIM,
    build_observation,
)
from haps_isac.envs.reward_builder import StageCost, build_stage_cost
from haps_isac.envs.scenario_sampler import (
    initial_target_covariance,
    initial_target_state,
    sample_channels,
    sample_ue_positions,
)
from haps_isac.envs.state import SensingJob, SimulatorState
from haps_isac.freshness.aoi import update_aoi
from haps_isac.freshness.aoli import update_aoli
from haps_isac.freshness.aosi import update_aosi
from haps_isac.physics.channels import (
    azimuth_rad,
    effective_beam_gain,
    ula_steering,
)
from haps_isac.physics.energy import (
    computation_energy_j,
    haps_slot_energy_j,
    processing_delay_slots,
)
from haps_isac.physics.noma import legitimate_noma_result, packet_completed
from haps_isac.physics.secrecy import SecrecyResult, evaluate_eavesdropper
from haps_isac.physics.sensing import (
    SensingResult,
    evaluate_sensing,
    sample_measurement,
    sensing_sinr,
)
from haps_isac.rng import RandomStreams
from haps_isac.tracking import ekf
from haps_isac.tracking.target_dynamics import predict_gaussian, propagate_true_state
from haps_isac.units import thermal_noise_watts


class HapsIsacEnv(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Causal one-HAPS, fixed-pair, one-target Version 1 environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: ExperimentConfig | str = "configs/system_v1.yaml") -> None:
        super().__init__()
        self.config = load_config(config) if isinstance(config, str) else config
        pairs = self.config.system.num_noma_pairs
        continuous_low = np.asarray(
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        continuous_high = np.asarray(
            [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "pair": spaces.Discrete(pairs + 1),
                "ris_code": spaces.Discrete(1),
                "continuous": spaces.Box(continuous_low, continuous_high),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "pair_tokens": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(pairs, PAIR_TOKEN_DIM),
                    dtype=np.float32,
                ),
                "pair_mask": spaces.Box(
                    0,
                    1,
                    shape=(pairs,),
                    dtype=np.int8,
                ),
                "global_features": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(GLOBAL_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                "virtual_queues": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.config.num_virtual_queues,),
                    dtype=np.float32,
                ),
                "previous_action": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(PREVIOUS_ACTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self._streams: RandomStreams | None = None
        self._state: SimulatorState | None = None
        self._receiver_noise_w = thermal_noise_watts(
            self.config.system.bandwidth_hz,
            self.config.channels.thermal_noise_density_dbm_hz,
            self.config.channels.receiver_noise_figure_db,
        )

    @property
    def state(self) -> SimulatorState:
        if self._state is None:
            raise RuntimeError("environment must be reset before use")
        return self._state

    def _initial_state(self) -> SimulatorState:
        if self._streams is None:
            raise RuntimeError("random streams are not initialized")
        config = self.config
        ue_positions = sample_ue_positions(config, self._streams["scenario"])
        target_state = initial_target_state(config)
        initial_covariance = initial_target_covariance(config)
        channels = sample_channels(
            config,
            ue_positions,
            target_state,
            self._streams["channels"],
        )
        pairs = config.system.num_noma_pairs
        initial_mean = target_state.copy()
        return SimulatorState(
            slot=0,
            ue_positions_m=ue_positions,
            target_true_state=target_state,
            channels=channels,
            hidden_filter_mean=initial_mean.copy(),
            hidden_filter_covariance=initial_covariance.copy(),
            available_source_mean=initial_mean.copy(),
            available_source_covariance=initial_covariance.copy(),
            available_timestamp=0,
            available_mean=initial_mean.copy(),
            available_covariance=initial_covariance.copy(),
            pending_sensing_jobs=(),
            aoi=np.full(
                (pairs, 2),
                config.freshness.initial_aoi_slots,
                dtype=np.float64,
            ),
            aoli=np.full(
                (pairs, 2),
                config.freshness.initial_aoli_slots,
                dtype=np.float64,
            ),
            aosi=config.freshness.initial_aosi_slots,
            waiting_slots=np.zeros(pairs, dtype=np.float64),
            virtual_queues=np.zeros(config.num_virtual_queues, dtype=np.float64),
            last_rates_bps=np.zeros((pairs, 2), dtype=np.float64),
            last_delivery=np.zeros((pairs, 2), dtype=np.bool_),
            last_sic_margin=np.zeros(pairs, dtype=np.float64),
            previous_action=None,
            last_repair_distance=0.0,
            last_fallback_used=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        effective_seed = self.config.master_seed if seed is None else seed
        super().reset(seed=effective_seed)
        self._streams = RandomStreams.from_seed(effective_seed)
        self._state = self._initial_state()
        observation = build_observation(self.config, self.state)
        return observation, {
            "slot": 0,
            "master_seed": effective_seed,
            "hard_feasible": True,
            "fallback_used": False,
        }

    def _coerce_action(
        self,
        action: HighLevelAction | Mapping[str, Any],
    ) -> HighLevelAction:
        high_level = action if isinstance(action, HighLevelAction) else action_from_mapping(action)
        return clip_high_level_action(
            high_level,
            self.config.system.num_noma_pairs,
            self.config.features.aerial_ris,
        )

    def _evaluate_communication(
        self,
        executable: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        SecrecyResult | None,
    ]:
        config = self.config
        state = self.state
        physical = executable.physical
        pairs = config.system.num_noma_pairs
        rates = np.zeros((pairs, 2), dtype=np.float64)
        delivered = np.zeros((pairs, 2), dtype=np.bool_)
        intercepted = np.zeros((pairs, 2), dtype=np.bool_)
        secrecy_outage = np.zeros((pairs, 2), dtype=np.bool_)
        sic_margins = state.last_sic_margin.copy()

        if physical.high_level.pair == 0 or physical.communication_power_w <= 0.0:
            return rates, delivered, intercepted, secrecy_outage, sic_margins, None

        pair = physical.high_level.pair - 1
        near_channel = state.channels.haps_ue_true[pair, 0]
        far_channel = state.channels.haps_ue_true[pair, 1]
        near_gain = effective_beam_gain(near_channel, physical.communication_beam)
        far_gain = effective_beam_gain(far_channel, physical.communication_beam)
        near_sensing_gain = effective_beam_gain(near_channel, physical.sensing_beam)
        far_sensing_gain = effective_beam_gain(far_channel, physical.sensing_beam)
        near_disturbance = (
            config.channels.sensing_cancellation_fraction
            * physical.sensing_power_w
            * near_sensing_gain
            + self._receiver_noise_w
        )
        far_disturbance = (
            config.channels.sensing_cancellation_fraction
            * physical.sensing_power_w
            * far_sensing_gain
            + self._receiver_noise_w
        )
        noma = legitimate_noma_result(
            physical.near_power_w,
            physical.far_power_w,
            near_gain,
            far_gain,
            near_disturbance,
            far_disturbance,
            config.channels.legitimate_sic_residual_fraction,
            config.system.bandwidth_hz,
            config.constraints.sic_sinr_threshold,
        )
        rates[pair] = [noma.near_rate_bps, noma.far_rate_bps]
        delivered[pair] = [
            packet_completed(
                noma.near_rate_bps,
                config.system.slot_duration_s,
                config.traffic.packet_bits_near,
            ),
            packet_completed(
                noma.far_rate_bps,
                config.system.slot_duration_s,
                config.traffic.packet_bits_far,
            ),
        ]
        sic_margins[pair] = noma.sic_margin

        target_channel = state.channels.haps_target_true
        target_gain = effective_beam_gain(
            target_channel,
            physical.communication_beam,
        )
        target_sensing_gain = effective_beam_gain(
            target_channel,
            physical.sensing_beam,
        )
        target_disturbance = (
            config.channels.sensing_cancellation_fraction
            * physical.sensing_power_w
            * target_sensing_gain
            + self._receiver_noise_w
        )
        secrecy = evaluate_eavesdropper(
            physical.near_power_w,
            physical.far_power_w,
            target_gain,
            target_disturbance,
            config.channels.target_sic_residual_fraction,
            config.system.bandwidth_hz,
            config.system.slot_duration_s,
            config.traffic.packet_bits_near,
            config.traffic.packet_bits_far,
            noma.near_rate_bps,
            noma.far_rate_bps,
            config.constraints.secrecy_rate_target_bps,
        )
        intercepted[pair] = [secrecy.near_intercepted, secrecy.far_intercepted]
        secrecy_outage[pair] = [secrecy.near_outage, secrecy.far_outage]
        return (
            rates,
            delivered,
            intercepted,
            secrecy_outage,
            sic_margins,
            secrecy,
        )

    def _evaluate_sensing(
        self,
        executable: Any,
    ) -> tuple[
        SensingResult,
        SensingJob | None,
        np.ndarray,
        np.ndarray,
        float,
        float | None,
    ]:
        if self._streams is None:
            raise RuntimeError("random streams are not initialized")
        config = self.config
        state = self.state
        physical = executable.physical
        haps_position = np.asarray(config.haps.position_m, dtype=np.float64)
        true_position = np.asarray([state.target_true_state[0], state.target_true_state[1], 0.0])
        actual_angle = azimuth_rad(haps_position, true_position)
        target_steering = ula_steering(
            config.haps.num_tx_antennas,
            actual_angle,
            unit_norm=True,
        )
        residual_si = (
            config.sensing.residual_self_interference_fraction
            if config.features.residual_self_interference
            else 0.0
        )
        sinr = sensing_sinr(
            physical.communication_power_w,
            physical.sensing_power_w,
            physical.communication_beam,
            physical.sensing_beam,
            target_steering,
            config.haps.num_rx_antennas,
            config.sensing.echo_gain,
            self._receiver_noise_w,
            residual_si,
        )
        sensing = evaluate_sensing(
            sinr,
            config.sensing.sinr_threshold,
            config.sensing.sinr_floor,
            config.sensing.range_variance_scale,
            config.sensing.angle_variance_scale,
            config.sensing.doppler_variance_scale,
        )
        posterior_mean = state.hidden_filter_mean.copy()
        posterior_covariance = state.hidden_filter_covariance.copy()
        job: SensingJob | None = None
        compute_energy = 0.0
        sensing_nis: float | None = None
        if sensing.detected:
            measurement = sample_measurement(
                state.target_true_state,
                haps_position,
                sensing.measurement_covariance,
                self._streams["sensing"],
            )
            sensing_nis = ekf.normalized_innovation_squared(
                posterior_mean,
                posterior_covariance,
                measurement,
                sensing.measurement_covariance,
                haps_position,
            )
            posterior_mean, posterior_covariance = ekf.update(
                posterior_mean,
                posterior_covariance,
                measurement,
                sensing.measurement_covariance,
                haps_position,
            )
            delay = processing_delay_slots(
                config.haps.sensing_cycles,
                physical.cpu_frequency_hz,
                config.system.slot_duration_s,
            )
            compute_energy = computation_energy_j(
                config.haps.effective_capacitance,
                config.haps.sensing_cycles,
                physical.cpu_frequency_hz,
            )
            if float(np.trace(posterior_covariance)) <= config.sensing.acceptance_covariance_trace:
                job = SensingJob(
                    timestamp=state.slot,
                    ready_slot=state.slot + delay,
                    posterior_mean=posterior_mean.copy(),
                    posterior_covariance=posterior_covariance.copy(),
                )
        return (
            sensing,
            job,
            posterior_mean,
            posterior_covariance,
            compute_energy,
            sensing_nis,
        )

    def _release_sensing_jobs(
        self,
        pending_jobs: tuple[SensingJob, ...],
        next_slot: int,
    ) -> tuple[
        tuple[SensingJob, ...],
        SensingJob | None,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
    ]:
        config = self.config
        state = self.state
        current_trace = float(np.trace(state.available_source_covariance))
        ready = [
            job
            for job in pending_jobs
            if job.ready_slot <= next_slot
            and (
                job.timestamp > state.available_timestamp
                or (
                    job.timestamp == state.available_timestamp
                    and float(np.trace(job.posterior_covariance)) < current_trace
                )
            )
        ]
        accepted = max(ready, key=lambda job: job.timestamp) if ready else None
        remaining = tuple(job for job in pending_jobs if job.ready_slot > next_slot)
        if accepted is None:
            source_mean = state.available_source_mean.copy()
            source_covariance = state.available_source_covariance.copy()
            source_timestamp = state.available_timestamp
        else:
            source_mean = accepted.posterior_mean.copy()
            source_covariance = accepted.posterior_covariance.copy()
            source_timestamp = accepted.timestamp
        steps = max(0, next_slot - source_timestamp)
        available_mean, available_covariance = predict_gaussian(
            source_mean,
            source_covariance,
            config.system.slot_duration_s,
            config.target.acceleration_std_mps2,
            steps=steps,
        )
        return (
            remaining,
            accepted,
            source_mean,
            source_covariance,
            source_timestamp,
            available_mean,
            available_covariance,
        )

    def step(
        self,
        action: HighLevelAction | Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._streams is None:
            raise RuntimeError("environment must be reset before stepping")
        config = self.config
        state = self.state
        high_level = self._coerce_action(action)
        completed = complete_action(config, state, high_level)
        executable = repair_action(
            config,
            state,
            completed,
            self._receiver_noise_w,
        )
        (
            rates,
            delivered,
            intercepted,
            secrecy_outage,
            sic_margins,
            secrecy,
        ) = self._evaluate_communication(executable)
        (
            sensing,
            new_job,
            hidden_posterior_mean,
            hidden_posterior_covariance,
            compute_energy,
            sensing_nis,
        ) = self._evaluate_sensing(executable)

        physical = executable.physical
        next_aoi = update_aoi(
            state.aoi,
            delivered,
            config.freshness.aoi_cap_slots,
        )
        next_aoli = update_aoli(
            state.aoli,
            intercepted,
            config.freshness.aoli_cap_slots,
        )
        waiting = np.minimum(
            state.waiting_slots + 1.0,
            config.freshness.aoi_cap_slots,
        )
        if physical.high_level.pair > 0:
            waiting[physical.high_level.pair - 1] = 0.0

        hidden_next_mean, hidden_next_covariance = ekf.predict(
            hidden_posterior_mean,
            hidden_posterior_covariance,
            config.system.slot_duration_s,
            config.target.acceleration_std_mps2,
        )
        next_target_state = propagate_true_state(
            state.target_true_state,
            config.system.slot_duration_s,
            config.target.acceleration_std_mps2,
            self._streams["target"],
        )
        next_slot = state.slot + 1
        pending_jobs = state.pending_sensing_jobs
        if new_job is not None:
            pending_jobs = (*pending_jobs, new_job)
        (
            remaining_jobs,
            accepted_job,
            source_mean,
            source_covariance,
            source_timestamp,
            available_mean,
            available_covariance,
        ) = self._release_sensing_jobs(pending_jobs, next_slot)
        next_aosi = update_aosi(
            state.aosi,
            next_slot,
            accepted_job.timestamp if accepted_job is not None else None,
            config.freshness.aosi_cap_slots,
        )
        covariance_trace = float(np.trace(available_covariance))
        increments = constraint_increments(
            config,
            next_aoli,
            delivered,
            secrecy_outage,
            physical.high_level.pair,
            sensing.detected,
            covariance_trace,
        )
        next_queues = update_virtual_queues(
            state.virtual_queues,
            increments,
            config.constraints.queue_clip,
        )
        slot_energy = haps_slot_energy_j(
            physical.communication_power_w,
            physical.sensing_power_w,
            config.system.slot_duration_s,
            compute_energy,
        )
        stage_cost = build_stage_cost(
            config,
            next_aoi,
            next_aosi,
            covariance_trace,
            slot_energy,
        )
        next_channels = sample_channels(
            config,
            state.ue_positions_m,
            next_target_state,
            self._streams["channels"],
        )
        self._state = SimulatorState(
            slot=next_slot,
            ue_positions_m=state.ue_positions_m.copy(),
            target_true_state=next_target_state,
            channels=next_channels,
            hidden_filter_mean=hidden_next_mean,
            hidden_filter_covariance=hidden_next_covariance,
            available_source_mean=source_mean,
            available_source_covariance=source_covariance,
            available_timestamp=source_timestamp,
            available_mean=available_mean,
            available_covariance=available_covariance,
            pending_sensing_jobs=remaining_jobs,
            aoi=next_aoi,
            aoli=next_aoli,
            aosi=next_aosi,
            waiting_slots=waiting,
            virtual_queues=next_queues,
            last_rates_bps=rates,
            last_delivery=delivered,
            last_sic_margin=sic_margins,
            previous_action=physical.high_level,
            last_repair_distance=executable.log.distance,
            last_fallback_used=executable.log.fallback_used,
        )
        observation = build_observation(config, self.state)
        truncated = next_slot >= config.system.episode_slots
        info = self._build_info(
            stage_cost,
            executable,
            sensing,
            secrecy,
            delivered,
            intercepted,
            secrecy_outage,
            increments,
            covariance_trace,
            compute_energy,
            sensing_nis,
            accepted_job,
        )
        return observation, stage_cost.reward, False, truncated, info

    def _build_info(
        self,
        stage_cost: StageCost,
        executable: Any,
        sensing: SensingResult,
        secrecy: SecrecyResult | None,
        delivered: np.ndarray,
        intercepted: np.ndarray,
        secrecy_outage: np.ndarray,
        increments: np.ndarray,
        covariance_trace: float,
        compute_energy: float,
        sensing_nis: float | None,
        accepted_job: SensingJob | None,
    ) -> dict[str, Any]:
        config = self.config
        state = self.state
        layout = queue_slices(config.system.num_noma_pairs)
        target_error = state.available_mean[:2] - state.target_true_state[:2]
        secrecy_rates = (
            [secrecy.near_secrecy_rate_bps, secrecy.far_secrecy_rate_bps]
            if secrecy is not None
            else [0.0, 0.0]
        )
        physical = executable.physical
        target_state_error = state.available_mean - state.target_true_state
        target_state_nees = float(
            target_state_error
            @ np.linalg.pinv(state.available_covariance)
            @ target_state_error
        )
        target_sinrs = (
            [secrecy.target_near_sinr, secrecy.target_far_sinr]
            if secrecy is not None
            else [0.0, 0.0]
        )
        target_rates = (
            [secrecy.target_near_rate_bps, secrecy.target_far_rate_bps]
            if secrecy is not None
            else [0.0, 0.0]
        )
        communication_energy = (
            physical.communication_power_w * config.system.slot_duration_s
        )
        sensing_energy = physical.sensing_power_w * config.system.slot_duration_s
        return {
            "slot": state.slot,
            "stage_cost": stage_cost.total,
            "normalized_aoi_cost": stage_cost.normalized_aoi,
            "normalized_aosi_cost": stage_cost.normalized_aosi,
            "normalized_uncertainty_cost": stage_cost.normalized_uncertainty,
            "normalized_energy_cost": stage_cost.normalized_energy,
            "sum_aoi": float(np.sum(state.aoi)),
            "mean_aoi": float(np.mean(state.aoi)),
            "max_aoi": float(np.max(state.aoi)),
            "aoi": state.aoi.copy(),
            "mean_aoli": float(np.mean(state.aoli)),
            "min_aoli": float(np.min(state.aoli)),
            "aoli": state.aoli.copy(),
            "aosi": state.aosi,
            "delivery": delivered.copy(),
            "rates_bps": state.last_rates_bps.copy(),
            "interception": intercepted.copy(),
            "secrecy_outage": secrecy_outage.copy(),
            "secrecy_rates_bps": np.asarray(secrecy_rates, dtype=np.float64),
            "mean_secrecy_rate_bps": float(np.mean(secrecy_rates)),
            "target_decoding_sinr": np.asarray(target_sinrs, dtype=np.float64),
            "target_decoding_rates_bps": np.asarray(target_rates, dtype=np.float64),
            "tracking_mse": float(np.dot(target_error, target_error)),
            "tracking_state_nees": target_state_nees,
            "tracking_cov_trace": covariance_trace,
            "total_energy_j": stage_cost.energy_j,
            "communication_energy_j": communication_energy,
            "sensing_energy_j": sensing_energy,
            "computation_energy_j": compute_energy,
            "propulsion_energy_j": 0.0,
            "total_power_w": (
                physical.communication_power_w + physical.sensing_power_w
            ),
            "communication_power_w": physical.communication_power_w,
            "sensing_power_w": physical.sensing_power_w,
            "cpu_frequency_hz": physical.cpu_frequency_hz,
            "sic_margin": state.last_sic_margin.copy(),
            "sensing_sinr": sensing.sinr,
            "sensing_detected": sensing.detected,
            "sensing_measurement_covariance": sensing.measurement_covariance.copy(),
            "sensing_nis": sensing_nis,
            "virtual_queues": state.virtual_queues.copy(),
            "constraint_increments": increments.copy(),
            "accepted_sensing_timestamp": (
                accepted_job.timestamp if accepted_job is not None else None
            ),
            "pending_sensing_jobs": len(state.pending_sensing_jobs),
            "reliability_violation": float(
                np.mean(np.maximum(increments[layout.reliability], 0.0))
            ),
            "secrecy_violation": float(np.mean(np.maximum(increments[layout.secrecy], 0.0))),
            "sensing_violation": float(max(increments[layout.sensing], 0.0)),
            "uncertainty_violation": float(max(increments[layout.uncertainty], 0.0)),
            "power_violation": 0.0,
            "mobility_violation": 0.0,
            "repair_distance": executable.log.distance,
            "repair_reasons": executable.log.reasons,
            "hard_feasible": executable.log.hard_feasible,
            "fallback_used": executable.log.fallback_used,
        }

    def clone(self) -> HapsIsacEnv:
        if self._streams is None:
            raise RuntimeError("environment must be reset before cloning")
        cloned = HapsIsacEnv(self.config)
        cloned._streams = self._streams.clone()
        cloned._state = self.state.clone()
        return cloned

    def fork_from_state(
        self,
        state: SimulatorState,
        rollout_seed: int,
    ) -> HapsIsacEnv:
        """Create an isolated environment with fresh common-random-number streams."""

        if rollout_seed < 0:
            raise ValueError("rollout_seed must be non-negative")
        forked = HapsIsacEnv(self.config)
        forked._streams = RandomStreams.from_seed(rollout_seed)
        forked._state = state.clone()
        return forked

    def evaluate_candidate(
        self,
        state: SimulatorState,
        high_level_action: HighLevelAction | Mapping[str, Any],
        rollout_seed: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Evaluate one action without mutating this environment."""

        candidate_env = self.fork_from_state(state, rollout_seed)
        return candidate_env.step(high_level_action)
