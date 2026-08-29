"""Versioned deterministic construction of causal teacher prompts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig
from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.physics_completion import complete_action
from haps_isac.envs.state import SimulatorState
from haps_isac.physics.channels import effective_beam_gain
from haps_isac.units import thermal_noise_watts

if TYPE_CHECKING:
    from haps_isac.teachers.base_teacher import VerificationConfig

Observation = dict[str, npt.NDArray[np.generic]]
SIC_TEMPLATE_SENSING_FRACTIONS = (0.05, 0.25, 0.45)
SIC_TEMPLATE_NEAR_FRACTION = 0.20
SIC_TEMPLATE_POWER_MARGIN = 1.15
PAIR_TOKEN_FIELDS = (
    "near_aoi_fraction",
    "far_aoi_fraction",
    "near_aoli_fraction",
    "far_aoli_fraction",
    "near_channel_gain_fraction",
    "far_channel_gain_fraction",
    "near_csi_uncertainty_fraction",
    "far_csi_uncertainty_fraction",
    "near_packet_fraction",
    "far_packet_fraction",
    "previous_sic_margin_tanh",
    "waiting_time_fraction",
)
COVARIANCE_CHOLESKY_FIELDS = (
    "L00",
    "L10",
    "L11",
    "L20",
    "L21",
    "L22",
    "L30",
    "L31",
    "L32",
    "L33",
)


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    state_id: str
    prompt_version: str
    prompt: str
    prompt_hash: str
    causal_state_hash: str
    causal_payload: dict[str, Any]
    semantic_state_packet: dict[str, Any]
    optimization_contract: dict[str, Any]
    sensing_only_template: dict[str, Any]

    sic_safe_templates: tuple[dict[str, Any], ...]


def _rounded_list(value: npt.NDArray[np.generic]) -> Any:
    return np.round(np.asarray(value, dtype=np.float64), decimals=6).tolist()


def causal_observation_payload(observation: Observation) -> dict[str, Any]:
    """Return only values exposed by the Gymnasium observation contract."""

    required = {
        "pair_tokens",
        "pair_mask",
        "global_features",
        "virtual_queues",
        "previous_action",
    }
    missing = required.difference(observation)
    if missing:
        raise ValueError(f"observation is missing keys: {sorted(missing)}")
    return {
        "pair_tokens": _rounded_list(observation["pair_tokens"]),
        "pair_mask": np.asarray(observation["pair_mask"], dtype=np.int8).tolist(),
        "global_features": _rounded_list(observation["global_features"]),
        "virtual_queues": _rounded_list(observation["virtual_queues"]),
        "previous_action": _rounded_list(observation["previous_action"]),
    }


def build_semantic_state_packet(
    config: ExperimentConfig,
    causal: dict[str, Any],
) -> dict[str, Any]:
    """Name every causal feature exposed to the numerical policy.

    The student continues to train on compact tensors in ``causal``. The teacher,
    however, needs feature names and directions rather than a hidden positional
    encoding inside a JSON array.
    """

    tokens = causal["pair_tokens"]
    queues = causal["virtual_queues"]
    users = config.num_users
    pairs: list[dict[str, Any]] = []
    queue_pairs: list[dict[str, Any]] = []
    for pair_index, token in enumerate(tokens):
        if len(token) != 14:
            raise ValueError("pair token contract must have 14 elements")
        pair_values = {
            field: float(token[index]) for index, field in enumerate(PAIR_TOKEN_FIELDS[:8])
        }
        pair_values.update(
            {
                "pair_id": pair_index + 1,
                "active": bool(causal["pair_mask"][pair_index]),
                "near_packet_fraction": float(token[10]),
                "far_packet_fraction": float(token[11]),
                "previous_sic_margin_tanh": float(token[12]),
                "waiting_time_fraction": float(token[13]),
            }
        )
        pairs.append(pair_values)
        near_user = 2 * pair_index
        far_user = near_user + 1
        queue_pairs.append(
            {
                "pair_id": pair_index + 1,
                "near": {
                    "aoli_deficit": float(queues[near_user]),
                    "delivery_deficit": float(queues[users + near_user]),
                    "secrecy_outage_deficit": float(queues[2 * users + near_user]),
                },
                "far": {
                    "aoli_deficit": float(queues[far_user]),
                    "delivery_deficit": float(queues[users + far_user]),
                    "secrecy_outage_deficit": float(queues[2 * users + far_user]),
                },
            }
        )

    global_features = causal["global_features"]
    if len(global_features) != 25:
        raise ValueError("global feature contract must have 25 elements")
    previous_action = causal["previous_action"]
    if len(previous_action) != 9:
        raise ValueError("previous action contract must have 9 elements")
    if len(queues) != config.num_virtual_queues:
        raise ValueError("virtual queue contract does not match the configured system")

    return {
        "packet_version": "2.0",
        "normalization": {
            "bounded_fraction_fields": (
                "[0, 1] where larger means more urgency, gain, uncertainty, or deficit "
                "unless explicitly signed"
            ),
            "signed_fields": [
                "target_position_x_normalized",
                "target_position_y_normalized",
                "target_velocity_x_normalized",
                "target_velocity_y_normalized",
                "target_covariance_cholesky_lower_normalized",
                "previous_heading_fraction_of_pi",
                "previous_sic_margin_tanh",
            ],
            "virtual_queue_transform": (
                "clip(log1p(raw_queue) / log1p(queue_reference), 0, 1); larger means "
                "a larger accumulated long-term constraint deficit"
            ),
        },
        "pairs": pairs,
        "global": {
            "aosi_fraction": float(global_features[0]),
            "target_position_x_normalized": float(global_features[1]),
            "target_position_y_normalized": float(global_features[2]),
            "target_velocity_x_normalized": float(global_features[3]),
            "target_velocity_y_normalized": float(global_features[4]),
            "target_covariance_cholesky_lower_normalized": {
                field: float(global_features[5 + index])
                for index, field in enumerate(COVARIANCE_CHOLESKY_FIELDS)
            },
        },
        "virtual_queues": {
            "per_pair": queue_pairs,
            "sensing_outage_deficit": float(queues[3 * users]),
            "tracking_uncertainty_deficit": float(queues[3 * users + 1]),
        },
        "previous_action": {
            "scheduled_pair_fraction": float(previous_action[0]),
            "eta_haps": float(previous_action[2]),
            "eta_communication": float(previous_action[3]),
            "eta_near_fraction_of_maximum": float(previous_action[4]),
            "eta_jamming": float(previous_action[5]),
            "previous_heading_fraction_of_pi": float(previous_action[6]),
            "aav_speed_fraction": float(previous_action[7]),
            "eta_cpu": float(previous_action[8]),
        },
    }


def build_optimization_contract(
    config: ExperimentConfig,
    verification: VerificationConfig | None,
) -> dict[str, Any]:
    """Expose the causal objective and verifier definition to the teacher."""

    contract: dict[str, Any] = {
        "decision_rule": (
            "minimize risk-sensitive verifier score; all hard-feasible actions dominate "
            "infeasible actions"
        ),
        "stage_cost_weights": {
            "mean_aoi": config.objective.weight_aoi,
            "aosi": config.objective.weight_aosi,
            "tracking_uncertainty": config.objective.weight_uncertainty,
            "energy": config.objective.weight_energy,
        },
        "long_term_constraint_targets": {
            "minimum_aoli_slots": config.constraints.minimum_aoli_slots,
            "minimum_delivery_rate": config.constraints.minimum_delivery_rate,
            "maximum_secrecy_outage_probability": config.constraints.secrecy_outage_probability,
            "maximum_sensing_outage_probability": config.constraints.sensing_outage_probability,
            "maximum_tracking_covariance_trace": config.constraints.maximum_covariance_trace,
        },
        "queue_interpretation": (
            "larger normalized queue values indicate greater accumulated violation pressure "
            "and should increase priority when physically feasible"
        ),
    }
    if verification is not None:
        contract["rollout_verifier"] = {
            "candidate_controlled_steps": [0],
            "continuation_policy_after_step_zero": "urgency_greedy",
            "horizon_slots": verification.rollout_horizon_slots,
            "initial_monte_carlo_rollouts": verification.monte_carlo_rollouts,
            "maximum_monte_carlo_rollouts": verification.max_monte_carlo_rollouts,
            "discount_factor": verification.discount_factor,
            "risk_score": {
                "mean_cost_weight": 1.0 - verification.cvar_weight,
                "cvar_alpha": verification.cvar_alpha,
                "cvar_weight": verification.cvar_weight,
                "mean_constraint_violation_weight": verification.constraint_weight,
                "mean_repair_distance_weight": verification.repair_weight,
                "fallback_rate_weight": verification.fallback_weight,
                "hard_infeasibility_penalty": 1000.0,
            },
        }
    return contract


def _round_up(value: float, decimals: int = 6) -> float:
    scale = float(10**decimals)
    return math.ceil(value * scale) / scale


def _round_down(value: float, decimals: int = 6) -> float:
    scale = float(10**decimals)
    return math.floor(value * scale) / scale


def build_sic_safe_templates(
    config: ExperimentConfig,
    state: SimulatorState,
) -> tuple[dict[str, Any], ...]:
    """Build causal NOMA templates with margin for sensing interference and SIC."""

    receiver_noise = thermal_noise_watts(
        config.system.bandwidth_hz,
        config.channels.thermal_noise_density_dbm_hz,
        config.channels.receiver_noise_figure_db,
    )
    maximum_power = config.haps.max_power_w
    sic_threshold = config.constraints.sic_sinr_threshold
    near_denominator_factor = 1.0 - SIC_TEMPLATE_NEAR_FRACTION * (1.0 + sic_threshold)
    templates: list[dict[str, Any]] = []
    for pair_index in range(config.system.num_noma_pairs):
        reference_action = HighLevelAction(
            pair=pair_index + 1,
            ris_code=0,
            eta_haps=1.0,
            eta_communication=0.5,
            eta_near=SIC_TEMPLATE_NEAR_FRACTION,
            eta_jamming=0.0,
            aav_heading_rad=0.0,
            aav_speed_fraction=0.0,
            eta_cpu=0.5,
        )
        completed = complete_action(config, state, reference_action)
        near_channel = state.channels.haps_ue_estimate[pair_index, 0]
        communication_gain = effective_beam_gain(
            near_channel,
            completed.communication_beam,
        )
        sensing_gain = effective_beam_gain(near_channel, completed.sensing_beam)
        if communication_gain <= 0.0:
            continue
        for sensing_fraction in SIC_TEMPLATE_SENSING_FRACTIONS:
            sensing_power = max(
                config.constraints.minimum_sensing_power_w,
                sensing_fraction * maximum_power,
            )
            disturbance = (
                config.channels.sensing_cancellation_fraction * sensing_power * sensing_gain
                + receiver_noise
            )
            required_communication = SIC_TEMPLATE_POWER_MARGIN * (
                sic_threshold * disturbance / (communication_gain * near_denominator_factor)
            )
            target_communication_power = _round_up(max(required_communication, 1e-6))
            target_total_power = target_communication_power + sensing_power
            if target_total_power > maximum_power:
                continue
            eta_haps = min(
                1.0,
                _round_up(target_total_power / maximum_power),
            )
            allocated_total_power = eta_haps * maximum_power
            eta_communication = _round_down(target_communication_power / allocated_total_power)
            communication_power = eta_communication * allocated_total_power
            sensing_power = allocated_total_power - communication_power
            disturbance = (
                config.channels.sensing_cancellation_fraction * sensing_power * sensing_gain
                + receiver_noise
            )
            maximum_near_power = (
                communication_power * communication_gain - sic_threshold * disturbance
            ) / (communication_gain * (1.0 + sic_threshold))
            maximum_near_fraction = _round_down(
                min(0.5, max(0.0, maximum_near_power / communication_power))
            )
            templates.append(
                {
                    "template_id": (f"p{pair_index + 1}_sense{int(round(sensing_power))}w"),
                    "pair": pair_index + 1,
                    "eta_haps": eta_haps,
                    "eta_communication": eta_communication,
                    "maximum_eta_near": maximum_near_fraction,
                    "recommended_eta_near": round(
                        min(SIC_TEMPLATE_NEAR_FRACTION, 0.8 * maximum_near_fraction),
                        6,
                    ),
                    "communication_power_w": round(communication_power, 6),
                    "sensing_power_w": round(sensing_power, 6),
                }
            )
    return tuple(templates)


def build_sensing_only_template(config: ExperimentConfig) -> dict[str, Any]:
    """Build the minimum-power sensing-only action template."""

    minimum_fraction = config.constraints.minimum_sensing_power_w / config.haps.max_power_w
    if minimum_fraction > 1.0:
        raise ValueError("minimum sensing power exceeds maximum HAPS power")
    eta_haps = _round_up(minimum_fraction)
    return {
        "template_id": "sensing_only_minimum_power",
        "pair": 0,
        "eta_haps": eta_haps,
        "eta_communication": 0.0,
        "eta_near": 0.0,
        "sensing_power_w": round(eta_haps * config.haps.max_power_w, 6),
    }


def build_teacher_prompt(
    config: ExperimentConfig,
    observation: Observation,
    state: SimulatorState,
    state_id: str,
    prompt_version: str,
    num_candidates: int,
    verification: VerificationConfig | None = None,
) -> PromptArtifact:
    """Build a causal, semantically named teacher request.

    The raw observation remains the distillation input. This packet is only the
    teacher-facing explanation of the same causal information.
    """

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    causal = causal_observation_payload(observation)
    canonical_state = json.dumps(causal, sort_keys=True, separators=(",", ":"))
    state_hash = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
    sic_templates = build_sic_safe_templates(config, state)
    sensing_only_template = build_sensing_only_template(config)
    semantic_state_packet = build_semantic_state_packet(config, causal)
    optimization_contract = build_optimization_contract(config, verification)
    action_bank = {
        "sensing_only_template": sensing_only_template,
        "noma_templates": sic_templates,
        "deterministic_verifier_coverage": {
            "description": (
                "Every listed template is independently added to the verifier pool at several "
                "CPU fractions, together with a selectable urgency-greedy baseline."
            ),
            "teacher_role": (
                "Provide complementary state-specific refinements; do not use output ordering "
                "to implement template coverage."
            ),
        },
    }
    task = {
        "schema_version": 1,
        "prompt_version": prompt_version,
        "state_id": state_id,
        "num_pairs": config.system.num_noma_pairs,
        "num_candidates": num_candidates,
        "features": config.features.model_dump(),
        "physical_constraints": {
            "maximum_haps_power_w": config.haps.max_power_w,
            "minimum_sensing_power_w": config.constraints.minimum_sensing_power_w,
            "sic_sinr_threshold": config.constraints.sic_sinr_threshold,
            "secrecy_rate_target_bps": config.constraints.secrecy_rate_target_bps,
            "sensing_sinr_threshold": config.constraints.sensing_sinr_threshold,
            "maximum_covariance_trace": config.constraints.maximum_covariance_trace,
        },
        "optimization_contract": optimization_contract,
        "causal_state": semantic_state_packet,
        "action_bank": action_bank,
    }
    instructions = (
        f"Propose exactly {num_candidates} distinct high-level action refinements for the "
        "supplied causal HAPS-ISAC state. The verifier minimizes the supplied "
        "risk-sensitive objective, so use the named state features, objective weights, "
        "long-term deficits, and rollout contract rather than treating vector positions "
        "as anonymous numbers. For each candidate, choose template_id by copying one "
        "exact string from action_bank: do not invent aliases, suffixes, or descriptions. "
        "Return only the free refinements eta_near and eta_cpu; the verifier reconstructs "
        "pair, eta_haps, eta_communication, ris_code, eta_jamming, heading, and speed "
        "from template_id. For pair=0, eta_near is ignored and the sensing-only value is "
        "used. For pair>0, eta_near must be in [0, maximum_eta_near] and eta_cpu in [0,1]. "
        "Do not put template_id in reason_codes; reason_codes are short explanatory text. "
        "The verifier already evaluates every safe template and a greedy baseline, so "
        "use these candidates for state-specific choices of template, eta_near, and "
        "eta_cpu rather than reserving positions for coverage. Return only JSON with "
        "schema_version=1, the exact state_id, and a candidates array. Each candidate "
        "must contain template_id, eta_near, eta_cpu, reason_codes, and confidence in [0,1]."
    )
    prompt = f"{instructions}\nINPUT_JSON={json.dumps(task, sort_keys=True)}"
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return PromptArtifact(
        state_id=state_id,
        prompt_version=prompt_version,
        prompt=prompt,
        prompt_hash=prompt_hash,
        causal_state_hash=state_hash,
        causal_payload=causal,
        semantic_state_packet=semantic_state_packet,
        optimization_contract=optimization_contract,
        sic_safe_templates=sic_templates,
        sensing_only_template=sensing_only_template,
    )
