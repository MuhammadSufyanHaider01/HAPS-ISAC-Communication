"""Versioned deterministic construction of causal teacher prompts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig
from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.physics_completion import complete_action
from haps_isac.envs.state import SimulatorState
from haps_isac.physics.channels import effective_beam_gain
from haps_isac.units import thermal_noise_watts

Observation = dict[str, npt.NDArray[np.generic]]
SIC_TEMPLATE_SENSING_FRACTIONS = (0.05, 0.25, 0.45)
SIC_TEMPLATE_NEAR_FRACTION = 0.20
SIC_TEMPLATE_POWER_MARGIN = 1.15


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    state_id: str
    prompt_version: str
    prompt: str
    prompt_hash: str
    causal_state_hash: str
    causal_payload: dict[str, Any]
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
) -> PromptArtifact:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    causal = causal_observation_payload(observation)
    canonical_state = json.dumps(causal, sort_keys=True, separators=(",", ":"))
    state_hash = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
    sic_templates = build_sic_safe_templates(config, state)
    sensing_only_template = build_sensing_only_template(config)
    task = {
        "schema_version": 1,
        "prompt_version": prompt_version,
        "state_id": state_id,
        "num_pairs": config.system.num_noma_pairs,
        "num_candidates": num_candidates,
        "features": config.features.model_dump(),
        "constraints": {
            "maximum_haps_power_w": config.haps.max_power_w,
            "minimum_sensing_power_w": config.constraints.minimum_sensing_power_w,
            "sic_sinr_threshold": config.constraints.sic_sinr_threshold,
            "secrecy_rate_target_bps": config.constraints.secrecy_rate_target_bps,
            "sensing_sinr_threshold": config.constraints.sensing_sinr_threshold,
            "maximum_covariance_trace": config.constraints.maximum_covariance_trace,
        },
        "sic_safe_templates": sic_templates,
        "sensing_only_template": sensing_only_template,
        "causal_observation": causal,
    }
    instructions = (
        f"Propose exactly {num_candidates} diverse high-level actions for the supplied "
        "causal HAPS-ISAC state. pair is 0 (sensing only) or "
        f"1..{config.system.num_noma_pairs}. All fractions are "
        "bounded: eta_haps, eta_communication, eta_jamming, aav_speed_fraction and "
        "eta_cpu in [0,1], eta_near in [0,0.5], heading in [-pi,pi]. For Version 1, "
        "every candidate MUST set the exact values ris_code=0, eta_jamming=0.0, "
        "aav_heading_rad=0.0 and aav_speed_fraction=0.0; there are no exceptions. "
        "Return exactly one sensing-only candidate copied from sensing_only_template: "
        "copy its pair, eta_haps, eta_communication, and eta_near exactly and put its "
        "template_id in reason_codes. Every other candidate MUST be a NOMA action copied from one "
        "entry in sic_safe_templates: copy that entry's pair, eta_haps, and "
        "eta_communication exactly, set eta_near no higher than maximum_eta_near, and "
        "put its template_id in reason_codes. Use distinct templates when possible and "
        "cover every available pair before repeating one. These templates already "
        "satisfy sensing_power_w=eta_haps*(1-eta_communication)*maximum_haps_power_w, "
        "the minimum sensing constraint, and estimated-channel SIC with a safety "
        "margin. Favor freshness while respecting sensing, SIC, secrecy and long-term "
        "queues. Return only JSON with schema_version=1, the exact "
        "state_id, and a candidates array. Each candidate must contain pair, ris_code, "
        "eta_haps, eta_communication, eta_near, eta_jamming, aav_heading_rad, "
        "aav_speed_fraction, eta_cpu, reason_codes (short strings), and confidence "
        "in [0,1]."
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
        sic_safe_templates=sic_templates,
        sensing_only_template=sensing_only_template,
    )
