"""Versioned deterministic construction of causal teacher prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig

Observation = dict[str, npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    state_id: str
    prompt_version: str
    prompt: str
    prompt_hash: str
    causal_state_hash: str
    causal_payload: dict[str, Any]


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


def build_teacher_prompt(
    config: ExperimentConfig,
    observation: Observation,
    state_id: str,
    prompt_version: str,
    num_candidates: int,
) -> PromptArtifact:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    causal = causal_observation_payload(observation)
    canonical_state = json.dumps(causal, sort_keys=True, separators=(",", ":"))
    state_hash = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
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
        "causal_observation": causal,
    }
    instructions = (
        f"Propose exactly {num_candidates} diverse high-level actions for the supplied "
        "causal HAPS-ISAC state. pair is 0 (sensing only) or "
        f"1..{config.system.num_noma_pairs}. All fractions are "
        "bounded: eta_haps, eta_communication, eta_jamming, aav_speed_fraction and "
        "eta_cpu in [0,1], eta_near in [0,0.5], heading in [-pi,pi]. Version 1 has "
        "ris_code=0, eta_jamming=0, heading=0 and speed=0. Favor freshness while "
        "respecting sensing, SIC, secrecy and long-term queues. Return only JSON with "
        "schema_version=1, the exact state_id, and a candidates array. Each candidate "
        "must contain pair, ris_code, eta_haps, eta_communication, eta_near, "
        "eta_jamming, aav_heading_rad, aav_speed_fraction, eta_cpu, reason_codes "
        "(short strings), and confidence in [0,1]."
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
    )
