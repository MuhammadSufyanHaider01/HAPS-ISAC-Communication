"""Hybrid policy-action transforms and deterministic decoding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from haps_isac.control.action_schema import HighLevelAction, RawPolicyAction


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def transform_raw_action(action: RawPolicyAction) -> HighLevelAction:
    """Map the canonical seven unconstrained controls into physical bounds."""

    raw = action.continuous
    return HighLevelAction(
        pair=int(action.pair),
        ris_code=int(action.ris_code),
        eta_haps=_sigmoid(float(raw[0])),
        eta_communication=_sigmoid(float(raw[1])),
        eta_near=0.5 * _sigmoid(float(raw[2])),
        eta_jamming=_sigmoid(float(raw[3])),
        aav_heading_rad=math.pi * math.tanh(float(raw[4])),
        aav_speed_fraction=_sigmoid(float(raw[5])),
        eta_cpu=_sigmoid(float(raw[6])),
    )


def clip_high_level_action(
    action: HighLevelAction,
    num_pairs: int,
    ris_enabled: bool = False,
) -> HighLevelAction:
    """Clip continuous controls and expose invalid categories for fallback."""

    pair = action.pair if 0 <= action.pair <= num_pairs else -1
    ris_code = action.ris_code if ris_enabled else 0
    return HighLevelAction(
        pair=pair,
        ris_code=ris_code,
        eta_haps=float(np.clip(action.eta_haps, 0.0, 1.0)),
        eta_communication=float(np.clip(action.eta_communication, 0.0, 1.0)),
        eta_near=float(np.clip(action.eta_near, 0.0, 0.5)),
        eta_jamming=float(np.clip(action.eta_jamming, 0.0, 1.0)),
        aav_heading_rad=float(np.clip(action.aav_heading_rad, -math.pi, math.pi)),
        aav_speed_fraction=float(np.clip(action.aav_speed_fraction, 0.0, 1.0)),
        eta_cpu=float(np.clip(action.eta_cpu, 0.0, 1.0)),
    )


def action_from_mapping(value: Mapping[str, Any]) -> HighLevelAction:
    """Parse a transformed action mapping used by scripts and baselines."""

    continuous = np.asarray(value["continuous"], dtype=np.float64)
    if continuous.shape != (7,):
        raise ValueError("mapped continuous action must have shape (7,)")
    return HighLevelAction(
        pair=int(value["pair"]),
        ris_code=int(value.get("ris_code", 0)),
        eta_haps=float(continuous[0]),
        eta_communication=float(continuous[1]),
        eta_near=float(continuous[2]),
        eta_jamming=float(continuous[3]),
        aav_heading_rad=math.pi * float(continuous[4]),
        aav_speed_fraction=float(continuous[5]),
        eta_cpu=float(continuous[6]),
    )
