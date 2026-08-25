"""Deterministic scenario-level dataset split assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0.0 for value in values):
            raise ValueError("split fractions must be non-negative")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to one")


DEFAULT_SPLIT_FRACTIONS = SplitFractions()


def assign_split(
    scenario_id: str,
    master_seed: int,
    fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS,
) -> str:
    """Assign every state from one scenario to the same stable split."""

    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    digest = hashlib.sha256(f"{master_seed}:{scenario_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < fractions.train:
        return "train"
    if unit < fractions.train + fractions.validation:
        return "validation"
    return "test"
