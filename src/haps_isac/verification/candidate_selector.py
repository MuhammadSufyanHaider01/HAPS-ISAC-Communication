"""Risk-sensitive candidate ranking, uncertainty, and quality weights."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from haps_isac.teachers.base_teacher import VerificationConfig
from haps_isac.verification.candidate_evaluator import OneStepEvaluation
from haps_isac.verification.rollout_verifier import (
    CandidateRolloutSummary,
    risk_score_from_rollouts,
)


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    candidate_index: int
    rank: int
    risk_score: float
    quality_weight: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_candidate_index: int
    rankings: tuple[CandidateRanking, ...]
    score_margin: float
    standardized_margin: float
    selection_uncertain: bool
    margin_confidence_lower: float
    margin_confidence_upper: float
    selection_probability: float


def _paired_risk_bootstrap(
    best: CandidateRolloutSummary,
    second: CandidateRolloutSummary,
    settings: VerificationConfig,
) -> tuple[float, float, float, float]:
    best_seeds = tuple(item.rollout_seed for item in best.rollouts)
    second_seeds = tuple(item.rollout_seed for item in second.rollouts)
    if best_seeds != second_seeds:
        raise ValueError("candidate rollouts must use aligned common random seeds")
    if not best_seeds:
        raise ValueError("candidate summaries must retain rollout records")

    seed_material = f"{best.candidate_index}:{second.candidate_index}:" + ",".join(
        str(seed) for seed in best_seeds
    )
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
        "big",
    )
    generator = np.random.default_rng(bootstrap_seed)
    sample_indices = generator.integers(
        0,
        len(best_seeds),
        size=(settings.uncertainty_bootstrap_samples, len(best_seeds)),
    )
    margins = np.empty(settings.uncertainty_bootstrap_samples, dtype=np.float64)
    for sample_index, indices in enumerate(sample_indices):
        best_sample = tuple(best.rollouts[int(index)] for index in indices)
        second_sample = tuple(second.rollouts[int(index)] for index in indices)
        margins[sample_index] = risk_score_from_rollouts(
            second_sample,
            settings,
        ) - risk_score_from_rollouts(best_sample, settings)

    alpha = (1.0 - settings.uncertainty_confidence_level) / 2.0
    lower = float(np.quantile(margins, alpha))
    upper = float(np.quantile(margins, 1.0 - alpha))
    probability = float(np.mean(margins > 0.0))
    standard_error = float(np.std(margins, ddof=1))
    return lower, upper, probability, standard_error


def select_candidate(
    one_step: dict[int, OneStepEvaluation],
    summaries: tuple[CandidateRolloutSummary, ...],
    settings: VerificationConfig,
) -> SelectionResult:
    quality_temperature = settings.quality_temperature
    if quality_temperature <= 0.0:
        raise ValueError("quality_temperature must be positive")
    if not summaries:
        raise ValueError("no verified candidates were provided")
    for summary in summaries:
        if summary.candidate_index not in one_step:
            raise ValueError("missing one-step evaluation for a rollout candidate")

    ordered = sorted(
        summaries,
        key=lambda item: (
            not one_step[item.candidate_index].hard_feasible,
            item.risk_score,
            one_step[item.candidate_index].repair_distance,
            item.candidate_index,
        ),
    )
    scores = np.asarray([item.risk_score for item in ordered], dtype=np.float64)
    logits = -(scores - float(np.min(scores))) / quality_temperature
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    weights /= float(np.sum(weights))

    if len(ordered) == 1:
        margin = math.inf
        standardized = math.inf
        confidence_lower = math.inf
        confidence_upper = math.inf
        selection_probability = 1.0
        uncertain = False
    else:
        margin = ordered[1].risk_score - ordered[0].risk_score
        (
            confidence_lower,
            confidence_upper,
            selection_probability,
            bootstrap_standard_error,
        ) = _paired_risk_bootstrap(ordered[0], ordered[1], settings)
        standardized = (
            margin / bootstrap_standard_error if bootstrap_standard_error > 0.0 else math.inf
        )
        uncertain = confidence_lower <= 0.0

    rankings = tuple(
        CandidateRanking(
            candidate_index=item.candidate_index,
            rank=rank + 1,
            risk_score=item.risk_score,
            quality_weight=float(weights[rank]),
        )
        for rank, item in enumerate(ordered)
    )
    return SelectionResult(
        selected_candidate_index=ordered[0].candidate_index,
        rankings=rankings,
        score_margin=float(margin),
        standardized_margin=float(standardized),
        selection_uncertain=uncertain,
        margin_confidence_lower=float(confidence_lower),
        margin_confidence_upper=float(confidence_upper),
        selection_probability=selection_probability,
    )
