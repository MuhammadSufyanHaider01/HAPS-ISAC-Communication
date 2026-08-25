"""Risk-sensitive candidate ranking, uncertainty, and quality weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from haps_isac.verification.candidate_evaluator import OneStepEvaluation
from haps_isac.verification.rollout_verifier import CandidateRolloutSummary


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


def select_candidate(
    one_step: dict[int, OneStepEvaluation],
    summaries: tuple[CandidateRolloutSummary, ...],
    quality_temperature: float,
) -> SelectionResult:
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
        uncertain = False
    else:
        margin = ordered[1].risk_score - ordered[0].risk_score
        combined_error = math.sqrt(
            ordered[0].cost_standard_error ** 2 + ordered[1].cost_standard_error ** 2
        )
        standardized = margin / combined_error if combined_error > 0.0 else math.inf
        uncertain = bool(combined_error > 0.0 and standardized < 2.0)

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
    )
