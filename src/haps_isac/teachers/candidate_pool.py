"""Coverage-complete, provenance-aware candidate construction for verification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.action_transform import action_from_mapping
from haps_isac.teachers.base_teacher import CandidatePoolConfig
from haps_isac.teachers.prompt_builder import PromptArtifact
from haps_isac.teachers.response_parser import ParsedCandidate, ParsedTeacherResponse


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Selection pool plus immutable diagnostics for one teacher request."""

    candidates: tuple[ParsedCandidate, ...]
    teacher_candidate_count: int
    safe_template_expected_count: int
    safe_template_coverage_rate: float
    source_counts: dict[str, int]
    greedy_candidate_index: int | None


def canonical_action_key(action: HighLevelAction) -> tuple[int | float, ...]:
    return (
        action.pair,
        action.ris_code,
        *tuple(round(float(value), 8) for value in action.continuous_vector()),
    )


def _safe_template_candidates(
    artifact: PromptArtifact,
    cpu_fractions: tuple[float, ...],
    start_index: int,
) -> tuple[ParsedCandidate, ...]:
    templates = (artifact.sensing_only_template, *artifact.sic_safe_templates)
    candidates: list[ParsedCandidate] = []
    for template in templates:
        template_id = str(template["template_id"])
        default_near = float(template["eta_near"]) if "eta_near" in template else 0.0
        maximum_near = float(template.get("maximum_eta_near", default_near))
        recommended_near = float(template.get("recommended_eta_near", default_near))
        for cpu in cpu_fractions:
            action = HighLevelAction(
                pair=int(template["pair"]),
                ris_code=0,
                eta_haps=float(template["eta_haps"]),
                eta_communication=float(template["eta_communication"]),
                eta_near=min(recommended_near, maximum_near),
                eta_jamming=0.0,
                aav_heading_rad=0.0,
                aav_speed_fraction=0.0,
                eta_cpu=float(cpu),
            )
            candidates.append(
                ParsedCandidate(
                    candidate_index=start_index + len(candidates),
                    action=action,
                    reason_codes=("deterministic_safe_template", template_id),
                    confidence=0.0,
                    canonical_key=canonical_action_key(action),
                    source="safe_template",
                    source_label=f"{template_id}:cpu={cpu:.3f}",
                )
            )
    return tuple(candidates)


def build_candidate_pool(
    parsed: ParsedTeacherResponse,
    artifact: PromptArtifact,
    observation: dict[str, np.ndarray],
    settings: CandidatePoolConfig,
) -> CandidatePool:
    """Combine teacher proposals with complete safe coverage and a greedy floor."""

    candidates = list(parsed.candidates)
    seen = {candidate.canonical_key for candidate in candidates}
    expected_safe: tuple[ParsedCandidate, ...] = ()
    if settings.include_safe_template_bank:
        expected_safe = _safe_template_candidates(
            artifact,
            settings.template_cpu_fractions,
            len(candidates),
        )
        for candidate in expected_safe:
            if candidate.canonical_key in seen:
                continue
            candidates.append(
                ParsedCandidate(
                    candidate_index=len(candidates),
                    action=candidate.action,
                    reason_codes=candidate.reason_codes,
                    confidence=candidate.confidence,
                    canonical_key=candidate.canonical_key,
                    source=candidate.source,
                    source_label=candidate.source_label,
                )
            )
            seen.add(candidate.canonical_key)

    greedy_candidate_index: int | None = None
    if settings.include_selectable_greedy_baseline:
        greedy_action = action_from_mapping(
            GreedyPolicy(len(observation["pair_mask"])).act(observation)
        )
        greedy = ParsedCandidate(
            candidate_index=len(candidates),
            action=greedy_action,
            reason_codes=("greedy_selectable_baseline",),
            confidence=0.0,
            canonical_key=canonical_action_key(greedy_action),
            source="greedy_baseline",
            source_label="urgency_greedy",
        )
        candidates.append(greedy)
        greedy_candidate_index = greedy.candidate_index

    expected_keys = {candidate.canonical_key for candidate in expected_safe}
    actual_keys = {candidate.canonical_key for candidate in candidates}
    coverage = len(expected_keys.intersection(actual_keys)) / max(1, len(expected_keys))
    source_counts = dict(sorted(Counter(candidate.source for candidate in candidates).items()))
    return CandidatePool(
        candidates=tuple(candidates),
        teacher_candidate_count=len(parsed.candidates),
        safe_template_expected_count=len(expected_safe),
        safe_template_coverage_rate=coverage,
        source_counts=source_counts,
        greedy_candidate_index=greedy_candidate_index,
    )


def template_compliance(
    candidate: ParsedCandidate,
    artifact: PromptArtifact,
) -> tuple[str | None, bool | None]:
    """Return referenced template ID and exact Version-1 template compliance."""

    if candidate.source == "greedy_baseline":
        return None, None
    templates = {
        str(template["template_id"]): template
        for template in (artifact.sensing_only_template, *artifact.sic_safe_templates)
    }
    template_id = next((code for code in candidate.reason_codes if code in templates), None)
    if template_id is None:
        return None, False
    template = templates[template_id]
    action = candidate.action
    tolerance = 1.5e-6
    default_near = float(template["eta_near"]) if "eta_near" in template else 0.0
    maximum_near = float(template.get("maximum_eta_near", default_near))
    matches = (
        action.pair == int(template["pair"])
        and abs(action.eta_haps - float(template["eta_haps"])) <= tolerance
        and abs(action.eta_communication - float(template["eta_communication"])) <= tolerance
        and action.eta_near <= maximum_near + tolerance
        and action.ris_code == 0
        and abs(action.eta_jamming) <= tolerance
        and abs(action.aav_heading_rad) <= tolerance
        and abs(action.aav_speed_fraction) <= tolerance
    )
    if int(template["pair"]) == 0:
        matches = matches and abs(action.eta_near - float(template["eta_near"])) <= tolerance
    return template_id, matches
