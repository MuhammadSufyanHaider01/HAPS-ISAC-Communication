"""Coverage-complete, provenance-aware candidate construction for verification."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.action_transform import action_from_mapping
from haps_isac.teachers.base_teacher import CandidatePoolConfig
from haps_isac.teachers.prompt_builder import PromptArtifact
from haps_isac.teachers.response_parser import (
    ParsedCandidate,
    ParsedTeacherResponse,
    TeacherResponseError,
)

MAX_NEAREST_TEMPLATE_WATT_DISTANCE = 8.0


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
                    template_id=template_id,
                    template_id_raw=template_id,
                    template_resolution="exact",
                    source="safe_template",
                    source_label=f"{template_id}:cpu={cpu:.3f}",
                )
            )
    return tuple(candidates)


def _template_lookup(artifact: PromptArtifact) -> dict[str, dict[str, Any]]:
    templates = (artifact.sensing_only_template, *artifact.sic_safe_templates)
    return {str(template["template_id"]): template for template in templates}


def _template_base_matches(action: HighLevelAction, template: dict[str, Any]) -> bool:
    return (
        action.pair == int(template["pair"])
        and abs(action.eta_haps - float(template["eta_haps"])) <= 1.5e-6
        and abs(action.eta_communication - float(template["eta_communication"])) <= 1.5e-6
    )


def _normalize_template_identifier(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _nearest_available_template_id(
    raw_id: str,
    templates: dict[str, dict[str, Any]],
) -> str | None:
    match = re.fullmatch(r"p(\d+)_sense(\d+(?:\.\d+)?)w", _normalize_template_identifier(raw_id))
    if match is None:
        return None
    pair = int(match.group(1))
    sensing_watts = float(match.group(2))
    candidates: list[tuple[float, str]] = []
    for template_id, template in templates.items():
        if int(template["pair"]) != pair:
            continue
        template_match = re.fullmatch(
            r"p(\d+)_sense(\d+(?:\.\d+)?)w",
            _normalize_template_identifier(template_id),
        )
        if template_match is not None:
            candidates.append((abs(float(template_match.group(2)) - sensing_watts), template_id))
    if not candidates:
        return None
    distance, template_id = min(candidates, key=lambda item: (item[0], item[1]))
    return template_id if distance <= MAX_NEAREST_TEMPLATE_WATT_DISTANCE else None


def _resolve_template_id(
    candidate: ParsedCandidate,
    templates: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    normalized_id = (candidate.template_id or "").strip()
    raw_id = (candidate.template_id_raw or normalized_id).strip()
    if normalized_id in templates:
        resolution = (
            "exact"
            if not candidate.template_id_raw or candidate.template_id_raw == normalized_id
            else "normalized_template_id"
        )
        return normalized_id, resolution
    if raw_id in templates:
        return raw_id, "exact"
    for reason in candidate.reason_codes:
        if reason in templates:
            return reason, "reason_code_alias"
    normalized = _normalize_template_identifier(raw_id)
    for template_id in sorted(templates, key=len, reverse=True):
        if normalized.startswith(_normalize_template_identifier(template_id) + "_"):
            return template_id, "template_id_suffix_alias"
    for reason in candidate.reason_codes:
        normalized_reason = _normalize_template_identifier(reason)
        for template_id in sorted(templates, key=len, reverse=True):
            if normalized_reason.startswith(_normalize_template_identifier(template_id) + "_"):
                return template_id, "reason_code_suffix_alias"
    matches = [
        template_id
        for template_id, template in templates.items()
        if _template_base_matches(candidate.action, template)
    ]
    if len(matches) == 1:
        return matches[0], "numeric_action_alias"
    nearest = _nearest_available_template_id(raw_id, templates)
    if nearest is not None:
        return nearest, "nearest_available_template"
    return None, "unresolved"


def canonicalize_teacher_response(
    parsed: ParsedTeacherResponse,
    artifact: PromptArtifact,
) -> ParsedTeacherResponse:
    """Project teacher refinements onto the exact per-state template bank.

    ``template_id`` is authoritative for pair and fixed physical controls. Known
    aliases from older model behavior are resolved using the explicit id, reason
    codes, an unambiguous numeric template match, or a bounded nearest watt
    match within the same pair. Nearest matching is limited to the adjacent
    configured sensing-power level, never an arbitrary hallucinated wattage.
    The raw identifier and resolution method are
    retained in candidate logs so repaired model output cannot be mistaken for
    exact contract compliance.
    """

    templates = _template_lookup(artifact)
    canonical_candidates: list[ParsedCandidate] = []
    for candidate in parsed.candidates:
        template_id, template_resolution = _resolve_template_id(candidate, templates)
        if template_id is None:
            raise TeacherResponseError(
                f"candidate {candidate.candidate_index} has no resolvable template_id "
                f"({candidate.template_id_raw or candidate.template_id!r})"
            )
        template = templates[template_id]
        default_near = float(template["eta_near"]) if "eta_near" in template else 0.0
        maximum_near = float(template.get("maximum_eta_near", default_near))
        eta_near = (
            default_near
            if int(template["pair"]) == 0
            else min(float(candidate.action.eta_near), maximum_near)
        )
        action = HighLevelAction(
            pair=int(template["pair"]),
            ris_code=0,
            eta_haps=float(template["eta_haps"]),
            eta_communication=float(template["eta_communication"]),
            eta_near=eta_near,
            eta_jamming=0.0,
            aav_heading_rad=0.0,
            aav_speed_fraction=0.0,
            eta_cpu=float(candidate.action.eta_cpu),
        )
        canonical_candidates.append(
            ParsedCandidate(
                candidate_index=candidate.candidate_index,
                action=action,
                reason_codes=candidate.reason_codes,
                confidence=candidate.confidence,
                canonical_key=canonical_action_key(action),
                template_id=template_id,
                template_id_raw=candidate.template_id_raw,
                template_resolution=template_resolution,
                source=candidate.source,
                source_label=candidate.source_label,
            )
        )
    return ParsedTeacherResponse(
        schema_version=parsed.schema_version,
        state_id=parsed.state_id,
        candidates=tuple(canonical_candidates),
        normalization_notes=parsed.normalization_notes,
    )


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
                    template_id=candidate.template_id,
                    template_id_raw=candidate.template_id_raw,
                    template_resolution=candidate.template_resolution,
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
            template_id=None,
            template_id_raw=None,
            template_resolution="baseline",
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
    """Return the canonical template ID and exact Version-1 compliance."""

    if candidate.source in {"greedy_baseline", "random_baseline"}:
        return None, None
    templates = _template_lookup(artifact)
    template_id = candidate.template_id
    if template_id is None or template_id not in templates:
        return template_id, False
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
        matches = matches and abs(action.eta_near - default_near) <= tolerance
    return template_id, matches
