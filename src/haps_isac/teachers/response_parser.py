"""Strict teacher response parsing and bounded action validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from haps_isac.control.action_schema import HighLevelAction


class CandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    template_id: str = Field(min_length=1, max_length=64)
    eta_near: float = Field(ge=0.0, le=0.5)
    eta_cpu: float = Field(ge=0.0, le=1.0)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=3)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Accepted for transport compatibility; canonicalization ignores these
    # template-controlled values and reconstructs them from ``template_id``.
    pair: int | None = Field(default=None, ge=0)
    ris_code: int | None = Field(default=None, ge=0)
    eta_haps: float | None = Field(default=None, ge=0.0, le=1.0)
    eta_communication: float | None = Field(default=None, ge=0.0, le=1.0)
    eta_jamming: float | None = Field(default=None, ge=0.0, le=1.0)
    aav_heading_rad: float | None = Field(
        default=None, ge=-3.141592653589793, le=3.141592653589793
    )
    aav_speed_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class ResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    state_id: str
    candidates: tuple[CandidatePayload, ...]


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    candidate_index: int
    action: HighLevelAction
    reason_codes: tuple[str, ...]
    confidence: float
    canonical_key: tuple[int | float, ...]
    template_id: str | None = None
    template_id_raw: str | None = None
    template_resolution: str = "raw"
    source: str = "teacher"
    source_label: str = "teacher_response"


@dataclass(frozen=True, slots=True)
class ParsedTeacherResponse:
    schema_version: int
    state_id: str
    candidates: tuple[ParsedCandidate, ...]
    normalization_notes: tuple[str, ...] = ()

    @property
    def unique_candidate_count(self) -> int:
        return len({candidate.canonical_key for candidate in self.candidates})


class TeacherResponseError(ValueError):
    """The teacher response is unusable as a verified candidate set."""


def _normalize_payload(value: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply only bounded repairs for recurring provider formatting mistakes."""

    normalized = dict(value)
    raw_candidates = normalized.get("candidates")
    if not isinstance(raw_candidates, (list, tuple)):
        return normalized, ()
    notes: list[str] = []
    candidates: list[Any] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            candidates.append(raw_candidate)
            continue
        candidate = dict(raw_candidate)
        if "template_id_actual" in candidate:
            # The provider has occasionally emitted a correction under
            # ``template_id_actual``.  Prefer that explicitly named corrected
            # value, while retaining the unmodified response in the request log.
            actual = candidate.pop("template_id_actual")
            if "template_id" not in candidate:
                candidate["template_id"] = actual
                notes.append(f"candidate[{index}].template_id_actual->template_id")
            elif candidate["template_id"] == actual:
                notes.append(f"candidate[{index}].template_id_actual_duplicate_removed")
            else:
                candidate["template_id"] = actual
                notes.append(f"candidate[{index}].template_id_conflict_prefer_actual")
        if isinstance(candidate.get("reason_codes"), str):
            candidate["reason_codes"] = [candidate["reason_codes"]]
            notes.append(f"candidate[{index}].reason_codes_string_to_array")
        if "reason_codes_override" in candidate:
            del candidate["reason_codes_override"]
            notes.append(f"candidate[{index}].reason_codes_override_removed")
        candidates.append(candidate)
    normalized["candidates"] = candidates
    return normalized, tuple(notes)


def _extract_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        final_fence = text.rfind("```")
        if first_newline >= 0 and final_fence > first_newline:
            text = text[first_newline + 1 : final_fence].strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise TeacherResponseError("response contains no valid JSON object")


def parse_teacher_response(
    raw_text: str,
    expected_state_id: str,
    expected_candidates: int,
    num_pairs: int,
) -> ParsedTeacherResponse:
    raw_payload = _extract_object(raw_text)
    normalized_payload, normalization_notes = _normalize_payload(raw_payload)
    try:
        payload = ResponsePayload.model_validate(normalized_payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise TeacherResponseError(str(error)) from error
    if payload.schema_version != 1:
        raise TeacherResponseError("unsupported teacher response schema")
    if payload.state_id != expected_state_id:
        raise TeacherResponseError("teacher response state_id does not match the request")
    if len(payload.candidates) != expected_candidates:
        raise TeacherResponseError(
            f"expected {expected_candidates} candidates, received {len(payload.candidates)}"
        )

    _ = num_pairs  # retained in the public API for caller compatibility
    raw_candidate_values = raw_payload.get("candidates")
    raw_template_ids = (
        tuple(
            raw_candidate.get("template_id")
            if isinstance(raw_candidate, dict)
            else None
            for raw_candidate in raw_candidate_values
        )
        if isinstance(raw_candidate_values, (list, tuple))
        else ()
    )
    parsed: list[ParsedCandidate] = []
    for index, item in enumerate(payload.candidates):
        action = HighLevelAction(
            pair=0,
            ris_code=0,
            eta_haps=0.0,
            eta_communication=0.0,
            eta_near=item.eta_near,
            eta_jamming=0.0,
            aav_heading_rad=0.0,
            aav_speed_fraction=0.0,
            eta_cpu=item.eta_cpu,
        )
        canonical = (
            action.pair,
            action.ris_code,
            *tuple(round(value, 8) for value in action.continuous_vector()),
        )
        parsed.append(
            ParsedCandidate(
                candidate_index=index,
                action=action,
                reason_codes=item.reason_codes,
                confidence=item.confidence,
                canonical_key=canonical,
                template_id=item.template_id,
                template_id_raw=(
                    raw_template_ids[index]
                    if index < len(raw_template_ids)
                    else item.template_id
                ),
            )
        )
    return ParsedTeacherResponse(
        schema_version=payload.schema_version,
        state_id=payload.state_id,
        candidates=tuple(parsed),
        normalization_notes=normalization_notes,
    )
