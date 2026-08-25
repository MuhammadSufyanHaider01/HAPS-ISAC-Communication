"""Strict teacher response parsing and bounded action validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from haps_isac.control.action_schema import HighLevelAction


class CandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pair: int = Field(ge=0)
    ris_code: int = 0
    eta_haps: float = Field(ge=0.0, le=1.0)
    eta_communication: float = Field(ge=0.0, le=1.0)
    eta_near: float = Field(ge=0.0, le=0.5)
    eta_jamming: float = Field(ge=0.0, le=1.0)
    aav_heading_rad: float = Field(ge=-3.141592653589793, le=3.141592653589793)
    aav_speed_fraction: float = Field(ge=0.0, le=1.0)
    eta_cpu: float = Field(ge=0.0, le=1.0)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=12)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


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


@dataclass(frozen=True, slots=True)
class ParsedTeacherResponse:
    schema_version: int
    state_id: str
    candidates: tuple[ParsedCandidate, ...]

    @property
    def unique_candidate_count(self) -> int:
        return len({candidate.canonical_key for candidate in self.candidates})


class TeacherResponseError(ValueError):
    """The teacher response is unusable as a verified candidate set."""


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
    try:
        payload = ResponsePayload.model_validate(_extract_object(raw_text))
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

    parsed: list[ParsedCandidate] = []
    for index, item in enumerate(payload.candidates):
        if item.pair > num_pairs:
            raise TeacherResponseError(f"candidate {index} has an invalid pair")
        action = HighLevelAction(
            pair=item.pair,
            ris_code=item.ris_code,
            eta_haps=item.eta_haps,
            eta_communication=item.eta_communication,
            eta_near=item.eta_near,
            eta_jamming=item.eta_jamming,
            aav_heading_rad=item.aav_heading_rad,
            aav_speed_fraction=item.aav_speed_fraction,
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
            )
        )
    return ParsedTeacherResponse(
        schema_version=payload.schema_version,
        state_id=payload.state_id,
        candidates=tuple(parsed),
    )
