"""Provider-independent offline teacher protocol and HTTP transport."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from haps_isac.teachers.query_cache import QueryCache, cache_key_for


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingConfig(FrozenModel):
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    presence_penalty: float = Field(ge=-2.0, le=2.0)


class VerificationConfig(FrozenModel):
    rollout_horizon_slots: int = Field(gt=0)
    monte_carlo_rollouts: int = Field(gt=0)
    shortlist_size: int = Field(gt=0)
    discount_factor: float = Field(gt=0.0, le=1.0)
    cvar_alpha: float = Field(gt=0.0, lt=1.0)
    cvar_weight: float = Field(ge=0.0)
    constraint_weight: float = Field(ge=0.0)
    repair_weight: float = Field(ge=0.0)
    fallback_weight: float = Field(ge=0.0)
    quality_temperature: float = Field(gt=0.0)


class LoggingConfig(FrozenModel):
    flush_every: int = Field(gt=0)
    retain_selected_trajectories: bool
    retain_failure_trajectories: bool
    trajectory_audit_fraction: float = Field(ge=0.0, le=1.0)
    export_parquet: bool


class TeacherConfig(FrozenModel):
    schema_version: int = Field(ge=1)
    provider: Literal["qwen", "gemma", "mock"]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    num_candidates: int = Field(gt=0, le=64)
    request_timeout_s: float = Field(gt=0.0, le=1800.0)
    max_retries: int = Field(ge=0, le=10)
    cache_enabled: bool
    cache_directory: str
    sampling: SamplingConfig
    verification: VerificationConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_shortlist(self) -> Self:
        if self.verification.shortlist_size > self.num_candidates:
            raise ValueError("shortlist_size cannot exceed num_candidates")
        return self


def load_teacher_config(path: str | Path) -> TeacherConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return TeacherConfig.model_validate(raw)


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    request_id: str
    state_id: str
    prompt: str
    prompt_hash: str
    seed: int


@dataclass(frozen=True, slots=True)
class TeacherCallResult:
    request_id: str
    cache_key: str
    raw_text: str
    reasoning_text: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    cached: bool
    retries: int
    status: str
    error: str | None
    response_body: dict[str, Any] | None


class BaseTeacher(ABC):
    """Small provider-independent interface used only by offline generation."""

    def __init__(self, config: TeacherConfig, cache: QueryCache | None = None) -> None:
        self.config = config
        self.cache = cache

    @abstractmethod
    def generate(self, request: TeacherRequest) -> TeacherCallResult:
        """Generate candidate actions or return a logged error result."""


class OpenAICompatibleTeacher(BaseTeacher):
    """Synchronous OpenAI-compatible chat-completions transport."""

    def request_overrides(self) -> dict[str, Any]:
        return {}

    def _request_body(self, request: TeacherRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You propose bounded numerical control candidates. "
                        "Return exactly one JSON object matching the requested schema."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": self.config.sampling.temperature,
            "top_p": self.config.sampling.top_p,
            "max_tokens": self.config.sampling.max_tokens,
            "presence_penalty": self.config.sampling.presence_penalty,
            "seed": request.seed,
            "response_format": {"type": "json_object"},
        }
        body.update(self.request_overrides())
        return body

    def generate(self, request: TeacherRequest) -> TeacherCallResult:
        body = self._request_body(request)
        key = cache_key_for(
            self.config.model_id,
            self.config.model_revision,
            request.prompt_hash,
            body,
            request.seed,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return TeacherCallResult(
                    request_id=request.request_id,
                    cache_key=key,
                    raw_text=str(cached["raw_text"]),
                    reasoning_text=(
                        str(cached["reasoning_text"])
                        if cached.get("reasoning_text") is not None
                        else None
                    ),
                    prompt_tokens=_optional_int(cached.get("prompt_tokens")),
                    completion_tokens=_optional_int(cached.get("completion_tokens")),
                    latency_s=0.0,
                    cached=True,
                    retries=0,
                    status="ok",
                    error=None,
                    response_body=None,
                )

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
        last_error: str | None = None
        started = time.perf_counter()
        for attempt in range(self.config.max_retries + 1):
            try:
                http_request = urllib.request.Request(
                    url,
                    data=encoded,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(
                    http_request,
                    timeout=self.config.request_timeout_s,
                ) as response:
                    response_body = json.loads(response.read().decode("utf-8"))
                raw_text, reasoning_text, prompt_tokens, completion_tokens = (
                    _extract_chat_completion(response_body)
                )
                latency = time.perf_counter() - started
                if self.cache is not None:
                    self.cache.put(
                        key,
                        {
                            "raw_text": raw_text,
                            "reasoning_text": reasoning_text,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    )
                return TeacherCallResult(
                    request_id=request.request_id,
                    cache_key=key,
                    raw_text=raw_text,
                    reasoning_text=reasoning_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_s=latency,
                    cached=False,
                    retries=attempt,
                    status="ok",
                    error=None,
                    response_body=response_body,
                )
            except (
                OSError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < self.config.max_retries:
                    time.sleep(min(8.0, float(2**attempt)))
        return TeacherCallResult(
            request_id=request.request_id,
            cache_key=key,
            raw_text="",
            reasoning_text=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_s=time.perf_counter() - started,
            cached=False,
            retries=self.config.max_retries,
            status="error",
            error=last_error,
            response_body=None,
        )


class MockTeacher(BaseTeacher):
    """Deterministic local teacher used to validate the complete pipeline."""

    def __init__(
        self,
        config: TeacherConfig,
        num_pairs: int,
        cache: QueryCache | None = None,
    ) -> None:
        super().__init__(config, cache)
        self.num_pairs = num_pairs

    def generate(self, request: TeacherRequest) -> TeacherCallResult:
        candidates: list[dict[str, Any]] = []
        for index in range(self.config.num_candidates):
            pair = 1 + ((request.seed + index) % self.num_pairs)
            sensing_fraction = 0.2 + 0.05 * (index % 5)
            candidates.append(
                {
                    "pair": pair,
                    "ris_code": 0,
                    "eta_haps": min(1.0, 0.65 + 0.04 * index),
                    "eta_communication": 1.0 - sensing_fraction,
                    "eta_near": 0.1 + 0.04 * (index % 5),
                    "eta_jamming": 0.0,
                    "aav_heading_rad": 0.0,
                    "aav_speed_fraction": 0.0,
                    "eta_cpu": 0.5 + 0.05 * (index % 6),
                    "reason_codes": ["mock_diverse_candidate"],
                    "confidence": 0.5,
                }
            )
        raw = json.dumps(
            {
                "schema_version": 1,
                "state_id": request.state_id,
                "candidates": candidates,
            },
            sort_keys=True,
        )
        key = cache_key_for(
            self.config.model_id,
            self.config.model_revision,
            request.prompt_hash,
            self.config.sampling.model_dump(),
            request.seed,
        )
        return TeacherCallResult(
            request_id=request.request_id,
            cache_key=key,
            raw_text=raw,
            reasoning_text=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_s=0.0,
            cached=False,
            retries=0,
            status="ok",
            error=None,
            response_body=None,
        )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _extract_chat_completion(
    body: Any,
) -> tuple[str, str | None, int | None, int | None]:
    if not isinstance(body, dict):
        raise ValueError("teacher response body must be an object")
    choices = body["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("teacher response contains no choices")
    message = choices[0]["message"]
    content = message["content"]
    if not isinstance(content, str):
        raise ValueError("teacher response content must be text")
    reasoning = message.get("reasoning_content")
    usage = body.get("usage", {})
    return (
        content,
        str(reasoning) if reasoning is not None else None,
        _optional_int(usage.get("prompt_tokens")),
        _optional_int(usage.get("completion_tokens")),
    )
