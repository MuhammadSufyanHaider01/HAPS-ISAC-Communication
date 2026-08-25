"""Gemma adapter for a local OpenAI-compatible inference server."""

from __future__ import annotations

from typing import Any

from haps_isac.teachers.base_teacher import OpenAICompatibleTeacher


class GemmaTeacher(OpenAICompatibleTeacher):
    """Gemma uses the common transport and explicit top-k sampling."""

    def request_overrides(self) -> dict[str, Any]:
        return {"top_k": self.config.sampling.top_k}
