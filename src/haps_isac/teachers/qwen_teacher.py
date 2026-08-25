"""Qwen adapter for a local OpenAI-compatible inference server."""

from __future__ import annotations

from typing import Any

from haps_isac.teachers.base_teacher import OpenAICompatibleTeacher


class QwenTeacher(OpenAICompatibleTeacher):
    """Enable Qwen thinking separation while requiring final structured JSON."""

    def request_overrides(self) -> dict[str, Any]:
        return {
            "top_k": self.config.sampling.top_k,
            "chat_template_kwargs": {"enable_thinking": True},
        }
