"""Qwen adapter for a local OpenAI-compatible inference server."""

from __future__ import annotations

from typing import Any

from haps_isac.teachers.base_teacher import OpenAICompatibleTeacher


def qwen_chat_template_kwargs(body: dict[str, Any]) -> dict[str, bool]:
    """Validate the Qwen chat-template options accepted by the local server."""

    raw = body.get("chat_template_kwargs", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("chat_template_kwargs must be an object")
    enable_thinking = raw.get("enable_thinking", True)
    if not isinstance(enable_thinking, bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be boolean")
    return {"enable_thinking": enable_thinking}


def qwen_device_map(cuda_device_count: int) -> str | None:
    """Select weight placement without sharding a model that fits one GPU."""

    if cuda_device_count <= 0:
        raise ValueError("cuda_device_count must be positive")
    return "balanced_low_0" if cuda_device_count > 1 else None


class QwenTeacher(OpenAICompatibleTeacher):
    """Request concise structured Qwen output for offline action proposals."""

    def request_overrides(self) -> dict[str, Any]:
        return {
            "top_k": self.config.sampling.top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }
