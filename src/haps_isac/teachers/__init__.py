"""Offline teacher adapters and strict candidate-response contracts."""

from haps_isac.teachers.base_teacher import (
    BaseTeacher,
    MockTeacher,
    TeacherCallResult,
    TeacherConfig,
    TeacherRequest,
    load_teacher_config,
)
from haps_isac.teachers.gemma_teacher import GemmaTeacher
from haps_isac.teachers.qwen_teacher import QwenTeacher

__all__ = [
    "BaseTeacher",
    "GemmaTeacher",
    "MockTeacher",
    "QwenTeacher",
    "TeacherCallResult",
    "TeacherConfig",
    "TeacherRequest",
    "load_teacher_config",
]
