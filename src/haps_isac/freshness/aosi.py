"""Delayed accepted-estimate age-of-sensing-information recursion."""

from __future__ import annotations


def update_aosi(
    current_age: int,
    next_slot: int,
    accepted_timestamp: int | None,
    cap_slots: int,
) -> int:
    """Update AoSI at a slot boundary using only newly available estimates."""

    if current_age < 1 or next_slot < 0 or cap_slots <= 1:
        raise ValueError("invalid AoSI state")
    if accepted_timestamp is None:
        return min(current_age + 1, cap_slots)
    if accepted_timestamp > next_slot:
        raise ValueError("accepted timestamp cannot be in the future")
    return min(max(1, next_slot - accepted_timestamp), cap_slots)
