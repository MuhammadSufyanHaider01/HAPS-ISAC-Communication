"""Integrity and teacher-quality audit for generated demonstration datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from haps_isac.data.dataset_loader import DatasetLoader


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, int]
    metrics: dict[str, float]


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([float(record[key]) for record in records]))


def _quantile(records: list[dict[str, Any]], key: str, probability: float) -> float:
    if not records:
        return 0.0
    return float(np.quantile([float(record[key]) for record in records], probability))


def audit_dataset(directory: str) -> DatasetAudit:
    loader = DatasetLoader(directory)
    errors: list[str] = []
    warnings: list[str] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "states",
        "teacher_requests",
        "candidates",
        "rollouts",
        "selections",
        "demonstrations",
    ):
        path = loader.directory / f"{name}.jsonl"
        if path.exists():
            tables[name] = list(loader.iter_table(name))
        else:
            tables[name] = []
            errors.append(f"missing canonical table: {name}.jsonl")

    states = tables["states"]
    requests = tables["teacher_requests"]
    candidates = tables["candidates"]
    rollouts = tables["rollouts"]
    selections = tables["selections"]
    demonstrations = tables["demonstrations"]
    state_ids = {str(record["state_id"]) for record in states}

    for table_name, records in tables.items():
        orphaned = {
            str(record["state_id"])
            for record in records
            if "state_id" in record and str(record["state_id"]) not in state_ids
        }
        if orphaned:
            errors.append(f"{table_name} contains {len(orphaned)} orphaned state IDs")

    valid_requests = [record for record in requests if record["schema_valid"]]
    expected_candidates = int(loader.manifest["num_candidates"])
    candidates_by_state: dict[str, int] = {}
    for record in candidates:
        state_id = str(record["state_id"])
        candidates_by_state[state_id] = candidates_by_state.get(state_id, 0) + 1
    for request in valid_requests:
        state_id = str(request["state_id"])
        if candidates_by_state.get(state_id, 0) != expected_candidates:
            errors.append(
                f"{state_id} has {candidates_by_state.get(state_id, 0)} candidates; "
                f"expected {expected_candidates}"
            )

    if any(not bool(record["hard_feasible"]) for record in candidates):
        errors.append("at least one candidate remained hard-infeasible after repair")
    if len(selections) != len(states):
        errors.append("every state must have exactly one selection/rejection record")

    selected_keys = {
        (str(record["state_id"]), int(record["candidate_index"]))
        for record in candidates
        if record["selected"]
    }
    for record in demonstrations:
        key = (str(record["state_id"]), int(record["selected_candidate_index"]))
        if key not in selected_keys:
            errors.append(f"demonstration {key[0]} does not reference a selected candidate")

    scenario_splits: dict[str, set[str]] = {}
    for record in states:
        scenario_splits.setdefault(str(record["scenario_id"]), set()).add(str(record["split"]))
    leaking = [scenario for scenario, splits in scenario_splits.items() if len(splits) > 1]
    if leaking:
        errors.append(f"{len(leaking)} scenarios cross dataset splits")

    parse_rate = (
        float(np.mean([record["schema_valid"] for record in requests])) if requests else 0.0
    )
    unique_ratio = (
        float(
            np.mean(
                [
                    float(record["unique_candidates"])
                    / max(1.0, float(record["candidates_returned"]))
                    for record in valid_requests
                ]
            )
        )
        if valid_requests
        else 0.0
    )
    candidate_hard_rate = (
        float(np.mean([record["hard_feasible"] for record in candidates])) if candidates else 0.0
    )
    pre_repair_rate = (
        float(np.mean([record["pre_repair_feasible"] for record in candidates]))
        if candidates
        else 0.0
    )
    fallback_rate = (
        float(np.mean([record["fallback_used"] for record in candidates])) if candidates else 0.0
    )
    accepted_selections = [
        record for record in selections if record["acceptance_status"] == "accepted"
    ]
    uncertain_rate = (
        float(np.mean([record["selection_uncertain"] for record in accepted_selections]))
        if accepted_selections
        else 0.0
    )
    rollout_hard_rate = (
        float(np.mean([record["metrics"]["hard_feasible"] for record in rollouts]))
        if rollouts
        else 0.0
    )
    retained_rate = (
        float(np.mean([record["retained_trajectory"] for record in rollouts])) if rollouts else 0.0
    )
    completion_tokens = [
        float(record["completion_tokens"])
        for record in requests
        if record["completion_tokens"] is not None
    ]

    metrics = {
        "request_schema_valid_rate": parse_rate,
        "candidate_unique_ratio": unique_ratio,
        "candidate_pre_repair_feasible_rate": pre_repair_rate,
        "candidate_post_repair_hard_feasible_rate": candidate_hard_rate,
        "candidate_repair_rate": (
            float(np.mean([record["repair_distance"] > 1e-12 for record in candidates]))
            if candidates
            else 0.0
        ),
        "candidate_mean_repair_distance": _mean(candidates, "repair_distance"),
        "candidate_p95_repair_distance": _quantile(
            candidates,
            "repair_distance",
            0.95,
        ),
        "candidate_fallback_rate": fallback_rate,
        "selection_uncertain_rate": uncertain_rate,
        "demonstration_acceptance_rate": (len(demonstrations) / len(states) if states else 0.0),
        "rollout_hard_feasible_rate": rollout_hard_rate,
        "rollout_retained_trajectory_rate": retained_rate,
        "mean_request_latency_s": _mean(requests, "latency_s"),
        "p95_request_latency_s": _quantile(requests, "latency_s", 0.95),
        "mean_completion_tokens": (float(np.mean(completion_tokens)) if completion_tokens else 0.0),
    }

    if parse_rate < 0.99:
        warnings.append("teacher schema-valid rate is below 99%")
    if unique_ratio < 0.75:
        warnings.append("candidate uniqueness is below 75%")
    if fallback_rate > 0.05:
        warnings.append("candidate fallback rate is above 5%")
    if metrics["candidate_p95_repair_distance"] > 0.25:
        warnings.append("candidate p95 repair distance is above 0.25")
    if uncertain_rate > 0.25:
        warnings.append("more than 25% of accepted selections are statistically uncertain")
    if candidate_hard_rate < 1.0 or rollout_hard_rate < 1.0:
        errors.append("post-repair hard feasibility must be 100%")

    return DatasetAudit(
        passed=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        counts={name: len(records) for name, records in tables.items()},
        metrics=metrics,
    )
