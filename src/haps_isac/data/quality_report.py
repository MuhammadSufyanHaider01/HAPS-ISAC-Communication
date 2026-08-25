"""Plotting-ready teacher quality diagnostics and scale-up gates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from haps_isac.data.dataset_loader import DatasetLoader

ACTION_FIELDS = (
    "pair",
    "ris_code",
    "eta_haps",
    "eta_communication",
    "eta_near",
    "eta_jamming",
    "aav_heading_rad",
    "aav_speed_fraction",
    "eta_cpu",
)

SCALE_UP_THRESHOLDS = {
    "request_schema_valid_rate": (">=", 0.99),
    "candidate_unique_ratio": (">=", 0.75),
    "candidate_post_repair_hard_feasible_rate": (">=", 1.0),
    "candidate_fallback_rate": ("<=", 0.05),
    "candidate_p95_repair_distance": ("<=", 0.25),
    "demonstration_acceptance_rate": (">=", 0.99),
}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else 0.0


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return _mean([float(bool(record[key])) for record in records])


def _candidate_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    repair_distances = [float(record["repair_distance"]) for record in records]
    return {
        "count": len(records),
        "pre_repair_feasible_rate": _rate(records, "pre_repair_feasible"),
        "post_repair_hard_feasible_rate": _rate(records, "hard_feasible"),
        "fallback_rate": _rate(records, "fallback_used"),
        "repair_rate": _mean([float(distance > 1e-12) for distance in repair_distances]),
        "mean_repair_distance": _mean(repair_distances),
        "p95_repair_distance": _quantile(repair_distances, 0.95),
    }


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.5:
        return "low_[0,0.5)"
    if confidence < 0.8:
        return "medium_[0.5,0.8)"
    return "high_[0.8,1]"


def _action_field_changes(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for field in ACTION_FIELDS:
        absolute_changes: list[float] = []
        for record in candidates:
            proposed = record["proposed_action"][field]
            executed = record["executed_action"][field]
            absolute_changes.append(abs(float(executed) - float(proposed)))
        output[field] = {
            "count": len(absolute_changes),
            "changed_count": sum(change > 1e-12 for change in absolute_changes),
            "changed_rate": _mean([float(change > 1e-12) for change in absolute_changes]),
            "mean_absolute_change": _mean(absolute_changes),
            "p95_absolute_change": _quantile(absolute_changes, 0.95),
        }
    return output


def _baseline_comparison(
    candidates: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> dict[str, float | int]:
    selected_by_state = {
        str(record["state_id"]): record
        for record in candidates
        if bool(record["selected"])
    }
    selected_costs: list[float] = []
    greedy_costs: list[float] = []
    random_costs: list[float] = []
    for selection in selections:
        selected = selected_by_state.get(str(selection["state_id"]))
        baselines = selection.get("baseline_scores", {})
        if (
            selected is None
            or "greedy_one_step_cost" not in baselines
            or "random_one_step_cost" not in baselines
        ):
            continue
        selected_costs.append(float(selected["one_step_stage_cost"]))
        greedy_costs.append(float(baselines["greedy_one_step_cost"]))
        random_costs.append(float(baselines["random_one_step_cost"]))
    return {
        "compared_states": len(selected_costs),
        "mean_selected_one_step_cost": _mean(selected_costs),
        "mean_greedy_one_step_cost": _mean(greedy_costs),
        "mean_random_one_step_cost": _mean(random_costs),
        "selected_vs_greedy_win_rate": _mean(
            [
                float(selected < baseline)
                for selected, baseline in zip(selected_costs, greedy_costs, strict=True)
            ]
        ),
        "selected_vs_random_win_rate": _mean(
            [
                float(selected < baseline)
                for selected, baseline in zip(selected_costs, random_costs, strict=True)
            ]
        ),
        "mean_greedy_minus_selected_cost": _mean(
            [
                baseline - selected
                for selected, baseline in zip(selected_costs, greedy_costs, strict=True)
            ]
        ),
        "mean_random_minus_selected_cost": _mean(
            [
                baseline - selected
                for selected, baseline in zip(selected_costs, random_costs, strict=True)
            ]
        ),
    }


def _state_difficulty_rows(
    states: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_state: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_state.setdefault(str(candidate["state_id"]), []).append(candidate)
    selections_by_state = {str(record["state_id"]): record for record in selections}
    rows: list[dict[str, Any]] = []
    for state in states:
        state_id = str(state["state_id"])
        state_candidates = candidates_by_state.get(state_id, [])
        selected = next(
            (record for record in state_candidates if bool(record["selected"])),
            None,
        )
        selection = selections_by_state.get(state_id, {})
        metrics = state["state_metrics"]
        baselines = selection.get("baseline_scores", {})
        row = {
            "state_id": state_id,
            "scenario_id": str(state["scenario_id"]),
            "split": str(state["split"]),
            "slot": int(state["slot"]),
            "mean_aoi": float(metrics["mean_aoi"]),
            "max_aoi": float(metrics["max_aoi"]),
            "aosi": float(metrics["aosi"]),
            "queue_l1": float(metrics["queue_l1"]),
            "queue_max": float(metrics["queue_max"]),
            "tracking_covariance_trace": float(metrics["tracking_covariance_trace"]),
            "candidate_quality": _candidate_summary(state_candidates),
            "selected_candidate_index": int(selection.get("selected_candidate_index", -1)),
            "selected_one_step_cost": (
                float(selected["one_step_stage_cost"]) if selected is not None else None
            ),
            "selected_repair_distance": (
                float(selected["repair_distance"]) if selected is not None else None
            ),
            "selected_fallback_used": (
                bool(selected["fallback_used"]) if selected is not None else None
            ),
            "greedy_one_step_cost": (
                float(baselines["greedy_one_step_cost"])
                if "greedy_one_step_cost" in baselines
                else None
            ),
            "random_one_step_cost": (
                float(baselines["random_one_step_cost"])
                if "random_one_step_cost" in baselines
                else None
            ),
        }
        rows.append(row)
    return rows


def build_teacher_quality_report(directory: str | Path) -> dict[str, Any]:
    """Summarize teacher validity, repairs, confidence, state difficulty, and baselines."""

    loader = DatasetLoader(directory)
    states = list(loader.iter_table("states"))
    requests = list(loader.iter_table("teacher_requests"))
    candidates = list(loader.iter_table("candidates"))
    selections = list(loader.iter_table("selections"))
    demonstrations = list(loader.iter_table("demonstrations"))

    valid_requests = [record for record in requests if bool(record["schema_valid"])]
    request_schema_valid_rate = _rate(requests, "schema_valid")
    candidate_unique_ratio = _mean(
        [
            float(record["unique_candidates"])
            / max(1.0, float(record["candidates_returned"]))
            for record in valid_requests
        ]
    )
    candidate_summary = _candidate_summary(candidates)
    metrics = {
        "request_schema_valid_rate": request_schema_valid_rate,
        "candidate_unique_ratio": candidate_unique_ratio,
        "candidate_post_repair_hard_feasible_rate": candidate_summary[
            "post_repair_hard_feasible_rate"
        ],
        "candidate_fallback_rate": candidate_summary["fallback_rate"],
        "candidate_p95_repair_distance": candidate_summary["p95_repair_distance"],
        "demonstration_acceptance_rate": (
            len(demonstrations) / len(states) if states else 0.0
        ),
    }

    gates: dict[str, dict[str, Any]] = {}
    for name, (operator, threshold) in SCALE_UP_THRESHOLDS.items():
        value = float(metrics[name])
        passed = value >= threshold if operator == ">=" else value <= threshold
        gates[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }

    repair_reason_counts = Counter(
        str(reason)
        for candidate in candidates
        for reason in candidate.get("repair_reasons", [])
    )
    confidence_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        bucket = _confidence_bucket(float(candidate["teacher_confidence"]))
        confidence_groups.setdefault(bucket, []).append(candidate)

    return {
        "dataset": str(Path(directory)),
        "run_id": str(loader.manifest["run_id"]),
        "counts": {
            "states": len(states),
            "teacher_requests": len(requests),
            "candidates": len(candidates),
            "selections": len(selections),
            "demonstrations": len(demonstrations),
        },
        "candidate_summary": candidate_summary,
        "repair_reasons": {
            reason: {
                "count": count,
                "candidate_rate": count / len(candidates) if candidates else 0.0,
            }
            for reason, count in sorted(
                repair_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        },
        "action_field_changes": _action_field_changes(candidates),
        "confidence_buckets": {
            bucket: _candidate_summary(records)
            for bucket, records in sorted(confidence_groups.items())
        },
        "state_difficulty": _state_difficulty_rows(states, candidates, selections),
        "baseline_comparison": _baseline_comparison(candidates, selections),
        "scale_up_gates": gates,
        "scale_up_passed": all(bool(gate["passed"]) for gate in gates.values()),
    }
