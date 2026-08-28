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
    "candidate_role_compliance_rate": (">=", 0.95),
    "executed_candidate_unique_ratio": (">=", 0.75),
    "candidate_post_repair_hard_feasible_rate": (">=", 1.0),
    "candidate_fallback_rate": ("<=", 0.05),
    "candidate_p95_repair_distance": ("<=", 0.25),
    "selected_fallback_rate": ("<=", 0.0),
    "selected_p95_repair_distance": ("<=", 0.05),
    "selection_unresolved_rate": ("<=", 0.05),
    "distillation_target_valid_rate": (">=", 1.0),
    "duplicate_causal_state_rate": ("<=", 0.0),
    "state_distribution_max_abs_error": ("<=", 0.02),
    "split_distribution_max_abs_error": ("<=", 0.02),
    "demonstration_acceptance_rate": (">=", 0.99),
    "selected_vs_greedy_verified_win_rate": (">=", 0.55),
    "selected_vs_random_verified_win_rate": (">=", 0.55),
    "mean_greedy_minus_selected_verified_risk": (">=", 0.0),
    "mean_random_minus_selected_verified_risk": (">=", 0.0),
}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else 0.0


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return _mean([float(bool(record[key])) for record in records])


def _decision_status(record: dict[str, Any]) -> str:
    value = record.get("decision_status")
    if value in {"decisive", "practically_equivalent", "unresolved"}:
        return str(value)
    return "unresolved" if bool(record.get("selection_uncertain", True)) else "decisive"


def _target_is_valid(record: dict[str, Any]) -> bool:
    targets = record.get("target_candidates")
    if targets is None:
        return "selected_action" in record
    if not isinstance(targets, list) or not targets:
        return False
    if any(not isinstance(target, dict) for target in targets):
        return False
    try:
        weights = [float(target.get("weight", -1.0)) for target in targets]
        candidate_indices = [int(target.get("candidate_index", -1)) for target in targets]
    except (TypeError, ValueError):
        return False
    return (
        all(np.isfinite(weight) and weight >= 0.0 for weight in weights)
        and abs(sum(weights) - 1.0) <= 1e-6
        and len(candidate_indices) == len(set(candidate_indices))
        and all(isinstance(target.get("action"), dict) for target in targets)
    )


def _distribution_summary(
    records: list[dict[str, Any]],
    field: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(record.get(field, "unknown")) for record in records)
    denominator = max(1, len(records))
    labels = sorted(set(counts) | set(target))
    rows = {
        label: {
            "count": counts.get(label, 0),
            "actual_fraction": counts.get(label, 0) / denominator,
            "target_fraction": float(target.get(label, 0.0)),
            "absolute_error": abs(
                counts.get(label, 0) / denominator - float(target.get(label, 0.0))
            ),
        }
        for label in labels
    }
    return {
        "records": len(records),
        "categories": rows,
        "max_absolute_error": max(
            (float(row["absolute_error"]) for row in rows.values()),
            default=0.0,
        ),
    }


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


def _action_key(action: dict[str, Any]) -> tuple[int | float, ...]:
    return tuple(round(float(action[field]), 8) for field in ACTION_FIELDS)


def _action_diversity(candidates: list[dict[str, Any]]) -> dict[str, float | int]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_state.setdefault(str(candidate["state_id"]), []).append(candidate)
    proposed_ratios: list[float] = []
    executed_ratios: list[float] = []
    for records in by_state.values():
        denominator = max(1, len(records))
        proposed_ratios.append(
            len({_action_key(record["proposed_action"]) for record in records}) / denominator
        )
        executed_ratios.append(
            len({_action_key(record["executed_action"]) for record in records}) / denominator
        )
    return {
        "states": len(by_state),
        "mean_proposed_unique_ratio": _mean(proposed_ratios),
        "mean_executed_unique_ratio": _mean(executed_ratios),
        "states_below_0.75_executed_unique_ratio": sum(ratio < 0.75 for ratio in executed_ratios),
    }


def _candidate_role_compliance_rate(candidates: list[dict[str, Any]]) -> float:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_state.setdefault(str(candidate["state_id"]), []).append(candidate)
    return _mean(
        [
            float(sum(int(record["proposed_action"]["pair"]) == 0 for record in records) == 1)
            for records in by_state.values()
        ]
    )


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
        str(record["state_id"]): record for record in candidates if bool(record["selected"])
    }
    selected_costs: list[float] = []
    greedy_costs: list[float] = []
    random_costs: list[float] = []
    selected_risks: list[float] = []
    greedy_risks: list[float] = []
    random_risks: list[float] = []
    for selection in selections:
        selected = selected_by_state.get(str(selection["state_id"]))
        baselines = selection.get("baseline_scores", {})
        if selected is None:
            continue
        if "greedy_one_step_cost" in baselines and "random_one_step_cost" in baselines:
            selected_costs.append(float(selected["one_step_stage_cost"]))
            greedy_costs.append(float(baselines["greedy_one_step_cost"]))
            random_costs.append(float(baselines["random_one_step_cost"]))
        rollout_summary = selected.get("rollout_summary") or {}
        if (
            "risk_score" in rollout_summary
            and "greedy_verified_risk_score" in baselines
            and "random_verified_risk_score" in baselines
        ):
            selected_risks.append(float(rollout_summary["risk_score"]))
            greedy_risks.append(float(baselines["greedy_verified_risk_score"]))
            random_risks.append(float(baselines["random_verified_risk_score"]))
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
        "verified_compared_states": len(selected_risks),
        "mean_selected_verified_risk": _mean(selected_risks),
        "mean_greedy_verified_risk": _mean(greedy_risks),
        "mean_random_verified_risk": _mean(random_risks),
        "selected_vs_greedy_verified_win_rate": _mean(
            [
                float(selected < baseline)
                for selected, baseline in zip(selected_risks, greedy_risks, strict=True)
            ]
        ),
        "selected_vs_random_verified_win_rate": _mean(
            [
                float(selected < baseline)
                for selected, baseline in zip(selected_risks, random_risks, strict=True)
            ]
        ),
        "mean_greedy_minus_selected_verified_risk": _mean(
            [
                baseline - selected
                for selected, baseline in zip(selected_risks, greedy_risks, strict=True)
            ]
        ),
        "mean_random_minus_selected_verified_risk": _mean(
            [
                baseline - selected
                for selected, baseline in zip(selected_risks, random_risks, strict=True)
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
        selected_rollout = (selected or {}).get("rollout_summary") or {}
        row = {
            "state_id": state_id,
            "scenario_id": str(state["scenario_id"]),
            "split": str(state["split"]),
            "global_state_index": state.get("global_state_index"),
            "state_category": state.get("state_category", "ordinary"),
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
            "selected_verified_risk": (
                float(selected_rollout["risk_score"]) if "risk_score" in selected_rollout else None
            ),
            "selection_uncertain": selection.get("selection_uncertain"),
            "decision_status": _decision_status(selection),
            "verification_rollouts": selection.get("verification_rollouts"),
            "adaptive_rounds": selection.get("adaptive_rounds"),
            "selection_probability": selection.get("selection_probability"),
            "margin_confidence_lower": selection.get("margin_confidence_lower"),
            "margin_confidence_upper": selection.get("margin_confidence_upper"),
            "greedy_verified_risk": baselines.get("greedy_verified_risk_score"),
            "random_verified_risk": baselines.get("random_verified_risk_score"),
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
            float(record["unique_candidates"]) / max(1.0, float(record["candidates_returned"]))
            for record in valid_requests
        ]
    )
    candidate_summary = _candidate_summary(candidates)
    selected_candidates = [candidate for candidate in candidates if bool(candidate["selected"])]
    selected_summary = _candidate_summary(selected_candidates)
    action_diversity = _action_diversity(candidates)
    candidate_role_compliance_rate = _candidate_role_compliance_rate(candidates)
    accepted_selections = [
        selection for selection in selections if selection["acceptance_status"] == "accepted"
    ]
    selection_uncertain_rate = _rate(accepted_selections, "selection_uncertain")
    decision_counts = Counter(_decision_status(record) for record in accepted_selections)
    selection_unresolved_rate = (
        decision_counts["unresolved"] / len(accepted_selections) if accepted_selections else 0.0
    )
    target_valid_rate = _mean([float(_target_is_valid(record)) for record in demonstrations])
    causal_hash_counts = Counter(str(record["causal_state_hash"]) for record in states)
    duplicate_state_count = sum(count - 1 for count in causal_hash_counts.values() if count > 1)
    duplicate_causal_state_rate = duplicate_state_count / len(states) if states else 0.0
    state_distribution = _distribution_summary(
        states,
        "state_category",
        dict(loader.manifest.get("state_distribution", {})),
    )
    split_distribution = _distribution_summary(
        states,
        "split",
        dict(loader.manifest.get("split_fractions", {})),
    )
    verification_rollouts = [
        float(record.get("verification_rollouts", 0)) for record in accepted_selections
    ]
    adaptive_rounds = [float(record.get("adaptive_rounds", 0)) for record in accepted_selections]
    baseline_comparison = _baseline_comparison(candidates, selections)
    metrics = {
        "request_schema_valid_rate": request_schema_valid_rate,
        "candidate_unique_ratio": candidate_unique_ratio,
        "candidate_role_compliance_rate": candidate_role_compliance_rate,
        "executed_candidate_unique_ratio": action_diversity["mean_executed_unique_ratio"],
        "candidate_post_repair_hard_feasible_rate": candidate_summary[
            "post_repair_hard_feasible_rate"
        ],
        "candidate_fallback_rate": candidate_summary["fallback_rate"],
        "candidate_p95_repair_distance": candidate_summary["p95_repair_distance"],
        "selected_fallback_rate": selected_summary["fallback_rate"],
        "selected_p95_repair_distance": selected_summary["p95_repair_distance"],
        "selection_uncertain_rate": selection_uncertain_rate,
        "selection_unresolved_rate": selection_unresolved_rate,
        "distillation_target_valid_rate": target_valid_rate,
        "duplicate_causal_state_rate": duplicate_causal_state_rate,
        "state_distribution_max_abs_error": state_distribution["max_absolute_error"],
        "split_distribution_max_abs_error": split_distribution["max_absolute_error"],
        "demonstration_acceptance_rate": (len(demonstrations) / len(states) if states else 0.0),
        "selected_vs_greedy_verified_win_rate": baseline_comparison[
            "selected_vs_greedy_verified_win_rate"
        ],
        "selected_vs_random_verified_win_rate": baseline_comparison[
            "selected_vs_random_verified_win_rate"
        ],
        "mean_greedy_minus_selected_verified_risk": baseline_comparison[
            "mean_greedy_minus_selected_verified_risk"
        ],
        "mean_random_minus_selected_verified_risk": baseline_comparison[
            "mean_random_minus_selected_verified_risk"
        ],
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
        str(reason) for candidate in candidates for reason in candidate.get("repair_reasons", [])
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
        "selected_candidate_summary": selected_summary,
        "action_diversity": action_diversity,
        "candidate_role_compliance_rate": candidate_role_compliance_rate,
        "selection_quality": {
            "accepted_count": len(accepted_selections),
            "uncertain_rate": selection_uncertain_rate,
            "decision_counts": dict(sorted(decision_counts.items())),
            "unresolved_rate": selection_unresolved_rate,
            "practically_equivalent_rate": (
                decision_counts["practically_equivalent"] / len(accepted_selections)
                if accepted_selections
                else 0.0
            ),
            "mean_verification_rollouts": _mean(verification_rollouts),
            "p95_verification_rollouts": _quantile(verification_rollouts, 0.95),
            "max_verification_rollouts": max(verification_rollouts, default=0.0),
            "mean_adaptive_rounds": _mean(adaptive_rounds),
        },
        "distillation_targets": {
            "valid_rate": target_valid_rate,
            "mean_candidates_per_state": _mean(
                [float(len(record.get("target_candidates") or [None])) for record in demonstrations]
            ),
            "mean_entropy": _mean(
                [float(record.get("target_entropy", 0.0)) for record in demonstrations]
            ),
        },
        "sampling_quality": {
            "duplicate_causal_state_count": duplicate_state_count,
            "duplicate_causal_state_rate": duplicate_causal_state_rate,
            "state_distribution": state_distribution,
            "split_distribution": split_distribution,
        },
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
        "baseline_comparison": baseline_comparison,
        "scale_up_gates": gates,
        "scale_up_passed": all(bool(gate["passed"]) for gate in gates.values()),
    }
