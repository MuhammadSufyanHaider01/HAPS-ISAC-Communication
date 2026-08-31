"""Integrity and teacher-quality audit for generated demonstration datasets."""

from __future__ import annotations

from dataclasses import dataclass
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
    schema_version = int(loader.manifest.get("schema_version", 0))
    candidates_by_state: dict[str, int] = {}
    teacher_candidates_by_state: dict[str, int] = {}
    for record in candidates:
        state_id = str(record["state_id"])
        candidates_by_state[state_id] = candidates_by_state.get(state_id, 0) + 1
        if str(record.get("candidate_source", "teacher")) == "teacher":
            teacher_candidates_by_state[state_id] = teacher_candidates_by_state.get(state_id, 0) + 1
    selections_by_state = {str(record["state_id"]): record for record in selections}
    for request in valid_requests:
        state_id = str(request["state_id"])
        if schema_version >= 5:
            actual_teacher = teacher_candidates_by_state.get(state_id, 0)
            expected_pool = int(
                selections_by_state.get(state_id, {}).get("candidate_pool_count", 0)
            )
            if actual_teacher != expected_candidates:
                errors.append(
                    f"{state_id} has {actual_teacher} teacher candidates; "
                    f"expected {expected_candidates}"
                )
            if candidates_by_state.get(state_id, 0) != expected_pool:
                errors.append(
                    f"{state_id} has {candidates_by_state.get(state_id, 0)} pool candidates; "
                    f"expected {expected_pool}"
                )
        elif candidates_by_state.get(state_id, 0) != expected_candidates:
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

    candidate_lookup = {
        (str(record["state_id"]), int(record["candidate_index"])): record for record in candidates
    }
    valid_targets = 0
    for demonstration in demonstrations:
        state_id = str(demonstration["state_id"])
        targets = demonstration.get("target_candidates")
        if targets is None:
            valid_targets += 1
            continue
        target_errors: list[str] = []
        if not isinstance(targets, list) or not targets:
            target_errors.append("has no distillation targets")
        elif any(not isinstance(target, dict) for target in targets):
            target_errors.append("contains a non-object distillation target")
        else:
            try:
                weights = [float(target.get("weight", -1.0)) for target in targets]
                target_indices = [int(target.get("candidate_index", -1)) for target in targets]
            except (TypeError, ValueError):
                weights = [-1.0]
                target_indices = [-1]
                target_errors.append("contains malformed distillation target values")
            if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
                target_errors.append("has invalid distillation weights")
            if abs(sum(weights) - 1.0) > 1e-6:
                target_errors.append("distillation weights do not sum to one")
            if len(target_indices) != len(set(target_indices)):
                target_errors.append("contains duplicate distillation candidates")
            for target, candidate_index in zip(targets, target_indices, strict=True):
                candidate = candidate_lookup.get((state_id, candidate_index))
                if candidate is None:
                    target_errors.append(f"references missing candidate {candidate_index}")
                elif target.get("action") != candidate.get("executed_action"):
                    target_errors.append(f"action differs from candidate {candidate_index}")
        if target_errors:
            errors.extend(f"demonstration {state_id} {message}" for message in target_errors)
        else:
            valid_targets += 1

    if schema_version >= 4:
        global_indices = [int(record.get("global_state_index", -1)) for record in states]
        if len(global_indices) != len(set(global_indices)):
            errors.append("global state indices are duplicated")
        start = int(loader.manifest.get("global_state_start", 0))
        stop = int(loader.manifest.get("global_state_stop", start + len(states)))
        if set(global_indices) != set(range(start, stop)):
            errors.append("global state indices do not exactly cover the manifest range")
        causal_hashes = [str(record.get("causal_state_hash", "")) for record in states]
        if len(causal_hashes) != len(set(causal_hashes)):
            errors.append("causal states are duplicated")

        rollout_counts_by_state: dict[str, int] = {}
        for record in rollouts:
            state_id = str(record["state_id"])
            rollout_counts_by_state[state_id] = rollout_counts_by_state.get(state_id, 0) + 1
        for selection in selections:
            if selection.get("acceptance_status") != "accepted":
                continue
            state_id = str(selection["state_id"])
            verification_rollouts = int(selection.get("verification_rollouts", 0))
            verified_candidates = sum(
                bool(record.get("rollout_verified"))
                for record in candidates
                if str(record["state_id"]) == state_id
            )
            external_baselines = (
                int(selection.get("external_baseline_rollout_count", 0))
                if schema_version >= 5
                else 2
            )
            expected_rollouts = (verified_candidates + external_baselines) * verification_rollouts
            if (
                verification_rollouts <= 0
                or rollout_counts_by_state.get(state_id, 0) != expected_rollouts
            ):
                errors.append(f"{state_id} has an incomplete adaptive rollout transaction")

    scenario_splits: dict[str, set[str]] = {}
    for record in states:
        scenario_splits.setdefault(str(record["scenario_id"]), set()).add(str(record["split"]))
    leaking = [scenario for scenario, splits in scenario_splits.items() if len(splits) > 1]
    if leaking:
        errors.append(f"{len(leaking)} scenarios cross dataset splits")

    metric_candidates = (
        [
            record
            for record in candidates
            if str(record.get("candidate_source", "teacher")) == "teacher"
        ]
        if schema_version >= 5
        else candidates
    )
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
        float(np.mean([record["hard_feasible"] for record in metric_candidates]))
        if metric_candidates
        else 0.0
    )
    pre_repair_rate = (
        float(np.mean([record["pre_repair_feasible"] for record in metric_candidates]))
        if metric_candidates
        else 0.0
    )
    fallback_rate = (
        float(np.mean([record["fallback_used"] for record in metric_candidates]))
        if metric_candidates
        else 0.0
    )
    selected_candidates = [record for record in candidates if bool(record["selected"])]
    selected_fallback_rate = (
        float(np.mean([record["fallback_used"] for record in selected_candidates]))
        if selected_candidates
        else 0.0
    )
    selected_p95_repair_distance = _quantile(
        selected_candidates,
        "repair_distance",
        0.95,
    )
    candidate_records_by_state: dict[str, list[dict[str, Any]]] = {}
    for candidate in metric_candidates:
        candidate_records_by_state.setdefault(str(candidate["state_id"]), []).append(candidate)
    executed_unique_ratios = [
        len(
            {
                tuple(round(float(record["executed_action"][field]), 8) for field in ACTION_FIELDS)
                for record in records
            }
        )
        / max(1, len(records))
        for records in candidate_records_by_state.values()
    ]
    executed_candidate_unique_ratio = (
        float(np.mean(executed_unique_ratios)) if executed_unique_ratios else 0.0
    )
    candidate_role_compliance_rate = (
        float(
            np.mean(
                [
                    sum(int(record["proposed_action"]["pair"]) == 0 for record in records) == 1
                    for records in candidate_records_by_state.values()
                ]
            )
        )
        if candidate_records_by_state
        else 0.0
    )
    selected_by_state = {str(record["state_id"]): record for record in selected_candidates}
    selected_risks: list[float] = []
    greedy_risks: list[float] = []
    random_risks: list[float] = []
    for selection in selections:
        selected = selected_by_state.get(str(selection["state_id"]))
        baselines = selection.get("baseline_scores", {})
        summary = (selected or {}).get("rollout_summary") or {}
        if (
            "risk_score" not in summary
            or "greedy_verified_risk_score" not in baselines
            or "random_verified_risk_score" not in baselines
        ):
            continue
        selected_risks.append(float(summary["risk_score"]))
        greedy_risks.append(float(baselines["greedy_verified_risk_score"]))
        random_risks.append(float(baselines["random_verified_risk_score"]))
    accepted_selections = [
        record for record in selections if record["acceptance_status"] == "accepted"
    ]
    uncertain_rate = (
        float(np.mean([record["selection_uncertain"] for record in accepted_selections]))
        if accepted_selections
        else 0.0
    )
    unresolved_rate = (
        float(
            np.mean(
                [
                    record.get("decision_status") == "unresolved"
                    or (
                        record.get("decision_status") is None
                        and bool(record.get("selection_uncertain", True))
                    )
                    for record in accepted_selections
                ]
            )
        )
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

    template_compliance_rate = (
        float(
            np.mean([bool(record.get("template_compliant", False)) for record in metric_candidates])
        )
        if metric_candidates
        else 0.0
    )
    request_normalization_rate = (
        float(np.mean([bool(record.get("response_normalization_notes")) for record in requests]))
        if requests
        else 0.0
    )
    template_resolution_repair_rate = (
        float(
            np.mean(
                [
                    str(record.get("template_resolution", "exact")) != "exact"
                    for record in metric_candidates
                ]
            )
        )
        if metric_candidates
        else 0.0
    )
    safe_template_coverage_rate = (
        float(
            np.mean(
                [
                    float(record.get("safe_template_coverage_rate", 0.0))
                    for record in accepted_selections
                ]
            )
        )
        if accepted_selections
        else 0.0
    )
    greedy_selectable_rate = (
        float(
            np.mean(
                [record.get("greedy_candidate_index") is not None for record in accepted_selections]
            )
        )
        if accepted_selections
        else 0.0
    )

    metrics = {
        "request_schema_valid_rate": parse_rate,
        "request_normalization_rate": request_normalization_rate,
        "candidate_unique_ratio": unique_ratio,
        "candidate_role_compliance_rate": candidate_role_compliance_rate,
        "teacher_template_compliance_rate": template_compliance_rate,
        "teacher_template_resolution_repair_rate": template_resolution_repair_rate,
        "safe_template_coverage_rate": safe_template_coverage_rate,
        "greedy_selectable_rate": greedy_selectable_rate,
        "executed_candidate_unique_ratio": executed_candidate_unique_ratio,
        "selected_fallback_rate": selected_fallback_rate,
        "selected_p95_repair_distance": selected_p95_repair_distance,
        "verified_baseline_compared_states": float(len(selected_risks)),
        "selected_vs_greedy_verified_win_rate": (
            float(
                np.mean(
                    [
                        selected < baseline
                        for selected, baseline in zip(
                            selected_risks,
                            greedy_risks,
                            strict=True,
                        )
                    ]
                )
            )
            if selected_risks
            else 0.0
        ),
        "selected_vs_random_verified_win_rate": (
            float(
                np.mean(
                    [
                        selected < baseline
                        for selected, baseline in zip(
                            selected_risks,
                            random_risks,
                            strict=True,
                        )
                    ]
                )
            )
            if selected_risks
            else 0.0
        ),
        "mean_greedy_minus_selected_verified_risk": (
            float(np.mean(np.asarray(greedy_risks) - np.asarray(selected_risks)))
            if selected_risks
            else 0.0
        ),
        "mean_random_minus_selected_verified_risk": (
            float(np.mean(np.asarray(random_risks) - np.asarray(selected_risks)))
            if selected_risks
            else 0.0
        ),
        "candidate_pre_repair_feasible_rate": pre_repair_rate,
        "candidate_post_repair_hard_feasible_rate": candidate_hard_rate,
        "candidate_repair_rate": (
            float(np.mean([record["repair_distance"] > 1e-12 for record in metric_candidates]))
            if metric_candidates
            else 0.0
        ),
        "candidate_mean_repair_distance": _mean(metric_candidates, "repair_distance"),
        "candidate_p95_repair_distance": _quantile(
            metric_candidates,
            "repair_distance",
            0.95,
        ),
        "candidate_fallback_rate": fallback_rate,
        "selection_uncertain_rate": uncertain_rate,
        "selection_unresolved_rate": unresolved_rate,
        "distillation_target_valid_rate": (
            valid_targets / len(demonstrations) if demonstrations else 0.0
        ),
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
    if request_normalization_rate > 0.05:
        warnings.append("more than 5% of teacher requests required response normalization")
    if template_resolution_repair_rate > 0.05:
        warnings.append("more than 5% of teacher candidates required template-id repair")
    if fallback_rate > 0.05:
        warnings.append("candidate fallback rate is above 5%")
    if metrics["candidate_p95_repair_distance"] > 0.25:
        warnings.append("candidate p95 repair distance is above 0.25")
    if schema_version < 5 and candidate_role_compliance_rate < 0.95:
        warnings.append("fewer than 95% of states have exactly one sensing-only candidate")
    if schema_version >= 5 and template_compliance_rate < 0.95:
        warnings.append(
            "fewer than 95% of teacher candidates comply with the referenced safe template"
        )
    if schema_version >= 5 and safe_template_coverage_rate < 1.0:
        warnings.append("deterministic safe-template bank coverage is incomplete")
    if schema_version >= 5 and greedy_selectable_rate < 1.0:
        warnings.append("the selectable greedy floor is missing for at least one accepted state")
    if executed_candidate_unique_ratio < 0.75:
        warnings.append("post-repair candidate uniqueness is below 75%")
    if selected_fallback_rate > 0.0:
        warnings.append("at least one selected demonstration used fallback")
    if selected_p95_repair_distance > 0.05:
        warnings.append("selected-candidate p95 repair distance is above 0.05")
    if unresolved_rate > 0.05:
        warnings.append("more than 5% of accepted selections remain unresolved after verification")
    elif uncertain_rate > 0.05:
        warnings.append("confidence intervals overlap, but labels are practically equivalent")
    if selected_risks:
        greedy_win_rate = metrics["selected_vs_greedy_verified_win_rate"]
        random_win_rate = metrics["selected_vs_random_verified_win_rate"]
        if greedy_win_rate < 0.55:
            warnings.append("selected candidates do not beat verified greedy in 55% of states")
        if random_win_rate < 0.55:
            warnings.append("selected candidates do not beat verified random in 55% of states")
    if candidate_hard_rate < 1.0 or rollout_hard_rate < 1.0:
        errors.append("post-repair hard feasibility must be 100%")

    return DatasetAudit(
        passed=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        counts={name: len(records) for name, records in tables.items()},
        metrics=metrics,
    )
