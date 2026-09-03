#!/usr/bin/env python3
"""Materialize a validated demonstration-only view without new teacher queries."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _valid_target_set(record: dict[str, Any]) -> bool:
    targets = record.get("target_candidates")
    if targets is None:
        return "selected_action" in record
    if not isinstance(targets, list) or not targets:
        return False
    if any(not isinstance(target, dict) for target in targets):
        return False
    try:
        weights = [float(target.get("weight", -1.0)) for target in targets]
        indices = [int(target.get("candidate_index", -1)) for target in targets]
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(weight) and weight >= 0.0 for weight in weights)
        and abs(sum(weights) - 1.0) <= 1e-6
        and len(indices) == len(set(indices))
        and all(isinstance(target.get("action"), dict) for target in targets)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_view(source: Path, output: Path, run_id: str | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    demonstrations = _read_jsonl(source / "demonstrations.jsonl")
    valid = [record for record in demonstrations if _valid_target_set(record)]
    excluded = [record for record in demonstrations if not _valid_target_set(record)]
    output.mkdir(parents=True, exist_ok=True)

    with (output / "demonstrations.jsonl").open("w", encoding="utf-8") as stream:
        for record in valid:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    view_run_id = run_id or f"{source_manifest['run_id']}-distillation"
    view_manifest = dict(source_manifest)
    view_manifest.update(
        {
            "run_id": view_run_id,
            "source_run_id": source_manifest["run_id"],
            "view_type": "demonstration_only",
            "created_at": _utc_now(),
            "completed_at": _utc_now(),
            "table_counts": {"demonstrations": len(valid)},
            "total_requested_states": len(valid),
            "global_state_start": 0,
            "global_state_stop": len(valid),
            "shard_index": 0,
            "shard_count": 1,
            "excluded_source_demonstrations": len(excluded),
        }
    )
    _write_json(output / "manifest.json", view_manifest)

    source_state_count = int(
        source_manifest.get("table_counts", {}).get("states", 0)
        or source_manifest.get("total_requested_states", 0)
        or 0
    )
    source_report = {}
    report_path = source / "teacher_quality_report.json"
    if report_path.exists():
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "run_id": view_run_id,
        "source_run_id": source_manifest["run_id"],
        "ready_for_distillation": bool(valid),
        "quality_gates_passed": bool(source_report.get("scale_up_passed", False)),
        "source_state_count": source_state_count,
        "source_demonstration_count": len(demonstrations),
        "usable_demonstration_count": len(valid),
        "excluded_demonstration_count": len(excluded),
        "coverage_rate": (
            len(valid) / max(1, source_state_count)
        ),
        "excluded_state_ids": [str(record.get("state_id")) for record in excluded],
        "source_quality_report": str(report_path) if report_path.exists() else None,
        "selection_policy": "retain only source demonstrations with valid normalized soft targets",
        "generated_at": _utc_now(),
    }
    _write_json(output / "distillation_manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = build_view(arguments.source, arguments.output, arguments.run_id)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
