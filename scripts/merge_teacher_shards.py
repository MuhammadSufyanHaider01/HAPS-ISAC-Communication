#!/usr/bin/env python3
"""Merge complete teacher shards, audit them, and enforce aggregate quality gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from haps_isac.data.audit import audit_dataset
from haps_isac.data.demonstration_schema import json_safe
from haps_isac.data.merge import discover_shards, merge_shards
from haps_isac.data.quality_report import build_teacher_quality_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument(
        "--allow-failed-gates",
        action="store_true",
        help="retain a diagnostic merge with exit status zero even when scale-up gates fail",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    arguments = build_parser().parse_args()
    shards = discover_shards(arguments.shard_root)
    manifest = merge_shards(
        shards,
        arguments.output,
        arguments.run_id,
        expected_shards=arguments.expected_shards,
        export_parquet=not arguments.no_parquet,
    )
    output = Path(arguments.output)
    audit = audit_dataset(str(output))
    report = build_teacher_quality_report(output)
    _write_json(output / "audit.json", asdict(audit))
    _write_json(output / "teacher_quality_report.json", report)
    summary = {
        "run_id": manifest.run_id,
        "shards": len(shards),
        "states": manifest.table_counts.get("states", 0),
        "audit_passed": audit.passed,
        "scale_up_passed": report["scale_up_passed"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not arguments.allow_failed_gates and (
        not audit.passed or not bool(report["scale_up_passed"])
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
