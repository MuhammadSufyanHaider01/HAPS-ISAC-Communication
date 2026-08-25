"""Create a plotting-ready teacher quality and scale-up report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from haps_isac.data.quality_report import build_teacher_quality_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    report = build_teacher_quality_report(arguments.dataset)
    output = (
        Path(arguments.output)
        if arguments.output is not None
        else Path(arguments.dataset) / "teacher_quality_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
