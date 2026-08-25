"""Run a frozen-state, common-random-number teacher tournament."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from haps_isac.config import load_config
from haps_isac.data.demonstration_schema import json_safe
from haps_isac.teachers.base_teacher import TeacherConfig, load_teacher_config
from haps_isac.teachers.benchmark import run_tournament


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", default="configs/system_v1.yaml")
    parser.add_argument("--teacher-config", default="configs/teacher.yaml")
    parser.add_argument(
        "--teacher",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "PROVIDER", "MODEL_ID", "BASE_URL"),
        help="Repeat for each teacher; PROVIDER is qwen, gemma, or mock.",
    )
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--shortlist", type=int)
    parser.add_argument("--output", required=True)
    return parser


def _configured_teacher(
    base: TeacherConfig,
    specification: list[str],
    arguments: argparse.Namespace,
) -> tuple[str, TeacherConfig]:
    label, provider, model_id, base_url = specification
    if provider not in {"qwen", "gemma", "mock"}:
        raise ValueError(f"unsupported provider for {label}: {provider}")
    verification_updates: dict[str, int] = {}
    if arguments.rollouts is not None:
        verification_updates["monte_carlo_rollouts"] = arguments.rollouts
    if arguments.horizon is not None:
        verification_updates["rollout_horizon_slots"] = arguments.horizon
    if arguments.shortlist is not None:
        verification_updates["shortlist_size"] = arguments.shortlist
    payload: dict[str, Any] = base.model_dump()
    payload.update(
        {
            "provider": provider,
            "model_id": model_id,
            "base_url": base_url,
            "cache_directory": str(Path(base.cache_directory) / label),
        }
    )
    if arguments.candidates is not None:
        payload["num_candidates"] = arguments.candidates
    if verification_updates:
        verification = dict(payload["verification"])
        verification.update(verification_updates)
        payload["verification"] = verification
    return label, TeacherConfig.model_validate(payload)


def main() -> None:
    arguments = build_parser().parse_args()
    system = load_config(arguments.system_config)
    base = load_teacher_config(arguments.teacher_config)
    teacher_configs = dict(
        _configured_teacher(base, specification, arguments) for specification in arguments.teacher
    )
    if len(teacher_configs) != len(arguments.teacher):
        raise ValueError("teacher labels must be unique")
    master_seed = system.master_seed if arguments.seed is None else arguments.seed
    tournament = run_tournament(
        system,
        teacher_configs,
        arguments.states,
        master_seed,
    )
    output = Path(arguments.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "winner": tournament.winner,
        "summaries": json_safe(tournament.summaries),
        "state_count": arguments.states,
        "master_seed": master_seed,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "benchmark_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in tournament.records:
            stream.write(
                json.dumps(json_safe(record), sort_keys=True, separators=(",", ":")) + "\n"
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
