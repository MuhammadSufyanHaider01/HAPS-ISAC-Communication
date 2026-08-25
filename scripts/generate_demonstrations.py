"""Generate simulator-verified teacher demonstrations and plotting logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from haps_isac.config import load_config
from haps_isac.data.demonstration_schema import json_safe
from haps_isac.data.generation import generate_demonstrations
from haps_isac.teachers.base_teacher import load_teacher_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", default="configs/system_v1.yaml")
    parser.add_argument("--teacher-config", default="configs/teacher.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--provider", choices=("qwen", "gemma", "mock"))
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--base-url")
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--shortlist", type=int)
    parser.add_argument("--flush-every", type=int)
    parser.add_argument("--no-parquet", action="store_true")
    return parser


def _teacher_overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if arguments.provider is not None:
        updates["provider"] = arguments.provider
        if arguments.provider == "mock" and arguments.model_id is None:
            updates["model_id"] = "mock/deterministic-teacher"
        if arguments.provider == "mock" and arguments.model_revision is None:
            updates["model_revision"] = "deterministic-v1"
    if arguments.model_id is not None:
        updates["model_id"] = arguments.model_id
    if arguments.model_revision is not None:
        updates["model_revision"] = arguments.model_revision
    if arguments.base_url is not None:
        updates["base_url"] = arguments.base_url
    if arguments.candidates is not None:
        updates["num_candidates"] = arguments.candidates
    return updates


def main() -> None:
    arguments = build_parser().parse_args()
    system = load_config(arguments.system_config)
    teacher = load_teacher_config(arguments.teacher_config)
    verification_updates: dict[str, int] = {}
    if arguments.rollouts is not None:
        verification_updates["monte_carlo_rollouts"] = arguments.rollouts
    if arguments.horizon is not None:
        verification_updates["rollout_horizon_slots"] = arguments.horizon
    if arguments.shortlist is not None:
        verification_updates["shortlist_size"] = arguments.shortlist
    if verification_updates:
        teacher = teacher.model_copy(
            update={"verification": teacher.verification.model_copy(update=verification_updates)}
        )
    teacher = teacher.model_copy(update=_teacher_overrides(arguments))
    if arguments.flush_every is not None:
        teacher = teacher.model_copy(
            update={
                "logging": teacher.logging.model_copy(update={"flush_every": arguments.flush_every})
            }
        )
    if teacher.verification.shortlist_size > teacher.num_candidates:
        raise ValueError("shortlist cannot exceed the number of candidates")
    master_seed = system.master_seed if arguments.seed is None else arguments.seed
    manifest = generate_demonstrations(
        system_config=system,
        teacher_config=teacher,
        system_config_path=arguments.system_config,
        teacher_config_path=arguments.teacher_config,
        output_directory=Path(arguments.output),
        requested_states=arguments.states,
        master_seed=master_seed,
        run_id=arguments.run_id,
        export_parquet=False if arguments.no_parquet else None,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
