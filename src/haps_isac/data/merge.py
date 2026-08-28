"""Validated merge of independently resumable teacher-dataset shards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from haps_isac.data.dataset_writer import TABLES, DatasetWriter
from haps_isac.data.demonstration_schema import RunManifest, utc_now


def _load_manifest(directory: Path) -> dict[str, Any]:
    with (directory / "manifest.json").open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{directory} manifest must be an object")
    return value


def discover_shards(root: str | Path) -> tuple[Path, ...]:
    directories = [
        path.parent for path in Path(root).glob("shard-*/manifest.json") if path.parent.is_dir()
    ]
    return tuple(sorted(directories))


def _validate_shards(
    shard_directories: tuple[Path, ...], expected_shards: int | None
) -> tuple[list[dict[str, Any]], int]:
    if not shard_directories:
        raise ValueError("no shard manifests were found")
    manifests = [_load_manifest(directory) for directory in shard_directories]
    shard_count = int(manifests[0].get("shard_count", len(manifests)))
    if expected_shards is not None and shard_count != expected_shards:
        raise ValueError(f"manifests expect {shard_count} shards, not {expected_shards}")
    if len(manifests) != shard_count:
        raise ValueError(f"found {len(manifests)} of {shard_count} required shards")
    configuration_hashes = {str(manifest["configuration_hash"]) for manifest in manifests}
    total_states = {int(manifest["total_requested_states"]) for manifest in manifests}
    master_seeds = {int(manifest["master_seed"]) for manifest in manifests}
    commits = {str(manifest["git_commit"]) for manifest in manifests}
    if len(configuration_hashes) != 1 or len(total_states) != 1 or len(master_seeds) != 1:
        raise ValueError("shard configuration, total-state count, or master seed differs")
    if len(commits) != 1 or any(bool(manifest.get("git_dirty", True)) for manifest in manifests):
        raise ValueError("all shards must use the same clean Git commit")
    by_index = {int(manifest["shard_index"]): manifest for manifest in manifests}
    if set(by_index) != set(range(shard_count)):
        raise ValueError("shard indices are incomplete or duplicated")
    cursor = 0
    for index in range(shard_count):
        manifest = by_index[index]
        start = int(manifest["global_state_start"])
        stop = int(manifest["global_state_stop"])
        if start != cursor or stop <= start:
            raise ValueError("shard global ranges are not contiguous")
        if manifest.get("completed_at") is None:
            raise ValueError(f"shard {index} is not finalized")
        if int(manifest.get("table_counts", {}).get("states", -1)) != stop - start:
            raise ValueError(f"shard {index} state count does not match its global range")
        cursor = stop
    total = total_states.pop()
    if cursor != total:
        raise ValueError(f"shards cover {cursor} states but the plan requires {total}")
    ordered = [by_index[index] for index in range(shard_count)]
    return ordered, total


def _merged_manifest(
    run_id: str,
    manifests: list[dict[str, Any]],
    total_states: int,
) -> RunManifest:
    first = manifests[0]
    return RunManifest(
        schema_version=int(first["schema_version"]),
        run_id=run_id,
        created_at=utc_now(),
        completed_at=None,
        git_commit=str(first["git_commit"]),
        git_dirty=False,
        configuration_hash=str(first["configuration_hash"]),
        system_config_path=str(first["system_config_path"]),
        teacher_config_path=str(first["teacher_config_path"]),
        teacher_provider=str(first["teacher_provider"]),
        teacher_model_id=str(first["teacher_model_id"]),
        teacher_model_revision=str(first["teacher_model_revision"]),
        prompt_version=str(first["prompt_version"]),
        master_seed=int(first["master_seed"]),
        num_candidates=int(first["num_candidates"]),
        rollout_horizon_slots=int(first["rollout_horizon_slots"]),
        monte_carlo_rollouts=int(first["monte_carlo_rollouts"]),
        max_monte_carlo_rollouts=int(first["max_monte_carlo_rollouts"]),
        hardware={
            "merged_shards": [
                {
                    "shard_index": int(manifest["shard_index"]),
                    "slurm_job_id": manifest.get("slurm_job_id"),
                    "hardware": manifest.get("hardware", {}),
                }
                for manifest in manifests
            ]
        },
        software=dict(first["software"]),
        slurm_job_id=None,
        table_counts={},
        total_requested_states=total_states,
        global_state_start=0,
        global_state_stop=total_states,
        shard_index=0,
        shard_count=len(manifests),
        sampling_strategy=str(first.get("sampling_strategy", "sequential")),
        state_distribution=dict(first.get("state_distribution", {})),
        split_fractions=dict(first.get("split_fractions", {})),
    )


def _append_telemetry(shard_directories: tuple[Path, ...], output_directory: Path) -> None:
    metrics_output = output_directory / "teacher_server_metrics.jsonl"
    with metrics_output.open("w", encoding="utf-8") as destination:
        for shard_index, directory in enumerate(shard_directories):
            source = directory / "teacher_server_metrics.jsonl"
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record["shard_index"] = shard_index
                    destination.write(json.dumps(record, sort_keys=True) + "\n")

    gpu_output = output_directory / "gpu_metrics.csv"
    with gpu_output.open("w", newline="", encoding="utf-8") as destination_stream:
        writer: Any = None
        for shard_index, directory in enumerate(shard_directories):
            source = directory / "gpu_metrics.csv"
            if not source.exists():
                continue
            with source.open("r", newline="", encoding="utf-8") as source_stream:
                reader = csv.reader(source_stream)
                header = next(reader, None)
                if header is None:
                    continue
                if writer is None:
                    writer = csv.writer(destination_stream)
                    writer.writerow(["shard_index", *header])
                for row in reader:
                    writer.writerow([shard_index, *row])


def merge_shards(
    shard_directories: tuple[Path, ...],
    output_directory: str | Path,
    run_id: str,
    expected_shards: int | None = None,
    export_parquet: bool = True,
) -> RunManifest:
    shard_directories = tuple(
        directory
        for _, directory in sorted(
            (int(_load_manifest(directory)["shard_index"]), directory)
            for directory in shard_directories
        )
    )
    manifests, total_states = _validate_shards(shard_directories, expected_shards)
    output = Path(output_directory)
    manifest = _merged_manifest(run_id, manifests, total_states)
    writer = DatasetWriter(output, manifest, flush_every=128)
    seen_global_indices: set[int] = set()
    try:
        for directory in shard_directories:
            for table in TABLES:
                with (directory / f"{table}.jsonl").open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        record["run_id"] = run_id
                        writer.append(table, record)
                        if table == "states":
                            index = int(record["global_state_index"])
                            if index in seen_global_indices:
                                raise ValueError(f"duplicate global state index {index}")
                            seen_global_indices.add(index)
        if seen_global_indices != set(range(total_states)):
            raise ValueError("merged states do not cover the complete global index plan")
        merged = writer.finalize(export_parquet=export_parquet)
    except BaseException:
        writer.close()
        raise
    _append_telemetry(shard_directories, output)
    provenance = {
        "run_id": run_id,
        "source_shards": [
            {
                "directory": str(directory),
                "run_id": manifest["run_id"],
                "shard_index": manifest["shard_index"],
                "slurm_job_id": manifest.get("slurm_job_id"),
            }
            for directory, manifest in zip(shard_directories, manifests, strict=True)
        ],
    }
    with (output / "shards.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return merged
