"""Crash-resumable verified-demonstration and evaluation-log persistence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from haps_isac.data.demonstration_schema import RunManifest, json_safe, utc_now

TABLES = (
    "states",
    "teacher_requests",
    "candidates",
    "rollouts",
    "selections",
    "demonstrations",
)


def _read_valid_json_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                break
            records.append(value)
    return records


def prepare_resume_directory(
    directory: str | Path,
    expected_candidates: int,
    shortlist_size: int,
) -> int:
    """Trim an interrupted shard to its last complete per-state transaction."""

    root = Path(directory)
    if not (root / "manifest.json").exists():
        return 0
    tables = {table: _read_valid_json_objects(root / f"{table}.jsonl") for table in TABLES}
    requests = {str(record["state_id"]): record for record in tables["teacher_requests"]}
    selections = {str(record["state_id"]): record for record in tables["selections"]}
    demonstration_ids = {str(record["state_id"]) for record in tables["demonstrations"]}
    candidate_counts: dict[str, int] = {}
    rollout_counts: dict[str, int] = {}
    for record in tables["candidates"]:
        state_id = str(record["state_id"])
        candidate_counts[state_id] = candidate_counts.get(state_id, 0) + 1
    for record in tables["rollouts"]:
        state_id = str(record["state_id"])
        rollout_counts[state_id] = rollout_counts.get(state_id, 0) + 1

    complete_ids: list[str] = []
    for state in tables["states"]:
        state_id = str(state["state_id"])
        request = requests.get(state_id)
        selection = selections.get(state_id)
        if request is None or selection is None:
            break
        if selection.get("acceptance_status") == "accepted":
            verification_rollouts = int(selection.get("verification_rollouts", 0))
            expected_rollouts = (shortlist_size + 2) * verification_rollouts
            complete = (
                state_id in demonstration_ids
                and candidate_counts.get(state_id, 0) == expected_candidates
                and verification_rollouts > 0
                and rollout_counts.get(state_id, 0) == expected_rollouts
            )
        else:
            complete = candidate_counts.get(state_id, 0) == 0 and state_id not in demonstration_ids
        if not complete:
            break
        complete_ids.append(state_id)

    keep = set(complete_ids)
    for table, records in tables.items():
        destination = root / f"{table}.jsonl"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{table}.resume.", suffix=".tmp", dir=root, text=True
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                if str(record.get("state_id", "")) in keep:
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    for stale in (
        *root.glob("*.parquet"),
        root / "audit.json",
        root / "teacher_quality_report.json",
    ):
        if stale.exists():
            stale.unlink()
    return len(complete_ids)


class DatasetWriter:
    """Append-only JSONL is canonical; Parquet is an optional plotting export."""

    def __init__(
        self,
        directory: str | Path,
        manifest: RunManifest,
        flush_every: int = 32,
        resume: bool = False,
    ) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.exists() and not resume:
            raise FileExistsError(f"{self.directory} already contains a dataset; pass resume=True")
        self.flush_every = flush_every
        self.manifest = manifest
        self.counts = {name: 0 for name in TABLES}
        self._handles: dict[str, TextIO] = {}
        self._pending = 0
        if resume:
            self._load_resume_state()
        self._write_manifest()

    def _load_resume_state(self) -> None:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("run_id") != self.manifest.run_id:
            raise ValueError("resume run_id does not match the existing dataset")
        if existing.get("configuration_hash") != self.manifest.configuration_hash:
            raise ValueError("resume configuration does not match the existing dataset")
        expected_provenance = {
            "global_state_start": self.manifest.global_state_start,
            "global_state_stop": self.manifest.global_state_stop,
            "total_requested_states": self.manifest.total_requested_states,
            "shard_index": self.manifest.shard_index,
            "shard_count": self.manifest.shard_count,
            "master_seed": self.manifest.master_seed,
        }
        if any(
            existing.get(key, 0 if key == "global_state_start" else None) != value
            for key, value in expected_provenance.items()
        ):
            raise ValueError("resume global state range does not match the existing dataset")
        self.manifest = replace(
            self.manifest,
            created_at=str(existing.get("created_at", self.manifest.created_at)),
        )
        for table in TABLES:
            path = self.directory / f"{table}.jsonl"
            if path.exists():
                with path.open("r", encoding="utf-8") as stream:
                    self.counts[table] = sum(1 for line in stream if line.strip())

    def _write_manifest(self) -> None:
        current = replace(self.manifest, table_counts=dict(self.counts))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".manifest.",
            suffix=".tmp",
            dir=self.directory,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(json_safe(current), stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.manifest_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _handle(self, table: str) -> TextIO:
        if table not in TABLES:
            raise ValueError(f"unknown dataset table: {table}")
        if table not in self._handles:
            self._handles[table] = (self.directory / f"{table}.jsonl").open(
                "a",
                encoding="utf-8",
            )
        return self._handles[table]

    def append(self, table: str, record: Any) -> None:
        payload = json_safe(record)
        if not isinstance(payload, Mapping):
            raise TypeError("dataset records must serialize to objects")
        self._handle(table).write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self.counts[table] += 1
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        self._pending = 0
        self._write_manifest()

    def close(self) -> None:
        if any(not handle.closed for handle in self._handles.values()):
            self.flush()
        for handle in self._handles.values():
            handle.close()

    def finalize(self, export_parquet: bool = False) -> RunManifest:
        self.close()
        errors: tuple[str, ...] = ()
        if export_parquet:
            errors = self.export_parquet()
        self.manifest = replace(
            self.manifest,
            completed_at=utc_now(),
            table_counts=dict(self.counts),
            export_errors=errors,
        )
        self._write_manifest()
        return self.manifest

    def export_parquet(self) -> tuple[str, ...]:
        """Export flattened plotting tables when the optional data stack is installed."""

        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            return ("Parquet export skipped: install the project data extra",)

        errors: list[str] = []
        for table in TABLES:
            source = self.directory / f"{table}.jsonl"
            if not source.exists():
                continue
            try:
                with source.open("r", encoding="utf-8") as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                frame = pd.json_normalize(rows, sep=".")
                frame.to_parquet(self.directory / f"{table}.parquet", index=False)
            except (OSError, TypeError, ValueError) as error:
                errors.append(f"{table}: {type(error).__name__}: {error}")
        return tuple(errors)

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()
