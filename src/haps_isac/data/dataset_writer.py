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
