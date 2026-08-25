"""Content-addressed, atomic teacher-query caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible content deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key_for(
    model_id: str,
    model_revision: str,
    prompt_hash: str,
    sampling: Mapping[str, Any],
    seed: int,
) -> str:
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_hash": prompt_hash,
        "sampling": dict(sampling),
        "seed": seed,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class QueryCache:
    """One JSON object per key, written atomically for safe job resumption."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"invalid cache entry at {path}")
        return value

    def put(self, key: str, value: Mapping[str, Any]) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(dict(value), stream, sort_keys=True, ensure_ascii=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return path
