"""Numerical loading and batching for verified demonstration datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class DemonstrationBatch:
    pair_tokens: npt.NDArray[np.float32]
    pair_mask: npt.NDArray[np.int8]
    global_features: npt.NDArray[np.float32]
    virtual_queues: npt.NDArray[np.float32]
    previous_action: npt.NDArray[np.float32]
    scheduled_pair: npt.NDArray[np.int64]
    continuous_action: npt.NDArray[np.float32]
    quality_weight: npt.NDArray[np.float32]


class DatasetLoader:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest: dict[str, Any] = json.load(stream)

    def iter_table(self, table: str) -> Iterator[dict[str, Any]]:
        path = self.directory / f"{table}.jsonl"
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object record")
                yield value

    def demonstrations(self, split: str | None = None) -> list[dict[str, Any]]:
        return [
            record
            for record in self.iter_table("demonstrations")
            if split is None or record["split"] == split
        ]

    def batches(
        self,
        batch_size: int,
        split: str = "train",
        shuffle_seed: int | None = None,
    ) -> Iterator[DemonstrationBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records = self.demonstrations(split)
        indices = np.arange(len(records))
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = [records[int(index)] for index in indices[start : start + batch_size]]
            yield _stack_batch(selected)


def _stack_batch(records: list[dict[str, Any]]) -> DemonstrationBatch:
    if not records:
        raise ValueError("cannot stack an empty demonstration batch")
    observations = [record["observation"] for record in records]
    actions = [record["selected_action"] for record in records]
    continuous_names = (
        "eta_haps",
        "eta_communication",
        "eta_near",
        "eta_jamming",
        "aav_heading_rad",
        "aav_speed_fraction",
        "eta_cpu",
    )
    return DemonstrationBatch(
        pair_tokens=np.asarray([item["pair_tokens"] for item in observations], dtype=np.float32),
        pair_mask=np.asarray([item["pair_mask"] for item in observations], dtype=np.int8),
        global_features=np.asarray(
            [item["global_features"] for item in observations], dtype=np.float32
        ),
        virtual_queues=np.asarray(
            [item["virtual_queues"] for item in observations], dtype=np.float32
        ),
        previous_action=np.asarray(
            [item["previous_action"] for item in observations], dtype=np.float32
        ),
        scheduled_pair=np.asarray([item["pair"] for item in actions], dtype=np.int64),
        continuous_action=np.asarray(
            [[item[name] for name in continuous_names] for item in actions],
            dtype=np.float32,
        ),
        quality_weight=np.asarray(
            [record["quality_weight"] for record in records], dtype=np.float32
        ),
    )
