"""Random high-level baseline followed by the common repair layer."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, num_pairs: int, seed: int) -> None:
        self.num_pairs = num_pairs
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, object]:
        del observation
        return {
            "pair": int(self.rng.integers(0, self.num_pairs + 1)),
            "ris_code": 0,
            "continuous": np.asarray(
                [
                    self.rng.uniform(0.2, 1.0),
                    self.rng.uniform(0.0, 1.0),
                    self.rng.uniform(0.0, 0.5),
                    0.0,
                    self.rng.uniform(-1.0, 1.0),
                    0.0,
                    self.rng.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            ),
        }
