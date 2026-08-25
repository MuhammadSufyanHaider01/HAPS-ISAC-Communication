"""Freshness- and queue-urgency greedy baseline."""

from __future__ import annotations

import numpy as np


class GreedyPolicy:
    def __init__(self, num_pairs: int) -> None:
        self.num_pairs = num_pairs

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, object]:
        pair_tokens = observation["pair_tokens"]
        age_urgency = pair_tokens[:, 0] + pair_tokens[:, 1]
        waiting_urgency = pair_tokens[:, 13]
        pair = int(np.argmax(age_urgency + 0.5 * waiting_urgency)) + 1
        aosi = float(observation["global_features"][0])
        uncertainty = float(
            np.linalg.norm(observation["global_features"][5:15])
        )
        sensing_fraction = float(np.clip(0.2 + 0.4 * aosi + 0.1 * uncertainty, 0.2, 0.7))
        near_age = float(pair_tokens[pair - 1, 0])
        far_age = float(pair_tokens[pair - 1, 1])
        near_fraction = 0.15 if far_age >= near_age else 0.30
        return {
            "pair": pair,
            "ris_code": 0,
            "continuous": np.asarray(
                [
                    0.9,
                    1.0 - sensing_fraction,
                    near_fraction,
                    0.0,
                    0.0,
                    0.0,
                    float(np.clip(0.4 + aosi, 0.0, 1.0)),
                ],
                dtype=np.float32,
            ),
        }
