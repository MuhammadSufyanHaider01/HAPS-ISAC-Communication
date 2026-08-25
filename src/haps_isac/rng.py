"""Independent reproducible random-number streams."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

STREAM_NAMES = ("scenario", "channels", "target", "sensing", "policy")


@dataclass
class RandomStreams:
    """Named generators derived deterministically from one master seed."""

    generators: dict[str, np.random.Generator]

    @classmethod
    def from_seed(cls, seed: int) -> RandomStreams:
        sequence = np.random.SeedSequence(seed)
        children = sequence.spawn(len(STREAM_NAMES))
        return cls(
            {
                name: np.random.default_rng(child)
                for name, child in zip(STREAM_NAMES, children, strict=True)
            }
        )

    def __getitem__(self, name: str) -> np.random.Generator:
        return self.generators[name]

    def clone(self) -> RandomStreams:
        cloned: dict[str, np.random.Generator] = {}
        for name, generator in self.generators.items():
            new_generator = np.random.default_rng()
            new_generator.bit_generator.state = copy.deepcopy(generator.bit_generator.state)
            cloned[name] = new_generator
        return RandomStreams(cloned)
