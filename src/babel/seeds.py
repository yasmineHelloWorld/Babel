from dataclasses import dataclass

import numpy as np


@dataclass
class RandomStreams:
    data: np.random.Generator
    environment: np.random.Generator
    speaker: np.random.Generator
    listener: np.random.Generator
    evaluation: np.random.Generator


def make_random_streams(seed: int) -> RandomStreams:
    """Create independent, reproducible RNG streams from one run seed."""
    children = np.random.SeedSequence(seed).spawn(5)
    return RandomStreams(*(np.random.default_rng(child) for child in children))
