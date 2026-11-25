import numpy as np

from babel.datasets.synthetic import generate_synthetic_world


class SyntheticEnvironment:
    """Samples objects from the nine-object synthetic grammar world."""

    def __init__(self, rng: np.random.Generator):
        self.world = generate_synthetic_world()
        self.rng = rng

    def sample(self) -> dict:
        return self.world["objects"][int(self.rng.integers(len(self.world["objects"]))) ]

    @property
    def objects(self) -> list[dict]:
        return self.world["objects"]
