"""World environment: 4 shapes × 4 colours = 16 objects."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

NUM_SHAPES: int = 4
NUM_COLORS: int = 4
INPUT_DIM: int = NUM_SHAPES + NUM_COLORS  # 8


@dataclass(frozen=True)
class WorldObject:
    s: int  # shape index  0-3
    c: int  # colour index 0-3

    def __post_init__(self) -> None:
        assert 0 <= self.s < NUM_SHAPES, f"shape {self.s} out of range"
        assert 0 <= self.c < NUM_COLORS, f"colour {self.c} out of range"


def all_objects() -> list[WorldObject]:
    """Return all 16 canonical objects in row-major order."""
    return [WorldObject(s=s, c=c) for s in range(NUM_SHAPES) for c in range(NUM_COLORS)]


def random_object(rng: np.random.Generator) -> WorldObject:
    """Sample a uniformly random object."""
    return WorldObject(s=int(rng.integers(NUM_SHAPES)), c=int(rng.integers(NUM_COLORS)))


def obj_to_tensor(obj: WorldObject) -> torch.Tensor:
    """One-hot encode: [shape_0..shape_3, colour_0..colour_3] → 8-dim float tensor."""
    x = torch.zeros(INPUT_DIM)
    x[obj.s] = 1.0
    x[NUM_SHAPES + obj.c] = 1.0
    return x


def obj_id(obj: WorldObject) -> int:
    """Canonical integer index (0-15) for the object."""
    return obj.s * NUM_COLORS + obj.c
