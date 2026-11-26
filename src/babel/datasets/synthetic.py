from collections.abc import Iterable
from itertools import product

import numpy as np

SHAPES = ("circle", "square", "triangle")
COLORS = ("red", "green", "blue")


def generate_synthetic_world(rng: np.random.Generator | None = None) -> dict:
    """Return all shape-color combinations with one-hot feature vectors."""
    objects = []
    for object_id, (shape_id, color_id) in enumerate(product(range(len(SHAPES)), range(len(COLORS)))):
        features = np.zeros(len(SHAPES) + len(COLORS), dtype=np.float32)
        features[shape_id] = 1.0
        features[len(SHAPES) + color_id] = 1.0
        objects.append({"object_id": object_id, "shape_id": shape_id, "color_id": color_id, "features": features})
    if rng is not None:
        rng.shuffle(objects)
    return {"objects": objects, "shapes": SHAPES, "colors": COLORS}


def generate_synthetic_split(
    held_out_pairs: Iterable[tuple[int, int]] = ((2, 0),),
) -> dict:
    """Partition the 3x3 world into seen training objects and held-out pairs.

    Default holds out ``(2, 0)`` = ``("triangle", "red")`` so both attributes
    appear in training on other objects while the specific combination is unseen.
    """
    held_out = {tuple(map(int, pair)) for pair in held_out_pairs}
    world = generate_synthetic_world()
    objects = world["objects"]
    held_out_mask = np.asarray(
        [(obj["shape_id"], obj["color_id"]) in held_out for obj in objects],
        dtype=bool,
    )
    if not np.any(held_out_mask) or np.all(held_out_mask):
        raise ValueError("held_out_pairs must select a strict non-empty subset of the 3x3 world")
    train_objects = [obj for obj, is_held in zip(objects, held_out_mask) if not is_held]
    held_out_objects = [obj for obj, is_held in zip(objects, held_out_mask) if is_held]
    return {
        **world,
        "held_out_pairs": tuple(sorted(held_out)),
        "held_out_mask": held_out_mask,
        "train_objects": train_objects,
        "held_out_objects": held_out_objects,
    }
