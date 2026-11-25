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
