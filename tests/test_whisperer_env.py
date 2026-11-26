"""Unit tests for backend.env — world objects and tensor encoding."""
import numpy as np
import torch
import pytest

from backend.env import WorldObject, all_objects, random_object, obj_to_tensor, obj_id


def test_all_objects_count():
    objs = all_objects()
    assert len(objs) == 16


def test_all_objects_unique():
    objs = all_objects()
    pairs = {(o.s, o.c) for o in objs}
    assert len(pairs) == 16


def test_all_objects_valid_range():
    for o in all_objects():
        assert 0 <= o.s <= 3
        assert 0 <= o.c <= 3


def test_random_object_valid():
    rng = np.random.default_rng(42)
    for _ in range(200):
        o = random_object(rng)
        assert 0 <= o.s <= 3
        assert 0 <= o.c <= 3


def test_obj_to_tensor_shape():
    t = obj_to_tensor(WorldObject(s=0, c=0))
    assert t.shape == (8,)
    assert t.dtype == torch.float32


def test_obj_to_tensor_one_hot_shape():
    for s in range(4):
        t = obj_to_tensor(WorldObject(s=s, c=0))
        # shape one-hot: index s should be 1, others 0
        assert t[s].item() == pytest.approx(1.0)
        assert sum(t[:4]).item() == pytest.approx(1.0)


def test_obj_to_tensor_one_hot_color():
    for c in range(4):
        t = obj_to_tensor(WorldObject(s=0, c=c))
        # color one-hot: index 4+c should be 1
        assert t[4 + c].item() == pytest.approx(1.0)
        assert sum(t[4:]).item() == pytest.approx(1.0)


def test_obj_to_tensor_total_sum():
    for o in all_objects():
        t = obj_to_tensor(o)
        assert t.sum().item() == pytest.approx(2.0)


def test_obj_id_unique():
    ids = {obj_id(o) for o in all_objects()}
    assert len(ids) == 16


def test_obj_id_range():
    for o in all_objects():
        assert 0 <= obj_id(o) < 16
