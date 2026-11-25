"""Persistent, hash-checked train/validation/test indices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _indices_hash(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        array = np.asarray(values, dtype="<i8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    sha256: str


class SplitGuard:
    """Create splits once, persist them, and reject changed or overlapping indices."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".json")

    def create(self, labels: np.ndarray, *, seed: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> DataSplit:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) < 3:
            raise ValueError("labels must be a one-dimensional array with at least three entries")
        if len(fractions) != 3 or any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("fractions must be three positive values summing to 1")
        if self.path.exists() or self.metadata_path.exists():
            return self.load(labels=labels)

        rng = np.random.default_rng(seed)
        partitions: list[list[int]] = [[], [], []]
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            rng.shuffle(members)
            counts = _stratified_counts(len(members), fractions)
            offsets = np.cumsum((0, *counts))
            for partition, start, end in zip(partitions, offsets[:-1], offsets[1:]):
                partition.extend(int(index) for index in members[start:end])
        arrays = [np.asarray(partition, dtype=np.int64) for partition in partitions]
        for array in arrays:
            rng.shuffle(array)
        self._validate(*arrays, len(labels))
        digest = _indices_hash(*arrays)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, train=arrays[0], validation=arrays[1], test=arrays[2])
        self.metadata_path.write_text(json.dumps({"seed": seed, "fractions": fractions, "n_samples": len(labels), "labels_sha256": _array_hash(labels), "sha256": digest}, indent=2) + "\n", encoding="utf-8")
        return DataSplit(*arrays, digest)

    def create_held_out_combinations(
        self,
        features: np.ndarray,
        *,
        held_out: set[tuple[int, int]],
        columns: tuple[int, int] = (0, 5),
        seed: int = 0,
        validation_fraction: float = 0.15,
    ) -> DataSplit:
        """Keep selected categorical pairs entirely out of train/validation.

        The test partition contains every row matching a held-out pair. Remaining
        rows are stratified into train and validation using their observed pair as
        the stratum. Values in ``held_out`` are integer-coded feature values.
        """
        features = np.asarray(features)
        if features.ndim != 2 or max(columns) >= features.shape[1]:
            raise ValueError("features must be 2D and columns must refer to existing features")
        if not held_out:
            raise ValueError("at least one held-out combination is required")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        pairs = [tuple(int(value) for value in features[row, columns]) for row in range(len(features))]
        test = np.asarray([i for i, pair in enumerate(pairs) if pair in held_out], dtype=np.int64)
        seen = np.asarray([i for i, pair in enumerate(pairs) if pair not in held_out], dtype=np.int64)
        if len(test) == 0 or len(seen) == 0:
            raise ValueError("held_out combinations must match some rows and leave some rows for training")

        if self.path.exists() or self.metadata_path.exists():
            saved = self.load()
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if (metadata.get("split_type") != "held_out_combinations"
                    or metadata.get("held_out") != sorted([list(pair) for pair in held_out])
                    or metadata.get("columns") != list(columns)
                    or metadata.get("features_sha256") != _array_hash(features)):
                raise ValueError("Saved split does not match this dataset and held-out-combination configuration")
            return saved

        rng = np.random.default_rng(seed)
        train_parts: list[int] = []
        validation_parts: list[int] = []
        for pair in sorted(set(pairs[i] for i in seen)):
            members = np.asarray([i for i in seen if pairs[i] == pair], dtype=np.int64)
            rng.shuffle(members)
            n_validation = int(round(len(members) * validation_fraction))
            if len(members) > 1:
                n_validation = min(max(1, n_validation), len(members) - 1)
            validation_parts.extend(int(i) for i in members[:n_validation])
            train_parts.extend(int(i) for i in members[n_validation:])
        train = np.asarray(train_parts, dtype=np.int64)
        validation = np.asarray(validation_parts, dtype=np.int64)
        rng.shuffle(train)
        rng.shuffle(validation)
        rng.shuffle(test)
        self._validate(train, validation, test, len(features))
        digest = _indices_hash(train, validation, test)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, train=train, validation=validation, test=test)
        self.metadata_path.write_text(json.dumps({
            "seed": seed,
            "n_samples": len(features),
            "features_sha256": _array_hash(features),
            "sha256": digest,
            "split_type": "held_out_combinations",
            "held_out": sorted([list(pair) for pair in held_out]),
            "columns": list(columns),
            "validation_fraction": validation_fraction,
        }, indent=2) + "\n", encoding="utf-8")
        return DataSplit(train, validation, test, digest)

    def load(self, *, labels: np.ndarray | None = None) -> DataSplit:
        if not self.path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"Split files do not exist: {self.path} and {self.metadata_path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        with np.load(self.path, allow_pickle=False) as saved:
            arrays = [np.asarray(saved[key], dtype=np.int64) for key in ("train", "validation", "test")]
        n_samples = int(metadata["n_samples"])
        self._validate(*arrays, n_samples)
        actual_hash = _indices_hash(*arrays)
        if actual_hash != metadata.get("sha256"):
            raise ValueError("Saved split indices do not match their SHA-256 metadata")
        if labels is not None and len(labels) != n_samples:
            raise ValueError("Saved split was created for a different number of samples")
        if labels is not None and metadata.get("labels_sha256") not in (None, _array_hash(np.asarray(labels))):
            raise ValueError("Saved split was created for different labels or row ordering")
        return DataSplit(*arrays, actual_hash)

    @staticmethod
    def _validate(train: np.ndarray, validation: np.ndarray, test: np.ndarray, n_samples: int) -> None:
        arrays = (train, validation, test)
        combined = np.concatenate(arrays)
        if np.any(combined < 0) or np.any(combined >= n_samples):
            raise ValueError("Split index is outside the dataset")
        if len(np.unique(combined)) != len(combined):
            raise ValueError("Train, validation, and test indices must not overlap")
        if len(combined) != n_samples:
            raise ValueError("Splits must cover every dataset row exactly once")


def _stratified_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate each class across partitions while keeping all counts positive when possible."""
    raw = np.asarray(fractions) * size
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: size - int(counts.sum())]:
        counts[index] += 1
    if size >= 3:
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return tuple(int(value) for value in counts)
