"""Reproducible Fashion-MNIST IDX loader with local caching."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

FASHION_MNIST_LABELS = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
FASHION_MNIST_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
FASHION_MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


@dataclass(frozen=True)
class VisionDataset:
    name: str
    features: np.ndarray
    labels: np.ndarray
    label_names: tuple[str, ...]
    official_train_count: int
    source_url: str = FASHION_MNIST_BASE_URL


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic number {magic} in {path}")
        buf = stream.read(count * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(count, rows * cols)
    return (arr.astype(np.float32) / 255.0)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic number {magic} in {path}")
        buf = stream.read(count)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_fashion_mnist(
    data_dir: str | Path = "data",
    *,
    download: bool = True,
    max_samples: int | None = None,
) -> VisionDataset:
    """Load Fashion-MNIST from cached IDX .gz files, downloading once if permitted."""
    root = Path(data_dir) / "fashion-mnist"
    root.mkdir(parents=True, exist_ok=True)
    for filename in FASHION_MNIST_FILES:
        target = root / filename
        if not target.exists():
            if not download:
                raise FileNotFoundError(f"Fashion-MNIST file not found: {target}")
            url = f"{FASHION_MNIST_BASE_URL}/{filename}"
            try:
                with urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
            except Exception:
                target.unlink(missing_ok=True)
                raise

    train_x = _read_idx_images(root / "train-images-idx3-ubyte.gz")
    train_y = _read_idx_labels(root / "train-labels-idx1-ubyte.gz")
    test_x = _read_idx_images(root / "t10k-images-idx3-ubyte.gz")
    test_y = _read_idx_labels(root / "t10k-labels-idx1-ubyte.gz")
    features = np.concatenate([train_x, test_x], axis=0)
    labels = np.concatenate([train_y, test_y], axis=0)
    official_train = len(train_y)
    if max_samples is not None:
        if max_samples < 3:
            raise ValueError("max_samples must be >= 3")
        features = features[:max_samples]
        labels = labels[:max_samples]
        official_train = min(official_train, max_samples)
    return VisionDataset(
        name="fashion_mnist",
        features=features,
        labels=labels,
        label_names=FASHION_MNIST_LABELS,
        official_train_count=official_train,
        source_url=FASHION_MNIST_BASE_URL,
    )
