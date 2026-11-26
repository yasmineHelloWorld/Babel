"""Tests for Fashion-MNIST IDX loading, train-only PCA leakage guard, and fair baselines."""

import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from babel.baselines import fit_predict_baselines
from babel.datasets.vision import load_fashion_mnist
from babel.metrics import compute_classification_metrics
from babel.preprocessing import prepare_vision_pca_data
from babel.split_guard import SplitGuard


def _write_synthetic_idx(root: Path, n_train: int = 40, n_test: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fdir = root / "fashion-mnist"
    fdir.mkdir(parents=True, exist_ok=True)
    for prefix, count in (("train", n_train), ("t10k", n_test)):
        labels = np.arange(count, dtype=np.uint8) % 5
        # Class-correlated signal in pixel values so PCA captures separable variance.
        images = rng.integers(0, 30, size=(count, 28, 28), dtype=np.uint8)
        for idx, lbl in enumerate(labels):
            images[idx, lbl * 4 : lbl * 4 + 4, :] = 220
        with gzip.open(fdir / f"{prefix}-images-idx3-ubyte.gz", "wb") as stream:
            stream.write(struct.pack(">IIII", 2051, count, 28, 28))
            stream.write(images.tobytes())
        with gzip.open(fdir / f"{prefix}-labels-idx1-ubyte.gz", "wb") as stream:
            stream.write(struct.pack(">II", 2049, count))
            stream.write(labels.tobytes())


class VisionAndBaselineTests(unittest.TestCase):
    def test_fashion_mnist_idx_loader_reads_cached_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_idx(Path(tmp), n_train=30, n_test=15)
            dataset = load_fashion_mnist(tmp, download=False)
            self.assertEqual(dataset.features.shape, (45, 784))
            self.assertEqual(dataset.labels.shape, (45,))
            self.assertEqual(dataset.official_train_count, 30)
            self.assertTrue(np.all((0.0 <= dataset.features) & (dataset.features <= 1.0)))

    def test_vision_pca_fits_strictly_on_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_idx(Path(tmp), n_train=40, n_test=20)
            dataset = load_fashion_mnist(tmp, download=False)
            split = SplitGuard(Path(tmp) / "fmnist.npz").create(dataset.labels, seed=3)

            clean = prepare_vision_pca_data(dataset.features, dataset.labels, split, n_components=8, seed=0)
            corrupted_features = dataset.features.copy()
            # Extreme values on test rows must have ZERO effect on scaler/PCA or train/val output!
            corrupted_features[split.test] = 9999.0
            corrupted = prepare_vision_pca_data(corrupted_features, dataset.labels, split, n_components=8, seed=0)

            np.testing.assert_allclose(clean.scaler.mean_, corrupted.scaler.mean_)
            np.testing.assert_allclose(clean.pca.components_, corrupted.pca.components_)
            np.testing.assert_allclose(clean.train_features, corrupted.train_features)
            np.testing.assert_allclose(clean.validation_features, corrupted.validation_features)

    def test_baselines_respect_symbol_budget_and_output_probabilities(self):
        rng = np.random.default_rng(12)
        x_tr = rng.normal(size=(90, 8))
        y_tr = np.repeat(np.arange(3), 30)
        x_tr[y_tr == 0, 0] += 3.0
        x_tr[y_tr == 1, 1] += 3.0
        x_tr[y_tr == 2, 2] += 3.0
        x_va = rng.normal(size=(30, 8))
        y_va = np.repeat(np.arange(3), 10)
        x_te = rng.normal(size=(30, 8))
        y_te = np.repeat(np.arange(3), 10)
        x_te[y_te == 0, 0] += 3.0
        x_te[y_te == 1, 1] += 3.0
        x_te[y_te == 2, 2] += 3.0

        res = fit_predict_baselines(
            x_tr, y_tr, x_va, y_va, x_te, y_te, message_symbols=4, message_length=1, seed=0
        )
        self.assertEqual(res.budget["code_capacity"], 4)
        self.assertLessEqual(res.budget["kmeans_clusters"], 4)
        self.assertLessEqual(res.budget["tree_actual_leaves"], 4)
        for method in ("full_data_oracle", "full_data_oracle_mlp", "kmeans_vq", "size_limited_tree"):
            self.assertIn(method, res.predictions)
            self.assertIn(method, res.probabilities)
            metrics = compute_classification_metrics(
                y_te, res.predictions[method]["test"], res.probabilities[method]["test"]
            )
            self.assertIsNotNone(metrics["roc_auc_ovr"])
            self.assertGreater(metrics["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()