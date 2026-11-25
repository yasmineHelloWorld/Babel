"""Capacity-matched classical baselines for tabular communication tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from babel.config import TabularConfig
from babel.datasets.tabular import load_tabular_dataset
from babel.preprocessing import prepare_tabular_data
from babel.split_guard import SplitGuard


@dataclass(frozen=True)
class BaselinePredictions:
    """Predictions on the same train/validation/test rows for each method."""

    predictions: dict[str, dict[str, np.ndarray]]
    budget: dict[str, int | str]

    def save(self, path: str | Path) -> Path:
        """Save prediction vectors and budget metadata in a compressed NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{method}__{split}": values
                  for method, splits in self.predictions.items()
                  for split, values in splits.items()}
        arrays.update({f"budget__{key}": np.asarray(value) for key, value in self.budget.items()})
        np.savez_compressed(path, **arrays)
        return path


def run_tabular_baselines(
    config: TabularConfig,
    *,
    message_symbols: int = 3,
    message_length: int = 2,
    output_path: str | Path | None = None,
) -> BaselinePredictions:
    """Run all Phase 4 baselines on the persisted Phase 3 split."""
    config.validate()
    dataset = load_tabular_dataset(config.dataset, config.data_dir)
    guard = SplitGuard(config.split_path)
    if config.held_out_combinations:
        split = guard.create_held_out_combinations(
            dataset.features,
            held_out=set(config.held_out_combinations),
            columns=config.held_out_columns,
            seed=config.seed,
        )
    else:
        split = guard.load(labels=dataset.labels)
    prepared = prepare_tabular_data(dataset.features, dataset.labels, split)
    result = fit_predict_baselines(
        prepared.train_features, prepared.train_labels,
        prepared.validation_features, prepared.validation_labels,
        prepared.test_features, prepared.test_labels,
        message_symbols=message_symbols, message_length=message_length, seed=config.seed,
    )
    # Include ground truth and provenance so each saved prediction is directly
    # usable for confusion matrices and split-aware evaluation.
    result = BaselinePredictions(
        {**result.predictions, "truth": {
            "train": prepared.train_labels,
            "validation": prepared.validation_labels,
            "test": prepared.test_labels,
        }},
        {**result.budget, "dataset": config.dataset, "split_sha256": split.sha256},
    )
    if output_path is not None:
        result.save(output_path)
    return result


def fit_predict_baselines(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    message_symbols: int,
    message_length: int,
    seed: int = 0,
    oracle_c: float = 1.0,
) -> BaselinePredictions:
    """Fit full-data, vector-quantized, and size-limited tree baselines.

    K-Means and the tree are limited to at most ``K ** L`` discrete codes/leaves.
    Their decoders are trained with labels from training rows only. Validation
    arrays are accepted to keep the split contract explicit; these fixed
    baselines do not tune on validation data.
    """
    arrays = [np.asarray(x) for x in (train_features, validation_features, test_features)]
    labels = [np.asarray(y) for y in (train_labels, validation_labels, test_labels)]
    if any(x.ndim != 2 for x in arrays):
        raise ValueError("features must be two-dimensional encoded matrices")
    if any(len(x) != len(y) for x, y in zip(arrays, labels)):
        raise ValueError("each feature split must align with its labels")
    if len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("all feature splits must have the same encoded columns")
    if message_symbols < 2 or message_length < 1:
        raise ValueError("message_symbols must be >= 2 and message_length must be >= 1")
    capacity = message_symbols ** message_length
    n_clusters = min(capacity, len(arrays[0]))
    if n_clusters < 1:
        raise ValueError("training split must not be empty")

    # Oracle sees raw encoded features and establishes an upper reference.
    oracle = LogisticRegression(C=oracle_c, max_iter=2000, random_state=seed)
    oracle.fit(arrays[0], labels[0])
    output: dict[str, dict[str, np.ndarray]] = {
        "full_data_oracle": {name: oracle.predict(x) for name, x in zip(("train", "validation", "test"), arrays)}
    }

    # Quantization creates discrete codes; the listener/decoder sees codes only.
    quantizer = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    train_codes = quantizer.fit_predict(arrays[0])
    cluster_labels = np.unique(labels[0])
    code_to_label = np.empty(n_clusters, dtype=labels[0].dtype)
    for code in range(n_clusters):
        members = labels[0][train_codes == code]
        if len(members):
            values, counts = np.unique(members, return_counts=True)
            code_to_label[code] = values[np.argmax(counts)]
        else:
            # K-Means should not produce empty clusters, but remain deterministic
            # and well-defined if a backend does.
            code_to_label[code] = cluster_labels[0]
    output["kmeans_vq"] = {
        name: code_to_label[quantizer.predict(x)]
        for name, x in zip(("train", "validation", "test"), arrays)
    }

    tree = DecisionTreeClassifier(max_leaf_nodes=capacity, random_state=seed)
    tree.fit(arrays[0], labels[0])
    output["size_limited_tree"] = {
        name: tree.predict(x) for name, x in zip(("train", "validation", "test"), arrays)
    }
    return BaselinePredictions(output, {
        "message_symbols": message_symbols,
        "message_length": message_length,
        "code_capacity": capacity,
        "kmeans_clusters": n_clusters,
        "tree_max_leaf_nodes": capacity,
        "oracle": "LogisticRegression",
    })
