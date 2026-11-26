"""Train-only transformations and reproducible stratified splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from babel.split_guard import DataSplit


@dataclass
class PreparedTabularData:
    encoder: OneHotEncoder
    train_features: np.ndarray
    validation_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray


@dataclass
class PreparedVisionData:
    scaler: StandardScaler
    pca: PCA
    train_features: np.ndarray
    validation_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray
    explained_variance_ratio_sum: float


def prepare_tabular_data(features: np.ndarray, labels: np.ndarray, split: DataSplit) -> PreparedTabularData:
    """Fit categorical one-hot encoding only on training rows."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be 2D and have the same number of rows as labels")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    train = encoder.fit_transform(features[split.train])
    validation = encoder.transform(features[split.validation])
    test = encoder.transform(features[split.test])
    return PreparedTabularData(encoder, train, validation, test, labels[split.train], labels[split.validation], labels[split.test])


def prepare_vision_pca_data(
    features: np.ndarray,
    labels: np.ndarray,
    split: DataSplit,
    *,
    n_components: int = 32,
    seed: int = 0,
) -> PreparedVisionData:
    """Fit StandardScaler and PCA strictly on training rows, then transform val and test."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be 2D and have the same number of rows as labels")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    effective_components = min(n_components, len(split.train), features.shape[1])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features[split.train])
    val_scaled = scaler.transform(features[split.validation])
    test_scaled = scaler.transform(features[split.test])

    pca = PCA(n_components=effective_components, random_state=seed)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    test_pca = pca.transform(test_scaled)
    return PreparedVisionData(
        scaler=scaler,
        pca=pca,
        train_features=train_pca,
        validation_features=val_pca,
        test_features=test_pca,
        train_labels=labels[split.train],
        validation_labels=labels[split.validation],
        test_labels=labels[split.test],
        explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
    )
