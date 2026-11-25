"""Train-only transformations and reproducible stratified splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
