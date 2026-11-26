"""Core protocol contracts and typed data containers enforcing the information boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def assert_discrete_messages(
    messages: np.ndarray,
    *,
    vocab_size: int | None = None,
    message_length: int | None = None,
) -> np.ndarray:
    """Validate that messages are integer discrete symbol IDs within {0..K-1}^L."""
    arr = np.asarray(messages)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Messages must have an integer dtype, got {arr.dtype}")
    if arr.ndim not in (1, 2):
        raise ValueError(f"Messages must be 1D (L,) or 2D (batch, L), got shape {arr.shape}")
    if message_length is not None and arr.shape[-1] != message_length:
        raise ValueError(f"Expected message_length={message_length}, got {arr.shape[-1]}")
    if np.any(arr < 0):
        raise ValueError("Message symbol IDs must be nonnegative")
    if vocab_size is not None and np.any(arr >= vocab_size):
        raise ValueError(f"Message symbol IDs must be < vocab_size={vocab_size}")
    return arr


@dataclass(frozen=True)
class Batch:
    """A mini-batch or full evaluation split for communication games."""

    features: np.ndarray
    labels: np.ndarray
    object_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have matching batch length")
        if self.object_ids is not None and len(self.object_ids) != len(self.features):
            raise ValueError("object_ids must have matching batch length")


@runtime_checkable
class SpeakerAgent(Protocol):
    """Speaker observes input features x and emits discrete message IDs in {0..K-1}^L."""

    def act(self, features: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Return discrete integer message IDs for the given input features."""
        ...


@runtime_checkable
class ListenerAgent(Protocol):
    """Listener receives ONLY integer message IDs m — never inputs, labels, or Speaker state."""

    def predict(self, messages: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Predict target class distribution(s) from discrete integer message IDs only."""
        ...
