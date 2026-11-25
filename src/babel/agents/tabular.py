import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TabularSpeaker:
    """A learned categorical message policy indexed by observed objects."""

    def __init__(self, object_count: int, message_length: int, vocab_size: int, rng: np.random.Generator):
        self.logits = rng.normal(0.0, 0.01, (object_count, message_length, vocab_size))
        self.rng = rng

    def probabilities(self, object_id: int) -> np.ndarray:
        return _softmax(self.logits[object_id])

    def speak(self, object_id: int) -> tuple[np.ndarray, np.ndarray]:
        probs = self.probabilities(object_id)
        message = np.array([self.rng.choice(len(row), p=row) for row in probs], dtype=np.int64)
        return message, probs

    def update(self, object_id: int, message: np.ndarray, probs: np.ndarray, advantage: float,
               learning_rate: float, entropy_bonus: float) -> None:
        for position, (symbol, p) in enumerate(zip(message, probs)):
            grad_log_prob = -p.copy()
            grad_log_prob[symbol] += 1.0
            entropy_grad = -p * (np.log(np.maximum(p, 1e-12)) + (-np.sum(p * np.log(np.maximum(p, 1e-12)))))
            self.logits[object_id, position] += learning_rate * (advantage * grad_log_prob + entropy_bonus * entropy_grad)


class TabularListener:
    """Maps message IDs to attribute distributions; receives no input features."""

    def __init__(self, message_length: int, vocab_size: int, shape_count: int, color_count: int,
                 rng: np.random.Generator):
        self.shape_counts = np.ones((vocab_size,) * message_length + (shape_count,), dtype=np.float64)
        self.color_counts = np.ones((vocab_size,) * message_length + (color_count,), dtype=np.float64)
        # Tiny seeded perturbations keep the initialization reproducible without favoring a label.
        self.shape_counts += rng.random(self.shape_counts.shape) * 1e-6
        self.color_counts += rng.random(self.color_counts.shape) * 1e-6

    def predict(self, message: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(int(symbol) for symbol in message)
        shape = self.shape_counts[key]
        color = self.color_counts[key]
        return shape / shape.sum(), color / color.sum()

    def observe(self, message: np.ndarray, shape_id: int, color_id: int) -> None:
        key = tuple(int(symbol) for symbol in message)
        self.shape_counts[key][shape_id] += 1.0
        self.color_counts[key][color_id] += 1.0
