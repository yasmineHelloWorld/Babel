"""Predictive and language-structure metrics for discrete messages."""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


def _entropy(values) -> float:
    counts = np.asarray(list(Counter(map(tuple, values) if np.asarray(values).ndim > 1 else values).values()), dtype=float)
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information(left, right) -> float:
    """Empirical MI in bits for two categorical variables."""
    left, right = list(left), list(right)
    if len(left) != len(right):
        raise ValueError("mutual information inputs must have equal lengths")
    if not left:
        return 0.0
    joint = Counter(zip(left, right))
    left_counts, right_counts = Counter(left), Counter(right)
    total = len(left)
    result = 0.0
    for (x, y), count in joint.items():
        p_xy = count / total
        result += p_xy * np.log2((count * total) / (left_counts[x] * right_counts[y]))
    return float(result)


def compute_language_metrics(messages, shape_ids, color_ids, object_ids=None) -> dict:
    """Measure codes from messages shaped ``(objects, length)`` or ``(objects, draws, length)``.

    Repeated draws support stochastic Speakers and make synonymy measurable. Labels and
    object IDs are repeated across draws so all empirical quantities use the same samples.
    """
    messages = np.asarray(messages, dtype=np.int64)
    if messages.ndim == 2:
        messages = messages[:, np.newaxis, :]
    if messages.ndim != 3 or messages.shape[0] == 0 or messages.shape[2] == 0:
        raise ValueError("messages must have shape (objects, length) or (objects, draws, length)")
    shapes, colors = np.asarray(shape_ids), np.asarray(color_ids)
    object_count, draw_count, message_length = messages.shape
    if len(shapes) != object_count or len(colors) != object_count:
        raise ValueError("one shape and color label is required per object")
    if draw_count == 0 or np.any(messages < 0):
        raise ValueError("messages must contain nonnegative symbol IDs and at least one draw")
    if object_ids is None:
        object_ids = np.arange(object_count)
    object_ids = np.asarray(object_ids)
    if len(object_ids) != object_count:
        raise ValueError("one object ID is required per object")

    flat_messages = messages.reshape(-1, message_length)
    flat_shapes = np.repeat(shapes, draw_count).tolist()
    flat_colors = np.repeat(colors, draw_count).tolist()
    flat_objects = np.repeat(object_ids, draw_count).tolist()
    message_tuples = [tuple(int(symbol) for symbol in message) for message in flat_messages]
    active_symbols = set(int(symbol) for symbol in flat_messages.flat)

    # Ambiguity: conditional entropy of object identity given its message, in bits.
    by_message = defaultdict(list)
    by_object = defaultdict(list)
    for message, object_id in zip(message_tuples, flat_objects):
        by_message[message].append(object_id)
        by_object[object_id].append(message)
    ambiguity = sum((len(ids) / len(flat_objects)) * _entropy(ids)
                    for ids in by_message.values())
    # Synonymy: conditional entropy of message given the object, in bits.
    synonymy = sum((len(codes) / len(flat_objects)) * _entropy(codes)
                   for codes in by_object.values())

    position_mi = {}
    position_entropies = {}
    for position in range(message_length):
        symbols = flat_messages[:, position].tolist()
        position_mi[f"{position}:shape"] = _mutual_information(symbols, flat_shapes)
        position_mi[f"{position}:color"] = _mutual_information(symbols, flat_colors)
        position_entropies[str(position)] = _entropy(symbols)

    return {
        "message_entropy_bits": _entropy(message_tuples),
        "position_entropies_bits": position_entropies,
        "shape_mutual_information_bits": _mutual_information(message_tuples, flat_shapes),
        "color_mutual_information_bits": _mutual_information(message_tuples, flat_colors),
        "position_property_mutual_information_bits": position_mi,
        "active_vocabulary": len(active_symbols),
        "active_symbols": sorted(active_symbols),
        "ambiguity_bits": float(ambiguity),
        "synonymy_bits": float(synonymy),
    }


def evaluate_language(objects: list[dict], speaker, listener) -> dict:
    """Evaluate deterministic argmax messages and report predictive and code metrics."""
    shape_correct = color_correct = exact_correct = 0
    messages = []
    rows = []
    for object_id, obj in enumerate(objects):
        probabilities = speaker.probabilities(object_id)
        message = np.argmax(probabilities, axis=-1).astype(np.int64)
        shape_probs, color_probs = listener.predict(message)
        shape_ok = int(np.argmax(shape_probs) == obj["shape_id"])
        color_ok = int(np.argmax(color_probs) == obj["color_id"])
        shape_correct += shape_ok
        color_correct += color_ok
        exact_correct += shape_ok * color_ok
        messages.append(message.tolist())
        rows.append({"shape": obj["shape_id"], "color": obj["color_id"], "message": message.tolist()})
    total = len(objects)
    language_metrics = compute_language_metrics(
        messages, [obj["shape_id"] for obj in objects], [obj["color_id"] for obj in objects]
    )
    return {"shape_accuracy": shape_correct / total, "color_accuracy": color_correct / total,
            "exact_accuracy": exact_correct / total, **language_metrics,
            "messages": messages, "assignments": rows}


def topographic_similarity(messages, feature_vectors) -> float:
    """Spearman correlation between pairwise Hamming distances, averaged-rank ties."""
    messages = np.asarray(messages)
    features = np.asarray(feature_vectors)
    if messages.ndim != 2 or features.ndim != 2 or len(messages) != len(features):
        raise ValueError("messages and feature_vectors must be 2-D arrays with matching rows")
    if len(messages) < 3:
        raise ValueError("at least three objects are required")
    message_distances, feature_distances = [], []
    for left, right in combinations(range(len(messages)), 2):
        message_distances.append(float(np.mean(messages[left] != messages[right])))
        feature_distances.append(float(np.mean(features[left] != features[right])))
    message_ranks, feature_ranks = _average_ranks(message_distances), _average_ranks(feature_distances)
    if np.std(message_ranks) == 0 or np.std(feature_ranks) == 0:
        return 0.0
    return float(np.corrcoef(message_ranks, feature_ranks)[0, 1])


def _average_ranks(values) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks
