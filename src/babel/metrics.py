"""Predictive and language-structure metrics for discrete messages."""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    return float(max(0.0, result))


def positional_disentanglement(messages, attributes) -> float:
    """Chaabouni et al. (2020) PosDis over discrete messages and ground-truth attributes.

    For message length L and A attributes:
        PosDis = (1/L) * sum_i (I(m_i; a_{i*}) - max_{j != i*} I(m_i; a_j)) / H(m_i)
    where i* = argmax_j I(m_i; a_j) and positions with H(m_i) == 0 contribute 0.
    """
    msg_arr = np.asarray(messages, dtype=np.int64)
    attr_arr = np.asarray(attributes)
    if msg_arr.ndim != 2 or attr_arr.ndim != 2:
        raise ValueError("messages (N, L) and attributes (N, A) must both be 2D arrays")
    if len(msg_arr) != len(attr_arr) or len(msg_arr) == 0:
        raise ValueError("messages and attributes must have matching non-zero row counts")
    if msg_arr.shape[1] == 0 or attr_arr.shape[1] < 2:
        raise ValueError("PosDis requires at least 1 message position and 2 attributes")

    message_length = msg_arr.shape[1]
    n_attrs = attr_arr.shape[1]
    total_score = 0.0
    for pos in range(message_length):
        symbols = msg_arr[:, pos].tolist()
        h_m = _entropy(symbols)
        if h_m <= 1e-12:
            continue
        mis = [_mutual_information(symbols, attr_arr[:, j].tolist()) for j in range(n_attrs)]
        best_idx = int(np.argmax(mis))
        best_mi = mis[best_idx]
        second_mi = max(mis[j] for j in range(n_attrs) if j != best_idx)
        total_score += (best_mi - second_mi) / h_m
    return float(total_score / message_length)


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

    attr_matrix = np.column_stack([flat_shapes, flat_colors])
    posdis = positional_disentanglement(flat_messages, attr_matrix)
    h_objects = _entropy(flat_objects)
    object_mi = max(0.0, float(h_objects - ambiguity))

    return {
        "message_entropy_bits": _entropy(message_tuples),
        "position_entropies_bits": position_entropies,
        "shape_mutual_information_bits": _mutual_information(message_tuples, flat_shapes),
        "color_mutual_information_bits": _mutual_information(message_tuples, flat_colors),
        "object_mutual_information_bits": object_mi,
        "position_property_mutual_information_bits": position_mi,
        "posdis": posdis,
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


def topographic_similarity_with_pvalue(
    messages,
    feature_vectors,
    *,
    n_permutations: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute topographic similarity and a two-sided permutation p-value."""
    msg_arr = np.asarray(messages)
    feat_arr = np.asarray(feature_vectors)
    observed = topographic_similarity(msg_arr, feat_arr)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    rng = np.random.default_rng(seed)
    extreme_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(msg_arr))
        null_val = topographic_similarity(msg_arr[perm], feat_arr)
        if abs(null_val) >= abs(observed) - 1e-12:
            extreme_count += 1
    p_value = (extreme_count + 1) / (n_permutations + 1)
    return float(observed), float(p_value)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    *,
    labels: Sequence[int] | None = None,
) -> dict:
    """Compute standard ML metrics: Accuracy, Precision/Recall/F1, Confusion Matrix, ROC AUC."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or len(yt) != len(yp):
        raise ValueError("y_true and y_pred must be 1D arrays of equal length")
    present_labels = list(labels) if labels is not None else sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=present_labels)
    roc_auc: float | None = None
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=np.float64)
        unique_true = np.unique(yt)
        if len(unique_true) >= 2:
            try:
                if proba.ndim == 2 and proba.shape[1] == 2 and len(present_labels) == 2:
                    roc_auc = float(roc_auc_score(yt, proba[:, 1]))
                elif proba.ndim == 2 and len(unique_true) == proba.shape[1]:
                    roc_auc = float(roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=present_labels))
                elif proba.ndim == 1 and len(unique_true) == 2:
                    roc_auc = float(roc_auc_score(yt, proba))
            except ValueError:
                roc_auc = None

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(precision_score(yt, yp, labels=present_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(yt, yp, labels=present_labels, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, labels=present_labels, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(yt, yp, labels=present_labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, labels=present_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(yt, yp, labels=present_labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": [int(v) for v in present_labels],
        "roc_auc_ovr": roc_auc,
    }


def compute_multi_attribute_metrics(y_true, y_pred) -> dict:
    """Compute exact joint and per-attribute accuracy for multi-attribute targets (N, A)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 2 or yp.shape != yt.shape:
        raise ValueError("y_true and y_pred must be 2D arrays with identical shape")
    per_attr = [float(np.mean(yt[:, col] == yp[:, col])) for col in range(yt.shape[1])]
    exact = float(np.mean(np.all(yt == yp, axis=1)))
    return {
        "exact_accuracy": exact,
        "per_attribute_accuracy": per_attr,
        "mean_attribute_accuracy": float(np.mean(per_attr)),
    }


def _align_messages_hungarian(
    source_messages: np.ndarray,
    target_messages: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Align source_messages to target_messages via label-free Hungarian symbol matching."""
    n_samples, message_length = source_messages.shape
    if n_samples == 0:
        return source_messages.copy()
    pos_perms = list(permutations(range(message_length))) if message_length <= 3 else [tuple(range(message_length))]
    best_score = -1
    best_aligned = source_messages
    for pos_perm in pos_perms:
        reordered = source_messages[:, pos_perm]
        candidate = np.empty_like(reordered)
        score = 0
        for pos in range(message_length):
            cooccur = np.zeros((vocab_size, vocab_size), dtype=np.int64)
            np.add.at(cooccur, (reordered[:, pos], target_messages[:, pos]), 1)
            row_ind, col_ind = linear_sum_assignment(-cooccur)
            mapping = np.arange(vocab_size, dtype=np.int64)
            mapping[row_ind] = col_ind
            candidate[:, pos] = mapping[reordered[:, pos]]
            score += int(cooccur[row_ind, col_ind].sum())
        if score > best_score:
            best_score = score
            best_aligned = candidate
    return best_aligned


def _accuracy_from_listener(predict_fn: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, ...]], messages: np.ndarray, targets: np.ndarray) -> float:
    preds = predict_fn(messages)
    if isinstance(preds, tuple):
        pred_ids = np.column_stack([np.argmax(head, axis=-1) for head in preds])
        return float(np.mean(np.all(pred_ids == targets, axis=1)))
    pred_arr = np.asarray(preds)
    if targets.ndim == 2:
        if pred_arr.ndim == 3:
            pred_ids = np.argmax(pred_arr, axis=-1)
        else:
            pred_ids = pred_arr
        return float(np.mean(np.all(pred_ids == targets, axis=1)))
    pred_ids = np.argmax(pred_arr, axis=-1) if pred_arr.ndim == 2 else pred_arr
    return float(np.mean(pred_ids == targets))


def compute_dialect_matrix(
    speaker_messages: Sequence[np.ndarray],
    listener_predict_fns: Sequence[Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, ...]]],
    targets: np.ndarray,
    *,
    vocab_size: int | None = None,
) -> dict:
    """Evaluate N x N cross-play compatibility A_ij = Acc(S_i -> L_j), raw and Hungarian-aligned."""
    n_agents = len(speaker_messages)
    if n_agents != len(listener_predict_fns) or n_agents < 2:
        raise ValueError("Need at least two matched speaker message arrays and listener predict functions")
    msgs_list = [np.asarray(m, dtype=np.int64) for m in speaker_messages]
    target_arr = np.asarray(targets)
    k = vocab_size or (max(int(m.max()) for m in msgs_list) + 1)

    raw_matrix = np.zeros((n_agents, n_agents), dtype=np.float64)
    aligned_matrix = np.zeros((n_agents, n_agents), dtype=np.float64)
    for i in range(n_agents):
        for j in range(n_agents):
            raw_matrix[i, j] = _accuracy_from_listener(listener_predict_fns[j], msgs_list[i], target_arr)
            if i == j:
                aligned_matrix[i, j] = raw_matrix[i, j]
            else:
                remapped = _align_messages_hungarian(msgs_list[i], msgs_list[j], k)
                aligned_matrix[i, j] = _accuracy_from_listener(listener_predict_fns[j], remapped, target_arr)

    off_diag_mask = ~np.eye(n_agents, dtype=bool)
    return {
        "raw_matrix": raw_matrix.tolist(),
        "aligned_matrix": aligned_matrix.tolist(),
        "diagonal_mean": float(np.mean(np.diag(raw_matrix))),
        "raw_off_diagonal_mean": float(np.mean(raw_matrix[off_diag_mask])),
        "hungarian_off_diagonal_mean": float(np.mean(aligned_matrix[off_diag_mask])),
    }


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
