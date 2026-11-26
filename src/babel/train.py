from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from babel.agents.neural import (
    NeuralTrainingConfig,
    save_neural_checkpoint,
    train_neural_agents,
)
from babel.agents.tabular import TabularListener, TabularSpeaker
from babel.config import RunConfig
from babel.datasets.synthetic import COLORS, SHAPES
from babel.env import SyntheticEnvironment
from babel.metrics import evaluate_language
from babel.seeds import make_random_streams
from babel.store import RunStore


def train(config: RunConfig) -> dict:
    config.validate()
    streams = make_random_streams(config.seed)
    environment = SyntheticEnvironment(streams.environment)
    objects = environment.objects
    speaker = TabularSpeaker(len(objects), config.message_length, config.vocab_size, streams.speaker)
    listener = TabularListener(config.message_length, config.vocab_size, len(SHAPES), len(COLORS), streams.listener)
    store = RunStore(config.output_root, config.to_dict(), config.seed, run_name="synthetic_tabular")
    baseline = 0.5

    for episode in range(1, config.episodes + 1):
        obj = environment.sample()
        object_id = obj["object_id"]
        message, policy = speaker.speak(object_id)
        shape_probs, color_probs = listener.predict(message)
        reward = 0.5 * (shape_probs[obj["shape_id"]] + color_probs[obj["color_id"]])
        advantage = float(reward - baseline)
        baseline += config.baseline_rate * (float(reward) - baseline)
        speaker.update(object_id, message, policy, advantage, config.learning_rate, config.entropy_bonus)
        listener.observe(message, obj["shape_id"], obj["color_id"])

        if episode % config.log_every == 0 or episode == config.episodes:
            scores = evaluate_language(objects, speaker, listener)
            metric_keys = ("shape_accuracy", "color_accuracy", "exact_accuracy", "message_entropy_bits",
                           "position_entropies_bits", "shape_mutual_information_bits",
                           "color_mutual_information_bits", "position_property_mutual_information_bits",
                           "posdis", "object_mutual_information_bits",
                           "active_vocabulary", "ambiguity_bits", "synonymy_bits")
            store.log_metrics({"episode": episode, "reward": float(reward), "baseline": baseline,
                               **{key: scores[key] for key in metric_keys}})

    scores = evaluate_language(objects, speaker, listener)
    store.save_checkpoint(speaker_logits=speaker.logits, listener_shape_counts=listener.shape_counts,
                          listener_color_counts=listener.color_counts)
    store.save_predictions(
        messages=np.asarray(scores["messages"], dtype=np.int64),
        truth=np.array([[o["shape_id"], o["color_id"]] for o in objects], dtype=np.int64),
    )
    return {"run_dir": str(store.path), **scores}


def train_neural_run(
    *,
    run_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    test_features: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    target_cardinalities: Sequence[int] | None = None,
    config: NeuralTrainingConfig = NeuralTrainingConfig(),
    seed: int = 0,
    runs_dir: str | Path = "runs",
    metadata: dict | None = None,
) -> dict:
    """Run neural Speaker/Listener training and persist config, metrics, predictions, and checkpoint."""
    store_cfg = {
        "run_name": run_name,
        "vocab_size": config.vocab_size,
        "message_length": config.message_length,
        "hidden_dim": config.hidden_dim,
        "embed_dim": config.embed_dim,
        "listener_mode": config.listener_mode,
        "training_mode": config.training_mode,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "speaker_lr": config.speaker_lr,
        "listener_lr": config.listener_lr,
        "entropy_bonus": config.entropy_bonus,
        **(metadata or {}),
    }
    store = RunStore(runs_dir, store_cfg, seed, run_name=run_name)
    result = train_neural_agents(
        train_features,
        train_labels,
        val_features=val_features,
        val_labels=val_labels,
        target_cardinalities=target_cardinalities,
        config=config,
        seed=seed,
        metrics_callback=store.log_metrics,
    )
    speaker = result["speaker"]
    listener = result["listener"]
    save_neural_checkpoint(store.path / "checkpoint.pt", speaker, listener)

    pred_payload: dict[str, np.ndarray] = {
        "train_messages": result["train_eval"]["messages"],
        "train_preds": result["train_eval"]["preds"],
        "train_truth": np.asarray(train_labels, dtype=np.int64),
    }
    if len(result["train_eval"]["probs"]) == 1:
        pred_payload["train_proba"] = result["train_eval"]["probs"][0]

    if val_features is not None and val_labels is not None and "val_eval" in result:
        pred_payload["val_messages"] = result["val_eval"]["messages"]
        pred_payload["val_preds"] = result["val_eval"]["preds"]
        pred_payload["val_truth"] = np.asarray(val_labels, dtype=np.int64)
        if len(result["val_eval"]["probs"]) == 1:
            pred_payload["val_proba"] = result["val_eval"]["probs"][0]

    test_eval = None
    if test_features is not None and test_labels is not None:
        test_msgs = speaker.act(np.asarray(test_features, dtype=np.float32), deterministic=True)
        test_probs = listener.predict(test_msgs)
        if not isinstance(test_probs, tuple):
            test_probs = (test_probs,)
        test_preds = np.column_stack([np.argmax(p, axis=-1) for p in test_probs])
        y_te = np.asarray(test_labels, dtype=np.int64)
        y_te_2d = y_te[:, np.newaxis] if y_te.ndim == 1 else y_te
        test_eval = {
            "exact_accuracy": float(np.mean(np.all(test_preds == y_te_2d, axis=1))),
            "per_attribute_accuracy": [float(np.mean(test_preds[:, i] == y_te_2d[:, i])) for i in range(y_te_2d.shape[1])],
            "messages": test_msgs,
            "preds": test_preds,
            "probs": test_probs,
            "active_vocabulary": len(set(int(s) for s in test_msgs.flat)),
        }
        pred_payload["test_messages"] = test_msgs
        pred_payload["test_preds"] = test_preds
        pred_payload["test_truth"] = y_te
        if len(test_probs) == 1:
            pred_payload["test_proba"] = test_probs[0]

    store.save_predictions(**pred_payload)
    return {
        "run_dir": str(store.path),
        "speaker": speaker,
        "listener": listener,
        "history": result["history"],
        "train_eval": result["train_eval"],
        "val_eval": result.get("val_eval"),
        "test_eval": test_eval,
    }
