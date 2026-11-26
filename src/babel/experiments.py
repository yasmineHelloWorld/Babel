"""Reproducible multi-seed experiment runner for Phase 7 YAML manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from babel.agents.neural import NeuralTrainingConfig
from babel.agents.tabular import TabularListener, TabularSpeaker
from babel.baselines import fit_predict_baselines
from babel.config import ExperimentConfig, RunConfig
from babel.data_cards import write_data_cards
from babel.datasets.synthetic import COLORS, SHAPES, generate_synthetic_split, generate_synthetic_world
from babel.datasets.tabular import load_tabular_dataset
from babel.datasets.vision import load_fashion_mnist
from babel.metrics import (
    _entropy,
    _mutual_information,
    compute_classification_metrics,
    compute_dialect_matrix,
    compute_language_metrics,
    compute_multi_attribute_metrics,
    evaluate_language,
    positional_disentanglement,
    topographic_similarity_with_pvalue,
)
from babel.preprocessing import prepare_tabular_data, prepare_vision_pca_data
from babel.seeds import make_random_streams
from babel.split_guard import SplitGuard
from babel.store import RunStore
from babel.train import train_neural_run


def _mean_ci(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    arr = np.asarray(clean, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "ci95": ci95, "n": len(arr)}


def _save_summary(runs_dir: str | Path, name: str, payload: dict) -> Path:
    summary_dir = Path(runs_dir) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    target = summary_dir / f"{name}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _load_classification_dataset(config: ExperimentConfig) -> dict:
    """Load dataset and fit preprocessing strictly on SplitGuard training indices."""
    if config.dataset == "fashion_mnist":
        max_samples = config.extra.get("max_samples")
        ds = load_fashion_mnist(config.data_dir, download=True, max_samples=max_samples)
        split_path = Path(config.splits_dir) / f"fashion_mnist_{len(ds.labels)}.npz"
        split = SplitGuard(split_path).create(ds.labels, seed=0)
        prep = prepare_vision_pca_data(
            ds.features, ds.labels, split, n_components=config.pca_components, seed=0
        )
        return {
            "dataset": ds.name,
            "label_names": list(ds.label_names),
            "split_sha256": split.sha256,
            "train_x": prep.train_features,
            "train_y": prep.train_labels,
            "val_x": prep.validation_features,
            "val_y": prep.validation_labels,
            "test_x": prep.test_features,
            "test_y": prep.test_labels,
            "class_balance": {
                ds.label_names[int(k)]: int(v)
                for k, v in sorted(Counter(ds.labels.tolist()).items())
            },
            "pca_explained_variance": prep.explained_variance_ratio_sum,
        }

    ds = load_tabular_dataset(config.dataset, config.data_dir, download=True)
    guard_path = Path(config.splits_dir) / (
        f"{config.dataset}-held-out.npz" if config.held_out_combinations else f"{config.dataset}.npz"
    )
    guard = SplitGuard(guard_path)
    if config.held_out_combinations:
        split = guard.create_held_out_combinations(
            ds.features,
            held_out=set(config.held_out_combinations),
            columns=config.held_out_columns,
            seed=0,
        )
    else:
        split = guard.create(ds.labels, seed=0)
    prep = prepare_tabular_data(ds.features, ds.labels, split)
    return {
        "dataset": ds.name,
        "label_names": list(ds.label_names),
        "split_sha256": split.sha256,
        "train_x": prep.train_features,
        "train_y": prep.train_labels,
        "val_x": prep.validation_features,
        "val_y": prep.validation_labels,
        "test_x": prep.test_features,
        "test_y": prep.test_labels,
        "class_balance": {
            ds.label_names[int(k)]: int(v)
            for k, v in sorted(Counter(ds.labels.tolist()).items())
        },
        "test_class_coverage": {
            ds.label_names[int(k)]: int(v)
            for k, v in sorted(Counter(prep.test_labels.tolist()).items())
        },
    }


def _single_label_code_metrics(messages: np.ndarray, labels: np.ndarray, features: np.ndarray, seed: int = 0) -> dict:
    msg_tuples = [tuple(int(s) for s in row) for row in messages]
    lbl_list = labels.tolist()
    h_m = _entropy(msg_tuples)
    h_o = _entropy(lbl_list)
    by_msg: dict[tuple[int, ...], list[int]] = {}
    for m, y in zip(msg_tuples, lbl_list):
        by_msg.setdefault(m, []).append(y)
    h_o_given_m = sum((len(ys) / len(lbl_list)) * _entropy(ys) for ys in by_msg.values())
    mi_bits = _mutual_information(msg_tuples, lbl_list)
    active = sorted(set(int(s) for s in messages.flat))
    # Cap topsim sample count for O(N^2) runtime on large datasets like Fashion-MNIST.
    if len(messages) > 350:
        idx = np.random.default_rng(seed).choice(len(messages), size=350, replace=False)
        topsim, topsim_p = topographic_similarity_with_pvalue(messages[idx], features[idx], n_permutations=100, seed=seed)
    else:
        topsim, topsim_p = topographic_similarity_with_pvalue(messages, features, n_permutations=150, seed=seed)
    return {
        "message_entropy_bits": float(h_m),
        "target_entropy_bits": float(h_o),
        "mutual_information_bits": float(mi_bits),
        "conditional_entropy_bits": float(h_o_given_m),
        "active_vocabulary": len(active),
        "topsim": float(topsim),
        "topsim_pvalue": float(topsim_p),
    }


def _tune_on_validation(data: dict, config: ExperimentConfig) -> tuple[dict, list[dict]]:
    """Select (entropy_bonus, hidden_dim) using validation accuracy ONLY (never test)."""
    tune_k = config.vocab_sizes[min(len(config.vocab_sizes) - 1, 1)]
    candidates = [
        {"entropy_bonus": 0.01, "hidden_dim": config.hidden_dim},
        {"entropy_bonus": 0.03, "hidden_dim": config.hidden_dim},
        {"entropy_bonus": 0.02, "hidden_dim": max(32, config.hidden_dim // 2)},
    ]
    records = []
    best_cfg = candidates[0]
    best_val = -1.0
    for cand in candidates:
        res = train_neural_run(
            run_name=f"{config.name}_val_tune",
            train_features=data["train_x"],
            train_labels=data["train_y"],
            val_features=data["val_x"],
            val_labels=data["val_y"],
            config=NeuralTrainingConfig(
                vocab_size=tune_k,
                message_length=config.message_length,
                hidden_dim=cand["hidden_dim"],
                embed_dim=min(32, cand["hidden_dim"]),
                listener_mode=config.listener_mode,
                training_mode=config.training_mode,
                epochs=min(60, config.epochs),
                batch_size=config.batch_size,
                speaker_lr=config.speaker_lr,
                listener_lr=config.listener_lr,
                entropy_bonus=cand["entropy_bonus"],
                baseline_ema=config.baseline_ema,
                log_every=max(20, min(60, config.epochs) // 2),
            ),
            seed=config.seeds[0],
            runs_dir=Path(config.runs_dir) / "tuning",
        )
        val_acc = res["val_eval"]["exact_accuracy"]
        records.append({**cand, "tune_vocab_size": tune_k, "val_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_cfg = cand
    return best_cfg, records


def run_rate_accuracy_experiment(config: ExperimentConfig) -> dict:
    """Experiment 1 / Phase 6-7: Rate-accuracy curves on real data across vocab sizes & 5 seeds."""
    config.validate()
    write_data_cards(json_dir=config.data_dir)
    data = _load_classification_dataset(config)
    best_hp, tuning_records = _tune_on_validation(data, config)

    all_labels = list(range(len(data["label_names"])))
    by_vocab: dict[str, dict] = {}
    for vocab_size in config.vocab_sizes:
        neural_seed_metrics = []
        baseline_seed_metrics: dict[str, list[dict]] = {
            "full_data_oracle": [],
            "full_data_oracle_mlp": [],
            "kmeans_vq": [],
            "size_limited_tree": [],
        }
        confusion_accum = None
        for seed in config.seeds:
            run_res = train_neural_run(
                run_name=f"{config.name}_K{vocab_size}",
                train_features=data["train_x"],
                train_labels=data["train_y"],
                val_features=data["val_x"],
                val_labels=data["val_y"],
                test_features=data["test_x"],
                test_labels=data["test_y"],
                target_cardinalities=(len(data["label_names"]),),
                config=NeuralTrainingConfig(
                    vocab_size=vocab_size,
                    message_length=config.message_length,
                    hidden_dim=best_hp["hidden_dim"],
                    embed_dim=min(32, best_hp["hidden_dim"]),
                    listener_mode=config.listener_mode,
                    training_mode=config.training_mode,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    speaker_lr=config.speaker_lr,
                    listener_lr=config.listener_lr,
                    entropy_bonus=best_hp["entropy_bonus"],
                    baseline_ema=config.baseline_ema,
                    log_every=max(10, config.epochs // 6),
                ),
                seed=seed,
                runs_dir=config.runs_dir,
                metadata={"experiment": config.name, "dataset": data["dataset"], "split_sha256": data["split_sha256"]},
            )
            te = run_res["test_eval"]
            clf = compute_classification_metrics(
                data["test_y"], te["preds"][:, 0], te["probs"][0], labels=all_labels
            )
            code = _single_label_code_metrics(te["messages"], data["test_y"], data["test_x"], seed=seed)
            neural_seed_metrics.append({"seed": seed, "run_dir": run_res["run_dir"], **clf, **code})
            cm = np.asarray(clf["confusion_matrix"], dtype=np.int64)
            confusion_accum = cm if confusion_accum is None else (confusion_accum + cm)

            base = fit_predict_baselines(
                data["train_x"], data["train_y"],
                data["val_x"], data["val_y"],
                data["test_x"], data["test_y"],
                message_symbols=vocab_size,
                message_length=config.message_length,
                seed=seed,
            )
            if seed == config.seeds[0]:
                base_save = Path(config.runs_dir) / "baselines" / f"{config.name}_K{vocab_size}.npz"
                base.save(base_save)
            for method in baseline_seed_metrics:
                b_clf = compute_classification_metrics(
                    data["test_y"],
                    base.predictions[method]["test"],
                    base.probabilities[method]["test"] if base.probabilities else None,
                    labels=all_labels,
                )
                baseline_seed_metrics[method].append(b_clf)

        metric_fields = (
            "accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr",
            "mutual_information_bits", "message_entropy_bits", "active_vocabulary", "topsim",
        )
        neural_agg = {
            field: _mean_ci([m[field] for m in neural_seed_metrics if m[field] is not None])
            for field in metric_fields
        }
        baselines_agg = {
            method: {
                field: _mean_ci([m[field] for m in rows if m[field] is not None])
                for field in ("accuracy", "f1_macro", "roc_auc_ovr")
            }
            for method, rows in baseline_seed_metrics.items()
        }
        by_vocab[str(vocab_size)] = {
            "vocab_size": vocab_size,
            "bits_capacity": float(config.message_length * np.log2(vocab_size)),
            "neural": neural_agg,
            "baselines": baselines_agg,
            "confusion_matrix_sum": confusion_accum.tolist() if confusion_accum is not None else [],
            "runs": [
                {"seed": m["seed"], "run_dir": m["run_dir"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                 "roc_auc_ovr": m["roc_auc_ovr"], "active_vocabulary": m["active_vocabulary"]}
                for m in neural_seed_metrics
            ],
        }

    summary = {
        "experiment": config.name,
        "kind": "rate_accuracy",
        "dataset": data["dataset"],
        "label_names": data["label_names"],
        "split_sha256": data["split_sha256"],
        "message_length": config.message_length,
        "seeds": list(config.seeds),
        "class_balance": data["class_balance"],
        "validation_tuning": {"selected": best_hp, "candidates": tuning_records},
        "results_by_vocab": by_vocab,
    }
    if "pca_explained_variance" in data:
        summary["pca_components"] = config.pca_components
        summary["pca_explained_variance"] = data["pca_explained_variance"]
    if "test_class_coverage" in data:
        summary["test_class_coverage"] = data["test_class_coverage"]
    summary_path = _save_summary(config.runs_dir, config.name, summary)
    return {"summary_path": str(summary_path), **summary}


def _train_tabular_synthetic_heldout(
    train_objs: list[dict],
    all_objs: list[dict],
    held_out_objs: list[dict],
    *,
    vocab_size: int,
    message_length: int,
    seed: int,
    runs_dir: str | Path,
) -> dict:
    streams = make_random_streams(seed)
    speaker = TabularSpeaker(len(all_objs), message_length, vocab_size, streams.speaker)
    listener = TabularListener(message_length, vocab_size, len(SHAPES), len(COLORS), streams.listener)
    store = RunStore(runs_dir, {"vocab_size": vocab_size, "message_length": message_length, "agent": "tabular"}, seed, run_name=f"tabular_K{vocab_size}")
    baseline = 0.5
    episodes = 15_000
    for ep in range(1, episodes + 1):
        obj = train_objs[int(streams.environment.integers(len(train_objs)))]
        oid = obj["object_id"]
        msg, policy = speaker.speak(oid)
        sp, cp = listener.predict(msg)
        reward = 0.5 * (sp[obj["shape_id"]] + cp[obj["color_id"]])
        adv = float(reward - baseline)
        baseline += 0.05 * (float(reward) - baseline)
        speaker.update(oid, msg, policy, adv, 0.08, 0.015)
        listener.observe(msg, obj["shape_id"], obj["color_id"])
        if ep % 3000 == 0 or ep == episodes:
            sc = evaluate_language(train_objs, speaker, listener)
            store.log_metrics({"episode": ep, "reward": float(reward), "exact_accuracy": sc["exact_accuracy"]})

    # Deterministic eval across seen train and held-out objects
    def eval_objs(objs):
        msgs, preds, truth = [], [], []
        for o in objs:
            m = np.argmax(speaker.probabilities(o["object_id"]), axis=-1).astype(np.int64)
            sp, cp = listener.predict(m)
            msgs.append(m)
            preds.append([int(np.argmax(sp)), int(np.argmax(cp))])
            truth.append([o["shape_id"], o["color_id"]])
        return np.asarray(msgs), np.asarray(preds), np.asarray(truth)

    tr_m, tr_p, tr_y = eval_objs(train_objs)
    ho_m, ho_p, ho_y = eval_objs(held_out_objs)
    all_m, _, all_y = eval_objs(all_objs)
    all_x = np.stack([o["features"] for o in all_objs])
    lang = compute_language_metrics(all_m, all_y[:, 0], all_y[:, 1])
    topsim, _ = topographic_similarity_with_pvalue(all_m, all_x, n_permutations=100, seed=seed)
    store.save_checkpoint(speaker_logits=speaker.logits, listener_shape_counts=listener.shape_counts, listener_color_counts=listener.color_counts)
    store.save_predictions(train_messages=tr_m, train_preds=tr_p, train_truth=tr_y, test_messages=ho_m, test_preds=ho_p, test_truth=ho_y)
    return {
        "run_dir": str(store.path),
        "seen_exact_accuracy": compute_multi_attribute_metrics(tr_y, tr_p)["exact_accuracy"],
        "heldout_exact_accuracy": compute_multi_attribute_metrics(ho_y, ho_p)["exact_accuracy"],
        "heldout_attr_accuracy": compute_multi_attribute_metrics(ho_y, ho_p)["mean_attribute_accuracy"],
        "posdis": lang["posdis"],
        "topsim": topsim,
        "ambiguity_bits": lang["ambiguity_bits"],
        "synonymy_bits": lang["synonymy_bits"],
        "message_entropy_bits": lang["message_entropy_bits"],
        "active_vocabulary": lang["active_vocabulary"],
    }


def run_grammar_experiment(config: ExperimentConfig) -> dict:
    """Experiment 2 / Phase 7: L=2 grammar vs bandwidth & zero-shot held-out combinations."""
    config.validate()
    write_data_cards(json_dir=config.data_dir)

    if config.dataset == "synthetic":
        held_pairs = config.held_out_combinations or ((2, 0),)
        split_data = generate_synthetic_split(held_out_pairs=held_pairs)
        train_x = np.stack([o["features"] for o in split_data["train_objects"]])
        train_y = np.array([[o["shape_id"], o["color_id"]] for o in split_data["train_objects"]], dtype=np.int64)
        val_x, val_y = train_x, train_y
        test_x = np.stack([o["features"] for o in split_data["held_out_objects"]])
        test_y = np.array([[o["shape_id"], o["color_id"]] for o in split_data["held_out_objects"]], dtype=np.int64)
        all_x = np.stack([o["features"] for o in split_data["objects"]])
        all_y = np.array([[o["shape_id"], o["color_id"]] for o in split_data["objects"]], dtype=np.int64)
        cardinalities = (len(SHAPES), len(COLORS))
        split_sha = "synthetic-3x3"
        heldout_support = len(test_y)
        heldout_coverage = {"pairs": [list(p) for p in held_pairs]}
        modes = ("tabular", "holistic", "structured")
    else:
        ds = load_tabular_dataset(config.dataset, config.data_dir, download=True)
        cols = config.target_columns or config.held_out_columns
        held_pairs = config.held_out_combinations or ((0, 2), (3, 0))
        guard = SplitGuard(Path(config.splits_dir) / f"{config.dataset}-grammar-held-out.npz")
        split = guard.create_held_out_combinations(
            ds.features, held_out=set(held_pairs), columns=tuple(cols), seed=0
        )
        # Speaker observes the full row (all 6 attributes); Listener predicts ONLY the 2 chosen attributes.
        multi_targets = ds.features[:, cols].astype(np.int64)
        prep = prepare_tabular_data(ds.features, multi_targets, split)
        train_x, train_y = prep.train_features, prep.train_labels
        val_x, val_y = prep.validation_features, prep.validation_labels
        test_x, test_y = prep.test_features, prep.test_labels
        all_x = np.concatenate([train_x, val_x, test_x], axis=0)
        all_y = np.concatenate([train_y, val_y, test_y], axis=0)
        cardinalities = tuple(int(ds.features[:, c].max() + 1) for c in cols)
        split_sha = split.sha256
        heldout_support = int(len(test_y))
        heldout_coverage = {
            f"col_{c}": {str(int(k)): int(v) for k, v in sorted(Counter(test_y[:, i].tolist()).items())}
            for i, c in enumerate(cols)
        }
        modes = ("holistic", "structured")

    results_by_mode: dict[str, dict] = {}
    for mode in modes:
        by_vocab: dict[str, dict] = {}
        for k in config.vocab_sizes:
            seed_rows = []
            for seed in config.seeds:
                if mode == "tabular":
                    row = _train_tabular_synthetic_heldout(
                        split_data["train_objects"],
                        split_data["objects"],
                        split_data["held_out_objects"],
                        vocab_size=k,
                        message_length=config.message_length,
                        seed=seed,
                        runs_dir=config.runs_dir,
                    )
                    seed_rows.append(row)
                else:
                    run_res = train_neural_run(
                        run_name=f"{config.name}_{mode}_K{k}",
                        train_features=train_x,
                        train_labels=train_y,
                        val_features=val_x,
                        val_labels=val_y,
                        test_features=test_x,
                        test_labels=test_y,
                        target_cardinalities=cardinalities,
                        config=NeuralTrainingConfig(
                            vocab_size=k,
                            message_length=config.message_length,
                            hidden_dim=config.hidden_dim,
                            embed_dim=min(32, config.hidden_dim),
                            listener_mode=mode,
                            training_mode=config.training_mode,
                            epochs=config.epochs,
                            batch_size=config.batch_size,
                            speaker_lr=config.speaker_lr,
                            listener_lr=config.listener_lr,
                            entropy_bonus=config.entropy_bonus,
                            baseline_ema=config.baseline_ema,
                            log_every=max(20, config.epochs // 5),
                        ),
                        seed=seed,
                        runs_dir=config.runs_dir,
                        metadata={"experiment": config.name, "dataset": config.dataset, "listener_mode": mode},
                    )
                    all_msgs = run_res["speaker"].act(np.asarray(all_x, dtype=np.float32), deterministic=True)
                    lang = compute_language_metrics(all_msgs, all_y[:, 0], all_y[:, 1])
                    if len(all_msgs) > 350:
                        idx = np.random.default_rng(seed).choice(len(all_msgs), size=350, replace=False)
                        topsim, _ = topographic_similarity_with_pvalue(all_msgs[idx], all_x[idx], n_permutations=100, seed=seed)
                    else:
                        topsim, _ = topographic_similarity_with_pvalue(all_msgs, all_x, n_permutations=100, seed=seed)
                    val_m = compute_multi_attribute_metrics(val_y, run_res["val_eval"]["preds"])
                    ho_m = compute_multi_attribute_metrics(test_y, run_res["test_eval"]["preds"])
                    seed_rows.append({
                        "run_dir": run_res["run_dir"],
                        "seen_exact_accuracy": val_m["exact_accuracy"],
                        "heldout_exact_accuracy": ho_m["exact_accuracy"],
                        "heldout_attr_accuracy": ho_m["mean_attribute_accuracy"],
                        "posdis": lang["posdis"],
                        "topsim": topsim,
                        "ambiguity_bits": lang["ambiguity_bits"],
                        "synonymy_bits": lang["synonymy_bits"],
                        "message_entropy_bits": lang["message_entropy_bits"],
                        "active_vocabulary": lang["active_vocabulary"],
                    })

            fields = (
                "seen_exact_accuracy", "heldout_exact_accuracy", "heldout_attr_accuracy",
                "posdis", "topsim", "ambiguity_bits", "synonymy_bits",
                "message_entropy_bits", "active_vocabulary",
            )
            by_vocab[str(k)] = {
                "vocab_size": k,
                "metrics": {f: _mean_ci([r[f] for r in seed_rows]) for f in fields},
                "sample_run_dir": seed_rows[0]["run_dir"],
            }
        results_by_mode[mode] = by_vocab

    summary = {
        "experiment": config.name,
        "kind": "grammar",
        "dataset": config.dataset,
        "split_sha256": split_sha,
        "message_length": config.message_length,
        "seeds": list(config.seeds),
        "held_out_support": heldout_support,
        "held_out_coverage": heldout_coverage,
        "results_by_mode": results_by_mode,
    }
    summary_path = _save_summary(config.runs_dir, config.name, summary)
    return {"summary_path": str(summary_path), **summary}


def run_dialect_experiment(config: ExperimentConfig) -> dict:
    """Experiment 3 / Phase 7: N x N Dialect Cross-Play Matrix (raw vs Hungarian-aligned)."""
    config.validate()
    datasets_to_run = config.extra.get("datasets", ["synthetic", "congressional_voting_records"])
    summaries = {}
    for ds_name in datasets_to_run:
        if ds_name == "synthetic":
            world = generate_synthetic_world()
            x_tr = np.stack([o["features"] for o in world["objects"]])
            y_tr = np.array([[o["shape_id"], o["color_id"]] for o in world["objects"]], dtype=np.int64)
            x_ev, y_ev = x_tr, y_tr
            msg_len, vocab = 2, 3
            cards = (3, 3)
            epochs, bs = 600, 36
        else:
            sub_cfg = ExperimentConfig(
                name=f"{config.name}_{ds_name}", dataset=ds_name,
                data_dir=config.data_dir, splits_dir=config.splits_dir,
            )
            data = _load_classification_dataset(sub_cfg)
            x_tr, y_tr = data["train_x"], data["train_y"]
            x_ev, y_ev = data["test_x"], data["test_y"]
            msg_len, vocab = 1, 4
            cards = (len(data["label_names"]),)
            epochs, bs = config.epochs, config.batch_size

        speaker_msgs = []
        listener_fns = []
        for seed in config.seeds:
            res = train_neural_run(
                run_name=f"{config.name}_{ds_name}",
                train_features=x_tr,
                train_labels=y_tr,
                test_features=x_ev,
                test_labels=y_ev,
                target_cardinalities=cards,
                config=NeuralTrainingConfig(
                    vocab_size=vocab,
                    message_length=msg_len,
                    hidden_dim=config.hidden_dim,
                    embed_dim=min(32, config.hidden_dim),
                    listener_mode="holistic",
                    training_mode="hybrid_ce",
                    epochs=epochs,
                    batch_size=bs,
                    speaker_lr=config.speaker_lr,
                    listener_lr=config.listener_lr,
                    entropy_bonus=config.entropy_bonus,
                    log_every=max(20, epochs // 4),
                ),
                seed=seed,
                runs_dir=config.runs_dir,
            )
            speaker_msgs.append(res["speaker"].act(np.asarray(x_ev, dtype=np.float32), deterministic=True))
            listener_fns.append(res["listener"].predict)

        matrix = compute_dialect_matrix(speaker_msgs, listener_fns, y_ev, vocab_size=vocab)
        summaries[ds_name] = {
            "dataset": ds_name,
            "vocab_size": vocab,
            "message_length": msg_len,
            "seeds": list(config.seeds),
            **matrix,
        }

    payload = {"experiment": config.name, "kind": "dialect_matrix", "datasets": summaries}
    summary_path = _save_summary(config.runs_dir, config.name, payload)
    return {"summary_path": str(summary_path), **payload}


def run_ablation_experiment(config: ExperimentConfig) -> dict:
    """Phase 5 ablation: hybrid_ce vs both_reinforce across seeds."""
    config.validate()
    data = _load_classification_dataset(config)
    vocab = config.vocab_sizes[0]
    by_mode = {}
    for training_mode in ("hybrid_ce", "both_reinforce"):
        rows = []
        for seed in config.seeds:
            res = train_neural_run(
                run_name=f"{config.name}_{training_mode}",
                train_features=data["train_x"],
                train_labels=data["train_y"],
                val_features=data["val_x"],
                val_labels=data["val_y"],
                test_features=data["test_x"],
                test_labels=data["test_y"],
                target_cardinalities=(len(data["label_names"]),),
                config=NeuralTrainingConfig(
                    vocab_size=vocab,
                    message_length=config.message_length,
                    hidden_dim=config.hidden_dim,
                    embed_dim=min(32, config.hidden_dim),
                    listener_mode=config.listener_mode,
                    training_mode=training_mode,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    speaker_lr=config.speaker_lr,
                    listener_lr=config.listener_lr,
                    entropy_bonus=config.entropy_bonus,
                    log_every=max(20, config.epochs // 5),
                ),
                seed=seed,
                runs_dir=config.runs_dir,
            )
            clf = compute_classification_metrics(
                data["test_y"], res["test_eval"]["preds"][:, 0], res["test_eval"]["probs"][0]
            )
            rows.append({
                "accuracy": clf["accuracy"],
                "f1_macro": clf["f1_macro"],
                "active_vocabulary": res["test_eval"]["active_vocabulary"],
            })
        by_mode[training_mode] = {
            f: _mean_ci([r[f] for r in rows]) for f in ("accuracy", "f1_macro", "active_vocabulary")
        }

    payload = {
        "experiment": config.name,
        "kind": "ablation",
        "dataset": data["dataset"],
        "vocab_size": vocab,
        "message_length": config.message_length,
        "seeds": list(config.seeds),
        "results_by_training_mode": by_mode,
    }
    summary_path = _save_summary(config.runs_dir, config.name, payload)
    return {"summary_path": str(summary_path), **payload}


def run_manifest(path: str | Path) -> dict:
    """Dispatch a YAML experiment manifest by its ``extra.kind`` field."""
    config = ExperimentConfig.from_yaml(path)
    kind = config.extra.get("kind", "rate_accuracy")
    if kind == "rate_accuracy":
        return run_rate_accuracy_experiment(config)
    if kind == "grammar":
        return run_grammar_experiment(config)
    if kind == "dialect_matrix":
        return run_dialect_experiment(config)
    if kind == "ablation":
        return run_ablation_experiment(config)
    raise ValueError(f"Unknown experiment kind {kind!r} in {path}")
