"""Prepare reproducible tabular data for Phase 3 experiments."""

import json
from pathlib import Path

from babel.config import TabularConfig
from babel.datasets.tabular import load_tabular_dataset
from babel.preprocessing import prepare_tabular_data
from babel.split_guard import SplitGuard


def prepare_tabular_experiment(config: TabularConfig) -> dict:
    """Load a dataset, persist its split, and fit categorical encoding on train only."""
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
        split = guard.create(dataset.labels, seed=config.seed)
    prepared = prepare_tabular_data(dataset.features, dataset.labels, split)
    summary = {
        "config": config.to_dict(),
        "dataset": dataset.name,
        "source_url": dataset.source_url,
        "n_samples": int(len(dataset.labels)),
        "n_features": int(dataset.features.shape[1]),
        "feature_names": list(dataset.feature_names),
        "label_names": list(dataset.label_names),
        "split_sha256": split.sha256,
        "split_type": "held_out_combinations" if config.held_out_combinations else "stratified",
        "split_sizes": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "encoded_feature_count": int(prepared.train_features.shape[1]),
        "training_categories": [list(map(str, values)) for values in prepared.encoder.categories_],
    }
    summary_path = Path(config.splits_dir) / f"{config.dataset}-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"summary_path": str(summary_path), **summary}
