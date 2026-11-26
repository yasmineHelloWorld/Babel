import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class RunStore:
    def __init__(self, root: str | Path, config: dict, seed: int, *, run_name: str | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder = f"{run_name}_seed-{seed}_{stamp}" if run_name else f"seed-{seed}_{stamp}"
        self.path = Path(root) / folder
        self.path.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.path / "metrics.jsonl"
        payload = {"seed": seed, "config": config, "created_utc": datetime.now(timezone.utc).isoformat()}
        (self.path / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def log_metrics(self, metrics: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")

    def save_checkpoint(self, **arrays) -> Path:
        target = self.path / "checkpoint.npz"
        np.savez_compressed(target, **arrays)
        return target

    def save_predictions(self, **arrays) -> Path:
        target = self.path / "predictions.npz"
        np.savez_compressed(target, **arrays)
        return target

    def save_summary(self, summary: dict) -> Path:
        target = self.path / "summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return target
