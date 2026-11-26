"""Read-only artifact loaders for the Streamlit UI (never trains during demo)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from babel.agents.neural import NeuralListener


def list_run_dirs(runs_dir: str | Path = "runs") -> list[Path]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "config.json").is_file() and (p / "predictions.npz").is_file()
    ]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def load_run_bundle(run_dir: str | Path) -> dict:
    path = Path(run_dir)
    config_payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics_rows: list[dict] = []
    metrics_path = path / "metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                metrics_rows.append(json.loads(line))
    with np.load(path / "predictions.npz", allow_pickle=False) as data:
        predictions = {key: data[key] for key in data.files}
    return {
        "run_dir": str(path),
        "name": path.name,
        "config": config_payload,
        "metrics": metrics_rows,
        "predictions": predictions,
        "has_npz_ckpt": (path / "checkpoint.npz").is_file(),
        "has_pt_ckpt": (path / "checkpoint.pt").is_file(),
    }


def load_summaries(runs_dir: str | Path = "runs") -> dict[str, dict]:
    summary_dir = Path(runs_dir) / "summaries"
    if not summary_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for file_path in sorted(summary_dir.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        out[payload.get("experiment", file_path.stem)] = payload
    return out


def predict_with_saved_listener(run_dir: str | Path, message: list[int] | np.ndarray) -> tuple[np.ndarray, ...]:
    """Query a saved Listener checkpoint with discrete symbol IDs only (for Human Arena)."""
    path = Path(run_dir)
    msg_arr = np.asarray(message, dtype=np.int64)
    if (path / "checkpoint.npz").is_file():
        with np.load(path / "checkpoint.npz", allow_pickle=False) as ckpt:
            key = tuple(int(s) for s in msg_arr)
            shape = ckpt["listener_shape_counts"][key]
            color = ckpt["listener_color_counts"][key]
            return (shape / shape.sum(), color / color.sum())

    if (path / "checkpoint.pt").is_file():
        ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        meta = ckpt["listener_meta"]
        listener = NeuralListener(
            vocab_size=meta["vocab_size"],
            message_length=meta["message_length"],
            target_cardinalities=meta["target_cardinalities"],
            hidden_dim=meta["hidden_dim"],
            embed_dim=meta["embed_dim"],
            mode=meta["mode"],
        )
        listener.load_state_dict(ckpt["listener_state"])
        listener.eval()
        out = listener.predict(msg_arr)
        return out if isinstance(out, tuple) else (out,)

    raise FileNotFoundError(f"No listener checkpoint found in {path}")
