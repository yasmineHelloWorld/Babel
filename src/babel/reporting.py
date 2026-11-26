"""Programmatic Markdown report generator reading saved runs/summaries/*.json only."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_ci(stat: dict | None, digits: int = 3) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "N/A"
    return f"{stat['mean']:.{digits}f} ± {stat['ci95']:.{digits}f}"


def generate_results_markdown(
    runs_dir: str | Path = "runs",
    output_path: str | Path = "docs/results.md",
) -> Path:
    """Compile saved experiment summaries into docs/results.md without hardcoded numbers."""
    summary_dir = Path(runs_dir) / "summaries"
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory not found: {summary_dir}")

    summaries = {}
    for file_path in sorted(summary_dir.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        summaries[payload.get("experiment", file_path.stem)] = payload

    lines: list[str] = [
        "# Babel Experimental Results",
        "",
        "All tables and metrics below are generated programmatically from saved `runs/summaries/*.json` artifacts.",
        "Historical 1984 U.S. House voting records (**Congressional Voting Records**), vision PCA features (**Fashion-MNIST**),",
        "and controlled rule-generated data (**Car Evaluation**) are reported separately and never conflated.",
        "",
    ]

    for exp_name, payload in summaries.items():
        kind = payload.get("kind")
        if kind == "rate_accuracy":
            lines.extend(_render_rate_accuracy(exp_name, payload))
        elif kind == "grammar":
            lines.extend(_render_grammar(exp_name, payload))
        elif kind == "dialect_matrix":
            lines.extend(_render_dialect(exp_name, payload))
        elif kind == "ablation":
            lines.extend(_render_ablation(exp_name, payload))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def _render_rate_accuracy(name: str, payload: dict) -> list[str]:
    ds = payload["dataset"]
    lines = [
        f"## Rate-Accuracy Benchmark: `{name}` (`{ds}`, $L = {payload['message_length']}$)",
        "",
        f"- **Split SHA-256:** `{payload['split_sha256']}` | **Seeds:** `{payload['seeds']}`",
        f"- **Class balance:** `{payload.get('class_balance', {})}`",
        f"- **Validation-selected hyperparameters:** `{payload['validation_tuning']['selected']}`",
    ]
    if "pca_explained_variance" in payload:
        lines.append(
            f"- **PCA components:** {payload['pca_components']} (train-only cumulative variance = {payload['pca_explained_variance']:.3f})"
        )
    if "test_class_coverage" in payload:
        lines.append(f"- **Held-out test class coverage:** `{payload['test_class_coverage']}`")
    lines.extend([
        "",
        "| $K$ | Bits ($\\log_2 K$) | Emergent Acc | Emergent Macro-F1 | ROC AUC (OvR) | $I(O; M)$ (bits) | Active Vocab | k-Means VQ Acc | Size-Limited Tree Acc | Oracle (LogReg / MLP) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    parity_notes = []
    for k_str, row in sorted(payload["results_by_vocab"].items(), key=lambda item: int(item[0])):
        n = row["neural"]
        b = row["baselines"]
        em_acc = _fmt_ci(n["accuracy"])
        em_f1 = _fmt_ci(n["f1_macro"])
        em_auc = _fmt_ci(n["roc_auc_ovr"])
        mi = _fmt_ci(n["mutual_information_bits"])
        av = _fmt_ci(n["active_vocabulary"], digits=1)
        km = _fmt_ci(b["kmeans_vq"]["accuracy"])
        tr = _fmt_ci(b["size_limited_tree"]["accuracy"])
        or_lr = b["full_data_oracle"]["accuracy"]["mean"]
        or_mlp = b["full_data_oracle_mlp"]["accuracy"]["mean"]
        lines.append(
            f"| {k_str} | {row['bits_capacity']:.2f} | {em_acc} | {em_f1} | {em_auc} | {mi} | {av} | {km} | {tr} | {or_lr:.3f} / {or_mlp:.3f} |"
        )
        if n["active_vocabulary"]["mean"] <= 1.5:
            parity_notes.append(f"At $K={k_str}$, active vocabulary averaged {n['active_vocabulary']['mean']:.1f} (symbol collapse observed).")
        if n["accuracy"]["mean"] + 0.01 < b["kmeans_vq"]["accuracy"]["mean"]:
            parity_notes.append(f"At $K={k_str}$, emergent accuracy ({n['accuracy']['mean']:.3f}) trailed k-Means VQ ({b['kmeans_vq']['accuracy']['mean']:.3f}).")
        elif abs(n["accuracy"]["mean"] - b["kmeans_vq"]["accuracy"]["mean"]) <= 0.02:
            parity_notes.append(f"At $K={k_str}$, emergent code achieved rough parity with k-Means VQ ({n['accuracy']['mean']:.3f} vs {b['kmeans_vq']['accuracy']['mean']:.3f}).")

    lines.append("")
    if parity_notes:
        lines.append("**Honest Interpretation:** " + " ".join(parity_notes))
        lines.append("")
    return lines


def _render_grammar(name: str, payload: dict) -> list[str]:
    lines = [
        f"## Grammar & Held-Out Generalization: `{name}` (`{payload['dataset']}`, $L = {payload['message_length']}$)",
        "",
        f"- **Split SHA-256:** `{payload['split_sha256']}` | **Held-out support:** {payload['held_out_support']} rows",
        f"- **Held-out coverage:** `{payload['held_out_coverage']}`",
        "",
        "| Listener Mode | $K$ | Seen Exact Acc | Held-Out Exact Acc | Held-Out Attr Acc | PosDis | $\\rho_{\\text{topsim}}$ | Ambiguity | Synonymy | Active Vocab |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, by_k in payload["results_by_mode"].items():
        for k_str, row in sorted(by_k.items(), key=lambda item: int(item[0])):
            m = row["metrics"]
            lines.append(
                f"| {mode} | {k_str} | {_fmt_ci(m['seen_exact_accuracy'])} | {_fmt_ci(m['heldout_exact_accuracy'])} | "
                f"{_fmt_ci(m['heldout_attr_accuracy'])} | {_fmt_ci(m['posdis'])} | {_fmt_ci(m['topsim'])} | "
                f"{_fmt_ci(m['ambiguity_bits'])} | {_fmt_ci(m['synonymy_bits'])} | {_fmt_ci(m['active_vocabulary'], 1)} |"
            )
    lines.append("")
    return lines


def _render_dialect(name: str, payload: dict) -> list[str]:
    lines = [
        f"## Dialect Cross-Play Matrix: `{name}`",
        "",
        "| Dataset | $K$ | $L$ | Seeds | Diagonal Mean | Raw Off-Diagonal Mean | Hungarian-Aligned Off-Diagonal Mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds_name, row in payload["datasets"].items():
        lines.append(
            f"| {ds_name} | {row['vocab_size']} | {row['message_length']} | {len(row['seeds'])} | "
            f"{row['diagonal_mean']:.3f} | {row['raw_off_diagonal_mean']:.3f} | {row['hungarian_off_diagonal_mean']:.3f} |"
        )
    lines.append("")
    return lines


def _render_ablation(name: str, payload: dict) -> list[str]:
    lines = [
        f"## Training Objective Ablation: `{name}` (`{payload['dataset']}`, $K={payload['vocab_size']}, L={payload['message_length']}$)",
        "",
        "| Training Mode | Test Accuracy | Macro-F1 | Active Vocab |",
        "|---|---:|---:|---:|",
    ]
    for mode, m in payload["results_by_training_mode"].items():
        lines.append(
            f"| {mode} | {_fmt_ci(m['accuracy'])} | {_fmt_ci(m['f1_macro'])} | {_fmt_ci(m['active_vocabulary'], 1)} |"
        )
    lines.append("")
    return lines
