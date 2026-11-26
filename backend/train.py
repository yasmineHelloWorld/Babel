"""Standalone training script – runs without the UI.

Usage
-----
  python -m backend.train --epochs 2000 --seed 0
  python -m backend.train --epochs 2000 --seed 0 --gumbel --plot
  python -m backend.train --epochs 500  --rival-lr-mult 0.1

Acceptance criterion (seed 0, REINFORCE):
  listener accuracy ≥ 95 % within ≈ 2000 epochs; rival clearly lower.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Babel agents (no UI required)")
    parser.add_argument("--epochs",        type=int,   default=2000)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--batch-size",    type=int,   default=64,
                        help="Objects per gradient step")
    parser.add_argument("--gumbel",        action="store_true",
                        help="Use Gumbel-softmax straight-through instead of REINFORCE")
    parser.add_argument("--rival-lr-mult", type=float, default=0.2,
                        help="Rival LR = base_lr × this multiplier (default 0.2)")
    parser.add_argument("--log-interval",  type=int,   default=100)
    parser.add_argument("--plot",          action="store_true",
                        help="Save accuracy+TopSim plot to runs/training_plot.png")
    args = parser.parse_args()

    # Import here so the script works as `python -m backend.train`
    from backend.trainer import BabelTrainer, BATCH_SIZE
    import backend.trainer as _t_mod
    if args.batch_size != BATCH_SIZE:
        _t_mod.BATCH_SIZE = args.batch_size

    trainer = BabelTrainer(
        seed=args.seed,
        use_gumbel=args.gumbel,
        rival_lr_mult=args.rival_lr_mult,
    )

    listener_hist, rival_hist, topsim_hist = [], [], []

    print(f"Training Babel for {args.epochs} epochs  "
          f"(seed={args.seed}, gumbel={args.gumbel}, rival_lr_mult={args.rival_lr_mult})")
    print(f"{'Epoch':>6}  {'Listener':>8}  {'Rival':>8}  {'Entropy':>8}  "
          f"{'TopSim':>7}  {'Consisten.':>10}")

    stats = None
    for ep in range(1, args.epochs + 1):
        stats = trainer.train_step()
        listener_hist.append(stats.listener_acc)
        rival_hist.append(stats.rival_acc)
        topsim_hist.append(stats.topsim)
        if ep % args.log_interval == 0 or ep == 1:
            print(f"{ep:>6}  {stats.listener_acc:>8.3f}  {stats.rival_acc:>8.3f}  "
                  f"{stats.entropy:>8.3f}  {stats.topsim:>7.3f}  {stats.consistency:>10.3f}")

    assert stats is not None
    print(f"\nFinal  listener={stats.listener_acc:.3f}  rival={stats.rival_acc:.3f}")
    if stats.listener_acc >= 0.95:
        print("✓  Listener accuracy target met (≥ 95 %)")
    else:
        print(f"✗  Listener accuracy {stats.listener_acc:.3f} < 0.95 (try more epochs or lower rival_lr_mult)")

    gap = stats.listener_acc - stats.rival_acc
    if gap > 0.05:
        print(f"✓  Rival is clearly behind  (gap = {gap:+.3f})")
    else:
        print(f"⚠  Gap between listener and rival is small  ({gap:+.3f})")

    if args.plot:
        _save_plot(listener_hist, rival_hist, topsim_hist, args.epochs)


def _save_plot(
    listener_hist: list[float],
    rival_hist: list[float],
    topsim_hist: list[float],
    epochs: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        window = 20

        def smooth(arr: list[float]) -> np.ndarray:
            return np.convolve(arr, np.ones(window) / window, mode="valid")

        x = list(range(window, epochs + 1))
        s_li = smooth(listener_hist)
        s_rv = smooth(rival_hist)
        s_ts = smooth(topsim_hist)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.patch.set_facecolor("#0b0a1e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#12102e")
            ax.tick_params(colors="#cfc8f5")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3c88")

        ax1.plot(x, s_li, color="#4fe39a", label="Listener", linewidth=2)
        ax1.plot(x, s_rv, color="#ff6b6b", label="Rival",    linewidth=2)
        ax1.axhline(0.95, color="#ffd23f", linestyle="--", alpha=0.7, label="95 % target")
        ax1.set_ylabel("Accuracy",          color="#cfc8f5")
        ax1.set_title("Babel Training",     color="#f1ecff", fontsize=14)
        ax1.legend(facecolor="#12102e", edgecolor="#3a3c88", labelcolor="#cfc8f5")

        ax2.plot(x, s_ts, color="#6d70d8", label="TopSim", linewidth=2)
        ax2.set_ylabel("Topographic Similarity", color="#cfc8f5")
        ax2.set_xlabel("Epoch",                   color="#cfc8f5")
        ax2.legend(facecolor="#12102e", edgecolor="#3a3c88", labelcolor="#cfc8f5")

        plt.tight_layout()
        out = Path("runs/training_plot.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"Plot saved → {out}")
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
