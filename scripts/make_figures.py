"""Make the README figures from saved results (no GPU needed, runs in seconds).

Run from the repo root:
    uv run python scripts/make_figures.py --metrics checkpoints/row4_dice/test_metrics_postprocessed.csv

Writes into figures/:
    loss_curves.png    train and validation loss per epoch for every ablation run
    ef_agreement.png   model vs doctors' LV ejection fraction: scatter + Bland-Altman plot
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = {
    "row1_plain": "Plain U-Net",
    "row2_residual": "+ residual",
    "row3_attention": "+ attention",
    "row4_dice": "+ Dice loss",
}


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def loss_curves(out_dir: Path) -> None:
    """One panel per run. Losses are only comparable WITHIN a run (CE vs CE+Dice differ in scale)."""
    fig, axes = plt.subplots(1, len(RUNS), figsize=(4 * len(RUNS), 3.2))
    for ax, (run, title) in zip(axes, RUNS.items()):
        path = REPO / "checkpoints" / run / "history.csv"
        if not path.exists():
            ax.set_title(f"{title}\n(no run yet)")
            continue
        rows = read_csv(path)
        epochs = [int(r["epoch"]) for r in rows]
        val = [float(r["val_loss"]) for r in rows]
        ax.plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
        ax.plot(epochs, val, label="validation")
        best = int(np.argmin(val))
        ax.scatter(
            epochs[best], val[best], color="black", zorder=3, label="best (saved)"
        )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def ef_agreement(metrics: Path, out_dir: Path) -> None:
    """Left: does the model's EF track the doctors'? Right: how big and how biased are the misses?"""
    rows = read_csv(metrics)
    pred = np.array([float(r["LV_EF_pred"]) for r in rows])
    true = np.array([float(r["LV_EF_true"]) for r in rows])
    ok = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[ok], true[ok]

    diff, mean = pred - true, (pred + true) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    low, high = bias - 1.96 * sd, bias + 1.96 * sd  # 95% limits of agreement
    r = np.corrcoef(pred, true)[0, 1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    left.scatter(true, pred)
    lims = [0, max(true.max(), pred.max()) + 5]
    left.plot(lims, lims, "k--", linewidth=1, label="perfect agreement")
    left.set_xlabel("LV EF from doctors' masks (%)")
    left.set_ylabel("LV EF from model (%)")
    left.set_title(f"Correlation r = {r:.3f}")
    left.legend()

    right.scatter(mean, diff)
    right.axhline(bias, color="black", label=f"bias {bias:+.1f} points")
    for line in (low, high):
        right.axhline(line, color="gray", linestyle="--")
    right.set_xlabel("mean of model and doctors (%)")
    right.set_ylabel("model - doctors (percentage points)")
    right.set_title(f"Bland-Altman: 95% limits of agreement {low:+.1f} to {high:+.1f}")
    right.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ef_agreement.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--metrics", type=Path, required=True, help="a *_metrics.csv from evaluate.py"
    )
    args = p.parse_args()

    out_dir = REPO / "figures"
    out_dir.mkdir(exist_ok=True)
    loss_curves(out_dir)
    ef_agreement(args.metrics, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
