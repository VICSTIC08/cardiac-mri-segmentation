"""Build the ablation table: one row per training run, the same validation metrics for each.

Run from the repo root after scoring each run with scripts/evaluate.py:
    uv run python scripts/ablation_table.py

Prints a Markdown table (ready to paste into EXPERIMENTS.md / the README) and saves it to
checkpoints/ablation_table.md. Each row adds ONE change to the row above, so any difference
between two neighbouring rows comes from that one change.
"""

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# (run folder, what changed compared with the row above, which results file)
ROWS = [
    ("row1_plain", "Plain U-Net, cross-entropy", "val_metrics.csv"),
    ("row2_residual", "+ residual blocks", "val_metrics.csv"),
    ("row3_attention", "+ attention gates", "val_metrics.csv"),
    ("row4_dice", "+ Dice loss term", "val_metrics.csv"),
    ("row4_dice", "+ largest-blob post-processing", "val_metrics_postprocessed.csv"),
]


def mean_over_phases(rows: list[dict], metric: str, structure: str) -> float:
    values = [
        float(r[f"{phase}_{structure}_{metric}"])
        for r in rows
        for phase in ("ED", "ES")
    ]
    return float(np.nanmean(values))


def mean_abs_error(rows: list[dict], key: str) -> float:
    errors = [abs(float(r[f"{key}_pred"]) - float(r[f"{key}_true"])) for r in rows]
    return float(np.nanmean(errors))


def main() -> None:
    header = (
        "| Change | Dice RV | Dice Myo | Dice LV | HD95 RV (mm) | HD95 Myo | HD95 LV "
        "| LV EF error (%) | RV EF error (%) |"
    )
    lines = [header, "|" + "---|" * 9]
    for run, change, filename in ROWS:
        path = REPO / "checkpoints" / run / filename
        if not path.exists():
            lines.append(f"| {change} | *not scored yet* |" + " |" * 7)
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        dice = [mean_over_phases(rows, "dice", s) for s in ("RV", "Myo", "LV")]
        hd = [mean_over_phases(rows, "hd95", s) for s in ("RV", "Myo", "LV")]
        ef = [mean_abs_error(rows, k) for k in ("LV_EF", "RV_EF")]
        lines.append(
            f"| {change} | {dice[0]:.3f} | {dice[1]:.3f} | {dice[2]:.3f} "
            f"| {hd[0]:.1f} | {hd[1]:.1f} | {hd[2]:.1f} | {ef[0]:.1f} | {ef[1]:.1f} |"
        )

    table = "\n".join(lines)
    print(table)
    out_file = REPO / "checkpoints" / "ablation_table.md"
    out_file.write_text(table + "\n")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
