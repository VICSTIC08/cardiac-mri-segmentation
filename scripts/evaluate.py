"""Score a trained model: Dice, HD95 and the heart measurements, patient by patient.

Run from the repo root, e.g.:
    uv run python scripts/evaluate.py --name row1_plain                  (validation)
    uv run python scripts/evaluate.py --name row4_dice --split test --data-dir data/acdc/testing --postprocess

Everything is measured in each patient's ORIGINAL scan grid (true mm and mL), for both
time points (ED = heart fullest, ES = heart emptiest). Writes
checkpoints/<name>/<split>_metrics[_postprocessed].csv with one row per patient, and prints a
summary table. The test split is for one final evaluation only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cardioseg.data.acdc import load_patient
from cardioseg.data.splits import load_splits
from cardioseg.evaluation.metrics import dice, hd95
from cardioseg.evaluation.predict import (
    MYOCARDIUM_DENSITY_G_PER_ML,
    ejection_fraction,
    keep_largest_blob,
    predict_scan,
    volume_ml,
)
from cardioseg.models.unet import UNet
from cardioseg.training.loop import pick_device

REPO = Path(__file__).resolve().parent.parent
STRUCTURES = {"RV": 1, "Myo": 2, "LV": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--name", required=True, help="run name (folder in checkpoints/)")
    p.add_argument(
        "--split", default="val", help="which split to score (val; test only ONCE)"
    )
    p.add_argument(
        "--postprocess",
        action="store_true",
        help="keep only the largest blob per structure",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="only score the first N patients"
    )
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "acdc" / "training")
    p.add_argument("--splits", type=Path, default=REPO / "splits" / "acdc.json")
    return p.parse_args()


def load_model(run_dir: Path, device: torch.device) -> UNet:
    """Rebuild the exact model the run trained (from config.json), then load its best weights."""
    config = json.loads((run_dir / "config.json").read_text())
    model = UNet(residual=config["residual"], attention=config["attention"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device)


def score_patient(
    model: UNet, patient_dir: Path, device: torch.device, postprocess: bool = False
) -> dict:
    """All metrics for one patient, comparing the model's masks with the doctors' masks."""
    patient = load_patient(patient_dir)
    row = {"patient": patient.patient_id, "group": patient.group}
    volumes = {}
    for phase, scan in (("ED", patient.ed), ("ES", patient.es)):
        pred = predict_scan(model, scan, device)
        if postprocess:
            pred = keep_largest_blob(pred)
        for name, label in STRUCTURES.items():
            row[f"{phase}_{name}_dice"] = dice(pred == label, scan.mask == label)
            row[f"{phase}_{name}_hd95"] = hd95(
                pred == label, scan.mask == label, scan.spacing
            )
            for source, mask in (("pred", pred), ("true", scan.mask)):
                volumes[(phase, name, source)] = volume_ml(mask, label, scan.spacing)

    for name in ("LV", "RV"):
        for source in ("pred", "true"):
            edv, esv = volumes[("ED", name, source)], volumes[("ES", name, source)]
            row[f"{name}_EDV_{source}"] = edv
            row[f"{name}_ESV_{source}"] = esv
            row[f"{name}_EF_{source}"] = ejection_fraction(edv, esv)
    for source in ("pred", "true"):
        row[f"Myo_mass_{source}"] = (
            volumes[("ED", "Myo", source)] * MYOCARDIUM_DENSITY_G_PER_ML
        )
    return row


def summarize(rows: list[dict]) -> None:
    """Print the averages a paper (or README) would report."""
    print("\nSegmentation (mean over patients, ED and ES together)")
    print(f"{'':6}{'Dice':>8}{'HD95 (mm)':>12}")
    for name in STRUCTURES:
        d = [r[f"{p}_{name}_dice"] for r in rows for p in ("ED", "ES")]
        h = [r[f"{p}_{name}_hd95"] for r in rows for p in ("ED", "ES")]
        print(f"{name:6}{np.mean(d):8.3f}{np.nanmean(h):12.1f}")

    print("\nClinical measurements (model vs doctors' masks)")
    print(f"{'':12}{'mean error':>12}{'mean |error|':>14}{'correlation':>13}")
    for key, unit in (
        ("LV_EF", "%"),
        ("RV_EF", "%"),
        ("LV_EDV", "mL"),
        ("Myo_mass", "g"),
    ):
        pred = np.array([r[f"{key}_pred"] for r in rows])
        true = np.array([r[f"{key}_true"] for r in rows])
        ok = np.isfinite(pred) & np.isfinite(
            true
        )  # skip patients where EF was undefined
        pred, true = pred[ok], true[ok]
        error = pred - true  # positive = model says MORE than the doctors
        r = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else float("nan")
        print(
            f"{key:12}{error.mean():+10.1f} {unit:2}{np.abs(error).mean():10.1f} {unit:3}{r:11.3f}"
        )


def main() -> None:
    args = parse_args()
    device = pick_device()
    run_dir = REPO / "checkpoints" / args.name
    model = load_model(run_dir, device)
    patient_ids = load_splits(args.splits)[args.split][: args.limit]

    rows = []
    for i, patient_id in enumerate(patient_ids, 1):
        row = score_patient(model, args.data_dir / patient_id, device, args.postprocess)
        rows.append(row)
        print(
            f"[{i:2d}/{len(patient_ids)}] {patient_id} ({row['group']})  "
            f"LV Dice {row['ED_LV_dice']:.3f}/{row['ES_LV_dice']:.3f}  "
            f"LV EF {row['LV_EF_pred']:.1f}% (true {row['LV_EF_true']:.1f}%)"
        )

    suffix = "_postprocessed" if args.postprocess else ""
    out_file = run_dir / f"{args.split}_metrics{suffix}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    print(f"\nPer-patient results saved to {out_file}")


if __name__ == "__main__":
    main()
