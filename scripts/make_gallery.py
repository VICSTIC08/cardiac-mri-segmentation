"""Make the README gallery: what the final model's outlines look like on held-out test patients.

Run from the repo root (needs the data and the trained row4_dice checkpoint, CPU is fine):
    uv run python scripts/make_gallery.py

Uses the same final model and post-processing as the reported test numbers. No patient is
hand-picked: for each diagnosis group it shows the patient with the MEDIAN mean Dice in
results/test_metrics_row4_postprocessed.csv, so the pictures are typical, not best case.

Writes into figures/:
    test_gallery.png     one typical test patient per diagnosis group: scan, experts, model
    heartbeat_stack.png  one patient base to apex, full (ED) and squeezed (ES), model outlines
    social_preview.png   1280 x 640 banner for GitHub's social preview and link cards
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from cardioseg.data.acdc import load_patient
from cardioseg.evaluation.predict import keep_largest_blob, predict_scan
from cardioseg.models.unet import UNet

REPO = Path(__file__).resolve().parent.parent
COLORS = {1: "#e8574a", 2: "#4cc38a", 3: "#4a9be8"}  # RV, myocardium, LV
NAMES = {1: "right ventricle", 2: "myocardium", 3: "left ventricle"}
GROUP_NAMES = {
    "NOR": "Normal",
    "MINF": "Previous heart attack",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}
DICE_COLUMNS = [f"{p}_{s}_dice" for p in ("ED", "ES") for s in ("RV", "Myo", "LV")]


def load_model(run_dir: Path) -> UNet:
    config = json.loads((run_dir / "config.json").read_text())
    model = UNet(residual=config["residual"], attention=config["attention"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.eval()


def median_patients(metrics_csv: Path) -> dict[str, tuple[str, float]]:
    """Per group, the patient whose mean Dice (6 scores) is the group median."""
    by_group: dict[str, list[tuple[float, str]]] = {}
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            score = float(np.mean([float(row[c]) for c in DICE_COLUMNS]))
            by_group.setdefault(row["group"], []).append((score, row["patient"]))
    picked = {}
    for group, rows in by_group.items():
        rows.sort()
        score, patient = rows[(len(rows) - 1) // 2]
        picked[group] = (patient, score)
    return picked


def heart_box(mask: np.ndarray, margin_mm: float, spacing) -> tuple[slice, slice]:
    """A square crop (in the displayed, transposed orientation) around all labelled tissue."""
    m = mask.any(axis=2).T
    rows, cols = np.nonzero(m)
    pad = int(margin_mm / spacing[0])
    r0, r1 = rows.min() - pad, rows.max() + pad
    c0, c1 = cols.min() - pad, cols.max() + pad
    side = max(r1 - r0, c1 - c0)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0, c0 = max(rc - side // 2, 0), max(cc - side // 2, 0)
    return slice(r0, min(r0 + side, m.shape[0])), slice(c0, min(c0 + side, m.shape[1]))


def mid_slice(mask: np.ndarray) -> int:
    """The middle of the slices that contain the left ventricle."""
    z = np.nonzero((mask == 3).any(axis=(0, 1)))[0]
    return int(z[len(z) // 2])


def show(ax, image, labels=None, box=None, fill=0.35, lw=1.4):
    """Scan in grey, with each structure as a filled, outlined region."""
    img = image.T
    if box is not None:
        img = img[box]
    lo, hi = np.percentile(img, (1, 99.5))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="bilinear")
    if labels is not None:
        lab = labels.T
        if box is not None:
            lab = lab[box]
        for value, color in COLORS.items():
            region = (lab == value).astype(float)
            if region.any():
                rgba = np.zeros((*region.shape, 4))
                rgba[..., :3] = plt.matplotlib.colors.to_rgb(color)
                rgba[..., 3] = region * fill
                ax.imshow(rgba, interpolation="nearest")
                ax.contour(region, levels=[0.5], colors=[color], linewidths=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend_handles():
    return [Patch(facecolor=c, label=NAMES[v]) for v, c in COLORS.items()]


def test_gallery(model, data_dir, picked, out: Path) -> None:
    groups = [g for g in GROUP_NAMES if g in picked]
    fig, axes = plt.subplots(len(groups), 3, figsize=(8.4, 2.9 * len(groups)))
    for i, group in enumerate(groups):
        patient_id, score = picked[group]
        patient = load_patient(data_dir / patient_id)
        scan = patient.ed
        pred = keep_largest_blob(predict_scan(model, scan, torch.device("cpu")))
        z = mid_slice(scan.mask)
        box = heart_box(scan.mask, 25, scan.spacing)
        show(axes[i, 0], scan.image[:, :, z], box=box)
        show(axes[i, 1], scan.image[:, :, z], scan.mask[:, :, z], box)
        show(axes[i, 2], scan.image[:, :, z], pred[:, :, z], box)
        axes[i, 0].set_ylabel(
            f"{GROUP_NAMES[group]}\n{patient_id}, mean Dice {score:.3f}", fontsize=9
        )
    for ax, title in zip(axes[0], ("MRI (end-diastole)", "Experts", "Model")):
        ax.set_title(title, fontsize=11)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def heartbeat_stack(model, data_dir, patient_id, out: Path) -> None:
    patient = load_patient(data_dir / patient_id)
    zs = np.nonzero((patient.ed.mask > 0).any(axis=(0, 1)))[0]
    zs = zs[np.linspace(0, len(zs) - 1, min(6, len(zs))).astype(int)]
    box = heart_box(patient.ed.mask, 20, patient.ed.spacing)
    fig, axes = plt.subplots(2, len(zs), figsize=(2.1 * len(zs), 4.6))
    for r, (phase, scan) in enumerate(
        (("ED: fullest", patient.ed), ("ES: squeezed", patient.es))
    ):
        pred = keep_largest_blob(predict_scan(model, scan, torch.device("cpu")))
        for c, z in enumerate(zs):
            show(axes[r, c], scan.image[:, :, z], pred[:, :, z], box, lw=1.2)
        axes[r, 0].set_ylabel(phase, fontsize=11)
    axes[0, 0].set_title("base", fontsize=10, loc="left")
    axes[0, -1].set_title("apex", fontsize=10, loc="right")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def social_preview(model, data_dir, patient_id, out: Path) -> None:
    patient = load_patient(data_dir / patient_id)
    scan = patient.ed
    pred = keep_largest_blob(predict_scan(model, scan, torch.device("cpu")))
    z = mid_slice(scan.mask)
    box = heart_box(scan.mask, 30, scan.spacing)
    fig = plt.figure(figsize=(12.8, 6.4), facecolor="#0d1117")
    ax_scan = fig.add_axes((0.035, 0.12, 0.27, 0.76))
    ax_pred = fig.add_axes((0.32, 0.12, 0.27, 0.76))
    show(ax_scan, scan.image[:, :, z], box=box)
    show(ax_pred, scan.image[:, :, z], pred[:, :, z], box, fill=0.4, lw=2.0)
    for ax, label in ((ax_scan, "MRI"), (ax_pred, "Model")):
        ax.set_title(label, color="#c9d1d9", fontsize=15, fontweight="bold", loc="left")
    text = {"color": "white", "transform": fig.transFigure}
    fig.text(0.625, 0.76, "Cardiac MRI", fontsize=38, fontweight="bold", **text)
    fig.text(0.625, 0.66, "Segmentation", fontsize=38, fontweight="bold", **text)
    lines = [
        ("Dice 0.943 / 0.909 / 0.903", "LV / RV / myocardium, 50 test patients"),
        ("Ejection fraction r = 0.988", "vs the experts' measurements"),
        ("Within 0.02 Dice", "of the 2017 ACDC challenge winner"),
    ]
    y = 0.50
    for big, small in lines:
        fig.text(0.625, y, big, fontsize=19, fontweight="bold", **text)
        fig.text(
            0.625,
            y - 0.05,
            small,
            fontsize=13,
            color="#9aa4b2",
            transform=fig.transFigure,
        )
        y -= 0.135
    fig.text(
        0.625,
        0.07,
        "PyTorch  ·  residual attention U-Net  ·  48 tests in CI",
        fontsize=12,
        color="#9aa4b2",
        transform=fig.transFigure,
    )
    for i, (value, color) in enumerate(COLORS.items()):
        x = 0.035 + i * 0.13
        fig.patches.append(
            plt.Rectangle(
                (x, 0.045), 0.012, 0.024, color=color, transform=fig.transFigure
            )
        )
        fig.text(
            x + 0.017,
            0.047,
            NAMES[value],
            fontsize=11,
            color="#c9d1d9",
            transform=fig.transFigure,
        )
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--run", type=Path, default=REPO / "checkpoints" / "row4_dice")
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "acdc" / "testing")
    p.add_argument(
        "--metrics",
        type=Path,
        default=REPO / "results" / "test_metrics_row4_postprocessed.csv",
    )
    p.add_argument("--out-dir", type=Path, default=REPO / "figures")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    model = load_model(args.run)
    picked = median_patients(args.metrics)
    print({g: f"{pid} ({s:.3f})" for g, (pid, s) in picked.items()})
    args.out_dir.mkdir(exist_ok=True)
    test_gallery(model, args.data_dir, picked, args.out_dir / "test_gallery.png")
    normal = picked["NOR"][0]
    heartbeat_stack(model, args.data_dir, normal, args.out_dir / "heartbeat_stack.png")
    social_preview(model, args.data_dir, normal, args.out_dir / "social_preview.png")
    print("wrote", *(f.name for f in sorted(args.out_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
