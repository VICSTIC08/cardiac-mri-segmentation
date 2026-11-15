"""Train a U-Net on ACDC and keep the version that does best on the validation patients.

Run from the repo root, e.g.:
    uv run python scripts/train.py --name smoke --epochs 2 --limit 4     (quick check, ~1 min)
    uv run python scripts/train.py --name row1_plain --epochs 50          (a real run)

Each run writes to checkpoints/<name>/:
    best.pt       the weights from the epoch with the lowest validation loss
    history.csv   train and validation loss for every epoch (for the loss-curve plot)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cardioseg.data.augment import RandomAugment
from cardioseg.data.dataset import ACDCSliceDataset
from cardioseg.data.splits import load_splits
from cardioseg.models.unet import UNet
from cardioseg.training.loop import evaluate, pick_device, train_one_epoch
from cardioseg.training.losses import CombinedLoss

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--name", required=True, help="run name, used for the output folder")
    p.add_argument("--residual", action="store_true", help="use residual blocks")
    p.add_argument("--attention", action="store_true", help="use attention gates")
    p.add_argument(
        "--no-dice", action="store_true", help="cross-entropy only (no Dice term)"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate (step size)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N patients per split",
    )
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "acdc" / "training")
    p.add_argument("--splits", type=Path, default=REPO / "splits" / "acdc.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Same seed -> same starting weights and same shuffling, so runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    out_dir = REPO / "checkpoints" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.splits)
    train_ids, val_ids = splits["train"][: args.limit], splits["val"][: args.limit]
    print(
        f"Loading {len(train_ids)} training and {len(val_ids)} validation patients..."
    )
    train_set = ACDCSliceDataset(
        args.data_dir, train_ids, augment=RandomAugment(seed=args.seed)
    )
    val_set = ACDCSliceDataset(args.data_dir, val_ids)  # never augment validation data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    print(
        f"{len(train_set)} training slices, {len(val_set)} validation slices, device: {device}"
    )

    model = UNet(residual=args.residual, attention=args.attention).to(device)
    loss_fn = CombinedLoss(use_dice=not args.no_dice)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Save the exact settings next to the results, so every number can be traced to its run.
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )

    best_val = float("inf")
    with open(out_dir / "history.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss = evaluate(model, val_loader, loss_fn, device)
            seconds = time.time() - start
            log.writerow(
                [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{seconds:.0f}"]
            )
            f.flush()

            marker = ""
            if val_loss < best_val:  # keep the best version, not just the last one
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "best.pt")
                marker = "  <- best so far, saved"
            print(
                f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.3f}  "
                f"val {val_loss:.3f}  ({seconds:.0f}s){marker}"
            )

    print(f"Done. Best validation loss {best_val:.3f}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
