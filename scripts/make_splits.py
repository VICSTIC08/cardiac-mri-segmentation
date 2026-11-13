"""Create splits/acdc.json: which patients are used for training, validation and testing.

Run from the repo root:  uv run python scripts/make_splits.py
"""

from pathlib import Path

from cardioseg.data.splits import make_splits, save_splits

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "acdc"
OUT_FILE = REPO / "splits" / "acdc.json"


def main() -> None:
    splits = make_splits(DATA_DIR / "training", DATA_DIR / "testing")
    save_splits(splits, OUT_FILE)
    for name, patients in splits.items():
        print(f"{name}: {len(patients)} patients")
    print("validation:", ", ".join(splits["val"]))
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
