"""Split the 100 ACDC training patients into training and validation sets, by patient."""

import json
import random
from collections import defaultdict
from pathlib import Path

from cardioseg.data.acdc import list_patients, read_info

VAL_PER_GROUP = 4  # 4 from each 20-patient diagnosis group = 20 validation patients
SEED = 42  # fixed, so the split is identical every time it is made


def make_splits(
    training_dir: Path,
    testing_dir: Path,
    val_per_group: int = VAL_PER_GROUP,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Pick validation patients evenly from each diagnosis group; the rest are training."""
    by_group = defaultdict(list)
    for patient_dir in list_patients(training_dir):
        group = read_info(patient_dir)["Group"]
        by_group[group].append(patient_dir.name)

    rng = random.Random(seed)
    val = set()
    for group in sorted(by_group):
        val.update(rng.sample(by_group[group], val_per_group))

    train = [name for names in by_group.values() for name in names if name not in val]
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": [patient_dir.name for patient_dir in list_patients(testing_dir)],
    }


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    """Write the split to a JSON file, creating its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")


def load_splits(path: Path) -> dict[str, list[str]]:
    """Read a split saved by save_splits."""
    return json.loads(Path(path).read_text())
