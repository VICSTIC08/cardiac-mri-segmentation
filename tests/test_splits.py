import pytest

from cardioseg.data.acdc import read_info
from cardioseg.data.splits import load_splits, make_splits, save_splits

GROUPS = ["DCM", "HCM", "MINF", "NOR", "RV"]


@pytest.fixture
def fake_dataset(tmp_path):
    """30 fake training patients (6 per diagnosis group) and 3 fake test patients."""
    training, testing = tmp_path / "training", tmp_path / "testing"
    for number in range(1, 31):
        patient_dir = training / f"patient{number:03d}"
        patient_dir.mkdir(parents=True)
        group = GROUPS[(number - 1) // 6]
        (patient_dir / "Info.cfg").write_text(f"ED: 1\nES: 10\nGroup: {group}\n")
    for number in range(101, 104):
        (testing / f"patient{number:03d}").mkdir(parents=True)
    return training, testing


def test_no_patient_is_in_two_splits(fake_dataset):
    splits = make_splits(*fake_dataset, val_per_group=2)
    train, val, test = (set(splits[name]) for name in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train) + len(val) == 30  # every training patient used exactly once


def test_validation_has_the_same_number_from_every_group(fake_dataset):
    training, testing = fake_dataset
    splits = make_splits(training, testing, val_per_group=2)
    groups = [read_info(training / name)["Group"] for name in splits["val"]]
    assert sorted(groups) == sorted(GROUPS * 2)


def test_same_seed_gives_the_same_split(fake_dataset):
    assert make_splits(*fake_dataset, seed=7) == make_splits(*fake_dataset, seed=7)


def test_save_and_load_round_trip(fake_dataset, tmp_path):
    splits = make_splits(*fake_dataset, val_per_group=2)
    path = tmp_path / "splits" / "acdc.json"
    save_splits(splits, path)
    assert load_splits(path) == splits
