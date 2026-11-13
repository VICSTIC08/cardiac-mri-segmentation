import nibabel as nib
import numpy as np
import pytest

from cardioseg.data.acdc import list_patients, load_patient, read_info


@pytest.fixture
def fake_patient(tmp_path):
    """A tiny fake ACDC patient: 8 x 6 pixels, 3 slices, ED = frame 1, ES = frame 5."""
    patient_dir = tmp_path / "patient999"
    patient_dir.mkdir()
    (patient_dir / "Info.cfg").write_text(
        "ED: 1\nES: 5\nGroup: NOR\nHeight: 170.0\nNbFrame: 20\nWeight: 60.0\n"
    )
    voxel_size = np.diag([1.5, 1.5, 10.0, 1.0])  # 1.5 x 1.5 mm pixels, 10 mm slices
    for frame in (1, 5):
        image = np.random.default_rng(frame).random((8, 6, 3)).astype(np.float32)
        mask = np.zeros((8, 6, 3), dtype=np.uint8)
        mask[2:5, 2:4, 1] = 3  # a small left-ventricle blob on slice 1
        stem = f"patient999_frame{frame:02d}"
        nib.save(nib.Nifti1Image(image, voxel_size), patient_dir / f"{stem}.nii.gz")
        nib.save(nib.Nifti1Image(mask, voxel_size), patient_dir / f"{stem}_gt.nii.gz")
    return patient_dir


def test_read_info(fake_patient):
    info = read_info(fake_patient)
    assert info["ED"] == "1"
    assert info["ES"] == "5"
    assert info["Group"] == "NOR"


def test_load_patient(fake_patient):
    patient = load_patient(fake_patient)
    assert patient.patient_id == "patient999"
    assert patient.group == "NOR"
    assert patient.ed.image.shape == (8, 6, 3)
    assert patient.ed.image.dtype == np.float32
    assert patient.ed.mask.dtype == np.uint8
    assert patient.ed.spacing == (1.5, 1.5, 10.0)
    assert set(np.unique(patient.es.mask)) == {0, 3}


def test_list_patients(fake_patient):
    (fake_patient.parent / "notes.txt").write_text("not a patient")
    assert list_patients(fake_patient.parent) == [fake_patient]
