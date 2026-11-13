import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cardioseg.data.dataset import ACDCSliceDataset
from cardioseg.data.preprocessing import CROP_SIZE


@pytest.fixture
def fake_data_dir(tmp_path):
    """Two fake patients: patient001 with 3 slices and patient002 with 5 slices."""
    voxel_size = np.diag([1.25, 1.25, 10.0, 1.0])
    for patient_id, n_slices in [("patient001", 3), ("patient002", 5)]:
        patient_dir = tmp_path / patient_id
        patient_dir.mkdir()
        (patient_dir / "Info.cfg").write_text("ED: 1\nES: 7\nGroup: NOR\n")
        for frame in (1, 7):
            image = np.random.default_rng(frame).random((200, 240, n_slices))
            mask = np.zeros((200, 240, n_slices), dtype=np.uint8)
            mask[90:110, 100:140, :] = 3
            stem = f"{patient_id}_frame{frame:02d}"
            nib.save(
                nib.Nifti1Image(image.astype(np.float32), voxel_size),
                patient_dir / f"{stem}.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(mask, voxel_size), patient_dir / f"{stem}_gt.nii.gz"
            )
    return tmp_path


def test_length_counts_every_slice_of_both_phases(fake_data_dir):
    dataset = ACDCSliceDataset(fake_data_dir, ["patient001", "patient002"])
    assert len(dataset) == (3 + 5) * 2  # slices x (ED + ES)


def test_only_the_requested_patients_are_loaded(fake_data_dir):
    dataset = ACDCSliceDataset(fake_data_dir, ["patient002"])
    assert len(dataset) == 5 * 2
    assert {info.patient_id for info in dataset.info} == {"patient002"}


def test_item_shapes_and_types(fake_data_dir):
    image, mask = ACDCSliceDataset(fake_data_dir, ["patient001"])[0]
    assert image.shape == (1, CROP_SIZE, CROP_SIZE)
    assert image.dtype == torch.float32
    assert mask.shape == (CROP_SIZE, CROP_SIZE)
    assert mask.dtype == torch.int64


def test_augment_is_applied_to_image_and_mask(fake_data_dir):
    def mark_corner(image, mask):
        image, mask = image.copy(), mask.copy()
        image[0, 0], mask[0, 0] = 99.0, 2
        return image, mask

    dataset = ACDCSliceDataset(fake_data_dir, ["patient001"], augment=mark_corner)
    image, mask = dataset[0]
    assert image[0, 0, 0] == 99.0
    assert mask[0, 0] == 2


def test_dataloader_makes_batches(fake_data_dir):
    dataset = ACDCSliceDataset(fake_data_dir, ["patient001", "patient002"])
    images, masks = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    assert images.shape == (4, 1, CROP_SIZE, CROP_SIZE)
    assert masks.shape == (4, CROP_SIZE, CROP_SIZE)
