"""A PyTorch Dataset that serves preprocessed 2D slices (image + mask) from ACDC patients."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cardioseg.data.acdc import load_patient
from cardioseg.data.preprocessing import preprocess_scan

# An augmentation takes one (image, mask) slice pair and returns a changed pair.
Augment = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class SliceInfo:
    """Where a slice came from, for debugging and figures."""

    patient_id: str
    phase: str  # "ED" or "ES"
    slice_index: int


class ACDCSliceDataset(Dataset):
    """Every slice of the ED and ES scans of the given patients, preprocessed and ready to train on.

    All slices are loaded into memory once, when the dataset is created, so training
    does not reread and re-resample files on every pass.
    """

    def __init__(
        self, data_dir: Path, patient_ids: list[str], augment: Augment | None = None
    ):
        self.augment = augment
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.info: list[SliceInfo] = []

        for patient_id in patient_ids:
            patient = load_patient(Path(data_dir) / patient_id)
            for phase, scan in (("ED", patient.ed), ("ES", patient.es)):
                image, mask = preprocess_scan(scan)
                for z in range(image.shape[2]):
                    self.images.append(image[:, :, z])
                    self.masks.append(mask[:, :, z])
                    self.info.append(SliceInfo(patient_id, phase, z))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.images[index], self.masks[index]
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        # Image: (1, H, W) float32. The 1 is the "channel": one greyscale value per pixel.
        # Mask: (H, W) int64, one class label per pixel, the format the loss function expects.
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor
