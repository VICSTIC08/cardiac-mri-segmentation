"""Predict a whole patient scan, undo the preprocessing, and turn masks into heart measurements.

The model works on resampled, cropped 224x224 slices. Doctors' masks (and all reported
numbers) live in the ORIGINAL scan grid. So every prediction goes back the way it came:
    224x224 crop -> uncrop to the resampled size -> resample back to the original pixel size.
"""

import numpy as np
import torch
from torch import nn

from cardioseg.data.acdc import Scan
from cardioseg.data.preprocessing import (
    CROP_SIZE,
    center_crop_or_pad,
    normalize,
    resample_in_plane,
)

# Standard value used to turn myocardium volume (mL) into mass (g).
MYOCARDIUM_DENSITY_G_PER_ML = 1.05


def uncrop(volume: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The reverse of center_crop_or_pad: put a size x size crop back into a (H, W) canvas."""
    size = volume.shape[0]
    out = np.zeros((*shape, volume.shape[2]), dtype=volume.dtype)
    source, target = [], []
    for length_out in shape:
        length = min(length_out, size)
        start_crop = (size - length) // 2
        start_out = (length_out - length) // 2
        source.append(slice(start_crop, start_crop + length))
        target.append(slice(start_out, start_out + length))
    out[target[0], target[1]] = volume[source[0], source[1]]
    return out


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize each slice of a label mask to exactly (H, W) by nearest neighbour (labels stay whole)."""
    rows = np.minimum(
        ((np.arange(shape[0]) + 0.5) * mask.shape[0] / shape[0]).astype(int),
        mask.shape[0] - 1,
    )
    cols = np.minimum(
        ((np.arange(shape[1]) + 0.5) * mask.shape[1] / shape[1]).astype(int),
        mask.shape[1] - 1,
    )
    return mask[rows][:, cols]


@torch.no_grad()
def predict_scan(model: nn.Module, scan: Scan, device: torch.device) -> np.ndarray:
    """Segment every slice of one scan. Returns labels 0-3 in the ORIGINAL scan's shape."""
    model.eval()
    resampled = normalize(resample_in_plane(scan.image, scan.spacing, is_mask=False))
    image = center_crop_or_pad(resampled, CROP_SIZE, fill=float(resampled.min()))

    # (H, W, slices) -> (slices, 1, H, W): every slice becomes one item in a batch.
    batch = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(
        1
    )
    labels = model(batch.to(device)).argmax(dim=1).cpu().numpy()  # (slices, H, W)
    labels = labels.transpose(1, 2, 0).astype(np.uint8)  # back to (H, W, slices)

    labels = uncrop(labels, resampled.shape[:2])
    return resize_nearest(labels, scan.image.shape[:2])


def volume_ml(mask: np.ndarray, label: int, spacing: tuple[float, ...]) -> float:
    """Volume of one structure: number of voxels x size of one voxel (mm^3), in millilitres."""
    return float((mask == label).sum() * np.prod(spacing) / 1000.0)  # 1 mL = 1000 mm^3


def ejection_fraction(edv: float, esv: float) -> float:
    """Percent of the blood pumped out per beat: (fullest - emptiest) / fullest x 100."""
    return 100.0 * (edv - esv) / edv
