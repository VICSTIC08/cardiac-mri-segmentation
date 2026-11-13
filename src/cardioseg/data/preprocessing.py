"""Make every scan look alike before training: same pixel size, same brightness scale, same size."""

import numpy as np
from scipy.ndimage import zoom

from cardioseg.data.acdc import Scan

TARGET_SPACING_MM = 1.25  # every pixel becomes 1.25 mm x 1.25 mm
CROP_SIZE = 224  # every slice becomes 224 x 224 pixels (280 mm x 280 mm)


def resample_in_plane(
    volume: np.ndarray, spacing: tuple[float, float, float], is_mask: bool
) -> np.ndarray:
    """Resize each slice so every pixel is TARGET_SPACING_MM wide. The slice gap is left alone."""
    factors = (spacing[0] / TARGET_SPACING_MM, spacing[1] / TARGET_SPACING_MM, 1.0)
    # Masks: nearest neighbour (order=0), so labels stay whole numbers 0-3.
    # Images: bilinear (order=1), so brightness changes smoothly.
    order = 0 if is_mask else 1
    return zoom(volume, factors, order=order)


def normalize(image: np.ndarray) -> np.ndarray:
    """Z-score: shift and scale the scan so its mean is 0 and its standard deviation is 1."""
    return ((image - image.mean()) / (image.std() + 1e-8)).astype(np.float32)


def center_crop_or_pad(
    volume: np.ndarray, size: int = CROP_SIZE, fill: float = 0
) -> np.ndarray:
    """Cut each slice down to size x size around its centre, or pad it with `fill` if smaller."""
    out = np.full((size, size, volume.shape[2]), fill, dtype=volume.dtype)
    source, target = [], []
    for length_in in volume.shape[:2]:
        length = min(length_in, size)
        start_in = (length_in - length) // 2
        start_out = (size - length) // 2
        source.append(slice(start_in, start_in + length))
        target.append(slice(start_out, start_out + length))
    out[target[0], target[1]] = volume[source[0], source[1]]
    return out


def preprocess_scan(scan: Scan) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline for one scan: resample, normalize, crop. Returns (image, mask)."""
    image = normalize(resample_in_plane(scan.image, scan.spacing, is_mask=False))
    mask = resample_in_plane(scan.mask, scan.spacing, is_mask=True)
    # Pad images with their darkest value and masks with background (0).
    image = center_crop_or_pad(image, fill=float(image.min()))
    mask = center_crop_or_pad(mask, fill=0)
    return image, mask
