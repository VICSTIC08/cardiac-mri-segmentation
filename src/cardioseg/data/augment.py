"""Random changes to training slices, so the model sees more variety than 80 patients provide.

The rule that matters: the image and its mask always get the SAME geometric change.
If they were moved differently, the answers would no longer line up with the picture.
"""

import numpy as np
from scipy.ndimage import affine_transform


def transform_slice(
    slice_2d: np.ndarray,
    angle_deg: float,
    scale: float,
    shift: tuple[float, float],
    is_mask: bool,
) -> np.ndarray:
    """Rotate and zoom a slice around its centre, then shift it, keeping its size."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    # affine_transform asks: "for each OUTPUT pixel, where in the INPUT do I look?"
    # So we give it the inverse of the movement we want.
    inverse = np.array([[cos, sin], [-sin, cos]]) / scale
    centre = (np.array(slice_2d.shape) - 1) / 2
    offset = centre - inverse @ (centre + np.array(shift))
    return affine_transform(
        slice_2d,
        inverse,
        offset=offset,
        order=0 if is_mask else 1,  # nearest for masks, bilinear for images
        mode="constant",
        cval=0 if is_mask else float(slice_2d.min()),  # fill with background
    )


class RandomAugment:
    """Applied to each training slice every time it is used, so every pass looks a bit different."""

    def __init__(
        self,
        max_rotation_deg: float = 15.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        max_shift_px: float = 10.0,
        seed: int | None = None,
    ):
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range
        self.max_shift_px = max_shift_px
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # 1. Geometry: draw ONE set of numbers and use it for both image and mask.
        angle = self.rng.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        scale = self.rng.uniform(*self.scale_range)
        shift = tuple(self.rng.uniform(-self.max_shift_px, self.max_shift_px, size=2))
        image = transform_slice(image, angle, scale, shift, is_mask=False)
        mask = transform_slice(mask, angle, scale, shift, is_mask=True)

        # 2. Brightness: image only. A mask's labels never change brightness.
        contrast = self.rng.uniform(0.9, 1.1)
        brightness = self.rng.uniform(-0.1, 0.1)
        image = image * contrast + brightness
        if self.rng.random() < 0.5:
            image = image + self.rng.normal(0, 0.05, size=image.shape)
        return image.astype(np.float32), mask
