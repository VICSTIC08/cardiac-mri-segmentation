import numpy as np

from cardioseg.data.acdc import Scan
from cardioseg.data.preprocessing import (
    CROP_SIZE,
    TARGET_SPACING_MM,
    center_crop_or_pad,
    normalize,
    preprocess_scan,
    resample_in_plane,
)


def make_scan(width=160, height=120, slices=4, spacing=(1.5, 1.5, 10.0)):
    """A fake scan with a square "left ventricle" in the middle of every slice."""
    rng = np.random.default_rng(0)
    image = rng.random((width, height, slices)).astype(np.float32) * 500
    mask = np.zeros((width, height, slices), dtype=np.uint8)
    mask[60:100, 40:80, :] = 3
    mask[50:60, 40:80, :] = 2
    return Scan(image=image, mask=mask, spacing=spacing)


def test_resampling_keeps_mask_labels_whole():
    scan = make_scan()
    mask = resample_in_plane(scan.mask, scan.spacing, is_mask=True)
    assert set(np.unique(mask)) <= {0, 1, 2, 3}  # no in-between labels like 1.5


def test_resampling_keeps_real_world_size():
    scan = make_scan(width=160, height=120, spacing=(1.5, 1.5, 10.0))
    image = resample_in_plane(scan.image, scan.spacing, is_mask=False)
    # 160 pixels x 1.5 mm = 240 mm wide, which is 192 pixels at 1.25 mm
    assert image.shape == (192, 144, 4)
    assert image.shape[0] * TARGET_SPACING_MM == 160 * 1.5


def test_image_and_mask_stay_the_same_shape():
    scan = make_scan(spacing=(1.37, 1.37, 5.0))
    image = resample_in_plane(scan.image, scan.spacing, is_mask=False)
    mask = resample_in_plane(scan.mask, scan.spacing, is_mask=True)
    assert image.shape == mask.shape


def test_normalize_gives_mean_0_std_1():
    image = normalize(make_scan().image)
    assert abs(image.mean()) < 1e-4
    assert abs(image.std() - 1) < 1e-4


def test_crop_and_pad_give_the_right_size_and_keep_the_centre():
    for width, height in [(300, 260), (100, 180)]:  # one too big, one partly too small
        volume = np.zeros((width, height, 2), dtype=np.uint8)
        volume[width // 2, height // 2, :] = 7  # mark the centre pixel
        out = center_crop_or_pad(volume, size=CROP_SIZE)
        assert out.shape == (CROP_SIZE, CROP_SIZE, 2)
        # The centre pixel must still be in the centre.
        assert out[CROP_SIZE // 2, CROP_SIZE // 2, 0] == 7


def test_preprocess_scan_end_to_end():
    image, mask = preprocess_scan(make_scan())
    assert image.shape == mask.shape == (CROP_SIZE, CROP_SIZE, 4)
    assert image.dtype == np.float32
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 2, 3}
