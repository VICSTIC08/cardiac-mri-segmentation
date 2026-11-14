"""Guards against the bug from my research: image and mask drifting apart during augmentation."""

import numpy as np

from cardioseg.data.augment import RandomAugment, transform_slice


def make_slice(size=64):
    """A mask with an off-centre left-ventricle blob and muscle ring, plus an image copying it."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[20:34, 26:44] = 2
    mask[23:31, 29:41] = 3
    image = mask.astype(np.float32) * 100  # bright exactly where the mask says
    return image, mask


def centre_of(region):
    rows, cols = np.nonzero(region)
    return np.array([rows.mean(), cols.mean()])


def test_image_and_mask_stay_aligned():
    """The core guard: after random augmentation, the bright blob and label 3 are in the same place."""
    augment = RandomAugment(seed=0)
    for _ in range(20):
        image, mask = make_slice()
        image, mask = augment(image, mask)
        image_blob = image > 250  # brightness was 300 inside label 3
        assert np.abs(centre_of(image_blob) - centre_of(mask == 3)).max() < 1.0


def test_90_degree_rotation_is_exact():
    """A known answer: rotating a mask by 90 degrees must equal NumPy's exact rotation."""
    _, mask = make_slice()
    rotated = transform_slice(mask, 90, 1.0, (0, 0), is_mask=True)
    assert np.array_equal(rotated, np.rot90(mask, k=1))


def test_mask_labels_stay_whole():
    augment = RandomAugment(seed=1)
    for _ in range(20):
        _, mask = augment(*make_slice())
        assert set(np.unique(mask)) <= {0, 1, 2, 3}


def test_shapes_and_types_are_unchanged():
    image, mask = RandomAugment(seed=2)(*make_slice())
    assert image.shape == mask.shape == (64, 64)
    assert image.dtype == np.float32
    assert mask.dtype == np.uint8


def test_same_seed_gives_the_same_augmentation():
    first = RandomAugment(seed=3)(*make_slice())
    second = RandomAugment(seed=3)(*make_slice())
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
