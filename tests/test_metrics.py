"""Known-answer tests for the metrics, plus a slow brute-force check of HD95."""

import math

import numpy as np
from scipy.spatial.distance import cdist

from cardioseg.evaluation.metrics import dice, hd95, surface


def box(shape, start, stop):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(a, b) for a, b in zip(start, stop))] = True
    return mask


def test_dice_known_answers():
    a = box((10, 10, 4), (0, 0, 0), (4, 10, 4))  # 160 voxels
    b = box((10, 10, 4), (2, 0, 0), (6, 10, 4))  # 160 voxels, half overlapping a
    assert dice(a, a) == 1.0
    assert dice(a, b) == 0.5  # 2 * 80 / (160 + 160)
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0


def test_hd95_is_zero_for_identical_masks():
    a = box((20, 20, 5), (5, 5, 1), (15, 15, 4))
    assert hd95(a, a, spacing=(1.25, 1.25, 10.0)) == 0.0


def test_hd95_measures_a_shift_in_millimetres():
    """Shift a box by 3 pixels along the first axis at 1.25 mm/pixel: edges are 3.75 mm off."""
    a = box((30, 30, 5), (5, 5, 1), (15, 15, 4))
    b = box((30, 30, 5), (8, 5, 1), (18, 15, 4))
    assert math.isclose(hd95(a, b, spacing=(1.25, 1.25, 10.0)), 3.75)


def test_hd95_matches_a_brute_force_calculation():
    """Compare every boundary point with every other one, the slow obvious way."""
    rng = np.random.default_rng(0)
    a = box((24, 24, 4), (4, 6, 0), (16, 18, 3))
    b = box((24, 24, 4), (6, 4, 1), (20, 15, 4)) | (rng.random((24, 24, 4)) > 0.97)
    spacing = (1.5, 1.5, 8.0)
    pa = np.argwhere(surface(a)) * spacing
    pb = np.argwhere(surface(b)) * spacing
    d = cdist(pa, pb)
    expected = max(np.percentile(d.min(axis=1), 95), np.percentile(d.min(axis=0), 95))
    assert math.isclose(hd95(a, b, spacing), expected, rel_tol=1e-6)


def test_hd95_is_nan_when_one_mask_is_empty():
    a = box((10, 10, 2), (2, 2, 0), (6, 6, 2))
    assert math.isnan(hd95(a, np.zeros_like(a), spacing=(1.0, 1.0, 1.0)))
