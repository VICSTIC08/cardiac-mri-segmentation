"""Tests that predictions go back to the original scan grid correctly, and the heart maths."""

import math

import numpy as np
import torch

from cardioseg.data.acdc import Scan
from cardioseg.data.preprocessing import center_crop_or_pad, resample_in_plane
from cardioseg.evaluation.metrics import dice
from cardioseg.evaluation.predict import (
    ejection_fraction,
    keep_largest_blob,
    predict_scan,
    resize_nearest,
    uncrop,
    volume_ml,
)
from cardioseg.models.unet import UNet


def blob_mask(shape=(216, 256, 6)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[80:140, 90:150, 1:5] = 2
    mask[95:125, 105:135, 1:5] = 3
    return mask


def test_uncrop_reverses_crop_for_big_and_small_scans():
    for shape in [(300, 260, 3), (180, 200, 3)]:  # bigger than 224, and smaller
        volume = np.random.default_rng(0).integers(1, 4, shape).astype(np.uint8)
        back = uncrop(center_crop_or_pad(volume), shape[:2])
        keep = back > 0  # only the part that survived the crop can come back
        assert back.shape == shape
        assert np.array_equal(back[keep], volume[keep])


def test_mask_survives_the_round_trip_to_224_and_back():
    """Resample + crop, then uncrop + resize back: almost every voxel should be unchanged."""
    mask, spacing = blob_mask(), (1.5, 1.5, 10.0)
    resampled = resample_in_plane(mask, spacing, is_mask=True)
    there = center_crop_or_pad(resampled)
    back = resize_nearest(uncrop(there, resampled.shape[:2]), mask.shape[:2])
    assert back.shape == mask.shape
    assert dice(back == 3, mask == 3) > 0.97


def test_predict_scan_returns_the_original_shape():
    scan = Scan(
        image=np.random.default_rng(1).random((216, 256, 6)).astype(np.float32),
        mask=blob_mask(),
        spacing=(1.5, 1.5, 10.0),
    )
    labels = predict_scan(UNet(base_channels=8), scan, torch.device("cpu"))
    assert labels.shape == scan.image.shape
    assert set(np.unique(labels)) <= {0, 1, 2, 3}


def test_volume_and_ejection_fraction_known_answers():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[:5, :4, :5] = 3  # 100 voxels
    assert math.isclose(
        volume_ml(mask, 3, spacing=(2.0, 2.0, 5.0)), 2.0
    )  # 100 x 20 mm^3
    assert math.isclose(ejection_fraction(edv=150.0, esv=60.0), 60.0)


def test_ejection_fraction_is_nan_when_no_ventricle_was_found():
    assert math.isnan(ejection_fraction(edv=0.0, esv=0.0))


def test_keep_largest_blob_removes_stray_islands():
    labels = np.zeros((40, 40, 4), dtype=np.uint8)
    labels[10:20, 10:20, 1:3] = 1  # the real RV: 200 voxels
    labels[35:37, 35:37, 0] = 1  # a stray "RV" island far away: 4 voxels
    labels[22:30, 22:30, 1:3] = 3  # the LV, untouched
    cleaned = keep_largest_blob(labels)
    assert cleaned[36, 36, 0] == 0  # island erased
    assert (cleaned == 1).sum() == 200  # real RV kept
    assert (cleaned == 3).sum() == (labels == 3).sum()  # other structures unchanged
