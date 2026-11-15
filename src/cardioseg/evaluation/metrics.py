"""Segmentation metrics, written from scratch with NumPy so every number can be explained.

Dice  = how much the predicted and real regions OVERLAP (1 = perfect, 0 = none).
HD95  = how far off the EDGES are, in millimetres: the 95th-percentile distance between
        the predicted boundary and the real boundary (ignoring the worst 5% of points,
        so one stray pixel cannot dominate).

Both work on whole 3D volumes (all slices of one patient at one time point), because a
patient's heart is one 3D object, and spacing is in mm so distances are physical.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice between two boolean masks. Both empty counts as a perfect match (1.0)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2 * (pred & truth).sum() / total)


def surface(mask: np.ndarray) -> np.ndarray:
    """The boundary voxels of a mask: voxels in the mask whose neighbour is outside it."""
    return mask & ~binary_erosion(mask)


def hd95(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...]) -> float:
    """95th-percentile Hausdorff distance in mm. NaN if exactly one of the two is empty."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return float("nan")  # no boundary to measure against

    pred_edge, truth_edge = surface(pred), surface(truth)
    # For every voxel: distance (in mm) to the nearest boundary voxel of the OTHER mask.
    dist_to_truth = distance_transform_edt(~truth_edge, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_edge, sampling=spacing)
    # Check both directions: predicted edge -> real edge, and real edge -> predicted edge.
    return float(
        max(
            np.percentile(dist_to_truth[pred_edge], 95),
            np.percentile(dist_to_pred[truth_edge], 95),
        )
    )
