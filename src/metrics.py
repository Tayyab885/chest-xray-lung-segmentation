"""Segmentation metrics on boolean 2D masks.

Overlap metrics (dice, iou) are defined for empty masks: two empty masks agree
perfectly, one empty and one not agree not at all. Distance metrics are
undefined when a surface does not exist, so they return NaN rather than a
number that would silently pollute an average. HD95 and ASSD are reported in
pixels at the resolution of the input masks, so results computed on resized
images are not in physical units.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _check(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected 2D masks, got {pred.ndim}D")
    if pred.dtype != np.bool_ or gt.dtype != np.bool_:
        raise ValueError("masks must be boolean, threshold predictions first")
    return pred, gt


def dice(pred, gt):
    pred, gt = _check(pred, gt)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def iou(pred, gt):
    pred, gt = _check(pred, gt)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _boundary(mask):
    """One-pixel inner boundary of a binary mask."""
    # border_value=0 treats everything outside the array as background, so a
    # mask that touches the array edge gets a boundary segment along that
    # edge. This matches the medpy convention, but it biases ASSD/HD95
    # optimistically when both masks are cut off by the same field of view.
    eroded = binary_erosion(mask, border_value=0)
    return np.logical_xor(mask, eroded)


def _surface_distances(pred, gt):
    """Distances from each pred surface pixel to gt, and the reverse."""
    pred_b = _boundary(pred)
    gt_b = _boundary(gt)
    if not pred_b.any() or not gt_b.any():
        return None, None
    dt_to_gt = distance_transform_edt(~gt_b)
    dt_to_pred = distance_transform_edt(~pred_b)
    return dt_to_gt[pred_b], dt_to_pred[gt_b]


def hd95(pred, gt):
    pred, gt = _check(pred, gt)
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg is None:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def assd(pred, gt):
    pred, gt = _check(pred, gt)
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg is None:
        return float("nan")
    return float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))


def all_metrics(pred, gt):
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "hd95": hd95(pred, gt),
        "assd": assd(pred, gt),
    }
