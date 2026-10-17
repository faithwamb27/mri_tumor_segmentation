"""
Evaluation metrics for BraTS tumour segmentation.

BraTS evaluation convention
──────────────────────────
  WT (whole tumour)   = foreground classes 1 + 2 + 3  (after label remap)
  TC (tumour core)    = classes 1 + 3
  ET (enhancing)      = class  3

Both Dice coefficient and 95th-percentile Hausdorff distance (HD95) are
computed for each region.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from scipy.ndimage import label as ndimage_label

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Core binary Dice / HD95
# ─────────────────────────────────────────────────────────────────────────────

def binary_dice(pred: np.ndarray, gt: np.ndarray,
                smooth: float = 1e-5) -> float:
    """Dice coefficient for binary arrays."""
    intersection = (pred & gt).sum()
    denom        = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect match
    return float((2.0 * intersection + smooth) / (denom + smooth))


def binary_hd95(pred: np.ndarray, gt: np.ndarray,
                voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance (mm). Returns NaN if both empty."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    if HAS_MEDPY:
        return float(medpy_hd95(pred, gt, voxelspacing=voxel_spacing))
    # Fallback: simple max surface distance if medpy not installed
    return float(_simple_hd95(pred, gt))


def _simple_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Approximate HD95 without medpy (slower, for fallback use)."""
    from scipy.ndimage import distance_transform_edt
    pred_surf = pred ^ _binary_erosion(pred)
    gt_surf   = gt   ^ _binary_erosion(gt)
    d1 = distance_transform_edt(~gt)[pred_surf]
    d2 = distance_transform_edt(~pred)[gt_surf]
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95)) if len(all_d) > 0 else float("nan")


def _binary_erosion(arr: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(arr)


# ─────────────────────────────────────────────────────────────────────────────
# BraTS region extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_regions(
    seg: np.ndarray,
    regions: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """
    Convert a label map to binary masks for each named BraTS region.

    Parameters
    ----------
    seg     : (D, H, W) integer label map (0-indexed, remapped)
    regions : {'WT': [1,2,3], 'TC': [1,3], 'ET': [3]}
    """
    out = {}
    for name, labels in regions.items():
        mask = np.zeros_like(seg, dtype=bool)
        for lbl in labels:
            mask |= (seg == lbl)
        out[name] = mask
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_subject_metrics(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    regions: Dict[str, List[int]],
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    compute_hd95: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute Dice and HD95 for each tumour region.

    Returns
    -------
    {'WT': {'dice': 0.92, 'hd95': 3.1}, 'TC': {...}, 'ET': {...}}
    """
    pred_regions = extract_regions(pred_seg, regions)
    gt_regions   = extract_regions(gt_seg,   regions)

    results = {}
    for name in regions:
        p = pred_regions[name]
        g = gt_regions[name]
        dice = binary_dice(p, g)
        hd   = binary_hd95(p, g, voxel_spacing) if compute_hd95 else float("nan")
        results[name] = {"dice": dice, "hd95": hd}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics over a list of subjects
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(
    per_subject: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean ± std for each region and metric over all subjects.

    Returns
    -------
    {'WT': {'dice_mean': ..., 'dice_std': ..., 'hd95_mean': ..., ...}, ...}
    """
    regions = list(per_subject[0].keys())
    agg = {}
    for region in regions:
        dices = [s[region]["dice"] for s in per_subject]
        hd95s = [s[region]["hd95"] for s in per_subject
                 if not np.isnan(s[region]["hd95"])]
        agg[region] = {
            "dice_mean": float(np.mean(dices)),
            "dice_std":  float(np.std(dices)),
            "hd95_mean": float(np.mean(hd95s)) if hd95s else float("nan"),
            "hd95_std":  float(np.std(hd95s))  if hd95s else float("nan"),
        }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level Dice for training monitoring (tensors)
# ─────────────────────────────────────────────────────────────────────────────

def batch_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    skip_bg: bool = True,
    smooth: float = 1e-5,
) -> float:
    """
    Fast mean Dice over a batch of predictions (no grad).

    Parameters
    ----------
    logits  : (B, C, D, H, W)
    targets : (B, D, H, W)  int64
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)                          # (B, D, H, W)

        dices = []
        start = 1 if skip_bg else 0
        for c in range(start, num_classes):
            p = (preds   == c).float()
            g = (targets == c).float()
            inter = (p * g).sum()
            denom = p.sum() + g.sum()
            if denom == 0:
                dices.append(1.0)
            else:
                dices.append(float((2.0 * inter + smooth) / (denom + smooth)))
    return float(np.mean(dices))
