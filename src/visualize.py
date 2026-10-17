"""
Visualisation helpers for MRI tumour segmentation.

Functions
─────────
• plot_subject        – axial/coronal/sagittal slices of all 4 modalities + GT + prediction
• plot_training_curve – loss / Dice curves from the history log
• save_prediction_nii – export prediction as a NIfTI file
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")           # headless backend – safe on servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Colour map for segmentation overlay
# ─────────────────────────────────────────────────────────────────────────────

# Remapped labels: 0=BG, 1=NCR, 2=ED, 3=ET
_LABEL_COLORS = {
    0: (0.0,  0.0,  0.0,  0.0),   # background – transparent
    1: (1.0,  0.0,  0.0,  0.7),   # NCR – red
    2: (0.0,  0.6,  1.0,  0.7),   # ED  – blue
    3: (0.0,  1.0,  0.0,  0.7),   # ET  – green
}
_LABEL_NAMES = {1: "NCR/NET", 2: "Edema", 3: "Enhancing"}


def _seg_to_rgba(seg: np.ndarray) -> np.ndarray:
    """Convert (H, W) integer label map to (H, W, 4) RGBA float array."""
    rgba = np.zeros((*seg.shape, 4), dtype=np.float32)
    for lbl, color in _LABEL_COLORS.items():
        mask = seg == lbl
        rgba[mask] = color
    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_subject(
    image: np.ndarray,          # (C, D, H, W)  float – C modalities
    gt_seg: Optional[np.ndarray] = None,    # (D, H, W)  int
    pred_seg: Optional[np.ndarray] = None,  # (D, H, W)  int
    modality_names: List[str] = None,
    slice_idx: Optional[int] = None,        # axial slice; None = auto midpoint
    save_path: Optional[str] = None,
    title: str = "Subject",
) -> plt.Figure:
    """
    Plot one axial slice showing all modalities with GT and/or prediction
    segmentation overlays.

    Parameters
    ----------
    image          : (C, D, H, W) numpy array (e.g., from dataset[i][0].numpy())
    gt_seg / pred_seg : (D, H, W) integer label maps (may be None)
    save_path      : if given, save the figure to this path
    """
    C, D, H, W = image.shape
    if modality_names is None:
        modality_names = [f"mod_{i}" for i in range(C)]
    if slice_idx is None:
        # Pick the axial slice with the most foreground (or midpoint)
        if gt_seg is not None:
            fg = (gt_seg > 0).sum(axis=(1, 2))
            slice_idx = int(fg.argmax())
        else:
            slice_idx = D // 2

    # Number of columns: C modalities + optional GT + optional pred
    cols = C + (1 if gt_seg is not None else 0) + (1 if pred_seg is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.5))
    if cols == 1:
        axes = [axes]

    ax_idx = 0
    for c in range(C):
        ax = axes[ax_idx]; ax_idx += 1
        vol_slice = image[c, slice_idx]
        ax.imshow(vol_slice, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(modality_names[c], fontsize=9)
        ax.axis("off")

    for seg, label in [(gt_seg, "Ground Truth"), (pred_seg, "Prediction")]:
        if seg is None:
            continue
        ax = axes[ax_idx]; ax_idx += 1
        # Background modality = T1 (first channel)
        ax.imshow(image[0, slice_idx], cmap="gray", origin="lower",
                  interpolation="nearest")
        ax.imshow(_seg_to_rgba(seg[slice_idx]), origin="lower",
                  interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=_LABEL_COLORS[k][:3], label=v)
               for k, v in _LABEL_NAMES.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.8)
    fig.suptitle(f"{title} – axial slice {slice_idx}", fontsize=10, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    history: List[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss + mean Dice over epochs.

    Parameters
    ----------
    history : list of dicts loaded from HistoryLogger (each has 'epoch' key)
    """
    epochs     = [r["epoch"] for r in history]
    train_loss = [r.get("train_loss",  np.nan) for r in history]
    val_loss   = [r.get("val_loss",    np.nan) for r in history]
    train_dice = [r.get("train_dice",  np.nan) for r in history]
    val_dice   = [r.get("val_dice",    np.nan) for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss,   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Combined Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_dice, label="train")
    ax2.plot(epochs, val_dice,   label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean Dice")
    ax2.set_title("Mean Foreground Dice"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save prediction as NIfTI
# ─────────────────────────────────────────────────────────────────────────────

_INV_REMAP = {0: 0, 1: 1, 2: 2, 3: 4}  # undo the 4→3 remap from dataset.py


def save_prediction_nii(
    pred_seg: np.ndarray,
    reference_path: str,
    save_path: str,
    inv_remap: Optional[Dict[int, int]] = None,
):
    """
    Save *pred_seg* (D, H, W) as a NIfTI file, re-applying the BraTS label
    map (ET back to 4) and copying affine/header from *reference_path*.

    Parameters
    ----------
    pred_seg       : (D, H, W) integer array with remapped labels (0,1,2,3)
    reference_path : any *.nii.gz from the same subject (for affine + header)
    save_path      : output path (should end in .nii.gz)
    inv_remap      : mapping to invert label remap; defaults to {0:0,1:1,2:2,3:4}
    """
    if inv_remap is None:
        inv_remap = _INV_REMAP

    ref  = nib.load(reference_path)
    out  = np.zeros_like(pred_seg, dtype=np.int16)
    for src, dst in inv_remap.items():
        out[pred_seg == src] = dst

    nii_out = nib.Nifti1Image(out, affine=ref.affine, header=ref.header)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib.save(nii_out, save_path)
    print(f"Saved prediction → {save_path}")
