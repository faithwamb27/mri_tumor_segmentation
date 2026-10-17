"""
predict.py – Run inference on a single BraTS subject or a directory of subjects.

Usage
─────
  # Single subject
  python predict.py --checkpoint checkpoints/best.pth \
                    --subject_dir data/BraTS2021_00001

  # Batch (all subjects in a directory)
  python predict.py --checkpoint checkpoints/best.pth \
                    --data_dir data/new_subjects \
                    --save_nii --save_figs

Outputs (per subject, inside --output_dir):
  <subject_id>_pred.nii.gz   – prediction in BraTS label space (0,1,2,4)
  <subject_id>.png            – visualisation figure
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(__file__))

from src.config    import Config
from src.dataset   import (load_nii, remap_labels, zscore_normalize,
                            minmax_normalize, pad_to, center_crop,
                            discover_subjects)
from src.model     import UNet3D
from src.utils     import get_device, load_checkpoint, get_logger
from src.visualize import plot_subject, save_prediction_nii


def parse_args():
    p = argparse.ArgumentParser("BraTS Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--subject_dir", default=None,
                   help="Path to a single subject folder")
    p.add_argument("--data_dir",    default=None,
                   help="Root of a directory containing multiple subject folders")
    p.add_argument("--output_dir",  default="outputs/predictions")
    p.add_argument("--save_nii",    action="store_true")
    p.add_argument("--save_figs",   action="store_true")
    p.add_argument("--tta",         action="store_true",
                   help="Test-time augmentation (flip ensemble)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (mirror of dataset.py without augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_subject(subject_dir: str, cfg: Config):
    """
    Load and preprocess a subject. Returns:
      image_tensor : torch.FloatTensor (1, C, D, H, W)
      orig_shape   : (D, H, W)  before crop/pad – for potential resampling back
      t1_path      : path to T1 file (for NIfTI reference affine)
    """
    subject_id = os.path.basename(subject_dir)
    modalities_np = []
    for mod in cfg.modalities:
        path = os.path.join(subject_dir, f"{subject_id}_{mod}.nii.gz")
        vol  = load_nii(path)
        if cfg.normalize_mode == "zscore":
            vol = zscore_normalize(vol)
        else:
            vol = minmax_normalize(vol)
        modalities_np.append(vol)

    orig_shape = modalities_np[0].shape
    all_arrays = pad_to(modalities_np, cfg.roi_size)
    all_arrays = center_crop(all_arrays, cfg.roi_size)

    image = np.stack(all_arrays, axis=0).astype(np.float32)   # (C,D,H,W)
    image_tensor = torch.from_numpy(image).unsqueeze(0)        # (1,C,D,H,W)

    t1_path = os.path.join(subject_dir,
                            f"{subject_id}_{cfg.modalities[0]}.nii.gz")
    return image_tensor, orig_shape, t1_path


# ─────────────────────────────────────────────────────────────────────────────
# Inference (with optional TTA)
# ─────────────────────────────────────────────────────────────────────────────

FLIP_AXES = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device,
            tta: bool = False) -> np.ndarray:
    """Return (D, H, W) int64 prediction."""
    model.eval()
    x = image_tensor.to(device)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = torch.softmax(model(x), dim=1)

        if tta:
            for axes in FLIP_AXES:
                flipped  = torch.flip(x, dims=list(axes))
                out_flip = torch.softmax(model(flipped), dim=1)
                logits  += torch.flip(out_flip, dims=list(axes))
            logits /= (1 + len(FLIP_AXES))

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = Config()
    device = get_device()
    logger = get_logger("brats_predict")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels    = cfg.in_channels,
        num_classes    = cfg.num_classes,
        base_filters   = cfg.base_filters,
        encoder_depths = cfg.encoder_depths,
        dropout        = 0.0,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Collect subjects ──────────────────────────────────────────────────────
    if args.subject_dir:
        subject_dirs = [args.subject_dir]
    elif args.data_dir:
        subject_dirs = discover_subjects(args.data_dir)
    else:
        raise ValueError("Provide either --subject_dir or --data_dir")

    logger.info(f"Running inference on {len(subject_dirs)} subject(s) …")

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        logger.info(f"  Processing {subject_id} …")

        try:
            image_tensor, orig_shape, t1_path = preprocess_subject(subject_dir, cfg)
        except FileNotFoundError as e:
            logger.warning(f"  Skipped – {e}")
            continue

        pred_seg = predict(model, image_tensor, device, tta=args.tta)

        if args.save_nii:
            nii_path = os.path.join(args.output_dir, f"{subject_id}_pred.nii.gz")
            save_prediction_nii(pred_seg, t1_path, nii_path)

        if args.save_figs:
            # Load GT seg if available (for side-by-side comparison)
            seg_path = os.path.join(subject_dir,
                                    f"{subject_id}_{cfg.seg_suffix}.nii.gz")
            gt_seg = None
            if os.path.isfile(seg_path):
                from src.dataset import load_nii as _load
                gt_raw = _load(seg_path).astype(np.int64)
                gt_raw = remap_labels(gt_raw, cfg.label_remap)
                # Pad + crop to match roi_size
                all_gt = pad_to([gt_raw.astype(np.float32)], cfg.roi_size)
                all_gt = center_crop(all_gt, cfg.roi_size)
                gt_seg = all_gt[0].astype(np.int64)

            fig_path = os.path.join(args.output_dir, f"{subject_id}.png")
            plot_subject(
                image_tensor.squeeze(0).numpy(),
                gt_seg         = gt_seg,
                pred_seg       = pred_seg,
                modality_names = cfg.modalities,
                save_path      = fig_path,
                title          = subject_id,
            )
            logger.info(f"  Figure → {fig_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
