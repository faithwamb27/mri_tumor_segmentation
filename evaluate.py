"""
evaluate.py – Full BraTS evaluation on the test set.

Computes per-subject and mean Dice (WT / TC / ET) and HD95.

Usage
─────
  python evaluate.py --checkpoint checkpoints/best.pth
  python evaluate.py --checkpoint checkpoints/best.pth --split val
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config   import Config
from src.dataset  import discover_subjects, split_subjects, BraTSDataset
from src.model    import UNet3D
from src.metrics  import compute_subject_metrics, aggregate_metrics
from src.utils    import get_logger, get_device, load_checkpoint, set_seed
from src.visualize import plot_subject, save_prediction_nii


def parse_args():
    p = argparse.ArgumentParser("BraTS Evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   default=None)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--save_nii",   action="store_true",
                   help="Save prediction .nii.gz files")
    p.add_argument("--save_figs",  action="store_true",
                   help="Save visualisation PNGs")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_hd95",    action="store_true",
                   help="Skip HD95 (faster, no medpy required)")
    return p.parse_args()


@torch.no_grad()
def predict_volume(model, image_tensor, device):
    """
    Run model on a single (1, C, D, H, W) tensor, return (D, H, W) int64 array.
    """
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)          # (1,C,D,H,W)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = model(image_tensor)                              # (1,cls,D,H,W)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()          # (D,H,W)
    return pred.astype(np.int64)


def main():
    args   = parse_args()
    cfg    = Config()
    if args.data_dir:   cfg.data_dir   = args.data_dir
    if args.output_dir: cfg.output_dir = args.output_dir

    set_seed(cfg.random_seed)
    device = get_device()
    logger = get_logger("brats_eval")

    # ── Data ──────────────────────────────────────────────────────────────────
    subjects = discover_subjects(cfg.data_dir)
    train_subs, val_subs, test_subs = split_subjects(
        subjects, cfg.val_fraction, cfg.test_fraction, cfg.random_seed
    )
    split_map = {"train": train_subs, "val": val_subs, "test": test_subs}
    eval_subs = split_map[args.split]
    logger.info(f"Evaluating on {args.split} split: {len(eval_subs)} subjects")

    dataset = BraTSDataset(eval_subs, cfg, augment=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels    = cfg.in_channels,
        num_classes    = cfg.num_classes,
        base_filters   = cfg.base_filters,
        encoder_depths = cfg.encoder_depths,
        dropout        = 0.0,                 # disable at inference
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    per_subject_results = []
    os.makedirs(cfg.output_dir, exist_ok=True)

    for idx, subject_dir in enumerate(eval_subs):
        subject_id = os.path.basename(subject_dir)
        image_tensor, label_tensor = dataset[idx]

        pred_seg = predict_volume(model, image_tensor, device)
        gt_seg   = label_tensor.numpy().astype(np.int64)

        metrics = compute_subject_metrics(
            pred_seg, gt_seg,
            regions       = cfg.eval_regions,
            voxel_spacing = cfg.voxel_spacing,
            compute_hd95  = not args.no_hd95,
        )
        per_subject_results.append(metrics)

        # Per-subject log
        row = f"{subject_id}  " + "  ".join(
            f"{r}: Dice={metrics[r]['dice']:.3f} HD95={metrics[r]['hd95']:.1f}"
            for r in metrics
        )
        logger.info(row)

        # Optional NIfTI save
        if args.save_nii:
            import glob as _glob
            ref = sorted(_glob.glob(
                os.path.join(subject_dir, "*.nii.gz")))[0]
            nii_path = os.path.join(cfg.output_dir, f"{subject_id}_pred.nii.gz")
            save_prediction_nii(pred_seg, ref, nii_path)

        # Optional figure save
        if args.save_figs:
            fig_path = os.path.join(cfg.output_dir, f"{subject_id}.png")
            plot_subject(
                image_tensor.numpy(),
                gt_seg   = gt_seg,
                pred_seg = pred_seg,
                modality_names = cfg.modalities,
                save_path      = fig_path,
                title          = subject_id,
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate_metrics(per_subject_results)
    logger.info("\n──── Aggregate Results ────")
    for region, vals in agg.items():
        logger.info(
            f"  {region:2s}  Dice = {vals['dice_mean']:.4f} ± {vals['dice_std']:.4f} | "
            f"HD95 = {vals['hd95_mean']:.2f} ± {vals['hd95_std']:.2f}"
        )

    # Save JSON
    results_path = os.path.join(cfg.output_dir, f"eval_{args.split}.json")
    with open(results_path, "w") as f:
        json.dump({"aggregate": agg, "per_subject": per_subject_results}, f, indent=2)
    logger.info(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
