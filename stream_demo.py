"""
stream_demo.py – End-to-end demo: stream BraTS2021 from HuggingFace and train.

This script replaces train.py when you want to use remote data.
It supports three data modes:

  --mode hf          Stream real BraTS NIfTI files from HuggingFace
                     Two repos available (no sign-in needed):
                       --repo brats2021  rocky93/BraTS_segmentation  (20 subjects, Apache-2.0)
                       --repo brats2023  obi77/brats23-first-10-examples (10 subjects, public)
                     Files are cached in data/.hf_cache on first access.

  --mode synthetic   Generate synthetic BraTS-style data in memory.
                     Useful for offline testing and CI.

  --mode local       Use downloaded .nii.gz files in data/ (original behaviour).

  --mode idc         Stream DICOM brain tumor MRI from NCI Imaging Data Commons
                     (AWS S3, no auth). Requires: pip install idc-index s3fs pydicom

Usage
─────
  # Stream from HuggingFace (recommended)
  python stream_demo.py --mode hf

  # Quick smoke-test with 10 synthetic subjects, 3 epochs
  python stream_demo.py --mode synthetic --n_synthetic 10 --epochs 3

  # Resume from a checkpoint using HF data
  python stream_demo.py --mode hf --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from src.config   import Config
from src.model    import UNet3D
from src.losses   import build_loss
from src.metrics  import batch_dice
from src.utils    import (
    set_seed, get_logger, get_device,
    save_checkpoint, load_checkpoint, build_checkpoint,
    MetricTracker, HistoryLogger, build_scheduler, Timer,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("BraTS streaming demo")
    p.add_argument("--mode",        default="hf",
                   choices=["hf", "synthetic", "local", "idc"],
                   help="Data source")
    # HuggingFace options
    p.add_argument("--repo",        default="brats2021",
                   choices=["brats2021", "brats2023"],
                   help="Which HF BraTS repo to use (--mode hf only)")
    p.add_argument("--hf_cache",    default="data/.hf_cache",
                   help="Local cache dir for HF-downloaded files")
    p.add_argument("--hf_token",    default=None,
                   help="HuggingFace token (not needed for these public repos)")
    p.add_argument("--no_refresh",  action="store_true",
                   help="Use hardcoded subject list (faster startup)")
    # IDC options
    p.add_argument("--idc_collection", default="tcga_gbm",
                   choices=["tcga_gbm", "upenn_gbm", "tcga_lgg"],
                   help="IDC brain tumor collection (--mode idc only)")
    p.add_argument("--idc_patients",   type=int, default=10,
                   help="Max patients to stream from IDC")
    p.add_argument("--idc_cache",      default="data/.idc_cache")
    # Synthetic options
    p.add_argument("--n_synthetic", type=int, default=20,
                   help="Number of synthetic subjects")
    # Training options
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--base_filters",type=int,   default=None)
    p.add_argument("--resume",      default=None)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--data_dir",    default=None)
    return p.parse_args()


def apply_args(cfg: Config, args) -> Config:
    if args.epochs:      cfg.epochs        = args.epochs
    if args.batch_size:  cfg.batch_size    = args.batch_size
    if args.lr:          cfg.learning_rate = args.lr
    if args.base_filters:cfg.base_filters  = args.base_filters
    if args.seed:        cfg.random_seed   = args.seed
    if args.data_dir:    cfg.data_dir      = args.data_dir
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders per mode
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(args, cfg):
    if args.mode == "hf":
        from src.hf_stream import get_hf_dataloaders, REPO_BRATS2021, REPO_BRATS2023
        repo = REPO_BRATS2023 if args.repo == "brats2023" else REPO_BRATS2021
        return get_hf_dataloaders(
            config           = cfg,
            cache_dir        = args.hf_cache,
            token            = args.hf_token,
            refresh_subjects = not args.no_refresh,
            repo             = repo,
        )

    elif args.mode == "idc":
        from src.idc_stream import query_brain_series, IDCStreamDataset
        from src.dataset    import split_subjects
        series = query_brain_series(
            collection   = args.idc_collection,
            max_patients = args.idc_patients,
        )
        train_s, val_s, test_s = split_subjects(
            series, cfg.val_fraction, cfg.test_fraction, cfg.random_seed
        )
        kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
        mk = lambda s: IDCStreamDataset(s, cfg, cache_dir=args.idc_cache)
        print(f"[IDC] Note: single-modality (no labels); use --mode hf for full BraTS pipeline.")
        return (
            DataLoader(mk(train_s), shuffle=True,  **kw),
            DataLoader(mk(val_s),   shuffle=False, **kw),
            DataLoader(mk(test_s),  shuffle=False, **kw),
        )

    elif args.mode == "synthetic":
        from src.synthetic import SyntheticBraTSDataset
        from src.dataset   import split_subjects
        import random

        n = args.n_synthetic
        rng = random.Random(cfg.random_seed)
        idxs = list(range(n)); rng.shuffle(idxs)
        n_test = max(1, int(n * cfg.test_fraction))
        n_val  = max(1, int(n * cfg.val_fraction))
        test_i = idxs[:n_test]
        val_i  = idxs[n_test:n_test + n_val]
        tr_i   = idxs[n_test + n_val:]

        def _ds(indices, aug):
            ds = SyntheticBraTSDataset(
                n_subjects = len(indices), config=cfg, augment=aug,
                seed=cfg.random_seed + indices[0] if indices else 0,
            )
            return ds

        kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
        return (
            DataLoader(_ds(tr_i,   True),  shuffle=True,  **kw),
            DataLoader(_ds(val_i,  False), shuffle=False, **kw),
            DataLoader(_ds(test_i, False), shuffle=False, **kw),
        )

    elif args.mode == "local":
        from src.dataset import get_dataloaders
        return get_dataloaders(cfg)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, cfg, logger, epoch):
    model.train()
    tracker = MetricTracker()
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        if cfg.gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        dice = batch_dice(logits, labels, cfg.num_classes)
        tracker.update(loss=loss.item(), dice=dice)
        if (batch_idx + 1) % cfg.log_interval == 0:
            avg = tracker.average()
            logger.info(f"  [E{epoch:03d} | B{batch_idx+1}/{len(loader)}] "
                        f"loss={avg['loss']:.4f}  dice={avg['dice']:.4f}")
    return tracker.average()


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    tracker = MetricTracker()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)
        dice = batch_dice(logits, labels, cfg.num_classes)
        tracker.update(loss=loss.item(), dice=dice)
    return tracker.average()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = apply_args(Config(), args)
    set_seed(cfg.random_seed)
    device = get_device()

    log_file = os.path.join(cfg.output_dir, f"stream_{args.mode}.log")
    logger   = get_logger("brats_stream", log_file=log_file)
    history  = HistoryLogger(os.path.join(cfg.output_dir, "history.jsonl"))
    timer    = Timer()

    logger.info(f"Data mode: {args.mode.upper()}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_loaders(args, cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels    = cfg.in_channels,
        num_classes    = cfg.num_classes,
        base_filters   = cfg.base_filters,
        encoder_depths = cfg.encoder_depths,
        dropout        = cfg.dropout,
    ).to(device)
    logger.info(f"Parameters: {UNet3D.count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    criterion = build_loss(cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch   = 1
    best_val_dice = -1.0
    no_improve    = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt          = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt["metrics"].get("val_dice", -1.0)

    logger.info(f"Training for {cfg.epochs} epochs …")
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             scaler, device, cfg, logger, epoch)
        vl = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.4f} | "
            f"val loss={vl['loss']:.4f} dice={vl['dice']:.4f} | "
            f"lr={lr:.2e} | {timer.elapsed()}"
        )
        history.log(epoch, {"train_loss": tr["loss"], "train_dice": tr["dice"],
                             "val_loss": vl["loss"],   "val_dice":  vl["dice"], "lr": lr})

        is_best = vl["dice"] > best_val_dice
        if is_best:
            best_val_dice = vl["dice"]; no_improve = 0
        else:
            no_improve += 1

        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        best_path = os.path.join(cfg.checkpoint_dir, "best.pth")
        m = {"val_dice": vl["dice"]}
        if epoch % cfg.save_every == 0 or is_best:
            save_checkpoint(build_checkpoint(epoch, model, optimizer, scheduler, m),
                            ckpt_path, is_best=is_best, best_path=best_path)
            if is_best:
                logger.info(f"  best val Dice = {best_val_dice:.4f}")

        save_checkpoint(build_checkpoint(epoch, model, optimizer, scheduler, m),
                        os.path.join(cfg.checkpoint_dir, "last.pth"))

        if no_improve >= cfg.early_stop_patience:
            logger.info("Early stopping."); break

    logger.info(f"Done. Best val Dice = {best_val_dice:.4f}  {timer.elapsed()}")


if __name__ == "__main__":
    main()
