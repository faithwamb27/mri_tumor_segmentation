"""
train.py – Training script for BraTS 3-D tumour segmentation.

Usage
─────
  python train.py                          # default config
  python train.py --epochs 200 --lr 2e-4  # override config fields
  python train.py --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config   import Config
from src.dataset  import get_dataloaders
from src.model    import UNet3D
from src.losses   import build_loss
from src.metrics  import batch_dice
from src.utils    import (
    set_seed, get_logger, get_device,
    save_checkpoint, load_checkpoint, build_checkpoint,
    MetricTracker, HistoryLogger, build_scheduler, Timer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing (thin override layer on top of Config)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("BraTS 3-D U-Net trainer")
    p.add_argument("--data_dir",        default=None)
    p.add_argument("--checkpoint_dir",  default=None)
    p.add_argument("--output_dir",      default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--base_filters",    type=int,   default=None)
    p.add_argument("--resume",          default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--num_workers",     type=int,   default=None)
    return p.parse_args()


def apply_args(cfg: Config, args) -> Config:
    if args.data_dir:       cfg.data_dir       = args.data_dir
    if args.checkpoint_dir: cfg.checkpoint_dir = args.checkpoint_dir
    if args.output_dir:     cfg.output_dir     = args.output_dir
    if args.epochs:         cfg.epochs         = args.epochs
    if args.batch_size:     cfg.batch_size     = args.batch_size
    if args.lr:             cfg.learning_rate  = args.lr
    if args.base_filters:   cfg.base_filters   = args.base_filters
    if args.seed:           cfg.random_seed    = args.seed
    if args.num_workers:    cfg.num_workers    = args.num_workers
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, cfg, logger, epoch):
    model.train()
    tracker = MetricTracker()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)   # (B,C,D,H,W)
        labels = labels.to(device, non_blocking=True)   # (B,D,H,W)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision
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
            logger.info(
                f"  [Epoch {epoch:03d} | Batch {batch_idx+1}/{len(loader)}] "
                f"loss={avg['loss']:.4f}  dice={avg['dice']:.4f}"
            )

    return tracker.average()


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

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

    log_file = os.path.join(cfg.output_dir, "train.log")
    logger   = get_logger("brats", log_file=log_file)
    history  = HistoryLogger(os.path.join(cfg.output_dir, "history.jsonl"))
    timer    = Timer()

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels    = cfg.in_channels,
        num_classes    = cfg.num_classes,
        base_filters   = cfg.base_filters,
        encoder_depths = cfg.encoder_depths,
        dropout        = cfg.dropout,
    ).to(device)
    logger.info(f"Model parameters: {UNet3D.count_parameters(model):,}")

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=cfg.learning_rate,
                                   weight_decay=cfg.weight_decay)
    criterion  = build_loss(cfg)
    scheduler  = build_scheduler(optimizer, cfg)
    scaler     = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch    = 1
    best_val_dice  = -1.0
    no_improve     = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt        = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["metrics"].get("val_dice", -1.0)
        logger.info(f"  Resumed at epoch {ckpt['epoch']}  best val Dice={best_val_dice:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training for {cfg.epochs} epochs …")
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg, logger, epoch
        )
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, cfg)
        # LR step
        scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} | "
            f"lr={lr:.2e} | elapsed={timer.elapsed()}"
        )

        epoch_metrics = {
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss":   val_metrics["loss"],
            "val_dice":   val_metrics["dice"],
            "lr":         lr,
        }
        history.log(epoch, epoch_metrics)

        # Checkpoint – periodic
        is_best = val_metrics["dice"] > best_val_dice
        if is_best:
            best_val_dice = val_metrics["dice"]
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % cfg.save_every == 0 or is_best:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            best_path = os.path.join(cfg.checkpoint_dir, "best.pth")
            save_checkpoint(
                build_checkpoint(epoch, model, optimizer, scheduler, epoch_metrics),
                ckpt_path,
                is_best=is_best,
                best_path=best_path,
            )
            if is_best:
                logger.info(f"  New best val Dice = {best_val_dice:.4f}; saved to {best_path}")

        # Save last checkpoint every epoch (cheap, overwrites)
        save_checkpoint(
            build_checkpoint(epoch, model, optimizer, scheduler, epoch_metrics),
            os.path.join(cfg.checkpoint_dir, "last.pth"),
        )

        # Early stopping
        if no_improve >= cfg.early_stop_patience:
            logger.info(
                f"No validation improvement for {cfg.early_stop_patience} epochs. "
                "Early stopping."
            )
            break

    logger.info(f"Training complete.  Best val Dice = {best_val_dice:.4f}  "
                f"Total time = {timer.elapsed()}")


if __name__ == "__main__":
    main()
