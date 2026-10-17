"""
Utility helpers: seeding, checkpointing, logging, LR schedulers.
"""

import os
import random
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "brats", log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger                   # already configured
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
):
    """Save training state; optionally copy to *best_path* if best so far."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best and best_path:
        import shutil
        shutil.copyfile(path, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer / scheduler).

    Returns the full checkpoint dict so the caller can read epoch, metrics, etc.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer  and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler  and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    metrics: dict,
) -> dict:
    return {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics":         metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """Accumulate scalar metrics over a loop and compute averages."""

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            v = float(v)
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def average(self) -> dict:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class HistoryLogger:
    """Write per-epoch metrics to a JSON-lines file."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def load(path: str):
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        return records


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedulers
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, config, steps_per_epoch: int = 1):
    """
    Build an LR scheduler from the config.

    Supported: 'cosine', 'step', 'poly'
    Also injects a linear warmup for the first *config.warmup_epochs* epochs.
    """
    total = max(int(config.epochs), 1)
    warmup = min(max(int(config.warmup_epochs), 0), max(total - 1, 0))

    if config.lr_scheduler == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total - warmup, eta_min=1e-7
        )
    elif config.lr_scheduler == "step":
        base = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total // 3), gamma=0.1
        )
    elif config.lr_scheduler == "poly":
        base = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: (1 - (e - warmup) / max(1, total - warmup)) ** 0.9
            if e >= warmup else 1.0,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")

    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, base], milestones=[warmup]
        )
    else:
        scheduler = base

    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> str:
        s = int(time.time() - self._start)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
