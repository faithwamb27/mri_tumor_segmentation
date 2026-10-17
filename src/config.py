"""
Configuration for MRI Brain Tumor Segmentation (BraTS-style).
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "data"           # root containing BraTS subject folders
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    # ── BraTS modalities (file suffixes inside each subject folder) ───────────
    # BraTS folder layout:
    #   data/
    #     BraTS2021_00001/
    #       BraTS2021_00001_t1.nii.gz
    #       BraTS2021_00001_t1ce.nii.gz
    #       BraTS2021_00001_t2.nii.gz
    #       BraTS2021_00001_flair.nii.gz
    #       BraTS2021_00001_seg.nii.gz
    modalities: List[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    seg_suffix: str = "seg"

    # ── BraTS label mapping ───────────────────────────────────────────────────
    # Original:  0=background, 1=NCR/NET, 2=ED, 4=ET
    # Remapped:  0=background, 1=NCR/NET, 2=ED, 3=ET  (contiguous for nn)
    num_classes: int = 4
    label_remap: dict = field(default_factory=lambda: {0: 0, 1: 1, 2: 2, 4: 3})

    # BraTS evaluation regions:
    #   WT (whole tumour)  = labels 1+2+4  → remapped 1+2+3
    #   TC (tumour core)   = labels 1+4    → remapped 1+3
    #   ET (enhancing)     = label  4      → remapped 3
    eval_regions: dict = field(default_factory=lambda: {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    })

    # ── Volume dimensions ─────────────────────────────────────────────────────
    roi_size: Tuple[int, int, int] = (128, 128, 128)   # crop/resize target
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ── Normalisation ─────────────────────────────────────────────────────────
    # z-score normalisation inside brain mask (non-zero voxels)
    normalize_mode: str = "zscore"  # "zscore" | "minmax"

    # ── Data split ────────────────────────────────────────────────────────────
    val_fraction: float = 0.15
    test_fraction: float = 0.10
    random_seed: int = 42

    # ── Augmentation (training only) ─────────────────────────────────────────
    aug_flip_prob: float = 0.5        # random flip along each axis
    aug_rotate_prob: float = 0.3      # random 90-degree rotation
    aug_scale_prob: float = 0.3       # random intensity scaling ±20 %
    aug_noise_prob: float = 0.2       # additive Gaussian noise (σ=0.1)
    aug_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_noise_std: float = 0.03

    # ── Model (3-D U-Net) ─────────────────────────────────────────────────────
    in_channels: int = 4              # one per modality
    base_filters: int = 32            # doubles each encoder level
    encoder_depths: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 1               # 3-D volumes are large
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"      # "cosine" | "step" | "poly"
    warmup_epochs: int = 10
    gradient_clip: float = 1.0

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_dice_weight: float = 0.5
    loss_ce_weight: float = 0.5
    dice_smooth: float = 1e-5

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_interval: int = 10            # batches
    save_every: int = 10              # epochs
    early_stop_patience: int = 50     # epochs with no val improvement

    # ── Inference ─────────────────────────────────────────────────────────────
    sliding_window_overlap: float = 0.5
    tta: bool = False                 # test-time augmentation (flips)

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# Default singleton
cfg = Config()
