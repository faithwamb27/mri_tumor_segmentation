"""
BraTS-style dataset loader.

Expected folder layout
─────────────────────
data/
  BraTS2021_00001/
    BraTS2021_00001_t1.nii.gz
    BraTS2021_00001_t1ce.nii.gz
    BraTS2021_00001_t2.nii.gz
    BraTS2021_00001_flair.nii.gz
    BraTS2021_00001_seg.nii.gz
  BraTS2021_00002/
    ...
"""

import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from typing import Dict, List, Optional, Tuple

from .config import Config, cfg as default_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def nifti_to_dhw(volume: np.ndarray) -> np.ndarray:
    """Convert nibabel's native (H, W, D) style array to (D, H, W)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D NIfTI volume, got shape {volume.shape}")
    return np.transpose(volume, (2, 0, 1))


def load_nii(path: str) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array (D, H, W)."""
    img = nib.load(path)
    return nifti_to_dhw(img.get_fdata(dtype=np.float32))


def remap_labels(seg: np.ndarray, remap: Dict[int, int]) -> np.ndarray:
    """Map original BraTS label values to contiguous indices."""
    out = np.zeros_like(seg, dtype=np.int64)
    for src, dst in remap.items():
        out[seg == src] = dst
    return out


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise within the non-zero (brain) mask."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std  = volume[mask].std() + 1e-8
    volume = (volume - mean) / std
    volume[~mask] = 0.0
    return volume


def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] within the brain mask."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    vmin, vmax = volume[mask].min(), volume[mask].max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)
    volume[~mask] = 0.0
    return volume


def resize_volume(volume: np.ndarray,
                  target: Tuple[int, int, int],
                  order: int = 1) -> np.ndarray:
    """Resize a 3-D volume to *target* shape using spline interpolation."""
    factors = [t / s for t, s in zip(target, volume.shape)]
    return zoom(volume, factors, order=order)


def random_crop(arrays: List[np.ndarray],
                size: Tuple[int, int, int]) -> List[np.ndarray]:
    """Random crop all arrays to *size* using the same offset."""
    d, h, w = arrays[0].shape
    sd, sh, sw = size
    z = random.randint(0, max(d - sd, 0))
    y = random.randint(0, max(h - sh, 0))
    x = random.randint(0, max(w - sw, 0))
    return [a[z:z+sd, y:y+sh, x:x+sw] for a in arrays]


def center_crop(arrays: List[np.ndarray],
                size: Tuple[int, int, int]) -> List[np.ndarray]:
    """Centre-crop all arrays to *size*."""
    d, h, w = arrays[0].shape
    sd, sh, sw = size
    z = max((d - sd) // 2, 0)
    y = max((h - sh) // 2, 0)
    x = max((w - sw) // 2, 0)
    return [a[z:z+sd, y:y+sh, x:x+sw] for a in arrays]


def foreground_crop(arrays: List[np.ndarray],
                    size: Tuple[int, int, int],
                    jitter: bool = False) -> List[np.ndarray]:
    """Crop around the foreground region stored in the last array."""
    seg = arrays[-1]
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        return random_crop(arrays, size) if jitter else center_crop(arrays, size)

    d, h, w = seg.shape
    sd, sh, sw = size
    center = coords.mean(axis=0)
    starts = []
    for axis_len, crop_len, center_coord in zip((d, h, w), (sd, sh, sw), center):
        start = int(round(center_coord - crop_len / 2))
        if jitter and axis_len > crop_len:
            max_jitter = max(1, crop_len // 8)
            start += random.randint(-max_jitter, max_jitter)
        start = min(max(start, 0), max(axis_len - crop_len, 0))
        starts.append(start)

    z, y, x = starts
    return [a[z:z+sd, y:y+sh, x:x+sw] for a in arrays]


def pad_to(arrays: List[np.ndarray],
           size: Tuple[int, int, int]) -> List[np.ndarray]:
    """Zero-pad all arrays to at least *size* symmetrically."""
    d, h, w = arrays[0].shape
    sd, sh, sw = size
    pd = max(sd - d, 0)
    ph = max(sh - h, 0)
    pw = max(sw - w, 0)
    if pd == 0 and ph == 0 and pw == 0:
        return arrays
    pad_width = [(pd//2, pd - pd//2),
                 (ph//2, ph - ph//2),
                 (pw//2, pw - pw//2)]
    return [np.pad(a, pad_width, mode="constant", constant_values=0)
            for a in arrays]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class Augmentor:
    """Stochastic augmentation applied to (modalities_list, seg) arrays."""

    def __init__(self, config: Config):
        self.cfg = config

    def __call__(
        self,
        modalities: List[np.ndarray],   # each (D,H,W)
        seg: np.ndarray,                 # (D,H,W) int
    ) -> Tuple[List[np.ndarray], np.ndarray]:

        # Random axis flip
        for axis in range(3):
            if random.random() < self.cfg.aug_flip_prob:
                modalities = [np.flip(m, axis=axis).copy() for m in modalities]
                seg        = np.flip(seg, axis=axis).copy()

        # Random 90° rotation in axial plane
        if random.random() < self.cfg.aug_rotate_prob:
            k = random.randint(1, 3)
            modalities = [np.rot90(m, k=k, axes=(1, 2)).copy() for m in modalities]
            seg        = np.rot90(seg, k=k, axes=(1, 2)).copy()

        # Intensity scale per modality
        if random.random() < self.cfg.aug_scale_prob:
            for i in range(len(modalities)):
                scale = random.uniform(*self.cfg.aug_scale_range)
                modalities[i] = modalities[i] * scale

        # Additive Gaussian noise per modality
        if random.random() < self.cfg.aug_noise_prob:
            for i in range(len(modalities)):
                noise = np.random.normal(
                    0.0, self.cfg.aug_noise_std, modalities[i].shape
                ).astype(np.float32)
                modalities[i] = modalities[i] + noise

        return modalities, seg


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    PyTorch dataset for BraTS-style multi-modal MRI segmentation.

    Returns
    -------
    image : torch.Tensor  shape (C, D, H, W)  float32
    label : torch.Tensor  shape (D, H, W)      int64
    """

    def __init__(
        self,
        subject_dirs: List[str],
        config: Optional[Config] = None,
        augment: bool = False,
    ):
        self.subjects = subject_dirs
        self.cfg      = config or default_cfg
        self.augment  = augment
        self.aug      = Augmentor(self.cfg) if augment else None

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_dir = self.subjects[idx]
        subject_id  = os.path.basename(subject_dir)

        # ── Load modalities ──────────────────────────────────────────────────
        modalities: List[np.ndarray] = []
        for mod in self.cfg.modalities:
            path = os.path.join(subject_dir, f"{subject_id}_{mod}.nii.gz")
            vol  = load_nii(path)
            # Normalise
            if self.cfg.normalize_mode == "zscore":
                vol = zscore_normalize(vol)
            elif self.cfg.normalize_mode == "minmax":
                vol = minmax_normalize(vol)
            modalities.append(vol)

        # ── Load segmentation ────────────────────────────────────────────────
        seg_path = os.path.join(subject_dir,
                                f"{subject_id}_{self.cfg.seg_suffix}.nii.gz")
        seg = load_nii(seg_path).astype(np.int64)
        seg = remap_labels(seg, self.cfg.label_remap)

        # ── Spatial pre-processing ───────────────────────────────────────────
        # Pad if smaller than roi_size, then crop
        all_arrays = modalities + [seg.astype(np.float32)]
        all_arrays = pad_to(all_arrays, self.cfg.roi_size)

        if self.augment:
            all_arrays = foreground_crop(all_arrays, self.cfg.roi_size, jitter=True)
        else:
            all_arrays = foreground_crop(all_arrays, self.cfg.roi_size, jitter=False)

        modalities = all_arrays[:-1]
        seg        = all_arrays[-1].astype(np.int64)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augment and self.aug is not None:
            modalities, seg = self.aug(modalities, seg)

        # ── To tensor ────────────────────────────────────────────────────────
        image = torch.from_numpy(
            np.stack(modalities, axis=0).astype(np.float32)  # (C,D,H,W)
        )
        label = torch.from_numpy(seg)                         # (D,H,W)

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Utilities: scan data directory and split subjects
# ─────────────────────────────────────────────────────────────────────────────

def discover_subjects(data_dir: str) -> List[str]:
    """Return sorted list of subject directories that contain all 5 NIfTI files."""
    subjects = []
    for entry in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(subject_dir):
            continue
        # Quick check: at least one .nii.gz file present
        nii_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
        if nii_files:
            subjects.append(subject_dir)
    return subjects


def split_subjects(
    subjects: List[str],
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Randomly split subjects into train / val / test."""
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    test    = shuffled[:n_test]
    val     = shuffled[n_test:n_test + n_val]
    train   = shuffled[n_test + n_val:]
    return train, val, test


def get_dataloaders(
    config: Optional[Config] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function — returns (train_loader, val_loader, test_loader)."""
    config = config or default_cfg
    subjects = discover_subjects(config.data_dir)
    if not subjects:
        raise RuntimeError(
            f"No BraTS subjects found in '{config.data_dir}'.\n"
            "Check that each subject folder contains *_t1.nii.gz, "
            "*_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz."
        )

    train_subs, val_subs, test_subs = split_subjects(
        subjects, config.val_fraction, config.test_fraction, config.random_seed
    )
    print(f"Subjects  train={len(train_subs)}  val={len(val_subs)}  test={len(test_subs)}")

    train_ds = BraTSDataset(train_subs, config, augment=True)
    val_ds   = BraTSDataset(val_subs,   config, augment=False)
    test_ds  = BraTSDataset(test_subs,  config, augment=False)

    loader_kwargs = dict(
        batch_size  = config.batch_size,
        num_workers = config.num_workers,
        pin_memory  = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
