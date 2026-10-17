"""
synthetic.py – Generate synthetic BraTS-style MRI data for offline testing.

Creates anatomically plausible fake brain volumes with tumour labels.
Useful for debugging pipelines and unit tests without any real data.

Usage
─────
  from src.synthetic import SyntheticBraTSDataset
  ds    = SyntheticBraTSDataset(n_subjects=10)
  image, label = ds[0]     # torch.Tensor (4,128,128,128), (128,128,128)

  # Or save to disk as .nii.gz and load them with the standard BraTSDataset:
  from src.synthetic import generate_subjects_to_disk
  generate_subjects_to_disk("data/synthetic", n_subjects=5)
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Low-level shape generators
# ─────────────────────────────────────────────────────────────────────────────

def _ellipsoid_mask(shape: Tuple[int, int, int],
                    centre: Tuple[float, float, float],
                    radii:  Tuple[float, float, float]) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipsoid."""
    d, h, w = shape
    z, y, x = np.ogrid[:d, :h, :w]
    return (
        ((z - centre[0]) / radii[0]) ** 2 +
        ((y - centre[1]) / radii[1]) ** 2 +
        ((x - centre[2]) / radii[2]) ** 2
    ) <= 1.0


def _smooth(vol: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur without scipy dependency (box approximation)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol.astype(np.float32), sigma=sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Subject generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_subject(
    shape: Tuple[int, int, int] = (155, 240, 240),
    seed:  Optional[int]         = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic BraTS-like subject.

    Returns
    -------
    image : (4, D, H, W)  float32  – [T1, T1ce, T2, FLAIR]
    seg   : (D, H, W)     int16    – labels 0=BG, 1=NCR, 2=ED, 4=ET (BraTS convention)
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    centre  = (D * 0.5, H * 0.5, W * 0.5)

    # ── Brain mask (large ellipsoid) ─────────────────────────────────────────
    brain = _ellipsoid_mask(shape, centre,
                            (D * 0.42, H * 0.46, W * 0.46))

    # ── Tumour regions (nested ellipsoids) ───────────────────────────────────
    tc   = rng.uniform(0.35, 0.65, 3) * np.array(shape)   # tumour core centre
    r_wt = rng.uniform(0.08, 0.14, 3) * np.array(shape)   # whole-tumour radii
    r_tc = r_wt * rng.uniform(0.5, 0.75, 3)               # tumour-core radii
    r_et = r_tc * rng.uniform(0.35, 0.6, 3)               # enhancing core

    wt_mask = _ellipsoid_mask(shape, tc, r_wt) & brain
    tc_mask = _ellipsoid_mask(shape, tc, r_tc) & brain
    et_mask = _ellipsoid_mask(shape, tc, r_et) & brain

    # Add slight Gaussian deformation to tumour masks
    def _deform(mask):
        noise = rng.normal(0, 1, shape).astype(np.float32)
        blurred = _smooth(noise, sigma=8)
        blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
        return mask & (blurred > 0.4)

    wt_mask = _deform(wt_mask)
    tc_mask = tc_mask & wt_mask
    et_mask = et_mask & tc_mask

    # ── Segmentation (BraTS original labels) ─────────────────────────────────
    seg = np.zeros(shape, dtype=np.int16)
    ed_mask      = wt_mask & ~tc_mask          # edema = WT - TC
    ncr_mask     = tc_mask & ~et_mask          # NCR/NET = TC - ET
    seg[ed_mask]  = 2    # edema
    seg[ncr_mask] = 1    # necrotic core
    seg[et_mask]  = 4    # enhancing tumour

    # ── MRI intensities ──────────────────────────────────────────────────────
    # Base tissue (white matter / grey matter mix)
    base = rng.normal(0.5, 0.1, shape).astype(np.float32)
    base[~brain] = 0.0

    # Each modality = base + tissue-specific + tumour perturbation + noise
    def _modality(bg_scale, ed_val, ncr_val, et_val, noise_std=0.05):
        vol = base * bg_scale
        vol[ed_mask]  = rng.normal(ed_val,  0.05, ed_mask.sum())
        vol[ncr_mask] = rng.normal(ncr_val, 0.05, ncr_mask.sum())
        vol[et_mask]  = rng.normal(et_val,  0.05, et_mask.sum())
        vol += rng.normal(0, noise_std, shape).astype(np.float32)
        vol = _smooth(vol, sigma=1.5)
        vol[~brain] = 0.0
        # Scale to plausible MRI range
        vol = vol * 1000
        return vol

    t1    = _modality(1.0,  0.5, 0.3, 0.6)
    t1ce  = _modality(0.9,  0.5, 0.3, 1.2)   # ET bright on T1ce
    t2    = _modality(0.8,  1.5, 0.5, 0.8)   # ED bright on T2
    flair = _modality(0.85, 1.4, 0.4, 0.9)   # ED bright on FLAIR

    image = np.stack([t1, t1ce, t2, flair], axis=0)  # (4, D, H, W)
    return image, seg


# ─────────────────────────────────────────────────────────────────────────────
# In-memory Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBraTSDataset(Dataset):
    """
    In-memory synthetic BraTS dataset.  No files needed.

    Parameters
    ----------
    n_subjects : number of synthetic subjects to generate
    shape      : (D, H, W) of each generated volume
    config     : Config (for normalisation / roi_size)
    augment    : apply training augmentations
    seed       : base random seed (each subject gets seed+idx)
    """

    def __init__(
        self,
        n_subjects: int = 20,
        shape: Tuple[int, int, int] = (155, 240, 240),
        config=None,
        augment: bool = False,
        seed: int = 0,
    ):
        from .config import cfg as _cfg
        from .dataset import remap_labels, zscore_normalize, pad_to, center_crop, Augmentor

        self.n        = n_subjects
        self.shape    = shape
        self.cfg      = config or _cfg
        self.augment  = augment
        self.seed     = seed
        self._remap   = remap_labels
        self._znorm   = zscore_normalize
        self._pad     = pad_to
        self._crop    = center_crop
        self._aug     = Augmentor(self.cfg) if augment else None

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from .dataset import pad_to, center_crop, random_crop, Augmentor

        image_np, seg_np = _generate_subject(self.shape, seed=self.seed + idx)

        # Normalise each modality
        mods = []
        for c in range(image_np.shape[0]):
            mods.append(self._znorm(image_np[c]))

        seg_np = self._remap(seg_np.astype(np.int64), self.cfg.label_remap)

        all_arrs = mods + [seg_np.astype(np.float32)]
        all_arrs = pad_to(all_arrs, self.cfg.roi_size)
        if self.augment:
            all_arrs = random_crop(all_arrs, self.cfg.roi_size)
        else:
            all_arrs = center_crop(all_arrs, self.cfg.roi_size)

        mods   = all_arrs[:-1]
        seg_np = all_arrs[-1].astype(np.int64)

        if self.augment and self._aug:
            mods, seg_np = self._aug(mods, seg_np)

        image = torch.from_numpy(np.stack(mods, 0).astype(np.float32))
        label = torch.from_numpy(seg_np)
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Save to disk (for testing standard BraTSDataset with local files)
# ─────────────────────────────────────────────────────────────────────────────

def generate_subjects_to_disk(
    output_dir: str,
    n_subjects: int = 5,
    shape: Tuple[int, int, int] = (155, 240, 240),
    seed: int = 42,
    prefix: str = "SynBraTS_",
):
    """
    Generate *n_subjects* synthetic BraTS subjects and save as .nii.gz files.

    Each subject is saved in a folder matching the BraTS naming convention so
    the standard BraTSDataset (dataset.py) can load them directly.

    Parameters
    ----------
    output_dir : root directory (e.g. "data/synthetic")
    n_subjects : how many subjects to generate
    """
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1 mm isotropic
    modality_names = ["t1", "t1ce", "t2", "flair"]

    for i in range(n_subjects):
        sid    = f"{prefix}{i:05d}"
        s_dir  = os.path.join(output_dir, sid)
        os.makedirs(s_dir, exist_ok=True)

        image_np, seg_np = _generate_subject(shape, seed=seed + i)

        for c, mod in enumerate(modality_names):
            path = os.path.join(s_dir, f"{sid}_{mod}.nii.gz")
            nib.save(nib.Nifti1Image(image_np[c], affine), path)

        seg_path = os.path.join(s_dir, f"{sid}_seg.nii.gz")
        nib.save(nib.Nifti1Image(seg_np, affine), seg_path)

        print(f"  [{i+1}/{n_subjects}] saved {sid}")

    print(f"\nGenerated {n_subjects} subjects in '{output_dir}'")
