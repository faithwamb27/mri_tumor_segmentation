"""
hf_stream.py – Stream BraTS2021 NIfTI files directly from HuggingFace Hub.

Source: rocky93/BraTS_segmentation (Apache-2.0)
        https://huggingface.co/datasets/rocky93/BraTS_segmentation

Files are fetched on-demand via HTTPS and read into memory (no full download
required).  Optionally, files are cached in a local temp directory to avoid
re-fetching between runs.

Usage
─────
  from src.hf_stream import HuggingFaceStreamDataset, list_subjects
  subjects = list_subjects()            # discover available subject IDs
  ds = HuggingFaceStreamDataset(subjects, cache_dir="data/.hf_cache")
  image, label = ds[0]                  # streams the file on first access
"""

import io
import os
import hashlib
import tempfile
import threading
import time
import numpy as np
import nibabel as nib
import requests
import torch
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from urllib3.util.retry import Retry

from .config import Config, cfg as default_cfg
from .dataset import (
    remap_labels, zscore_normalize, minmax_normalize,
    pad_to, foreground_crop, Augmentor, nifti_to_dhw,
)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Hub constants
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Supported HuggingFace repos
# ─────────────────────────────────────────────────────────────────────────────
#
# REPO_BRATS2021 – rocky93/BraTS_segmentation (Apache-2.0)
#   Modalities : t1, t1ce, t2, flair   Seg labels: 0/1/2/4
#
# REPO_BRATS2023 – obi77/brats23-first-10-examples (public, 10 subjects)
#   Modalities : t1n, t1c, t2w, t2f    Seg labels: 0/1/2/3  (BraTS2023 GLI)
#   BraTS2023 label map: 1=NCR, 2=SNFH (edema), 3=ET  → same remapped indices
# ─────────────────────────────────────────────────────────────────────────────

REPO_BRATS2021 = "rocky93/BraTS_segmentation"
REPO_BRATS2023 = "obi77/brats23-first-10-examples"

# BraTS2023 files live inside a subfolder inside the repo
_BRATS2023_PREFIX = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

HF_BRANCH = "main"


def _hf_base(repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{HF_BRANCH}"


# Modality name maps per repo
_MODALITY_MAP = {
    REPO_BRATS2021: {"t1": "t1",   "t1ce": "t1ce", "t2": "t2",   "flair": "flair"},
    REPO_BRATS2023: {"t1": "t1n",  "t1ce": "t1c",  "t2": "t2w",  "flair": "t2f"},
}

# BraTS2023 uses labels 0/1/2/3 (no label 4) — 3=ET directly
_LABEL_REMAP_2023 = {0: 0, 1: 1, 2: 2, 3: 3}

# Hardcoded subject lists (fallback when API call is skipped)
_KNOWN_SUBJECTS_2021 = [
    "BraTS2021_00000", "BraTS2021_00002", "BraTS2021_00003",
    "BraTS2021_00005", "BraTS2021_00006", "BraTS2021_00008",
    "BraTS2021_00009", "BraTS2021_00011", "BraTS2021_00012",
    "BraTS2021_00013", "BraTS2021_00014", "BraTS2021_00015",
    "BraTS2021_00016", "BraTS2021_00017", "BraTS2021_00018",
    "BraTS2021_00019", "BraTS2021_00021", "BraTS2021_00023",
    "BraTS2021_00024", "BraTS2021_00025",
]
_KNOWN_SUBJECTS_2023 = [
    "BraTS-GLI-00000-000", "BraTS-GLI-00002-000", "BraTS-GLI-00005-000",
    "BraTS-GLI-00006-000", "BraTS-GLI-00008-000", "BraTS-GLI-00008-001",
    "BraTS-GLI-00009-000", "BraTS-GLI-00011-000", "BraTS-GLI-00012-000",
    "BraTS-GLI-00014-000",
]

# Default repo used when caller does not specify
HF_REPO = REPO_BRATS2021

_TIMEOUT   = 120   # seconds per file request
_LOCK      = threading.Lock()   # serialise cache writes
_RETRIES   = 4


def _build_session() -> requests.Session:
    retry = Retry(
        total=_RETRIES,
        connect=_RETRIES,
        read=_RETRIES,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Subject list helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_subjects(
    repo: str = REPO_BRATS2021,
    refresh: bool = False,
) -> List[str]:
    """
    Return available subject IDs from the HF repo.

    Parameters
    ----------
    repo    : one of REPO_BRATS2021 or REPO_BRATS2023
    refresh : if True, query the HF API dynamically; otherwise use hardcoded list
    """
    if not refresh:
        if repo == REPO_BRATS2023:
            return list(_KNOWN_SUBJECTS_2023)
        return list(_KNOWN_SUBJECTS_2021)

    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
        seen, subjects = set(), []
        for s in siblings:
            fname = s.get("rfilename", "")
            parts = fname.split("/")
            # BraTS2023 has an extra prefix folder
            p_idx = 1 if repo == REPO_BRATS2023 else 0
            if len(parts) > p_idx:
                sid = parts[p_idx]
                if (sid.startswith("BraTS") or sid.startswith("brats")) and sid not in seen:
                    seen.add(sid)
                    subjects.append(sid)
        return sorted(subjects) or (
            _KNOWN_SUBJECTS_2023 if repo == REPO_BRATS2023 else _KNOWN_SUBJECTS_2021
        )
    except Exception as e:
        print(f"[hf_stream] Subject refresh failed: {e}. Using hardcoded list.")
        return _KNOWN_SUBJECTS_2023 if repo == REPO_BRATS2023 else list(_KNOWN_SUBJECTS_2021)


# ─────────────────────────────────────────────────────────────────────────────
# Core streaming / caching
# ─────────────────────────────────────────────────────────────────────────────

def _hf_url(subject_id: str, suffix: str, repo: str = REPO_BRATS2021) -> str:
    base = _hf_base(repo)
    if repo == REPO_BRATS2023:
        mod_alias = _MODALITY_MAP[REPO_BRATS2023].get(suffix, suffix)
        return f"{base}/{_BRATS2023_PREFIX}/{subject_id}/{subject_id}-{mod_alias}.nii.gz"
    mod_alias = _MODALITY_MAP[REPO_BRATS2021].get(suffix, suffix)
    return f"{base}/{subject_id}/{subject_id}_{mod_alias}.nii.gz"


def _cache_path(cache_dir: str, subject_id: str, suffix: str) -> str:
    return os.path.join(cache_dir, subject_id, f"{subject_id}_{suffix}.nii.gz")


def fetch_nii(
    subject_id: str,
    suffix: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    repo: str = REPO_BRATS2021,
) -> np.ndarray:
    """
    Fetch a single NIfTI volume for *subject_id*/*suffix*.

    Strategy:
      1. If *cache_dir* is given and the file already exists locally, read it.
      2. Otherwise, stream the file from HuggingFace over HTTPS into memory
         and parse it with nibabel.
      3. If *cache_dir* is given, persist the downloaded bytes for future use.

    Parameters
    ----------
    subject_id : e.g. "BraTS2021_00000"
    suffix     : one of "t1", "t1ce", "t2", "flair", "seg"
    cache_dir  : local directory for file caching (optional)
    token      : HuggingFace token (not needed for this public repo)

    Returns
    -------
    numpy.ndarray  float32  shape (D, H, W)
    """
    # ── Check disk cache ────────────────────────────────────────────────────
    if cache_dir:
        cp = _cache_path(cache_dir, subject_id, suffix)
        if os.path.isfile(cp):
            try:
                return nifti_to_dhw(nib.load(cp).get_fdata(dtype=np.float32))
            except Exception:
                os.remove(cp)

    # ── Stream from HF ──────────────────────────────────────────────────────
    url     = _hf_url(subject_id, suffix, repo)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err = None
    session = _build_session()
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = session.get(url, headers=headers, timeout=_TIMEOUT, stream=True)
            resp.raise_for_status()
            break
        except requests.RequestException as err:
            last_err = err
            if attempt == _RETRIES:
                raise
            time.sleep(min(attempt, 3))
    else:
        raise last_err

    buffer = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=65536):
        buffer.write(chunk)
    buffer.seek(0)

    # ── Persist to cache ────────────────────────────────────────────────────
    if cache_dir:
        with _LOCK:
            cp_dir = os.path.dirname(_cache_path(cache_dir, subject_id, suffix))
            os.makedirs(cp_dir, exist_ok=True)
            cp = _cache_path(cache_dir, subject_id, suffix)
            with open(cp, "wb") as f:
                buffer.seek(0)
                f.write(buffer.read())
        return nifti_to_dhw(nib.load(cp).get_fdata(dtype=np.float32))

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        buffer.seek(0)
        tmp.write(buffer.read())
        tmp_path = tmp.name
    try:
        return nifti_to_dhw(nib.load(tmp_path).get_fdata(dtype=np.float32))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceStreamDataset(Dataset):
    """
    PyTorch Dataset that streams BraTS NIfTI files from HuggingFace Hub.

    Supports two repos:
      REPO_BRATS2021 – rocky93/BraTS_segmentation  (20 subjects, Apache-2.0)
      REPO_BRATS2023 – obi77/brats23-first-10-examples (10 subjects, public)

    Parameters
    ----------
    subjects   : list of subject IDs (use list_subjects() to discover)
    config     : Config instance
    augment    : apply training augmentations
    cache_dir  : local directory to cache downloaded files (recommended)
    token      : HuggingFace access token (not needed for these public repos)
    repo       : which HF repo to stream from (default: REPO_BRATS2021)

    Returns
    -------
    image : torch.FloatTensor  (C, D, H, W)
    label : torch.LongTensor   (D, H, W)
    """

    def __init__(
        self,
        subjects: List[str],
        config: Optional[Config] = None,
        augment: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        repo: str = REPO_BRATS2021,
    ):
        self.subjects  = subjects
        self.cfg       = config or default_cfg
        self.augment   = augment
        self.cache_dir = cache_dir
        self.token     = token
        self.repo      = repo
        self.aug       = Augmentor(self.cfg) if augment else None
        # BraTS2023 uses contiguous labels 0/1/2/3 (no label 4)
        self._lremap   = (_LABEL_REMAP_2023
                          if repo == REPO_BRATS2023
                          else self.cfg.label_remap)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.subjects[idx]

        # ── Load modalities ──────────────────────────────────────────────────
        modalities = []
        for mod in self.cfg.modalities:
            vol = fetch_nii(sid, mod, self.cache_dir, self.token, self.repo)
            if self.cfg.normalize_mode == "zscore":
                vol = zscore_normalize(vol)
            else:
                vol = minmax_normalize(vol)
            modalities.append(vol)

        # ── Load segmentation ────────────────────────────────────────────────
        seg = fetch_nii(sid, self.cfg.seg_suffix, self.cache_dir, self.token, self.repo)
        seg = remap_labels(seg.astype(np.int64), self._lremap)

        # ── Spatial pre-processing ───────────────────────────────────────────
        all_arrays = modalities + [seg.astype(np.float32)]
        all_arrays = pad_to(all_arrays, self.cfg.roi_size)
        if self.augment:
            all_arrays = foreground_crop(all_arrays, self.cfg.roi_size, jitter=True)
        else:
            all_arrays = foreground_crop(all_arrays, self.cfg.roi_size, jitter=False)

        modalities = all_arrays[:-1]
        seg        = all_arrays[-1].astype(np.int64)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augment and self.aug:
            modalities, seg = self.aug(modalities, seg)

        image = torch.from_numpy(
            np.stack(modalities, axis=0).astype(np.float32))
        label = torch.from_numpy(seg)
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Convenience split factory
# ─────────────────────────────────────────────────────────────────────────────

def get_hf_dataloaders(
    config: Optional[Config] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    refresh_subjects: bool = False,
    repo: str = REPO_BRATS2021,
):
    """
    Return (train_loader, val_loader, test_loader) using HF-streamed data.

    Parameters
    ----------
    repo      : REPO_BRATS2021 (20 subjects) or REPO_BRATS2023 (10 subjects)
    cache_dir : local cache for downloaded files (default: data/.hf_cache)
    """
    from torch.utils.data import DataLoader
    from .dataset import split_subjects

    config     = config or default_cfg
    cache_dir  = cache_dir or os.path.join(config.data_dir, ".hf_cache")
    subjects   = list_subjects(refresh=refresh_subjects)

    subjects = list_subjects(repo=repo, refresh=refresh_subjects)
    print(f"[HuggingFace] {len(subjects)} subjects from {repo}")

    train_s, val_s, test_s = split_subjects(
        subjects, config.val_fraction, config.test_fraction, config.random_seed
    )
    print(f"  split  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")
    print(f"  cache  {os.path.abspath(cache_dir)}")

    mk = lambda subs, aug: HuggingFaceStreamDataset(
        subs, config, augment=aug, cache_dir=cache_dir, token=token, repo=repo
    )
    train_ds = mk(train_s, True)
    val_ds   = mk(val_s,   False)
    test_ds  = mk(test_s,  False)

    kw = dict(batch_size=config.batch_size,
              num_workers=0,       # streaming: keep single-process to avoid fork issues
              pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
