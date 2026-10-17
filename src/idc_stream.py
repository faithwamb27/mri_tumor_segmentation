"""
idc_stream.py – Stream brain tumor MRI from NCI Imaging Data Commons (IDC).

IDC is the live successor to TCIA. Data is hosted on AWS S3 / GCS with no
authentication required (public collections only).

Supported collections (all contain brain tumor MRI):
  tcga_gbm  – The Cancer Genome Atlas Glioblastoma (~260 patients)
  upenn_gbm – UPenn GBM (~671 patients, includes segmentations)
  tcga_lgg  – The Cancer Genome Atlas Lower-Grade Glioma (~199 patients)

Files are DICOM, converted on-the-fly to numpy arrays via pydicom.
Each DICOM series is assembled into a (D, H, W) 3-D volume.

Dependencies
────────────
  pip install idc-index s3fs pydicom

Usage
─────
  from src.idc_stream import IDCStreamDataset, query_brain_series
  series = query_brain_series(collection="tcga_gbm", max_patients=5)
  ds     = IDCStreamDataset(series)
  image, metadata = ds[0]   # torch.FloatTensor (1, D, H, W), dict

NOTE: IDC data is DICOM (single modality per series). To form a 4-channel
BraTS-like input, you must query multiple series (one per modality) for
each patient and align them. The convenience function
`build_multimodal_dataset()` handles this.
"""

import io
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

_MISSING = []
try:
    from idc_index import IDCClient
except ImportError:
    _MISSING.append("idc-index")

try:
    import s3fs
except ImportError:
    _MISSING.append("s3fs")

try:
    import pydicom
    from pydicom.pixels import pixel_array
except ImportError:
    _MISSING.append("pydicom")


def _check_deps():
    if _MISSING:
        raise ImportError(
            f"IDC streaming requires: {', '.join(_MISSING)}\n"
            f"Install with:  pip install {' '.join(_MISSING)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Series query helpers
# ─────────────────────────────────────────────────────────────────────────────

# Typical MRI series description keywords per BraTS modality
_MODALITY_KEYWORDS = {
    "t1":    ["t1w", "t1_se", "t1 weighted", "t1-weighted", "pre-contrast t1"],
    "t1ce":  ["t1ce", "t1 post", "post-contrast", "gd", "gadolinium", "t1+c"],
    "t2":    ["t2w", "t2 weighted", "t2-weighted"],
    "flair": ["flair", "fluid attenuated"],
}


def query_brain_series(
    collection: str = "tcga_gbm",
    max_patients: int = 10,
    modality_filter: str = "MR",
) -> List[Dict]:
    """
    Query IDC for brain MRI series in a given collection.

    Returns a list of dicts with keys:
      collection_id, PatientID, StudyInstanceUID, SeriesInstanceUID,
      SeriesDescription, series_size_MB, aws_url
    """
    _check_deps()
    client = IDCClient()

    sql = f"""
        SELECT
            collection_id,
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription,
            series_size_MB,
            CONCAT('s3://idc-open-data/', crdc_series_uuid, '/') AS aws_prefix
        FROM index
        WHERE collection_id = '{collection}'
          AND Modality = '{modality_filter}'
        LIMIT {max_patients * 10}
    """
    try:
        df = client.sql_query(sql)
    except Exception as e:
        raise RuntimeError(f"IDC SQL query failed: {e}\n"
                           "Make sure idc-index is installed and up to date.")

    # Group by patient, take first StudyUID per patient, limit patients
    seen_patients = set()
    results = []
    for _, row in df.iterrows():
        pid = row["PatientID"]
        if pid not in seen_patients:
            seen_patients.add(pid)
            if len(seen_patients) > max_patients:
                break
        results.append(row.to_dict())

    print(f"[IDC] Found {len(results)} series for {len(seen_patients)} patients "
          f"in collection '{collection}'")
    return results


def get_series_s3_urls(series_instance_uid: str) -> List[str]:
    """Return a list of S3 URLs for all DICOM instances in a series."""
    _check_deps()
    client = IDCClient()
    return client.get_series_file_URLs(
        seriesInstanceUID=series_instance_uid,
        source_bucket_location="aws",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DICOM → numpy volume
# ─────────────────────────────────────────────────────────────────────────────

def stream_series_to_volume(
    s3_urls: List[str],
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Stream all DICOM instances for a series from S3 and assemble into a
    3-D numpy volume (D, H, W), sorted by SliceLocation / InstanceNumber.

    Parameters
    ----------
    s3_urls   : list of s3:// URLs returned by get_series_s3_urls()
    cache_dir : optional local cache directory

    Returns
    -------
    volume   : (D, H, W) float32 numpy array
    metadata : dict with spacing, origin, series description, etc.
    """
    _check_deps()
    fs = s3fs.S3FileSystem(anon=True, default_block_size=1 << 20)

    slices = []
    for url in s3_urls:
        # Strip leading s3://
        s3_path = url.replace("s3://", "", 1)

        if cache_dir:
            local = os.path.join(cache_dir, "dicom", os.path.basename(s3_path))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if not os.path.isfile(local):
                with fs.open(s3_path, "rb") as f, open(local, "wb") as out:
                    out.write(f.read())
            dcm = pydicom.dcmread(local, force=True)
        else:
            with fs.open(s3_path, "rb") as f:
                dcm = pydicom.dcmread(io.BytesIO(f.read()), force=True)

        slices.append(dcm)

    # Sort slices
    def _sort_key(d):
        if hasattr(d, "SliceLocation"):
            return float(d.SliceLocation)
        if hasattr(d, "InstanceNumber"):
            return int(d.InstanceNumber)
        return 0

    slices.sort(key=_sort_key)

    # Stack pixel arrays
    arrays = []
    for dcm in slices:
        arr = dcm.pixel_array.astype(np.float32)
        # Apply rescale slope/intercept if present (Hounsfield / signal units)
        slope     = float(getattr(dcm, "RescaleSlope",     1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        arrays.append(arr)

    volume = np.stack(arrays, axis=0)   # (D, H, W)

    # Extract basic metadata
    ref = slices[0]
    pixel_spacing = getattr(ref, "PixelSpacing", [1.0, 1.0])
    slice_thick   = float(getattr(ref, "SliceThickness", 1.0))
    metadata = {
        "SeriesDescription": str(getattr(ref, "SeriesDescription", "")),
        "PatientID":         str(getattr(ref, "PatientID", "")),
        "StudyInstanceUID":  str(getattr(ref, "StudyInstanceUID", "")),
        "SeriesInstanceUID": str(getattr(ref, "SeriesInstanceUID", "")),
        "voxel_spacing":     (slice_thick, float(pixel_spacing[0]),
                              float(pixel_spacing[1])),
        "n_slices":          len(slices),
        "shape":             volume.shape,
    }
    return volume, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Single-modality dataset (1 DICOM series → 1 3-D volume)
# ─────────────────────────────────────────────────────────────────────────────

class IDCStreamDataset(Dataset):
    """
    Streams individual DICOM series from IDC S3.
    Returns one single-modality 3-D volume per item.

    Suitable for exploration / visualisation. For multi-modal training,
    use build_multimodal_dataset() instead.
    """

    def __init__(
        self,
        series_list: List[Dict],
        config=None,
        cache_dir: Optional[str] = None,
    ):
        from .config import cfg as _cfg
        from .dataset import zscore_normalize, pad_to, center_crop

        self.series    = series_list
        self.cfg       = config or _cfg
        self.cache_dir = cache_dir
        self._znorm    = zscore_normalize
        self._pad      = pad_to
        self._crop     = center_crop

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        row = self.series[idx]
        uid = row["SeriesInstanceUID"]

        urls = get_series_s3_urls(uid)
        volume, meta = stream_series_to_volume(urls, self.cache_dir)

        # Normalise
        volume = self._znorm(volume)

        # Resize to roi_size
        from .dataset import pad_to, center_crop
        arrs = pad_to([volume], self.cfg.roi_size)
        arrs = center_crop(arrs, self.cfg.roi_size)
        volume = arrs[0]

        tensor = torch.from_numpy(volume[np.newaxis].astype(np.float32))  # (1,D,H,W)
        return tensor, meta
