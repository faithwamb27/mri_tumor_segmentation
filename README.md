# MRI Brain Tumour Segmentation (BraTS 3-D U-Net)

A PyTorch implementation of a **3-D Residual U-Net** for brain tumour segmentation on BraTS-style multi-modal MRI data (`.nii.gz`).

---

## Project structure

```
mri_tumor_segmentation/
├── data/                  ← place BraTS subject folders here
├── checkpoints/           ← saved model weights (auto-created)
├── outputs/               ← logs, figures, predictions (auto-created)
├── src/
│   ├── config.py          ← all hyperparameters in one dataclass
│   ├── dataset.py         ← BraTS dataset loader + preprocessing
│   ├── model.py           ← 3-D Residual U-Net
│   ├── losses.py          ← Soft Dice + Cross-Entropy combined loss
│   ├── metrics.py         ← Dice (WT/TC/ET) + HD95
│   ├── utils.py           ← seeding, checkpointing, LR schedulers
│   └── visualize.py       ← slice plots + NIfTI export
├── train.py
├── evaluate.py
├── predict.py
└── requirements.txt
```

---

## Getting data — no download required

### Option A · Stream BraTS2021 from HuggingFace ✅ Recommended

```bash
python stream_demo.py --mode hf --repo brats2021
```

20 real BraTS2021 subjects from [`rocky93/BraTS_segmentation`](https://huggingface.co/datasets/rocky93/BraTS_segmentation) (Apache-2.0).
Streamed on-demand via HTTPS, cached in `data/.hf_cache`.

### Option B · Stream BraTS2023 from HuggingFace ✅ Public

```bash
python stream_demo.py --mode hf --repo brats2023
```

10 real BraTS2023-GLI subjects from [`obi77/brats23-first-10-examples`](https://huggingface.co/datasets/obi77/brats23-first-10-examples) (public).
BraTS2023 uses modalities `t1n/t1c/t2w/t2f` — handled automatically.

### Option C · Stream DICOM from NCI Imaging Data Commons (IDC)

```bash
pip install idc-index s3fs pydicom
python stream_demo.py --mode idc --idc_collection tcga_gbm --idc_patients 10
```

IDC is the successor to TCIA. Brain tumor collections available (no auth):

| Collection | Tumour type | Patients |
|---|---|---|
| `tcga_gbm` | Glioblastoma | ~260 |
| `upenn_gbm` | GBM (multi-parametric + seg) | ~671 |
| `tcga_lgg` | Lower-grade glioma | ~199 |

Note: IDC serves DICOM (single modality). Best for exploration / visualisation.

### Option D · Synthetic data (fully offline)

```bash
python stream_demo.py --mode synthetic --n_synthetic 20 --epochs 3
```

Generates anatomically plausible fake MRI volumes + tumour masks in memory.

### Option E · Local BraTS files (traditional)

```bash
python train.py
```

Download from [Synapse BraTS2021](https://www.synapse.org/#!Synapse:syn25829067) (requires registration) and place in `data/`.

---

## Dataset format

BraTS 2021 (or 2020) with the standard folder layout:

```
data/
  BraTS2021_00001/
    BraTS2021_00001_t1.nii.gz
    BraTS2021_00001_t1ce.nii.gz
    BraTS2021_00001_t2.nii.gz
    BraTS2021_00001_flair.nii.gz
    BraTS2021_00001_seg.nii.gz
  BraTS2021_00002/
    ...
```

| Modality | Description |
|---|---|
| T1 | T1-weighted |
| T1ce | T1 with gadolinium contrast |
| T2 | T2-weighted |
| FLAIR | Fluid Attenuated Inversion Recovery |

**Labels (original → remapped)**

| Original | Remapped | Region |
|---|---|---|
| 0 | 0 | Background |
| 1 | 1 | Necrotic / Non-enhancing Tumour Core (NCR/NET) |
| 2 | 2 | Peritumoral Edema (ED) |
| 4 | 3 | Enhancing Tumour (ET) |

**Evaluation regions**

| Region | Labels |
|---|---|
| WT – Whole Tumour | 1 + 2 + 3 |
| TC – Tumour Core  | 1 + 3 |
| ET – Enhancing    | 3 |

Download links:
- BraTS 2021: <https://www.synapse.org/#!Synapse:syn25829067>
- BraTS 2020: <https://www.med.upenn.edu/cbica/brats2020/data.html>

---

## Installation

```bash
pip install -r requirements.txt
```

> `medpy` is required for HD95. If unavailable a fallback implementation is used.

---

## Quick start

### 1 · Install

```bash
pip install -r requirements.txt
```

### 2 · Train (stream real data from HuggingFace)

```bash
python stream_demo.py --mode hf
```

### 3 · Smoke-test with synthetic data (no internet)

```bash
python stream_demo.py --mode synthetic --n_synthetic 10 --epochs 3
```

### 4 · Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth --split test
```

Add `--save_nii` and `--save_figs` to export predictions:

```bash
python evaluate.py --checkpoint checkpoints/best.pth \
                   --split test --save_nii --save_figs
```

### 5 · Predict (single subject)

```bash
python predict.py --checkpoint checkpoints/best.pth \
                  --subject_dir data/.hf_cache/BraTS2021_00000 \
                  --save_nii --save_figs
```

---

## Model architecture

**3-D Residual U-Net**

```
Input (4, 128, 128, 128)
    │
    ├── Encoder Level 1 → 32  ch  ResBlock × 1  + MaxPool
    ├── Encoder Level 2 → 64  ch  ResBlock × 2  + MaxPool
    ├── Encoder Level 3 → 128 ch  ResBlock × 2  + MaxPool
    ├── Encoder Level 4 → 256 ch  ResBlock × 2  + MaxPool
    │
    ├── Bottleneck      → 512 ch  ResBlock
    │
    ├── Decoder Level 4 → 256 ch  Upsample + skip + ResBlock
    ├── Decoder Level 3 → 128 ch  Upsample + skip + ResBlock
    ├── Decoder Level 2 → 64  ch  Upsample + skip + ResBlock
    └── Decoder Level 1 → 32  ch  Upsample + skip + ResBlock
                       → 4 logits  (1×1×1 conv)
```

~19 M parameters with `base_filters=32`.

---

## Training details

| Setting | Value |
|---|---|
| Input crop | 128 × 128 × 128 |
| Batch size | 1 (GPU memory) |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| LR schedule | Cosine with linear warmup (10 epochs) |
| Loss | 0.5 × Dice + 0.5 × Cross-Entropy |
| Augmentation | Random flip, 90° rotation, intensity scale, Gaussian noise |
| Early stopping | 50 epochs patience |
| Mixed precision | `torch.cuda.amp` |

---

## Configuration

All settings live in `src/config.py` as a `Config` dataclass — edit directly or override via CLI arguments in `train.py`.

---

## Expected results (BraTS 2021)

| Region | Dice | HD95 (mm) |
|---|---|---|
| WT | ~0.90 | ~4–6 |
| TC | ~0.85 | ~5–8 |
| ET | ~0.80 | ~5–10 |

*Results vary with GPU, batch size, and training duration.*
