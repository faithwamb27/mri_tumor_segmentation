# MRI Brain Tumour Segmentation: Research Report

## Abstract
This project implements a 3-D Residual U-Net pipeline for BraTS-style brain tumour segmentation from multi-modal MRI. The codebase supports local BraTS folders, HuggingFace streaming, IDC DICOM streaming, and synthetic data for offline testing. A CPU smoke test using synthetic data completed successfully, confirming the training pipeline can start, train for one epoch, and write checkpoints/logs. The project was also packaged into a new GitHub repository for sharing and collaboration.

## 1. Introduction
Brain tumour segmentation is a core task in medical image analysis because accurate delineation of tumour subregions supports diagnosis, treatment planning, and response monitoring. BraTS datasets provide multi-modal MRI volumes with voxel-level annotations for background, necrotic core, edema, and enhancing tumour. This project targets that problem using a 3-D convolutional neural network designed to learn spatial context directly from volumetric MRI.

## 2. Objectives
The project aims to:

1. Train a 3-D segmentation model for BraTS-style MRI data.
2. Support multiple data sources, including local NIfTI folders, HuggingFace streaming, IDC DICOM streaming, and synthetic volumes.
3. Produce segmentation outputs, evaluation metrics, and visualisation artifacts.
4. Provide an executable, shareable codebase suitable for experimentation and extension.

## 3. Materials and Methods

### 3.1 Data Pipeline
The pipeline accepts four-channel MRI inputs (`t1`, `t1ce`, `t2`, `flair`) and a remapped label space with four classes: background, necrotic/non-enhancing tumour core, edema, and enhancing tumour. The codebase includes:

- `src/dataset.py` for BraTS folder loading, preprocessing, cropping, and augmentation.
- `src/hf_stream.py` for on-demand HuggingFace streaming.
- `src/idc_stream.py` for IDC DICOM streaming.
- `src/synthetic.py` for fully offline synthetic data generation.

### 3.2 Model Architecture
The model is a 3-D Residual U-Net with:

- 4 encoder levels,
- residual convolution blocks,
- a bottleneck stage,
- symmetric decoder blocks with skip connections,
- a 1x1x1 convolution head for multi-class logits.

The implementation is in `src/model.py`.

### 3.3 Loss Function and Optimisation
Training uses a combined Dice + cross-entropy objective implemented in `src/losses.py`. Optimisation is performed with AdamW, and learning-rate scheduling is handled in `src/utils.py`. The training scripts also support mixed precision when CUDA is available.

### 3.4 Evaluation
Evaluation utilities compute Dice and HD95 for the standard BraTS regions:

- WT: whole tumour
- TC: tumour core
- ET: enhancing tumour

Metrics and aggregation logic are implemented in `src/metrics.py`, while inference and export functions are in `evaluate.py`, `predict.py`, and `src/visualize.py`.

## 4. Experimental Status
A recommended smoke test was run with synthetic data:

`python stream_demo.py --mode synthetic --n_synthetic 10 --epochs 1`

The run completed successfully on CPU, producing a one-epoch training result and saving checkpoints and logs. During testing, a Windows console encoding issue caused by non-ASCII log output was identified and fixed. After that change, the same smoke test completed cleanly.

## 5. Results
Observed smoke-test outcome:

- Training started successfully.
- Data loading worked in synthetic mode.
- One epoch completed without runtime failure.
- Best validation Dice was reported and checkpoints were written.

Because the test used synthetic data and only one epoch, these results are pipeline-validation results rather than scientific segmentation performance.

## 6. Repository Publication
The project was uploaded to GitHub as a new public repository:

`https://github.com/faithwamb27/mri_tumor_segmentation`

Generated checkpoints, outputs, and cached data were excluded from the published repository via `.gitignore`.

## 7. Limitations

- The one-epoch synthetic run does not measure real BraTS performance.
- Full training on real BraTS data was not executed in this session.
- Windows console encoding required a small logging fix to avoid non-ASCII output.

## 8. Conclusion
This project provides a complete and reproducible 3-D MRI tumour segmentation workflow with multiple data ingestion paths, a residual U-Net architecture, and standard BraTS evaluation support. The successful smoke test demonstrates that the codebase is operational, and the GitHub repository now makes the project easy to share, review, and extend.

## 9. References
1. BraTS Challenge datasets and documentation.
2. PyTorch documentation.
3. Nibabel documentation.
4. HuggingFace Datasets documentation.
5. IDC / idc-index documentation.
