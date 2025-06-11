# Generative Adversarial Networks for Super-Resolution 4D Flow MRI

This repository provides the implementation and trained models accompanying the manuscript:  **"Potential and challenges of generative adversarial networks for super-resolution of intracranial 4D Flow MRI"**.  

In this study, we investigate the use of GANs for denoising and super-resolving 4D Flow MRI data. We introduce a dedicated GAN architecture for velocity field super-resolution and conduct a systematic evaluation of three commonly used adversarial loss formulations - Vanilla, Relativistic, and Wasserstein - trained and validated on synthetic 4D Flow MRI data generated from patient-specific cerebrovascular in-silico models.

<p align="center">
  <img src="PATH_TO_YOUR_FIGURE" width="600">
</p>

---

## Overview

- Training and inference implemented in TensorFlow 2.x with Keras.
- Supports generator-only (GAN-Gen) and Wasserstein GAN (WGAN) models.
- Synthetic paired training data generated from CFD-based high-resolution velocity fields combined with simulated low-resolution acquisitions using k-space subsampling and dual-venc reconstruction.
- Example dataset and pre-trained weights are provided for demonstration and reproducibility.

---

## Repository structure

| Folder | Description |
| ------ | ----------- |
| `/src/` | Source code for network architectures, loss functions, utilities |
| `/trainer/` | Training scripts |
| `/predictor/` | Inference scripts for super-resolving new data |
| `/data/` | Example dataset (toy CFD model) |
| `/models/` | Folder for trained model weights |
| `/evaluation/` | Evaluation and visualization scripts |

---

## Data preparation

Training requires paired high-resolution (HR) and low-resolution (LR) data generated from CFD simulations:

### 1. Prepare HR dataset

- High-resolution CFD velocity fields (`u`, `v`, `w`) stored in HDF5 format.
- Each velocity component has shape `[T, X, Y, Z]`.
- Flow region mask provided as either static or dynamic mask.

### 2. Generate LR dataset

- Configure file paths in `prepare_lowres_dataset.py`.
- Perform k-space cropping and complex noise injection to simulate scanner acquisition.
- Apply dual-venc reconstruction to simulate clinical processing.
- Low-resolution datasets stored as separate HDF5 files.

### 3. Extract patches for training

- Configure patch size, number of patches, and augmentation options in `prepare_patches.py`.
- Run to generate CSV patch index files for training.

---

## Training

Two training modes are supported:

### Generator-only (GAN-Gen)

- Voxel-wise supervised training using MSE loss only.

### Wasserstein GAN (WGAN)

- Combined voxel-wise and adversarial loss.
- Wasserstein loss formulation stabilizes adversarial training.

### Running training

Configure paths, model name, and hyperparameters in `trainer.py`, then start training:

python trainer.py

## Adjustable hyperparameters

| Parameter             | Description                    | Default |
|-----------------------|----------------------------------|---------|
| `patch_size`          | 3D patch size                  | 24      |
| `res_increase`        | Upsampling factor              | 2       |
| `batch_size`          | Batch size                     | 8       |
| `initial_learning_rate` | Learning rate                | 1e-4    |
| `epochs`              | Max epochs                     | 1000    |
| `lambda_G`            | Adversarial loss weighting (WGAN only) | 1e-3 |

---

## Inference

Pre-trained weights for both **GAN-Gen** and **WGAN** models are provided.

### Running prediction

1. Place model weights under `/models/`.
2. Prepare input HDF5 files using the same format as training data.
3. Configure filenames and parameters in `predictor.py`.
4. Run:

python predictor.py

## Evaluation

Quantitative evaluation and visualization is provided via:

jupyter notebook evaluation_playground.ipynb

This notebook includes:

RMSE, MRE, DE computation

Cross-sectional visualization

Feature distribution analysis

## Example dataset

An example toy dataset is provided under /data/, consisting of:

- High-resolution CFD simulation.

- Simulated low-resolution dual-venc reconstruction.

- Pre-extracted paired patches.

This allows full training and inference to be demonstrated on a simplified dataset.

## Citation
If you use this repository in your research, please cite:
O. Welin Odeback et al.
Potential and Challenges of Generative Adversarial Networks for Super-Resolution 4D Flow MRI. 2024.

## Contact
For questions or feedback, please contact:

Oliver Welin Odeback
Karolinska Institutet
oliver.welin.odeback@ki.se
