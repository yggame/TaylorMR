# TaylorMR
Code for Taylor-Expansion Informed Structural Feature Learning for Multi-Contrast MRI Reconstruction

# MCMRI-TSFN Test Guide

## Overview

This test evaluates the **TSFN** (Two-Stage Fusion Network) model for 4x accelerated multi-channel MRI reconstruction (mcRec) on the IXI dataset (T2 + PD modalities).

- **Model**: TSFN (Two-Stage Fusion Network)
- **Model Weight Path**: `./save/ixi_mcRec_model_tsfn10_x4/iter_best.pth`
- **Dataset**: IXI (T2 + PD dual-modality)
- **Acceleration Rate**: 4x
- **Sampling**: Random Uniform Mask
- **Image Size**: 256x256

## Environment Requirements

```bash
pip install torch torchvision pyiqa opencv-python pandas pyyaml tqdm
```

## Test Procedure

### 1. Run the Test

```bash
python test_mcRec_ixi_csv_ablation.py \
    --config ./configs/test/test_ixi_mcRec_x4.yaml \
    --model ./save/ixi_mcRec_model_tsfn10_x4/iter_best.pth
```

### 2. Argument Description

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `./configs/test/test_ixi_mcRec_x4.yaml` | Test configuration file |
| `--model` | `./save/ixi_mcRec_model_tsfn9_x4/iter_best.pth` | Path to model checkpoint |
| `--save_img` | `True` | Whether to save reconstructed images |

### 3. Configuration File (`configs/test/test_ixi_mcRec_x4.yaml`)

```yaml
task_mode: mcRec

test_dataset:
  dataset:
    name: ixi_dataset_Reconstruction
    args:
      modality1: T2
      modality2: PD
      split: test
      list_dir: /hdd1/yg_data/data_path/med_dataset/IXI_dataset/list_file_IXI_T2_PD
      data_dir: /hdd1/yg_data/data_path/med_dataset/IXI_dataset/IXI-T2-PD-H5
      img_size: 256
      mask_type: random_uniform
      acceleration_rate: 4

  batch_size: 1
  n_workers: 0
```

**Data Path Notes**: Ensure the following paths point to valid data locations:
- `list_dir`: Test list file for the IXI dataset
- `data_dir`: IXI dataset in HDF5 format

### 4. Test Execution Flow

The test script follows this pipeline:

```
1. Load configuration (yaml)
   ↓
2. Load model weights (iter_best.pth)
   ↓
3. Build test dataset (IXI T2+PD dual-modality)
   ↓
4. Iterate over test set (batch_size=1)
   ├─ Load undersampled K-space data (under_sample_Kspace_target)
   ├─ Load reference modality image (reference_img, PD)
   ├─ Model inference: model(im1_lr, im2_gt) → fake_H (T2 reconstruction)
   └─ Compute metrics
   ↓
5. Output results (CSV + JSON)
```

**Model Inference**: For TSFN, the model takes the undersampled target modality image (`under_sample_target_img`) and the reference modality image (`reference_img`, PD) as inputs, and produces the reconstructed target modality image (`fake_H`, T2) as output.

### 5. Output Results

Test results are saved under `./output/{model_name}/`:

| File | Description |
|------|-------------|
| `TEST_folder/test_result/*.png` | Per-image reconstructed results |
| `TEST_folder/test_results_detailed.csv` | Detailed per-image metrics with summary rows (mean/std/stderr) |
| `TEST_folder/test_info.json` | Test metadata and aggregate metrics |
| `{model_name}_log.txt` | Test log file |

### 6. Evaluation Metrics

The script computes the following metrics:

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (`calculate_psnr`) |
| **SSIM** | Structural Similarity Index (`calculate_ssim`) |
| **RMSE** | Root Mean Square Error (`calculate_rmse`) |
| **LPIPS** | Learned Perceptual Image Patch Similarity (pyiqa) |
| **PIQE** | Blind/No-reference Image Quality Evaluator (pyiqa) |
| **NIQE** | Natural Image Quality Evaluator (pyiqa) |
| **BRISQUE** | Blind/No-reference Image Quality Evaluator (pyiqa) |

The CSV file includes three summary rows at the bottom: `AVERAGE` (mean), `STD` (standard deviation), and `STDERR` (standard error).

## Ablation Experiment Comparison

This test script supports evaluating multiple model architectures. Simply change the `--model` argument to test different models. The script automatically selects the correct inference logic based on the `model_name`. Supported models include:

- **TSFN** (our model)
- mmrmamba, dudretlu
- fsmnet, loformer, mccdic
- mambairv2, swinir, A2CDIC

## Notes

1. **GPU Requirement**: Model inference runs on CUDA; ensure a GPU is available.
2. **Data Paths**: Verify `list_dir` and `data_dir` point to the correct IXI dataset before running.
3. **Memory**: Batch size defaults to 1, suitable for most GPUs. Adjust in the config file if needed.
4. **Image Saving**: Reconstructed images are saved as PNG by default. Disable with `--save_img false`.
