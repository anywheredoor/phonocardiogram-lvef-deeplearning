# Deep-Learning Analysis of Smartphone and Electronic-Stethoscope Phonocardiograms for Detection of Reduced Left Ventricular Ejection Fraction

Final Year Project, Bachelor of Biomedical Sciences, Li Ka Shing Faculty of Medicine, The University of Hong Kong.

## Table of Contents
- [Project Summary](#project-summary)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data and Inputs](#data-and-inputs)
- [Preprocessing](#preprocessing)
- [Workflow](#workflow)
- [Command Reference](#command-reference)
- [Training and Evaluation Notes](#training-and-evaluation-notes)
- [Default Hyperparameters](#default-hyperparameters)
- [Study Design (Dissertation Workflow)](#study-design-dissertation-workflow)
- [QA and SNR Sanity Check (Optional)](#qa-and-snr-sanity-check-optional)
- [Outputs](#outputs)
- [Colab](#colab)

## Project Summary
This project builds a phonocardiogram-based (PCG-based) screening model for reduced LVEF (binary classification: EF <= 40% vs > 40%) using recordings from iPhone, Android, and digital stethoscope devices. The core comparisons are (1) MFCC vs gammatone time-frequency representations and (2) lightweight CNNs vs SwinV2 backbones, with emphasis on within-device performance, cross-device generalization, and pooled-device training.

## Repository Structure
- `src/data`: metadata, splits, stats, QA.
- `src/datasets`: on-the-fly datasets.
- `src/models`: backbone factory.
- `src/training`: training entry point.
- `src/experiments`: CV runner.
- `colab_pipeline.ipynb`: end-to-end Colab notebook.

## Requirements
Python 3.10+ and packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you need GPU support, install the appropriate PyTorch build per the official PyTorch instructions.

## Data and Inputs
Expected structure:
- `heart_sounds/` with per-patient subfolders containing WAV files.
- `lvef.csv` with columns `patient_id` and `ef`.
- `patient_id` is treated as a string (leading zeros preserved).
- Filename parsing is defined in `src/data/build_metadata.py` (`FILENAME_RE`, `DEVICE_MAP`); update if your naming differs.
- Sensitive data (raw audio, labels) and derived artifacts are gitignored by default for privacy.

## Preprocessing
Audio is resampled to 2000 Hz, band-pass filtered to 20-800 Hz, then center-cropped or zero-padded to 4.0 s. Each waveform is converted to MFCC or gammatone, resized to the model input size, repeated to 3 channels, and normalized using training-split statistics (global or per-device). These steps are identical across devices to avoid leakage.

## Workflow
1. Build `metadata.csv`, then create patient-level splits and 5-fold CV splits to avoid leakage.
2. Run within-device CV for model selection (F1_pos as the primary metric).
3. Train one final model per device using the selected config, then evaluate cross-device performance using the saved checkpoints (no retraining).
4. Train one pooled model using the best config from within-device results and report overall + per-device metrics.

## Command Reference
```bash
# 1) Build metadata
python -m src.data.build_metadata \
  --lvef_csv lvef.csv \
  --heart_dir heart_sounds \
  --output_csv metadata.csv

# 2) Patient-level, stratified splits (for final within-device checkpoint)
python -m src.data.make_patient_splits \
  --metadata_csv metadata.csv \
  --output_dir splits

# 3) Patient-level CV splits (default: 5-fold)
python -m src.data.make_patient_cv_splits \
  --metadata_csv metadata.csv \
  --output_dir splits/cv \
  --n_splits 5 \
  --n_repeats 1

# 4) Compute TF stats (train split only; used for non-CV training)
python -m src.data.compute_stats \
  --train_csv splits/metadata_train.csv \
  --representations mfcc gammatone
# Optional: add --per_device if you plan to use --normalization per_device

# 5) Run 5-fold CV (default evaluation path)
python -m src.experiments.run_cv \
  --cv_index splits/cv/index.csv \
  --results_dir results \
  --output_dir checkpoints \
  -- \
  --representation mfcc \
  --backbone mobilenetv2 \
  --auto_pos_weight \
  --tune_threshold
# Optional: add --train_device_filter/--val_device_filter/--test_device_filter
# (set all three to the same device) for within-device CV.
# For SwinV2 or EfficientNetV2-S, also set --image_size (256 or 384).

# 6) Train a final within-device model (single run)
python -m src.training.train \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --representation mfcc \
  --backbone mobilenetv2 \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --test_device_filter iphone \
  --auto_pos_weight \
  --tune_threshold \
  --amp \
  --save_predictions \
  --results_dir results

# 7) Cross-device evaluation using the saved checkpoint (no retraining)
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --test_device_filter android_phone digital_stethoscope \
  --per_device_eval \
  --save_predictions \
  --results_dir results
```

## Training and Evaluation Notes
- Primary metric is F1 for the positive class (low LVEF, label 1).
- Use `--auto_pos_weight` for class imbalance (neg/pos).
- Use `--tune_threshold` to select the best decision threshold on the validation set.
- `--normalization per_device` requires `compute_stats --per_device` on the training split only.
- `--eval_only` uses the checkpoint threshold and skips training (class weighting and tuning are ignored).
- `run_cv` computes TF stats per fold by default; disable with `--skip_compute_stats`.
- Input size per backbone: 224x224 for MobileNet and EfficientNet-B0, 256x256 for SwinV2-Tiny/Small, and 384x384 for EfficientNetV2-S (matches pretrained configs for more stable transfer).
- Save predictions only for final selected models to keep output size manageable.

## Default Hyperparameters
Defaults from `src/training/train.py` (unless overridden in the notebook or CLI):
- `batch_size`: 32
- `epochs`: 100
- `lr`: 1e-4
- `optimizer`: adamw
- `weight_decay`: 1e-4
- `scheduler`: cosine (min_lr 1e-6, warmup_epochs 5)
- `grad_accum_steps`: 1
- `eval_threshold`: 0.5
- `early_stopping_patience`: 15 (min_delta 0.0)
- `sample_rate`: 2000
- `fixed_duration`: 4.0 s
- `image_size`: 224 (overridden for SwinV2/EfficientNetV2-S as documented)
- `normalization`: global
- `amp`: off by default (enable with `--amp`)
- `auto_pos_weight`: off by default (enable with `--auto_pos_weight`)
- `tune_threshold`: off by default (enable with `--tune_threshold`)

## Study Design (Dissertation Workflow)
Within-device model selection runs 3 devices x 2 representations x 6 backbones (36 configs) with 5-fold CV. After selecting the best config per device, train one final checkpoint per device for cross-device evaluation (3 training runs). Cross-device evaluation uses those checkpoints to test on the other devices (6 eval-only runs). A pooled model is trained once using the best within-device config and reported overall and per-device.

## QA and SNR Sanity Check (Optional)
Run QA to summarize data quality and (optionally) a simple SNR proxy (20-800 Hz band-pass energy vs residual):
```bash
python -m src.data.qa_report \
  --metadata_csv metadata.csv \
  --output_json reports/qa_report.json \
  --output_csv reports/qa_records.csv \
  --fixed_duration 4.0 \
  --compute_snr \
  --max_files 200
```
Example from the current dataset: mean SNR ~2.6 dB (median ~4.3 dB), with iPhone higher than stethoscope and Android lower. This check is used to justify minimal preprocessing (band-pass only) without device-specific denoising.

## Outputs
- `results/summary.csv`: aggregated metrics per run (and per device when enabled).
- `results/<run_name>/metrics.json` and `metrics.csv`: run metadata + metrics.
- `results/<run_name>/predictions_{val,test}.csv`: per-example outputs when enabled.
- `results/<run_name>/history.csv`: per-epoch metrics when enabled.
- `results/selection/best_config_per_device.csv`: best config per device from summary.csv.
- `results/selection/config_summary_by_device.csv`: aggregated config stats per device.
- `checkpoints/<run_name>/best.pth`: best checkpoint by validation F1_pos.

## Colab
Use `colab_pipeline.ipynb` for a guided end-to-end run on Google Colab.
