# Deep-Learning Analysis of Smartphone and Electronic-Stethoscope Phonocardiograms for Detection of Reduced Left Ventricular Ejection Fraction

Final Year Project, Bachelor of Biomedical Sciences, Li Ka Shing Faculty of Medicine, The University of Hong Kong.

## Overview
- Binary classification: LVEF <= 40% (label 1) vs > 40% (label 0).
- Inputs: PCG recordings from smartphones (iPhone, Android) and electronic stethoscopes.
- Representations: MFCC and gammatone spectrograms.
- Backbones: MobileNet/EfficientNet/Swin via timm.
- Patient-level splits to avoid leakage; supports within-device, cross-device, and pooled evaluations.

## Repository Structure
- `src/data`: metadata, splits, stats, caching, QA.
- `src/datasets`: on-the-fly and cached datasets.
- `src/models`: backbone factory.
- `src/training`: training entry point.
- `src/experiments`: optional CV runner.
- `colab_pipeline.ipynb`: end-to-end Colab notebook.

## Requirements
Python 3.10+ and packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you need GPU support, install the appropriate PyTorch build per the official PyTorch instructions.

## Data Expectations
- `heart_sounds/` with per-patient subfolders containing WAV files.
- `lvef.csv` with columns `patient_id` and `ef`.
- Filename parsing is defined in `src/data/build_metadata.py` (`FILENAME_RE`, `DEVICE_MAP`); update if your naming differs.

Sensitive data (raw audio, labels) and derived artifacts are gitignored by default for privacy. Store them locally or in secure storage.

## End-to-End Workflow (From Scratch)
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

# 4) Compute TF stats (train split only)
python -m src.data.compute_stats \
  --train_csv splits/metadata_train.csv \
  --representations mfcc gammatone
# Optional: add --per_device if you plan to use --normalization per_device

# 5) Precompute cached tensors (optional but faster)
python -m src.data.precompute_cache --representation mfcc
python -m src.data.precompute_cache --representation gammatone

# Cached CSVs are named: splits/cached_<representation>_metadata_{train,val,test}.csv

# 6) Run 5-fold CV (default evaluation path)
python -m src.experiments.run_cv \
  --cv_index splits/cv/index.csv \
  --results_dir results \
  --output_dir checkpoints \
  -- \
  --representation mfcc \
  --backbone mobilenetv2 \
  --auto_pos_weight \
  --tune_threshold

# 7) Train a final within-device model (single run)
python -m src.training.train \
  --train_csv splits/cached_mfcc_metadata_train.csv \
  --val_csv splits/cached_mfcc_metadata_val.csv \
  --test_csv splits/cached_mfcc_metadata_test.csv \
  --representation mfcc \
  --backbone mobilenetv2 \
  --use_cache \
  --auto_pos_weight \
  --tune_threshold \
  --amp \
  --results_dir results

# 8) Cross-device evaluation using the saved checkpoint (no retraining)
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/cached_mfcc_metadata_train.csv \
  --val_csv splits/cached_mfcc_metadata_val.csv \
  --test_csv splits/cached_mfcc_metadata_test.csv \
  --test_device_filter iphone digital_stethoscope \
  --per_device_eval \
  --results_dir results
```

## Training Notes
- `--use_cache` ignores `--sample_rate`, `--fixed_duration`, `--image_size`, and `--normalization`; ensure cached tensors were built with the intended settings.
- `--normalization per_device` requires `compute_stats --per_device`.
- Primary metric is F1 for the positive class (low LVEF, label 1).
- Use `--auto_pos_weight` for class imbalance (neg/pos).
- Use `--tune_threshold` to select the best decision threshold on the validation set.
- Use `--grad_accum_steps` if GPU memory is limited.
- Use `--deterministic` for reproducibility (may reduce performance).
- Default model selection uses 5-fold CV; single-run training is intended for final checkpoints.
- `--eval_only` uses the checkpoint threshold and skips training (class weighting and tuning are ignored).
- For strict CV hygiene, compute TF stats per fold and pass `--tf_stats_json` via `run_cv` extra args.

## Experiments
Run within-device experiments manually (repeat per device/representation/backbone):
```bash
python -m src.training.train \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --representation mfcc \
  --backbone mobilenetv2 \
  --use_cache \
  --auto_pos_weight \
  --tune_threshold \
  --per_device_eval
```

Cross-device evaluation using the best within-device checkpoint (no retraining):
```bash
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --train_device_filter android_phone \
  --val_device_filter android_phone \
  --test_device_filter iphone digital_stethoscope \
  --per_device_eval
```

Repeated patient-level CV:
```bash
python -m src.data.make_patient_cv_splits \
  --metadata_csv metadata.csv \
  --output_dir splits/cv \
  --n_splits 5 \
  --n_repeats 3 \
  --val_size 0.1

python -m src.experiments.run_cv \
  --cv_index splits/cv/index.csv \
  --results_dir results \
  --output_dir checkpoints \
  -- \
  --representation mfcc \
  --backbone mobilenetv2 \
  --tune_threshold
```

## Dissertation Workflow (Counts)
- Within-device model selection: 3 devices × 2 representations × 6 backbones = 36 configurations. Default 5-fold CV means 36 × 5 = 180 training jobs (one per fold) to select the best config per device.
- Final within-device checkpoints: CV is only for model selection. After picking the best config per device, train once per device on the standard train/val split to create the reusable `best.pth` checkpoints for cross-device evaluation (3 training runs).
- Cross-device evaluation: evaluate each device’s checkpoint on the other two devices (3 sources × 2 targets = 6 eval-only runs; no retraining).
- Pooled model: train 1 pooled model using the config chosen from within-device results, then report overall and per-device performance.

## Outputs
- `results/summary.csv`: aggregated metrics per run (and per device when enabled).
- `results/<run_name>/metrics.json` and `metrics.csv`: run metadata + metrics.
- `results/<run_name>/predictions_{val,test}.csv`: per-example outputs (optional).
- `results/<run_name>/history.csv`: per-epoch metrics (optional).
- `checkpoints/<run_name>/best.pth`: best checkpoint by validation F1_pos.

## Colab
Use `colab_pipeline.ipynb` for a guided end-to-end run on Google Colab.
