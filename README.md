# Deep-Learning Analysis of Smartphone and Electronic-Stethoscope Phonocardiograms for Detection of Reduced Left Ventricular Ejection Fraction

Final Year Project, Bachelor of Biomedical Sciences, Li Ka Shing Faculty of Medicine, The University of Hong Kong (HKU).

## Overview
- Binary classification: LVEF <= 40% (label 1) vs > 40% (label 0).
- Inputs: PCG recordings from smartphones and electronic stethoscopes.
- Representations: MFCC and gammatone spectrograms.
- Backbones: MobileNet/EfficientNet/Swin via timm.
- Patient-level splits to avoid leakage; supports within-device, cross-device, and pooled evaluations.

## Repository Structure
- `src/data`: metadata, splits, stats, caching, QA.
- `src/datasets`: on-the-fly and cached datasets.
- `src/models`: backbone factory.
- `src/training`: training entry point.
- `src/experiments`: sweep + CV runners.
- `configs/sweep_example.json`: sweep template.
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

## End-to-End Workflow (From Scratch)
```bash
# 1) Build metadata
python -m src.data.build_metadata \
  --lvef_csv lvef.csv \
  --heart_dir heart_sounds \
  --output_csv metadata.csv

# 2) Patient-level, stratified splits
python -m src.data.make_patient_splits \
  --metadata_csv metadata.csv \
  --output_dir splits

# 3) Compute TF stats (train split only)
python -m src.data.compute_stats \
  --train_csv splits/metadata_train.csv \
  --representations mfcc gammatone
# Optional: add --per_device if you plan to use --normalization per_device

# 4) Precompute cached tensors (optional but faster)
python -m src.data.precompute_cache --representation mfcc
python -m src.data.precompute_cache --representation gammatone

# Cached CSVs are named: splits/cached_<representation>_metadata_{train,val,test}.csv

# 5) Train a baseline model
python -m src.training.train \
  --train_csv splits/cached_mfcc_metadata_train.csv \
  --val_csv splits/cached_mfcc_metadata_val.csv \
  --test_csv splits/cached_mfcc_metadata_test.csv \
  --representation mfcc \
  --backbone mobilenetv2 \
  --use_cache \
  --tune_threshold \
  --amp \
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

## Experiments
Sweep:
```bash
python -m src.experiments.run_sweep --config configs/sweep_example.json
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
  --use_cache \
  --tune_threshold
```

## Outputs
- `results/summary.csv`: aggregated metrics per run (and per device when enabled).
- `results/<run_name>/metrics.json` and `metrics.csv`: run metadata + metrics.
- `results/<run_name>/predictions_{val,test}.csv`: per-example outputs (optional).
- `results/<run_name>/history.csv`: per-epoch metrics (optional).
- `checkpoints/<run_name>/best.pth`: best checkpoint by validation F1_pos.

## Colab
Use `colab_pipeline.ipynb` for a guided end-to-end run on Google Colab.
