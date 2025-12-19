# PCG-based Low LVEF Detection

Deep-learning pipeline for screening reduced left ventricular ejection fraction (LVEF <= 40%) from phonocardiograms (PCG) recorded on smartphones and digital stethoscopes. The code supports MFCC and gammatone spectrograms, ImageNet-pretrained backbones, and patient-level train/val/test splits for within-device and cross-device experiments.

## Layout
- `src/data`: metadata build, patient-level splits, TF stats, feature caching.
- `src/datasets`: on-the-fly and cached PCG datasets.
- `src/models`: backbone factory (MobileNet/EfficientNet/Swin).
- `src/training`: unified training entry point.
- `src/utils`: metric helpers (F1_pos, AUROC, AUPRC, etc.).
- `heart_sounds/`, `lvef.csv`: raw data and labels.
- `metadata.csv`: derived metadata generated from `lvef.csv` + `heart_sounds/`.
- `splits/`, `cache/`, `checkpoints*/`: derived data, caches, and model checkpoints.

## Dependencies
Python 3.10+, PyTorch, torchaudio, timm, pandas, numpy, scikit-learn, soundfile, tqdm, and `gammatone` for gammatonegrams.

Example install (adjust torch index/extra deps as needed):
```bash
pip install -r requirements.txt
```

## End-to-end workflow
From repository root (assumes `heart_sounds/` and `lvef.csv` are present):

```bash
# 1) Build metadata (EF labels, device, position, file paths)
python -m src.data.build_metadata
# If your filename pattern differs, edit FILENAME_RE/DEVICE_MAP in src/data/build_metadata.py.

# 2) Make patient-level, stratified splits (no patient leakage)
python -m src.data.make_patient_splits

# 3) Compute TF stats for normalisation (train split only)
python -m src.data.compute_stats --representations mfcc gammatone
# Optional: add --per_device for device-specific stats (needed for --normalization per_device)

# 4) Precompute cached spectrogram tensors (optional but faster)
python -m src.data.precompute_cache --representation mfcc
# Optional: add --normalization per_device if tf_stats.json has per-device stats

# Cached CSVs are named: splits/cached_<representation>_metadata_{train,val,test}.csv

# 5) Train a baseline model (example: MobileNetV2 + MFCC + cached features)
python -m src.training.train \
    --train_csv splits/cached_mfcc_metadata_train.csv \
    --val_csv splits/cached_mfcc_metadata_val.csv \
    --test_csv splits/cached_mfcc_metadata_test.csv \
    --representation mfcc \
    --backbone mobilenetv2 \
    --batch_size 64 \
    --epochs 10 \
    --use_cache \
    --tune_threshold \
    --amp \
    --results_dir results
```

Note: if you change `--sample_rate`, `--fixed_duration`, or `--image_size`, recompute `tf_stats.json` and any cached tensors to keep preprocessing consistent.

Key CLI options for training:
- `--use_cache`: toggle between on-the-fly spectrograms and cached tensors; accepts either `splits/metadata_{train,val,test}.csv` or `splits/cached_<representation>_metadata_{train,val,test}.csv` and auto-resolves to the representation-specific cached CSVs. When enabled, preprocessing args like `--sample_rate`, `--fixed_duration`, `--image_size`, and `--normalization` are ignored; ensure cached tensors were built with the intended settings.
- `--device_filter` / `--position_filter`: restrict data to specific devices or auscultation sites (applied to train/val/test).
- `--train_device_filter` / `--val_device_filter` / `--test_device_filter`: split-specific device filters (for cross-device runs).
- `--train_position_filter` / `--val_position_filter` / `--test_position_filter`: split-specific position filters.
- `--pos_weight`: positive-class weight for `BCEWithLogitsLoss` (label 1 = EF <= 40).
- `--auto_pos_weight`: compute pos_weight from the training split (neg/pos).
- `--grad_accum_steps`: gradient accumulation steps for memory-limited GPUs.
- `--tune_threshold`: tune decision threshold on the validation set to maximise F1_pos (default grid 0.05..0.95 step 0.05; override via `--threshold_grid`).
- `--eval_threshold`: default threshold when not tuning (0.5).
- `--early_stopping_patience` / `--early_stopping_min_delta`: stop when validation F1_pos stops improving.
- `--optimizer` / `--weight_decay`: optimizer settings (default AdamW with weight decay).
- `--scheduler` / `--warmup_epochs` / `--min_lr`: learning rate schedule (default cosine with warmup).
- `--amp`: enable mixed precision on CUDA.
- `--normalization`: on-the-fly normalisation strategy (`global`, `per_device`, `none`).
- `--sample_rate` / `--fixed_duration`: on-the-fly preprocessing controls (must match stats/cache settings).
- `--per_device_eval`: compute test metrics per device (stored in `metrics.json`).
- `--save_predictions`: save per-example predictions for val/test into `predictions_{val,test}.csv`.
- `--save_history`: save per-epoch training history into `history.csv`.
- `--deterministic`: enable deterministic training (may reduce performance).
- `--results_dir`, `--run_name`: where to store per-run artifacts (`metrics.json`, `metrics.csv`, checkpoint path) and an aggregated `summary.csv`. On Colab/Kaggle, point this to a persisted mount (`/content/drive/MyDrive/...` or `/kaggle/working/results` and download after the run).
  - `summary.csv` includes one row per run (metric_scope=`overall`) plus optional per-device rows when `--per_device_eval` is enabled (metric_scope=`test_device`).

Colab/Kaggle persistence tips:
- Set `--results_dir` (and optionally `--output_dir`) to a mounted path so results survive runtime resets.
- Each run creates `results/<run_name>/metrics.json` and `metrics.csv`, plus updates `results/summary.csv` for easy aggregation.

## Metrics and outputs
- Primary metric: F1 score for the low-LVEF class (label = 1).
- Also logged: AUROC, AUPRC, accuracy, sensitivity, specificity (threshold default 0.5).
- Best checkpoint (by validation F1_pos) saved under `checkpoints/<run_name>/best.pth` (or `<output_dir>/<run_name>/best.pth` as configured).
- Derived artefacts:
  - `metadata.csv`: recording-level metadata with labels/devices.
  - `splits/metadata_{train,val,test}.csv`: patient-level splits.
  - `tf_stats.json`: mean/std per representation.
  - `cache/<representation>/<split>/*.pt`: cached spectrogram tensors (if enabled).
  - `results/<run_name>/predictions_{val,test}.csv`: per-example logits/probabilities (if enabled).
  - `results/<run_name>/history.csv`: per-epoch metrics (if enabled).

## Cross-device examples
Train on iPhone + Android, test on digital stethoscope (patient-level split is still respected):

```bash
python -m src.training.train \
    --train_device_filter iphone android_phone \
    --val_device_filter iphone android_phone \
    --test_device_filter digital_stethoscope \
    --representation mfcc \
    --backbone mobilenetv2 \
    --tune_threshold \
    --per_device_eval
```

## QA report
Generate a quick dataset audit (missing files, durations, label counts):

```bash
mkdir -p reports
python -m src.data.qa_report \
    --metadata_csv metadata.csv \
    --output_json reports/qa_report.json \
    --output_csv reports/qa_records.csv \
    --fixed_duration 4.0
```

To include a training-split pos_weight (neg/pos), pass the train CSV:

```bash
python -m src.data.qa_report \
    --metadata_csv metadata.csv \
    --train_csv splits/metadata_train.csv \
    --output_json reports/qa_report.json
```

## Repeated patient-level CV splits
Create repeated, stratified CV folds with patient-level grouping:

```bash
python -m src.data.make_patient_cv_splits \
    --metadata_csv metadata.csv \
    --output_dir splits/cv \
    --n_splits 5 \
    --n_repeats 3 \
    --val_size 0.1
```

Run training across CV folds:

```bash
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

## Sweep runner
Use a JSON config to run factorial sweeps:

```bash
python -m src.experiments.run_sweep --config configs/sweep_example.json
```
