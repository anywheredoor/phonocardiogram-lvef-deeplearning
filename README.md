# PCG-based Low LVEF Detection

Deep-learning pipeline for screening reduced left ventricular ejection fraction (LVEF <= 40%) from phonocardiograms (PCG) recorded on smartphones and digital stethoscopes. The code supports MFCC and gammatone spectrograms, ImageNet-pretrained backbones, and patient-level train/val/test splits for within-device and cross-device experiments.

## Layout
- `src/data`: metadata build, patient-level splits, TF stats, feature caching.
- `src/datasets`: on-the-fly and cached PCG datasets.
- `src/models`: backbone factory (MobileNet/EfficientNet/Swin).
- `src/training`: unified training entry point.
- `src/utils`: metric helpers (F1_pos, AUROC, AUPRC, etc.).
- `heart_sounds/`, `lvef.csv`, `metadata.csv`: raw data and labels (not modified here).
- `splits/`, `cache/`, `checkpoints*/`: derived data, caches, and model checkpoints.
- `archive/`: legacy/debug scripts kept for reference.

## Dependencies
Python 3.10+, PyTorch, torchaudio, timm, pandas, numpy, scikit-learn, soundfile, tqdm, and optionally `gammatone` for gammatonegrams.

Example install (adjust torch index/extra deps as needed):
```bash
pip install torch torchaudio timm pandas numpy scikit-learn soundfile tqdm gammatone
```

## End-to-end workflow
From repository root (assumes `heart_sounds/` and `lvef.csv` are present):

```bash
# 1) Build metadata (EF labels, device, position, file paths)
python -m src.data.build_metadata

# 2) Make patient-level, stratified splits (no patient leakage)
python -m src.data.make_patient_splits

# 3) Compute TF stats for normalisation (train split only)
python -m src.data.compute_stats --representations mfcc gammatone

# 4) Precompute cached spectrogram tensors (optional but faster)
python -m src.data.precompute_cache --representation mfcc

# 5) Train a baseline model (example: MobileNetV2 + MFCC + cached features)
python -m src.training.train \
    --train_csv splits/cached_metadata_train.csv \
    --val_csv splits/cached_metadata_val.csv \
    --test_csv splits/cached_metadata_test.csv \
    --representation mfcc \
    --backbone mobilenetv2 \
    --batch_size 64 \
    --epochs 10 \
    --use_cache \
    --tune_threshold \
    --amp \
    --results_dir results
```

Key CLI options for training:
- `--use_cache`: toggle between on-the-fly spectrograms and cached tensors.
- `--device_filter` / `--position_filter`: restrict data to specific devices or auscultation sites.
- `--pos_weight`: positive-class weight for `BCEWithLogitsLoss` (label 1 = EF <= 40).
- `--tune_threshold`: tune decision threshold on the validation set to maximise F1_pos (default grid 0.05..0.95 step 0.05; override via `--threshold_grid`).
- `--eval_threshold`: default threshold when not tuning (0.5).
- `--amp`: enable mixed precision on CUDA.
- `--results_dir`, `--run_name`: where to store per-run artifacts (`metrics.json`, `metrics.csv`, checkpoint path) and an aggregated `summary.csv`. On Colab/Kaggle, point this to a persisted mount (`/content/drive/MyDrive/...` or `/kaggle/working/results` and download after the run).

Colab/Kaggle persistence tips:
- Set `--results_dir` (and optionally `--output_dir`) to a mounted path so results survive runtime resets.
- Each run creates `results/<run_name>/metrics.json` and `metrics.csv`, plus updates `results/summary.csv` for easy aggregation.

## Metrics and outputs
- Primary metric: F1 score for the low-LVEF class (label = 1).
- Also logged: AUROC, AUPRC, accuracy, sensitivity, specificity (threshold default 0.5).
- Best checkpoint (by validation F1_pos) saved under `checkpoints/` (or `checkpoints_cpu/` as configured).
- Derived artefacts:
  - `metadata.csv`: recording-level metadata with labels/devices.
  - `splits/metadata_{train,val,test}.csv`: patient-level splits.
  - `tf_stats.json`: mean/std per representation.
  - `cache/<representation>/<split>/*.pt`: cached spectrogram tensors (if enabled).
