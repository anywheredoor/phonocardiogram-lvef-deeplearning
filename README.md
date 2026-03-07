# Deep Learning Analysis of Smartphone and Digital Stethoscope Phonocardiograms for Detection of Reduced Left Ventricular Ejection Fraction

This repository contains the code used for a final-year research project on phonocardiogram-based detection of reduced left ventricular ejection fraction (LVEF).

The study compares:
- device-specific versus pooled-device training
- within-device versus cross-device generalization
- MFCC versus gammatone representations
- lightweight CNN backbones versus Swin Transformer backbones

The task is binary classification: reduced LVEF (`EF <= 40%`) versus non-reduced LVEF (`EF > 40%`).

## Scope
- Research code only
- No raw audio or linked clinical labels are distributed here
- Not intended for clinical deployment

## Repository Layout
- `src/data/`: metadata building, split generation, quality checks, feature-statistics utilities
- `src/datasets/`: dataset loading and on-the-fly time-frequency feature generation
- `src/models/`: model factory and backbone definitions
- `src/training/`: training and evaluation entrypoint
- `src/experiments/`: cross-validation runner and best-configuration selection
- `src/reporting/dissertation/`: dissertation figure/table generation utilities
- `scripts/make_dissertation_outputs.py`: dissertation reporting entrypoint
- `colab_pipeline.ipynb`: notebook workflow used for the main experiments

## Setup
Python 3.10+ is recommended.

Flexible dependency install:
```bash
pip install -r requirements.txt
```

Pinned dependency install for stricter reproduction:
```bash
pip install -r requirements-lock.txt
```

If you need GPU support, install a compatible PyTorch build first, then install the remaining packages.

## Private Data
This repository expects local access to private study data, but those files are intentionally excluded from version control.

Expected local inputs:
- `heart_sounds/` with per-patient WAV files
- `lvef.csv` with `patient_id` and `ef`

The following are gitignored by default:
- raw audio and label files
- generated metadata and split files
- checkpoints, results, and reports
- dissertation summary outputs and other derived artifacts

## Typical Workflow
Build metadata:
```bash
python -m src.data.build_metadata \
  --lvef_csv lvef.csv \
  --heart_dir heart_sounds \
  --output_csv metadata.csv
```

Create final train/validation/test splits:
```bash
python -m src.data.make_patient_splits \
  --metadata_csv metadata.csv \
  --output_dir splits
```

Create patient-level CV folds:
```bash
python -m src.data.make_patient_cv_splits \
  --metadata_csv metadata.csv \
  --output_dir splits/cv \
  --n_splits 5 \
  --n_repeats 1
```

Run cross-validation for a candidate configuration:
```bash
python -m src.experiments.run_cv \
  --cv_index splits/cv/index.csv \
  --results_dir results \
  --output_dir checkpoints \
  -- \
  --representation mfcc \
  --backbone mobilenetv2 \
  --auto_pos_weight \
  --tune_threshold
```

Select the best configuration per device from a CV-only summary file:
```bash
python -m src.experiments.select_best_config \
  --summary_csv results/summary.csv \
  --expected_folds 5 \
  --output_csv results/selection/best_config_per_device.csv \
  --all_csv results/selection/config_summary_by_device.csv
```

Train a final model:
```bash
python -m src.training.train \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --representation mfcc \
  --backbone mobilenetv2 \
  --auto_pos_weight \
  --tune_threshold \
  --amp \
  --save_predictions \
  --results_dir results
```

Evaluate a saved checkpoint without retraining:
```bash
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --per_device_eval \
  --save_predictions \
  --results_dir results
```

Generate dissertation-ready figures and tables from a local `summary.csv` and saved run outputs:
```bash
python3 scripts/make_dissertation_outputs.py \
  --summary_csv summary.csv \
  --output_dir reports/dissertation_summary_outputs \
  --results_run_dir "results/first run" \
  --dpi 300
```

## Notes on Reproducibility
- Primary training entrypoint: `src/training/train.py`
- Primary early-stopping metric during training: positive-class F1 on validation data
- Cross-validation config selection uses mean test `F1_pos` across folds
- Cross-validation and final evaluation are patient-level split to avoid leakage across recordings from the same patient
- The dissertation reporting code reads locally generated summaries and prediction files; it does not bundle sensitive data into the repository

## Citation
Citation metadata is provided in `CITATION.cff`.

## License
Apache License 2.0. See `LICENSE`.
