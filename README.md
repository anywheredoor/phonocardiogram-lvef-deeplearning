# Deep Learning Analysis of Smartphone and Digital Stethoscope Phonocardiograms for Detection of Reduced Left Ventricular Ejection Fraction

This repository contains the code for my final year project for the Bachelor of Biomedical Sciences programme, Li Ka Shing Faculty of Medicine, The University of Hong Kong.

The project studies reduced left ventricular ejection fraction (LVEF) detection from phonocardiograms (PCGs) using deep learning. The main comparisons are:
- within-device training and evaluation
- cross-device transfer
- pooled-device training and evaluation
- MFCC versus gammatone representations
- lightweight CNN backbones versus Swin Transformer backbones

The task is binary classification: reduced LVEF (`EF <= 40%`) versus non-reduced LVEF (`EF > 40%`).

This repository is intended for research and educational use only, not for clinical deployment.

---

**Acknowledgements**

I am deeply grateful to **[Prof. Joshua W. K. Ho](https://www.sbms.hku.hk/staff/joshua-ho)** for supervising this project and for helping me see the broader importance of cardiovascular disease research, especially its relevance to real-world clinical problems.

I would like to sincerely thank **Dr Chi Yan Ooi** for kindly explaining the study setup and the heart-sound data collection process, which gave me a much clearer understanding of the practical background of this work.

I am also thankful to **Chutong Xiao**, a PhD student, for first introducing the dataset to me and for helping me understand its structure and context at an early stage of the project.

I am grateful to **[Prof. Chun-Ka Wong](https://medic.hku.hk/en/Staff/University-Academic-Staff/Prof-WONG-Chun-Ka/Prof-WONG-Chun-Ka-Profile)**, whose clinical study provided the foundation for the de-identified data used in this project and made this research possible.

---

## Table of Contents
- [Data Source and Study Context](#data-source-and-study-context)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Study Workflow](#study-workflow)
- [Command-Line Usage](#command-line-usage)
- [Preliminary Experiments](#preliminary-experiments)
- [Citation](#citation)
- [License](#license)

## Data Source and Study Context
The heart sound and LVEF data used in this project come from a de-identified stratified random sample derived from [ClinicalTrials.gov NCT06070298](https://clinicaltrials.gov/study/NCT06070298). The registered study concerns smartphone-based phonocardiography for murmur detection, whereas this repository focuses on reduced-LVEF screening using the same heart-sound acquisition protocol.

High-level study context shared with the registered study:
- observational study approved by the Institutional Review Board of the University of Hong Kong / Hospital Authority Hong Kong West Cluster
- adult cardiology outpatients aged 22 years or above who had undergone echocardiography within 3 years
- exclusion of participants with implanted active medical devices in the torso
- recordings collected by research personnel during routine clinic or day-centre visits after consent
- intended acquisition at the 4 standard cardiac auscultation sites using 3 devices: iPhone, Android phone, and a digital stethoscope
- up to 12 recordings per participant (`4 sites x 3 devices`), although a small number contributed fewer recordings
- real-world public hospital recording conditions rather than a controlled acoustic environment

The subset used for this project contains WAV heart-sound recordings plus an LVEF CSV. It does not include additional clinical covariates such as age, sex, or BMI.

The raw heart-sound recordings and LVEF labels are not distributed in this repository. They originate from human-participant data and remain restricted for participant privacy, ethics approval, and local data-governance reasons.

Expected local inputs:
- `heart_sounds/` with per-patient WAV files
- `lvef.csv` with `patient_id` and `ef`

The following are gitignored by default:
- raw audio and label files
- generated metadata and split files
- checkpoints, results, and reports
- other derived artifacts

## Repository Layout
- `src/data/`: metadata building, split generation, quality checks, and feature-statistics utilities
- `src/datasets/`: dataset loading and on-the-fly time-frequency feature generation
- `src/models/`: model factory and backbone definitions
- `src/training/`: training and evaluation entrypoint
- `src/experiments/`: cross-validation runner and best-configuration selection
- `colab_pipeline.ipynb`: guided end-to-end workflow for Google Colab

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

`colab_pipeline.ipynb` is a convenient way to run the full workflow on Google Colab, especially if local GPU resources are limited.

## Data Preprocessing
The training pipeline performs feature generation on the fly.

At a high level, each recording is:
- loaded from WAV
- resampled to the target sampling rate (default: 2 kHz)
- band-pass filtered (`20-800 Hz`)
- centre-cropped to a fixed duration (default: 4 s; zero-padding is supported when needed)
- converted to either MFCC or gammatone representation
- resized to the requested image size for ImageNet-pretrained backbones

Patient-level splits are used throughout to avoid leakage across multiple recordings from the same participant.

## Study Workflow
The main experiment grid covers:
- `3 devices x 2 representations x 6 backbones = 36` within-device configurations
- each configuration evaluated with 5-fold cross-validation

After configuration selection:
- one final checkpoint is trained per device using the selected configuration (`3` training runs)
- `3` cross-device eval-only launches are run from the saved within-device checkpoints, yielding `6` pairwise comparisons
- one pooled-device model is trained using the selected configuration

This structure supports within-device, cross-device, and pooled-device comparisons under a consistent pipeline.

### Within-device workflow
```mermaid
flowchart TD
    A["iPhone<br/>Android phone<br/>Digital stethoscope"]
    B["MFCC<br/>Gammatone"]
    C["MobileNetV2<br/>MobileNetV3-Large<br/>EfficientNet-B0<br/>EfficientNetV2-S<br/>SwinV2-Tiny<br/>SwinV2-Small"]
    D["Best config identified for each device"]
    A --> B --> C --> D
```

### Cross-device workflow
```mermaid
flowchart LR
    A["Best-config within-device model trained on iPhone"] --> A1["Evaluate on Android phone"]
    A --> A2["Evaluate on Digital stethoscope"]
```

```mermaid
flowchart LR
    B["Best-config within-device model trained on Android phone"] --> B1["Evaluate on iPhone"]
    B --> B2["Evaluate on Digital stethoscope"]
```

```mermaid
flowchart LR
    C["Best-config within-device model trained on Digital stethoscope"] --> C1["Evaluate on iPhone"]
    C --> C2["Evaluate on Android phone"]
```

### Pooled-device workflow
```mermaid
flowchart LR
    A["Pooled-device model trained on all devices"] --> E["Evaluate on pooled test set"]
    B["Best-config within-device model trained on iPhone"] --> E
    C["Best-config within-device model trained on Android phone"] --> E
    D["Best-config within-device model trained on Digital stethoscope"] --> E
```

## Command-Line Usage
Typical flow:
- build `metadata.csv`
- make one held-out split in `splits/` and one CV split in `splits/cv/`
- select device-specific configs with within-device CV
- train held-out within-device models, then run cross-device and pooled-device evaluation

Build metadata:
```bash
python -m src.data.build_metadata \
  --lvef_csv lvef.csv \
  --heart_dir heart_sounds \
  --output_csv metadata.csv
```

Create held-out train/validation/test split:
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

Run within-device CV for one candidate configuration (example: iPhone):
```bash
python -m src.experiments.run_cv \
  --cv_index splits/cv/index.csv \
  --results_dir results \
  --output_dir checkpoints \
  -- \
  --representation gammatone \
  --backbone swinv2_tiny \
  --image_size 256 \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --test_device_filter iphone \
  --auto_pos_weight \
  --tune_threshold
```

Select the best configuration per device:
```bash
python -m src.experiments.select_best_config \
  --summary_csv results/summary.csv \
  --expected_folds 5 \
  --output_csv results/selection/best_config_per_device.csv \
  --all_csv results/selection/config_summary_by_device.csv
```

The examples below use `gammatone + swinv2_tiny` with `image_size 256`.

Train a held-out within-device model (example: iPhone):
```bash
python -m src.data.compute_stats \
  --train_csv splits/metadata_train.csv \
  --representations gammatone \
  --image_size 256 \
  --device_filter iphone \
  --output_json tf_stats_iphone.json

python -m src.training.train \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --representation gammatone \
  --backbone swinv2_tiny \
  --image_size 256 \
  --tf_stats_json tf_stats_iphone.json \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --test_device_filter iphone \
  --auto_pos_weight \
  --tune_threshold \
  --amp \
  --save_predictions \
  --results_dir results
```

Run cross-device eval-only transfer (example: iPhone -> Android phone):
```bash
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --test_device_filter android_phone \
  --save_predictions \
  --results_dir results
```

Evaluate a within-device checkpoint on the pooled test set:
```bash
python -m src.training.train \
  --eval_only \
  --checkpoint_path checkpoints/<run_name>/best.pth \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --train_device_filter iphone \
  --val_device_filter iphone \
  --save_predictions \
  --per_device_eval \
  --results_dir results
```

Train the pooled-device model on the held-out pooled split (leave device filters unset):
```bash
python -m src.data.compute_stats \
  --train_csv splits/metadata_train.csv \
  --representations gammatone \
  --image_size 256 \
  --output_json tf_stats_pooled.json

python -m src.training.train \
  --train_csv splits/metadata_train.csv \
  --val_csv splits/metadata_val.csv \
  --test_csv splits/metadata_test.csv \
  --representation gammatone \
  --backbone swinv2_tiny \
  --image_size 256 \
  --tf_stats_json tf_stats_pooled.json \
  --auto_pos_weight \
  --tune_threshold \
  --amp \
  --save_predictions \
  --results_dir results
```

## Preliminary Experiments
These earlier repositories were completed to help me define the scope of this final project:
- [Multi-Task vs Single-Task Modeling for PCG Analysis](https://github.com/anywheredoor/pcg_experiment_1)
- [PCG-Only Baseline for Reduced LVEF Detection (ViT-B/16)](https://github.com/anywheredoor/pcg_experiment_2)
- [Phonocardiogram MIL Pipeline for Reduced LVEF Screening](https://github.com/anywheredoor/pcg_experiment_3)

## Citation
Citation metadata is provided in `CITATION.cff`.

## License
Apache License 2.0. See `LICENSE`.
