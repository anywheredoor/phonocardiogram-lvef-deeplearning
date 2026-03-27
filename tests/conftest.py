from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEVICE_CODES = {
    "android_phone": "a",
    "iphone": "i",
    "digital_stethoscope": "e",
}
POSITIONS = ["A", "M", "P", "T"]


def run_checked(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def run_cmd(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"{' '.join([sys.executable, *args])}\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def _write_wave(path: Path, sr: int, freq_hz: float, rng: np.random.Generator) -> None:
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    wave = 0.2 * np.sin(2.0 * np.pi * freq_hz * t)
    wave += 0.02 * rng.standard_normal(size=t.shape[0])
    sf.write(path, wave.astype(np.float32), sr)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("pcg_synth")
    heart_dir = root / "heart_sounds"
    heart_dir.mkdir()

    patient_rows = [
        {"patient_id": "3001", "ef": 30.0},
        {"patient_id": "3002", "ef": 35.0},
        {"patient_id": "3003", "ef": 38.0},
        {"patient_id": "3004", "ef": 40.0},
        {"patient_id": "3005", "ef": 50.0},
        {"patient_id": "3006", "ef": 55.0},
        {"patient_id": "3007", "ef": 60.0},
        {"patient_id": "3008", "ef": 65.0},
    ]

    rng = np.random.default_rng(1234)
    sr = 4000
    for patient_idx, row in enumerate(patient_rows):
        patient_dir = heart_dir / row["patient_id"]
        patient_dir.mkdir()
        base_freq = 110.0 if row["ef"] <= 40.0 else 220.0
        for device_idx, device_code in enumerate(DEVICE_CODES.values()):
            for position_idx, position_code in enumerate(POSITIONS):
                freq = base_freq + device_idx * 17.0 + position_idx * 7.0 + patient_idx
                wav_path = patient_dir / f"{device_code}Data{row['patient_id']}{position_code}.wav"
                _write_wave(wav_path, sr=sr, freq_hz=freq, rng=rng)

    lvef_csv = root / "lvef.csv"
    pd.DataFrame(patient_rows).to_csv(lvef_csv, index=False)

    return {
        "root": root,
        "heart_dir": heart_dir,
        "lvef_csv": lvef_csv,
        "patient_ids": [row["patient_id"] for row in patient_rows],
    }


@pytest.fixture(scope="session")
def smoke_workspace(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_dataset: dict[str, object],
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("pcg_smoke")
    metadata_csv = root / "metadata.csv"
    splits_dir = root / "splits"
    cv_dir = root / "splits_cv"
    qa_json = root / "qa_report.json"
    qa_csv = root / "qa_records.csv"
    tf_stats_json = root / "tf_stats.json"
    cv_index_csv = root / "smoke_cv_index.csv"
    report_results_dir = root / "report_results"
    report_checkpoints_dir = root / "report_checkpoints"
    cv_results_dir = root / "cv_results"
    cv_checkpoints_dir = root / "cv_checkpoints"
    dissertation_output_dir = root / "dissertation_outputs"

    run_checked(
        [
            sys.executable,
            "-m",
            "src.data.build_metadata",
            "--lvef_csv",
            str(synthetic_dataset["lvef_csv"]),
            "--heart_dir",
            str(synthetic_dataset["heart_dir"]),
            "--output_csv",
            str(metadata_csv),
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.data.make_patient_splits",
            "--metadata_csv",
            str(metadata_csv),
            "--output_dir",
            str(splits_dir),
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.data.make_patient_cv_splits",
            "--metadata_csv",
            str(metadata_csv),
            "--output_dir",
            str(cv_dir),
            "--n_splits",
            "5",
            "--n_repeats",
            "1",
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.data.qa_report",
            "--metadata_csv",
            str(metadata_csv),
            "--output_json",
            str(qa_json),
            "--output_csv",
            str(qa_csv),
            "--fixed_duration",
            "4.0",
            "--compute_snr",
            "--max_files",
            "12",
            "--snr_max_seconds",
            "0.5",
        ]
    )

    metadata_df = pd.read_csv(metadata_csv, dtype={"patient_id": str})
    smoke_splits = {
        "train": {"3001", "3002", "3005", "3006"},
        "val": {"3003", "3007"},
        "test": {"3004", "3008"},
    }
    smoke_csvs: dict[str, Path] = {}
    for split, patient_ids in smoke_splits.items():
        out_path = root / f"smoke_{split}.csv"
        (
            metadata_df[metadata_df["patient_id"].isin(patient_ids)]
            .reset_index(drop=True)
            .to_csv(out_path, index=False)
        )
        smoke_csvs[split] = out_path

    run_checked(
        [
            sys.executable,
            "-m",
            "src.data.compute_stats",
            "--train_csv",
            str(smoke_csvs["train"]),
            "--representations",
            "mfcc",
            "gammatone",
            "--image_size",
            "64",
            "--batch_size",
            "8",
            "--output_json",
            str(tf_stats_json),
        ]
    )

    common_train_args = [
        "--train_csv",
        str(smoke_csvs["train"]),
        "--val_csv",
        str(smoke_csvs["val"]),
        "--test_csv",
        str(smoke_csvs["test"]),
        "--tf_stats_json",
        str(tf_stats_json),
        "--representation",
        "mfcc",
        "--backbone",
        "mobilenetv2",
        "--batch_size",
        "8",
        "--epochs",
        "1",
        "--num_workers",
        "0",
        "--image_size",
        "64",
        "--normalization",
        "global",
        "--auto_pos_weight",
        "--tune_threshold",
        "--save_predictions",
        "--save_history",
        "--per_device_eval",
    ]

    final_run_name = "final_iphone_mfcc_mobilenetv2"
    pooled_run_name = "pooled_all_devices_mfcc_mobilenetv2"
    eval_run_name = "eval_final_iphone_to_android_digital"
    cv_run_name = (
        "cv_smoke_r00_f00_mobilenetv2_mfcc_im64_deviphone_posall"
    )

    run_checked(
        [
            sys.executable,
            "-m",
            "src.training.train",
            *common_train_args,
            "--train_device_filter",
            "iphone",
            "--val_device_filter",
            "iphone",
            "--test_device_filter",
            "iphone",
            "--results_dir",
            str(report_results_dir),
            "--output_dir",
            str(report_checkpoints_dir),
            "--run_name",
            final_run_name,
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.training.train",
            *common_train_args,
            "--results_dir",
            str(report_results_dir),
            "--output_dir",
            str(report_checkpoints_dir),
            "--run_name",
            pooled_run_name,
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.training.train",
            "--eval_only",
            "--checkpoint_path",
            str(report_checkpoints_dir / final_run_name / "best.pth"),
            "--train_csv",
            str(smoke_csvs["train"]),
            "--val_csv",
            str(smoke_csvs["val"]),
            "--test_csv",
            str(smoke_csvs["test"]),
            "--tf_stats_json",
            str(tf_stats_json),
            "--test_device_filter",
            "android_phone",
            "digital_stethoscope",
            "--save_predictions",
            "--per_device_eval",
            "--results_dir",
            str(report_results_dir),
            "--run_name",
            eval_run_name,
        ]
    )

    pd.DataFrame(
        [
            {
                "repeat": 0,
                "fold": 0,
                "train_csv": str(smoke_csvs["train"]),
                "val_csv": str(smoke_csvs["val"]),
                "test_csv": str(smoke_csvs["test"]),
                "train_n_patients": 4,
                "train_n_pos": 2,
                "train_n_neg": 2,
                "val_n_patients": 2,
                "val_n_pos": 1,
                "val_n_neg": 1,
                "test_n_patients": 2,
                "test_n_pos": 1,
                "test_n_neg": 1,
            }
        ]
    ).to_csv(cv_index_csv, index=False)

    run_checked(
        [
            sys.executable,
            "-m",
            "src.experiments.run_cv",
            "--cv_index",
            str(cv_index_csv),
            "--results_dir",
            str(cv_results_dir),
            "--output_dir",
            str(cv_checkpoints_dir),
            "--max_runs",
            "1",
            "--run_name_format",
            "cv_smoke_r{repeat:02d}_f{fold:02d}_{backbone}_{representation}_im{image_size}_dev{device}_pos{position}",
            "--",
            "--representation",
            "mfcc",
            "--backbone",
            "mobilenetv2",
            "--image_size",
            "64",
            "--batch_size",
            "8",
            "--epochs",
            "1",
            "--num_workers",
            "0",
            "--train_device_filter",
            "iphone",
            "--val_device_filter",
            "iphone",
            "--test_device_filter",
            "iphone",
            "--tune_threshold",
            "--save_predictions",
            "--save_history",
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "src.experiments.select_best_config",
            "--summary_csv",
            str(cv_results_dir / "summary.csv"),
            "--expected_folds",
            "1",
            "--output_csv",
            str(cv_results_dir / "best.csv"),
            "--all_csv",
            str(cv_results_dir / "all.csv"),
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/make_dissertation_outputs.py",
            "--summary_csv",
            str(report_results_dir / "summary.csv"),
            "--results_run_dir",
            str(report_results_dir),
            "--output_dir",
            str(dissertation_output_dir),
            "--dpi",
            "100",
        ]
    )

    return {
        "root": root,
        "metadata_csv": metadata_csv,
        "splits_dir": splits_dir,
        "cv_dir": cv_dir,
        "qa_json": qa_json,
        "qa_csv": qa_csv,
        "tf_stats_json": tf_stats_json,
        "smoke_csvs": smoke_csvs,
        "report_results_dir": report_results_dir,
        "report_checkpoints_dir": report_checkpoints_dir,
        "report_summary": report_results_dir / "summary.csv",
        "cv_results_dir": cv_results_dir,
        "cv_summary": cv_results_dir / "summary.csv",
        "cv_best_csv": cv_results_dir / "best.csv",
        "cv_run_name": cv_run_name,
        "final_run_name": final_run_name,
        "pooled_run_name": pooled_run_name,
        "eval_run_name": eval_run_name,
        "dissertation_output_dir": dissertation_output_dir,
    }


@pytest.fixture
def smoke_project(tmp_path: Path) -> Path:
    root = tmp_path / "smoke_project"
    heart_dir = root / "heart_sounds"
    heart_dir.mkdir(parents=True)

    rows = []
    patients = [
        ("1001", 35.0),
        ("1002", 55.0),
        ("1003", 30.0),
        ("1004", 60.0),
        ("1005", 38.0),
        ("1006", 65.0),
    ]

    sr = 4000
    duration = 7.5
    time_axis = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    for patient_id, ef in patients:
        rows.append({"patient_id": patient_id, "ef": ef})
        patient_dir = heart_dir / patient_id
        patient_dir.mkdir()

        base_wave = (
            0.10 * np.sin(2 * np.pi * 40 * time_axis)
            + 0.05 * np.sin(2 * np.pi * 80 * time_axis)
        )

        for device_code in ("i", "a", "e"):
            for position in ("A", "M"):
                seed = int(patient_id) + ord(device_code) + ord(position)
                noise = 0.01 * np.random.default_rng(seed).normal(size=time_axis.shape)
                waveform = (base_wave + noise).astype("float32")
                sf.write(
                    patient_dir / f"{device_code}Data{patient_id}{position}.wav",
                    waveform,
                    sr,
                )

    pd.DataFrame(rows).to_csv(root / "lvef.csv", index=False)
    return root


@pytest.fixture
def prepared_smoke_project(smoke_project: Path) -> Path:
    root = smoke_project
    metadata_csv = root / "metadata.csv"
    splits_dir = root / "splits"
    cv_dir = root / "splits_cv"
    stats_json = root / "artifacts" / "tf_stats.json"

    run_cmd(
        [
            "-m",
            "src.data.build_metadata",
            "--lvef_csv",
            str(root / "lvef.csv"),
            "--heart_dir",
            str(root / "heart_sounds"),
            "--output_csv",
            str(metadata_csv),
        ]
    )
    run_cmd(
        [
            "-m",
            "src.data.make_patient_splits",
            "--metadata_csv",
            str(metadata_csv),
            "--output_dir",
            str(splits_dir),
            "--test_size",
            "0.2",
            "--val_size",
            "0.15",
            "--seed",
            "7",
        ]
    )
    run_cmd(
        [
            "-m",
            "src.data.make_patient_cv_splits",
            "--metadata_csv",
            str(metadata_csv),
            "--output_dir",
            str(cv_dir),
            "--n_splits",
            "3",
            "--n_repeats",
            "1",
            "--val_size",
            "0.25",
            "--seed",
            "7",
        ]
    )
    run_cmd(
        [
            "-m",
            "src.data.compute_stats",
            "--train_csv",
            str(splits_dir / "metadata_train.csv"),
            "--representations",
            "mfcc",
            "gammatone",
            "--image_size",
            "32",
            "--batch_size",
            "4",
            "--output_json",
            str(stats_json),
        ]
    )

    return root


@pytest.fixture(autouse=True)
def clean_mpl_cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
