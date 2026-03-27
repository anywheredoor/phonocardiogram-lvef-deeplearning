from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import run_cmd


def test_data_preparation_clis_smoke(smoke_project: Path) -> None:
    metadata_csv = smoke_project / "metadata.csv"
    splits_dir = smoke_project / "splits"
    cv_dir = smoke_project / "splits_cv"
    stats_json = smoke_project / "nested" / "stats" / "tf_stats.json"
    qa_json = smoke_project / "nested" / "qa" / "qa_report.json"
    qa_csv = smoke_project / "nested" / "qa" / "qa_records.csv"

    run_cmd(
        [
            "-m",
            "src.data.build_metadata",
            "--lvef_csv",
            str(smoke_project / "lvef.csv"),
            "--heart_dir",
            str(smoke_project / "heart_sounds"),
            "--output_csv",
            str(metadata_csv),
        ]
    )

    metadata_df = pd.read_csv(metadata_csv)
    assert len(metadata_df) == 36
    assert set(metadata_df["device"]) == {
        "iphone",
        "android_phone",
        "digital_stethoscope",
    }

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
    run_cmd(
        [
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
            "10",
        ]
    )

    assert (splits_dir / "metadata_train.csv").is_file()
    assert (cv_dir / "index.csv").is_file()
    assert stats_json.is_file()
    assert qa_json.is_file()
    assert qa_csv.is_file()

    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    assert set(stats) == {"mfcc", "gammatone"}
    assert all("mean" in value and "std" in value for value in stats.values())

    qa_report = json.loads(qa_json.read_text(encoding="utf-8"))
    assert qa_report["recording_count"] == 10
    assert qa_report["patient_count"] > 0
    assert qa_report["missing_path_count"] == 0
