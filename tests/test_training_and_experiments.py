from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch.nn as nn

from src.datasets.pcg_dataset import PCGDataset
import src.experiments.run_cv as run_cv_mod
import src.experiments.select_best_config as select_best_config_mod
import src.training.train as train_mod


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_dataset_rejects_unknown_representation(prepared_smoke_project: Path) -> None:
    with pytest.raises(ValueError, match="representation must be"):
        PCGDataset(
            csv_path=str(prepared_smoke_project / "metadata.csv"),
            representation="unknown",
        )


def test_training_and_eval_only_smoke(
    prepared_smoke_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = prepared_smoke_project
    results_dir = root / "results"
    checkpoints_dir = root / "checkpoints"
    stats_json = root / "artifacts" / "tf_stats.json"

    monkeypatch.setattr(train_mod, "create_model", lambda **_: TinyModel())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--train_csv",
            str(root / "splits" / "metadata_train.csv"),
            "--val_csv",
            str(root / "splits" / "metadata_val.csv"),
            "--test_csv",
            str(root / "splits" / "metadata_test.csv"),
            "--tf_stats_json",
            str(stats_json),
            "--representation",
            "mfcc",
            "--backbone",
            "mobilenetv2",
            "--image_size",
            "32",
            "--epochs",
            "2",
            "--batch_size",
            "4",
            "--num_workers",
            "0",
            "--results_dir",
            str(results_dir),
            "--output_dir",
            str(checkpoints_dir),
            "--run_name",
            "smoke_train",
            "--save_predictions",
            "--save_history",
            "--tune_threshold",
            "--auto_pos_weight",
            "--deterministic",
        ],
    )
    train_mod.main()

    assert (checkpoints_dir / "smoke_train" / "best.pth").is_file()
    assert (results_dir / "smoke_train" / "metrics.json").is_file()
    assert (results_dir / "smoke_train" / "predictions_test.csv").is_file()
    assert (results_dir / "smoke_train" / "history.csv").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--eval_only",
            "--checkpoint_path",
            str(checkpoints_dir / "smoke_train" / "best.pth"),
            "--train_csv",
            str(root / "splits" / "metadata_train.csv"),
            "--val_csv",
            str(root / "splits" / "metadata_val.csv"),
            "--test_csv",
            str(root / "splits" / "metadata_test.csv"),
            "--tf_stats_json",
            str(stats_json),
            "--results_dir",
            str(results_dir),
            "--run_name",
            "smoke_eval",
            "--save_predictions",
            "--per_device_eval",
        ],
    )
    train_mod.main()

    summary_df = pd.read_csv(results_dir / "summary.csv")
    assert set(summary_df["run_name"]) == {"smoke_train", "smoke_eval"}
    assert (results_dir / "smoke_eval" / "predictions_test.csv").is_file()


def test_run_cv_dry_run_and_select_best_config(
    prepared_smoke_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cv.py",
            "--cv_index",
            str(prepared_smoke_project / "splits_cv" / "index.csv"),
            "--results_dir",
            str(tmp_path / "cv_results"),
            "--output_dir",
            str(tmp_path / "cv_checkpoints"),
            "--dry_run",
            "--max_runs",
            "1",
            "--",
            "--representation",
            "mfcc",
            "--backbone",
            "mobilenetv2",
            "--image_size",
            "32",
            "--normalization",
            "global",
            "--train_device_filter",
            "iphone",
            "--val_device_filter",
            "iphone",
            "--test_device_filter",
            "iphone",
        ],
    )
    run_cv_mod.main()
    out = capsys.readouterr().out
    assert "src.data.compute_stats" in out
    assert "src.training.train" in out

    summary_csv = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "run_name": "cv_iphone_a",
                "representation": "mfcc",
                "backbone": "mobilenetv2",
                "image_size": 224,
                "normalization": "global",
                "test_f1_pos": 0.40,
                "test_auprc": 0.45,
                "test_auroc": 0.55,
                "metric_scope": "overall",
                "eval_only": False,
                "train_device_filter": "iphone",
                "val_device_filter": "iphone",
                "test_device_filter": "iphone",
            },
            {
                "run_name": "cv_iphone_b",
                "representation": "gammatone",
                "backbone": "swinv2_tiny",
                "image_size": 256,
                "normalization": "global",
                "test_f1_pos": 0.65,
                "test_auprc": 0.60,
                "test_auroc": 0.70,
                "metric_scope": "overall",
                "eval_only": False,
                "train_device_filter": "iphone",
                "val_device_filter": "iphone",
                "test_device_filter": "iphone",
            },
            {
                "run_name": "final_iphone_model",
                "representation": "gammatone",
                "backbone": "swinv2_tiny",
                "image_size": 256,
                "normalization": "global",
                "test_f1_pos": 0.99,
                "test_auprc": 0.99,
                "test_auroc": 0.99,
                "metric_scope": "overall",
                "eval_only": False,
                "train_device_filter": "iphone",
                "val_device_filter": "iphone",
                "test_device_filter": "iphone",
            },
        ]
    ).to_csv(summary_csv, index=False)

    best_csv = tmp_path / "best.csv"
    all_csv = tmp_path / "all.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_best_config.py",
            "--summary_csv",
            str(summary_csv),
            "--output_csv",
            str(best_csv),
            "--all_csv",
            str(all_csv),
        ],
    )
    select_best_config_mod.main()

    best_df = pd.read_csv(best_csv)
    assert len(best_df) == 1
    assert best_df.iloc[0]["representation"] == "gammatone"
    assert best_df.iloc[0]["backbone"] == "swinv2_tiny"
