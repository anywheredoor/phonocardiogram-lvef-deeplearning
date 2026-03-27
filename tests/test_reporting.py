from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.make_dissertation_outputs import _resolve_summary_csv_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_report_script_resolves_summary_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "summary.csv").write_text("run_name\n", encoding="utf-8")

    assert _resolve_summary_csv_path("summary.csv") == "results/summary.csv"


def test_dissertation_reporting_script_smoke(tmp_path: Path) -> None:
    required_paths = [
        REPO_ROOT / "summary.csv",
        REPO_ROOT / "lvef.csv",
        REPO_ROOT / "heart_sounds",
        REPO_ROOT / "results",
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("Repository reporting artifacts are not available in this checkout.")

    output_dir = tmp_path / "report_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_dissertation_outputs.py",
            "--output_dir",
            str(output_dir),
            "--results_run_dir",
            "results",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    assert (output_dir / "README.md").is_file()
    assert (
        output_dir / "tables" / "table_01_dataset_summary_before_preprocessing.csv"
    ).is_file()
    assert (
        output_dir / "figures" / "figure_01_backbone_parameter_sizes_used_in_experiments.png"
    ).is_file()
