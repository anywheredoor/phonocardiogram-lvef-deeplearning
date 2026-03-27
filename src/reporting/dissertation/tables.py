#!/usr/bin/env python3
"""Table builders for dissertation summary outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Dict, List
import wave

import numpy as np
import pandas as pd

from src.reporting.dissertation.common import (
    DEVICE_ORDER,
    _assemble_pooled_test_predictions_for_best_within_model,
    _compute_binary_metrics_from_prob,
    _load_predictions_file,
    _patient_cluster_bootstrap_threshold_metrics,
    _pretty_device,
    _pretty_device_after_trained_on,
)

_RAW_DEVICE_LABELS = {
    "i": "iPhone",
    "a": "Android phone",
    "e": "Digital stethoscope",
}

_RAW_SITE_LABELS = {
    "A": "Aortic",
    "M": "Mitral",
    "P": "Pulmonic",
    "T": "Tricuspid",
}

_RAW_FILENAME_RE = re.compile(
    r"^(?P<device_code>[A-Za-z])Data(?P<patient_id>\d+)(?P<position>[A-Za-z])$"
)


def _pct_text(count: int, total: int) -> str:
    if total <= 0:
        return "NA"
    return f"{100.0 * count / total:.1f}%"


def _recordings_per_patient_text(
    patient_record_counts: List[int], patients_with_12: int, total_patients: int
) -> str:
    if not patient_record_counts:
        return "NA; no analyzable patient recordings were found"
    return (
        f"{min(patient_record_counts)}-{max(patient_record_counts)}; "
        f"{patients_with_12:,} of {total_patients:,} patients have 12 recordings"
    )


def _duration_text(duration_values: set[float]) -> str:
    if not duration_values:
        return "NA"
    if len(duration_values) == 1:
        return f"{next(iter(duration_values)):g} s for all recordings"
    return (
        f"{min(duration_values):g}-{max(duration_values):g} s across recordings"
    )


def _parse_raw_filename(stem: str) -> tuple[str, str] | None:
    match = _RAW_FILENAME_RE.match(stem)
    if not match:
        return None

    device_code = match.group("device_code").lower()
    site_code = match.group("position").upper()
    if device_code not in _RAW_DEVICE_LABELS or site_code not in _RAW_SITE_LABELS:
        return None

    return device_code, site_code


def build_raw_dataset_summary_table(
    lvef_csv_path: Path,
    heart_sounds_dir: Path,
) -> pd.DataFrame:
    lvef_df = pd.read_csv(lvef_csv_path)
    lvef_df["patient_id"] = lvef_df["patient_id"].astype(str)
    lvef_df["ef"] = pd.to_numeric(lvef_df["ef"], errors="coerce")
    labeled_df = lvef_df[lvef_df["ef"].notna()].copy().reset_index(drop=True)

    patient_record_counts: List[int] = []
    reduced_recordings = 0
    non_reduced_recordings = 0
    device_counts: Counter[str] = Counter()
    site_counts: Counter[str] = Counter()
    duration_values = set()
    sample_rates = set()
    channel_counts = set()

    analyzed_patient_count = 0
    reduced_patients = 0
    non_reduced_patients = 0

    for row in labeled_df.itertuples(index=False):
        patient_dir = heart_sounds_dir / row.patient_id
        if not patient_dir.is_dir():
            continue

        wav_paths = sorted(patient_dir.glob("*.wav"))
        if not wav_paths:
            continue

        valid_wav_paths = []
        for wav_path in wav_paths:
            parsed = _parse_raw_filename(wav_path.stem)
            if parsed is None:
                continue
            valid_wav_paths.append((wav_path, parsed))
        if not valid_wav_paths:
            continue

        analyzed_patient_count += 1
        patient_record_counts.append(len(valid_wav_paths))
        is_reduced = float(row.ef) <= 40.0
        if is_reduced:
            reduced_patients += 1
        else:
            non_reduced_patients += 1

        for wav_path, (device_code, site_code) in valid_wav_paths:
            device_counts[device_code] += 1
            site_counts[site_code] += 1
            if is_reduced:
                reduced_recordings += 1
            else:
                non_reduced_recordings += 1

            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rates.add(int(wav_file.getframerate()))
                channel_counts.add(int(wav_file.getnchannels()))
                duration_values.add(round(wav_file.getnframes() / wav_file.getframerate(), 6))

    total_patients = int(analyzed_patient_count)
    total_recordings = int(sum(patient_record_counts))
    patients_with_12 = int(sum(count == 12 for count in patient_record_counts))
    recordings_per_patient_text = _recordings_per_patient_text(
        patient_record_counts, patients_with_12, total_patients
    )
    duration_text = _duration_text(duration_values)

    if len(sample_rates) == 1:
        sample_rate_text = f"{next(iter(sample_rates)) / 1000:g} kHz"
    elif sample_rates:
        sample_rate_text = ", ".join(
            f"{sr / 1000:g} kHz" for sr in sorted(sample_rates)
        )
    else:
        sample_rate_text = "NA"
    if channel_counts == {1}:
        channel_text = "mono"
    elif channel_counts:
        channel_text = ", ".join(str(ch) for ch in sorted(channel_counts))
    else:
        channel_text = "NA"

    audio_format_text = (
        f"WAV, {channel_text}, {sample_rate_text} sampling rate"
        if sample_rates or channel_counts
        else "WAV (no readable recordings)"
    )

    rows = [
        ("Total patients", f"{total_patients:,}"),
        ("Total recordings", f"{total_recordings:,}"),
        (
            "Patients by class",
            f"{reduced_patients:,} reduced LVEF ({_pct_text(reduced_patients, total_patients)}); "
            f"{non_reduced_patients:,} non-reduced LVEF ({_pct_text(non_reduced_patients, total_patients)})",
        ),
        (
            "Recordings by class",
            f"{reduced_recordings:,} reduced LVEF ({_pct_text(reduced_recordings, total_recordings)}); "
            f"{non_reduced_recordings:,} non-reduced LVEF ({_pct_text(non_reduced_recordings, total_recordings)})",
        ),
        (
            "Recordings by device",
            "; ".join(
                f"{_RAW_DEVICE_LABELS[key]}: {device_counts[key]:,}"
                for key in ("i", "a", "e")
            ),
        ),
        (
            "Recordings by auscultation site",
            "; ".join(
                f"{_RAW_SITE_LABELS[key]}: {site_counts[key]:,}"
                for key in ("A", "M", "P", "T")
            ),
        ),
        (
            "Recordings per patient",
            recordings_per_patient_text,
        ),
        ("Duration per recording", duration_text),
        ("Audio format", audio_format_text),
    ]
    return pd.DataFrame(rows, columns=["Attribute", "Details"])


def build_shared_training_settings_table() -> pd.DataFrame:
    rows = [
        ("Loss function", "BCEWithLogitsLoss"),
        ("Optimizer", "AdamW"),
        ("Initial learning rate", "1e-4"),
        ("Weight decay", "1e-4"),
        ("Batch size", "32"),
        ("Learning-rate scheduler", "Cosine annealing"),
        ("Minimum learning rate", "1e-6"),
        ("Warmup epochs", "5"),
        ("Early stopping", "Patience 15 epochs"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Specification"])


def _format_pct_ci(point: float, low: float, high: float) -> str:
    if any(pd.isna(v) for v in [point, low, high]):
        return "NA"
    return f"{point * 100:.1f} ({low * 100:.1f}-{high * 100:.1f})"


def _resolve_threshold(default_threshold: object, pred_df: pd.DataFrame | None) -> float:
    if pd.notna(default_threshold):
        return float(default_threshold)
    if pred_df is not None and not pred_df.empty and "threshold" in pred_df.columns:
        thr = pd.to_numeric(pred_df["threshold"], errors="coerce").dropna()
        if not thr.empty:
            return float(thr.iloc[0])
    return 0.5


def build_pooled_test_performance_table(
    views: Dict[str, pd.DataFrame],
    results_run_dir: Path,
) -> pd.DataFrame:
    final_overall = views.get("final_overall", pd.DataFrame()).copy()
    pool_overall = views.get("pool_overall", pd.DataFrame()).copy()
    if final_overall.empty or pool_overall.empty or not results_run_dir.exists():
        return pd.DataFrame()

    rows: List[Dict[str, str]] = []

    pool_row = pool_overall.iloc[0]
    pool_run_name = str(pool_row["run_name"])
    pool_pred = _load_predictions_file(results_run_dir, pool_run_name, split="test")
    if pool_pred is not None and not pool_pred.empty:
        pool_threshold = _resolve_threshold(pool_row.get("tuned_threshold", np.nan), pool_pred)
        pool_metrics = _compute_binary_metrics_from_prob(
            pool_pred["label"].to_numpy(dtype=int),
            pool_pred["prob"].to_numpy(dtype=float),
            threshold=pool_threshold,
        )
        pool_boot = _patient_cluster_bootstrap_threshold_metrics(
            pool_pred,
            threshold=pool_threshold,
        )
        rows.append(
            {
                "Model": "Pooled-device model trained on all devices",
                "Tuned threshold": f"{pool_threshold:.2f}",
                "F1, % (95% CI)": _format_pct_ci(
                    pool_metrics["f1_pos"],
                    pool_boot["f1_pos_ci95_low"],
                    pool_boot["f1_pos_ci95_high"],
                ),
                "Sensitivity, % (95% CI)": _format_pct_ci(
                    pool_metrics["sensitivity"],
                    pool_boot["sensitivity_ci95_low"],
                    pool_boot["sensitivity_ci95_high"],
                ),
                "Specificity, % (95% CI)": _format_pct_ci(
                    pool_metrics["specificity"],
                    pool_boot["specificity_ci95_low"],
                    pool_boot["specificity_ci95_high"],
                ),
            }
        )

    for device in DEVICE_ORDER:
        row = final_overall[final_overall["train_device_filter"] == device]
        if row.empty:
            continue
        within_row = row.iloc[0]
        within_run_name = str(within_row["run_name"])
        pooled_test_pred = _assemble_pooled_test_predictions_for_best_within_model(
            views=views,
            results_run_dir=results_run_dir,
            within_run_name=within_run_name,
        )
        if pooled_test_pred is None or pooled_test_pred.empty:
            continue
        within_threshold = _resolve_threshold(
            within_row.get("tuned_threshold", np.nan),
            pooled_test_pred,
        )
        within_metrics = _compute_binary_metrics_from_prob(
            pooled_test_pred["label"].to_numpy(dtype=int),
            pooled_test_pred["prob"].to_numpy(dtype=float),
            threshold=within_threshold,
        )
        within_boot = _patient_cluster_bootstrap_threshold_metrics(
            pooled_test_pred,
            threshold=within_threshold,
        )
        rows.append(
            {
                "Model": (
                    "Best-config within-device model trained on "
                    f"{_pretty_device_after_trained_on(device)}"
                ),
                "Tuned threshold": f"{within_threshold:.2f}",
                "F1, % (95% CI)": _format_pct_ci(
                    within_metrics["f1_pos"],
                    within_boot["f1_pos_ci95_low"],
                    within_boot["f1_pos_ci95_high"],
                ),
                "Sensitivity, % (95% CI)": _format_pct_ci(
                    within_metrics["sensitivity"],
                    within_boot["sensitivity_ci95_low"],
                    within_boot["sensitivity_ci95_high"],
                ),
                "Specificity, % (95% CI)": _format_pct_ci(
                    within_metrics["specificity"],
                    within_boot["specificity_ci95_low"],
                    within_boot["specificity_ci95_high"],
                ),
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Tuned threshold",
            "F1, % (95% CI)",
            "Sensitivity, % (95% CI)",
            "Specificity, % (95% CI)",
        ],
    )
    return out.rename(columns={"Model": "Model evaluated on pooled test set"})


def _format_ci(point: float, low: float, high: float) -> str:
    if any(pd.isna(v) for v in [point, low, high]):
        return "NA"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"


def build_discrimination_vs_random_baseline_table(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_df.empty:
        return pd.DataFrame()

    order = {
        "final_within": 0,
        "cross_pairwise": 1,
        "pool_overall": 2,
        "final_within_pooled_test": 3,
    }
    table_df = bootstrap_df[
        bootstrap_df["evaluation_group"].isin(
            ["final_within", "final_within_pooled_test", "cross_pairwise", "pool_overall"]
        )
    ].copy()
    if table_df.empty:
        return pd.DataFrame()

    table_df["__group_sort"] = table_df["evaluation_group"].map(order).fillna(99).astype(int)
    table_df["__source_sort"] = pd.Categorical(
        table_df["source_device"].astype(str),
        categories=DEVICE_ORDER + ["pooled_all_devices"],
        ordered=True,
    )
    table_df["__target_sort"] = pd.Categorical(
        table_df["target_device"].astype(str),
        categories=DEVICE_ORDER + ["all_devices"],
        ordered=True,
    )
    table_df = (
        table_df.sort_values(["__group_sort", "__source_sort", "__target_sort"])
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "Model and evaluation setting": table_df["evaluation_label"].astype(str),
            "AUROC - 0.5 (95% CI)": [
                _format_ci(v - 0.5, lo, hi)
                for v, lo, hi in zip(
                    table_df["auroc"],
                    table_df["delta_auroc_vs_random_ci95_low"],
                    table_df["delta_auroc_vs_random_ci95_high"],
                )
            ],
            "95% CI above AUROC baseline?": table_df["auroc_above_random_95ci"].map(
                {True: "Yes", False: "No"}
            ),
        }
    )
    return out
