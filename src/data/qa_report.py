#!/usr/bin/env python3
"""
Generate a dataset QA report from metadata.csv and audio files.

Outputs a JSON summary and optionally a per-record CSV with audio stats.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a QA report for PCG metadata and audio files."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="metadata.csv",
        help="Path to metadata CSV (default: metadata.csv).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="qa_report.json",
        help="Output JSON report path (default: qa_report.json).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-record audio stats CSV.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Optional training split CSV to compute pos_weight (neg/pos).",
    )
    parser.add_argument(
        "--fixed_duration",
        type=float,
        default=None,
        help="Fixed duration in seconds to flag short recordings.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on number of files to scan (random sample).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--max_list",
        type=int,
        default=20,
        help="Max number of example paths to include in the report.",
    )
    return parser.parse_args()


def _summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _label_from_ef(ef: float) -> int:
    return int(float(ef) <= 40.0)


def _safe_value_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _df_to_nested_dict(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    if df is None:
        return {}
    data: Dict[str, Dict[str, int]] = {}
    for idx, row in df.iterrows():
        data[str(idx)] = {str(col): int(row[col]) for col in df.columns}
    return data


def _compute_pos_weight_from_csv(csv_path: str):
    if not os.path.isfile(csv_path):
        return None, None, None, f"train_csv not found: {csv_path}"
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
    elif "ef" in df.columns:
        labels = pd.to_numeric(df["ef"], errors="coerce").apply(
            lambda v: _label_from_ef(v) if pd.notna(v) else np.nan
        )
    else:
        return None, None, None, "train_csv missing label/ef columns"

    labels = labels.dropna().astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg, "train_csv has only one class"
    return float(n_neg / n_pos), n_pos, n_neg, ""


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.metadata_csv):
        print(f"ERROR: metadata CSV not found at {args.metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.metadata_csv)
    required_cols = {"patient_id", "ef", "label", "device", "path"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: metadata CSV missing required columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    sampled = False
    if args.max_files is not None and args.max_files < len(df):
        df = df.sample(n=args.max_files, random_state=args.seed).reset_index(drop=True)
        sampled = True

    df["ef"] = pd.to_numeric(df["ef"], errors="coerce")
    valid_ef = df["ef"].notna()
    df["label_from_ef"] = np.where(valid_ef, df["ef"].apply(_label_from_ef), np.nan)
    label_mismatch = (
        df.loc[valid_ef, "label"].astype(int)
        != df.loc[valid_ef, "label_from_ef"].astype(int)
    ).sum()

    label_conflicts = (
        df.groupby("patient_id")["label"].nunique().sort_values(ascending=False)
    )
    conflicting_patients = label_conflicts[label_conflicts > 1].index.tolist()

    ef_conflicts = (
        df.groupby("patient_id")["ef"].nunique().sort_values(ascending=False)
    )
    conflicting_efs = ef_conflicts[ef_conflicts > 1].index.tolist()

    record_counts = df.groupby("patient_id").size().tolist()

    device_label_counts = (
        df.groupby(["device", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    position_label_counts = None
    if "position" in df.columns:
        position_label_counts = (
            df.groupby(["position", "label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

    missing_paths: List[str] = []
    unreadable_paths: List[str] = []
    durations: List[float] = []
    sample_rates: List[int] = []
    durations_by_device: Dict[str, List[float]] = {}

    per_record_rows = []

    for _, row in df.iterrows():
        path = row["path"]
        device = row["device"]
        exists = os.path.isfile(path)
        duration = None
        samplerate = None
        frames = None
        error = ""

        if not exists:
            missing_paths.append(path)
        else:
            try:
                info = sf.info(path)
                frames = int(info.frames)
                samplerate = int(info.samplerate)
                if samplerate > 0:
                    duration = frames / samplerate
                    durations.append(duration)
                    sample_rates.append(samplerate)
                    durations_by_device.setdefault(device, []).append(duration)
            except Exception as exc:
                unreadable_paths.append(path)
                error = str(exc)

        if args.output_csv:
            per_record_rows.append(
                {
                    "path": path,
                    "device": device,
                    "exists": exists,
                    "duration_sec": duration,
                    "sample_rate": samplerate,
                    "frames": frames,
                    "error": error,
                }
            )

    short_count = None
    short_fraction = None
    if args.fixed_duration is not None and durations:
        short_count = int(np.sum(np.asarray(durations) < args.fixed_duration))
        short_fraction = float(short_count / len(durations))

    report: Dict[str, Any] = {
        "metadata_csv": args.metadata_csv,
        "sampled": sampled,
        "recording_count": int(len(df)),
        "patient_count": int(df["patient_id"].nunique()),
        "missing_ef_count": int((~valid_ef).sum()),
        "label_counts_recordings": _safe_value_counts(df["label"]),
        "label_counts_patients": _safe_value_counts(
            df.groupby("patient_id")["label"].first()
        ),
        "label_mismatch_count": int(label_mismatch),
        "conflicting_label_patients_count": int(len(conflicting_patients)),
        "conflicting_ef_patients_count": int(len(conflicting_efs)),
        "device_counts": _safe_value_counts(df["device"]),
        "device_label_counts": _df_to_nested_dict(device_label_counts),
        "position_label_counts": _df_to_nested_dict(position_label_counts),
        "missing_path_count": int(len(missing_paths)),
        "unreadable_path_count": int(len(unreadable_paths)),
        "missing_path_examples": missing_paths[: args.max_list],
        "unreadable_path_examples": unreadable_paths[: args.max_list],
        "duration_summary_sec": _summarize(durations),
        "duration_summary_by_device_sec": {
            device: _summarize(vals) for device, vals in durations_by_device.items()
        },
        "sample_rate_counts": _safe_value_counts(pd.Series(sample_rates)),
        "recordings_per_patient_summary": _summarize(record_counts),
        "fixed_duration_sec": args.fixed_duration,
        "shorter_than_fixed_duration_count": short_count,
        "shorter_than_fixed_duration_fraction": short_fraction,
        "duplicate_path_count": int(df["path"].duplicated().sum()),
    }

    train_csv = args.train_csv
    inferred_train = False
    if train_csv is None:
        candidate = os.path.join("splits", "metadata_train.csv")
        if os.path.isfile(candidate):
            train_csv = candidate
            inferred_train = True

    if train_csv is not None:
        pos_weight, n_pos, n_neg, err = _compute_pos_weight_from_csv(train_csv)
        report["pos_weight_train_csv"] = train_csv
        report["pos_weight_train_inferred"] = inferred_train
        report["pos_weight_train"] = pos_weight
        report["pos_weight_train_n_pos"] = n_pos
        report["pos_weight_train_n_neg"] = n_neg
        if err:
            report["pos_weight_train_error"] = err

    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote QA report to {args.output_json}")

    if args.output_csv:
        pd.DataFrame(per_record_rows).to_csv(args.output_csv, index=False)
        print(f"Wrote per-record audio stats to {args.output_csv}")


if __name__ == "__main__":
    main()
