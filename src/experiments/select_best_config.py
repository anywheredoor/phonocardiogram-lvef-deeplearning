#!/usr/bin/env python3
"""
Select the best config per device from results/summary.csv.

By default, ranks configs by mean test F1_pos, then mean AUPRC, then AUROC.
"""

import argparse
import os
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best config per device from summary.csv."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/summary.csv",
        help="Path to summary.csv (default: results/summary.csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/selection/best_config_per_device.csv",
        help="Output CSV for best config per device.",
    )
    parser.add_argument(
        "--all_csv",
        type=str,
        default="results/selection/config_summary_by_device.csv",
        help="Output CSV for aggregated configs (per device).",
    )
    parser.add_argument(
        "--metric_scope",
        type=str,
        default="overall",
        help="If present, filter metric_scope to this value (default: overall).",
    )
    parser.add_argument(
        "--include_eval_only",
        action="store_true",
        help="Include eval_only rows when selecting the best config.",
    )
    parser.add_argument(
        "--allow_cross_device",
        action="store_true",
        help="Allow cross-device rows (train/val/test device mismatch).",
    )
    return parser.parse_args()


def _resolve_summary_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if path == "results/summary.csv" and os.path.isfile("summary.csv"):
        print("Warning: results/summary.csv not found; using ./summary.csv")
        return "summary.csv"
    raise FileNotFoundError(f"summary.csv not found: {path}")


def _filter_eval_only(df: pd.DataFrame, include_eval_only: bool) -> pd.DataFrame:
    if include_eval_only or "eval_only" not in df.columns:
        return df
    col = df["eval_only"]
    if col.dtype == object:
        mask = col.astype(str).str.lower().isin(["false", "0", "no"])
    else:
        mask = col == False
    return df[mask]


def _require_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    summary_path = _resolve_summary_path(args.summary_csv)
    df = pd.read_csv(summary_path)

    _require_columns(
        df,
        [
            "run_name",
            "representation",
            "backbone",
            "image_size",
            "normalization",
            "test_f1_pos",
            "test_auprc",
            "test_auroc",
        ],
    )

    if args.metric_scope and "metric_scope" in df.columns:
        df = df[df["metric_scope"] == args.metric_scope]

    df = _filter_eval_only(df, args.include_eval_only)

    if not args.allow_cross_device:
        _require_columns(
            df,
            ["train_device_filter", "val_device_filter", "test_device_filter"],
        )
        mask = (
            df["train_device_filter"].notna()
            & (df["train_device_filter"] == df["val_device_filter"])
            & (df["train_device_filter"] == df["test_device_filter"])
        )
        df = df[mask]

    if df.empty:
        print("No rows left after filtering. Check your filters/summary.csv.")
        sys.exit(1)

    group_cols = [
        "train_device_filter",
        "representation",
        "backbone",
        "image_size",
        "normalization",
    ]
    _require_columns(df, group_cols)

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            folds=("run_name", "count"),
            mean_f1=("test_f1_pos", "mean"),
            sd_f1=("test_f1_pos", "std"),
            mean_auprc=("test_auprc", "mean"),
            mean_auroc=("test_auroc", "mean"),
        )
        .sort_values(
            ["train_device_filter", "mean_f1", "mean_auprc", "mean_auroc"],
            ascending=[True, False, False, False],
        )
    )

    best_rows = (
        agg.sort_values(
            ["train_device_filter", "mean_f1", "mean_auprc", "mean_auroc"],
            ascending=[True, False, False, False],
        )
        .groupby("train_device_filter", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    best_rows.to_csv(args.output_csv, index=False)
    print(f"Wrote best configs to: {args.output_csv}")

    if args.all_csv:
        os.makedirs(os.path.dirname(args.all_csv) or ".", exist_ok=True)
        agg.to_csv(args.all_csv, index=False)
        print(f"Wrote aggregated configs to: {args.all_csv}")


if __name__ == "__main__":
    main()
