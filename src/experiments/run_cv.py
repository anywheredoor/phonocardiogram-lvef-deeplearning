#!/usr/bin/env python3
"""
Run training across CV splits defined by splits/cv/index.csv.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV training sweeps.")
    parser.add_argument(
        "--cv_index",
        type=str,
        default="splits/cv/index.csv",
        help="Path to CV index CSV (default: splits/cv/index.csv).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of repeats to run.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of folds to run.",
    )
    parser.add_argument(
        "--run_name_format",
        type=str,
        default="cv_r{repeat:02d}_f{fold:02d}_{backbone}_{representation}",
        help="Run name format string.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints).",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip runs that already have metrics.json in results_dir.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on number of runs.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the training script.",
    )
    return parser.parse_args()


def _find_arg_value(args: List[str], key: str, default: Optional[str] = "") -> str:
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default or ""


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.cv_index):
        print(f"ERROR: CV index not found at {args.cv_index}")
        sys.exit(1)

    df = pd.read_csv(args.cv_index)
    required_cols = {"repeat", "fold", "train_csv", "val_csv", "test_csv"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CV index missing columns: {missing}")
        sys.exit(1)

    if args.repeat is not None:
        df = df[df["repeat"].isin(args.repeat)]
    if args.fold is not None:
        df = df[df["fold"].isin(args.fold)]

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    backbone = _find_arg_value(extra_args, "--backbone", "backbone")
    representation = _find_arg_value(extra_args, "--representation", "rep")

    run_count = 0
    for _, row in df.iterrows():
        repeat = int(row["repeat"])
        fold = int(row["fold"])
        train_csv = row["train_csv"]
        val_csv = row["val_csv"]
        test_csv = row["test_csv"]

        run_name = args.run_name_format.format(
            repeat=repeat, fold=fold, backbone=backbone, representation=representation
        )

        metrics_path = os.path.join(args.results_dir, run_name, "metrics.json")
        if args.skip_if_exists and os.path.exists(metrics_path):
            print(f"Skipping existing run: {run_name}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "src.training.train",
            "--train_csv",
            train_csv,
            "--val_csv",
            val_csv,
            "--test_csv",
            test_csv,
            "--run_name",
            run_name,
            "--results_dir",
            args.results_dir,
            "--output_dir",
            args.output_dir,
        ] + extra_args

        cmd_str = " ".join(cmd)
        print(f"[{run_count:03d}] {cmd_str}")

        if not args.dry_run:
            subprocess.run(cmd, check=True)

        run_count += 1
        if args.max_runs is not None and run_count >= args.max_runs:
            print("Reached --max_runs limit.")
            return


if __name__ == "__main__":
    main()
