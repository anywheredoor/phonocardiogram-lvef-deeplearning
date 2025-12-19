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
    parser = argparse.ArgumentParser(description="Run CV training jobs.")
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
        "--skip_compute_stats",
        action="store_true",
        help=(
            "Skip per-fold TF stats computation. Only use if you provide "
            "--tf_stats_json via extra args or use cached tensors."
        ),
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


def _find_arg_list(args: List[str], key: str) -> List[str]:
    if key not in args:
        return []
    idx = args.index(key) + 1
    values: List[str] = []
    while idx < len(args) and not args[idx].startswith("--"):
        values.append(args[idx])
        idx += 1
    return values


def _slug_list(values: List[str]) -> str:
    if not values:
        return "all"
    return "-".join(str(v) for v in values)


def _sanitize(value: str) -> str:
    safe = value.replace(os.sep, "-").replace(" ", "")
    for ch in [",", "[", "]", "(", ")", "'", '"', ":", ";"]:
        safe = safe.replace(ch, "")
    return safe


def _resolve_cached_csv(path: str, representation: str) -> Optional[str]:
    base = os.path.basename(path)
    directory = os.path.dirname(path)
    if base.startswith(f"cached_{representation}_"):
        return path
    if base.startswith("cached_metadata_"):
        return os.path.join(directory, f"cached_{representation}_{base[len('cached_'):]}")
    if base.startswith("metadata_"):
        return os.path.join(directory, f"cached_{representation}_{base}")
    if base.startswith("cached_"):
        return None
    return None


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
    representation = _find_arg_value(extra_args, "--representation", "mfcc")
    normalization = _find_arg_value(extra_args, "--normalization", "global")
    tf_stats_arg = _find_arg_value(extra_args, "--tf_stats_json", "")
    use_cache = "--use_cache" in extra_args

    train_device_filter = _find_arg_list(extra_args, "--train_device_filter")
    device_filter = train_device_filter or _find_arg_list(extra_args, "--device_filter")
    train_position_filter = _find_arg_list(extra_args, "--train_position_filter")
    position_filter = train_position_filter or _find_arg_list(
        extra_args, "--position_filter"
    )

    sample_rate = _find_arg_value(extra_args, "--sample_rate", "2000")
    fixed_duration = _find_arg_value(extra_args, "--fixed_duration", "4.0")
    image_size = _find_arg_value(extra_args, "--image_size", "224")

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

        fold_extra_args = list(extra_args)

        compute_stats = (
            not args.skip_compute_stats
            and not use_cache
            and normalization != "none"
            and not tf_stats_arg
        )
        if use_cache:
            expected = [
                _resolve_cached_csv(train_csv, representation),
                _resolve_cached_csv(val_csv, representation),
                _resolve_cached_csv(test_csv, representation),
            ]
            if any(p is None for p in expected) or any(
                p is not None and not os.path.exists(p) for p in expected
            ):
                print(
                    "ERROR: --use_cache requires cached CSVs for each fold. "
                    "Precompute caches per fold or disable --use_cache."
                )
                sys.exit(1)
        if compute_stats:
            fold_dir = os.path.dirname(train_csv)
            device_slug = _slug_list(device_filter)
            position_slug = _slug_list(position_filter)
            stats_tag = (
                f"{representation}_{normalization}"
                f"_dev{device_slug}_pos{position_slug}"
                f"_sr{sample_rate}_dur{fixed_duration}_im{image_size}"
            )
            stats_tag = _sanitize(stats_tag.replace(".", "p"))
            stats_path = os.path.join(fold_dir, f"tf_stats_{stats_tag}.json")
            if not os.path.isfile(stats_path):
                stats_cmd = [
                    sys.executable,
                    "-m",
                    "src.data.compute_stats",
                    "--train_csv",
                    train_csv,
                    "--representations",
                    representation,
                    "--output_json",
                    stats_path,
                ]
                if normalization == "per_device":
                    stats_cmd.append("--per_device")
                if device_filter:
                    stats_cmd.append("--device_filter")
                    stats_cmd.extend(device_filter)
                if position_filter:
                    stats_cmd.append("--position_filter")
                    stats_cmd.extend(position_filter)
                stats_cmd.extend(["--sample_rate", sample_rate])
                stats_cmd.extend(["--fixed_duration", fixed_duration])
                stats_cmd.extend(["--image_size", image_size])
                print(f"[{run_count:03d}] " + " ".join(stats_cmd))
                if not args.dry_run:
                    subprocess.run(stats_cmd, check=True)
            fold_extra_args.extend(["--tf_stats_json", stats_path])

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
        ] + fold_extra_args

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
