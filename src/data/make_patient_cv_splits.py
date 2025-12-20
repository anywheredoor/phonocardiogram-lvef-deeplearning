#!/usr/bin/env python3
"""
Create repeated, stratified patient-level CV splits from metadata.csv.

Outputs:
  - output_dir/repeat_XX/fold_YY/metadata_{train,val,test}.csv
  - output_dir/repeat_XX/fold_YY/patient_splits.csv
  - output_dir/index.csv (paths + label counts per fold)
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make repeated stratified patient-level CV splits."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="metadata.csv",
        help="Path to metadata CSV (default: metadata.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits/cv",
        help="Directory to write CV splits (default: splits/cv).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of repeats with different shuffles (default: 1).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation proportion within train+val (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_metadata(metadata_csv: str) -> pd.DataFrame:
    if not os.path.isfile(metadata_csv):
        print(f"ERROR: metadata CSV not found at {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv, dtype={"patient_id": str})
    required_cols = {"patient_id", "ef", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: metadata CSV missing required columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def make_patient_table(df_meta: pd.DataFrame) -> pd.DataFrame:
    df_pat = (
        df_meta.groupby("patient_id")
        .agg({"ef": "first", "label": "first"})
        .reset_index()
    )
    label_counts = df_meta.groupby("patient_id")["label"].nunique()
    conflicts = label_counts[label_counts > 1]
    if len(conflicts) > 0:
        print(
            "ERROR: Some patients have multiple labels in metadata.",
            file=sys.stderr,
        )
        print(conflicts.head(), file=sys.stderr)
        sys.exit(1)
    return df_pat


def make_stratified_folds(
    df_pat: pd.DataFrame, n_splits: int, seed: int
) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    for label in sorted(df_pat["label"].unique()):
        pids = df_pat[df_pat["label"] == label]["patient_id"].tolist()
        rng.shuffle(pids)
        for idx, pid in enumerate(pids):
            folds[idx % n_splits].append(pid)

    all_ids = [pid for fold in folds for pid in fold]
    if len(all_ids) != len(df_pat):
        raise RuntimeError("Fold assignment dropped or duplicated patient IDs.")
    return folds


def safe_train_val_split(
    df_pat: pd.DataFrame, val_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_size <= 0:
        return df_pat, df_pat.iloc[0:0].copy()

    labels = df_pat["label"]
    stratify = None
    if labels.nunique() > 1 and labels.value_counts().min() >= 2:
        stratify = labels

    try:
        train_df, val_df = train_test_split(
            df_pat,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            df_pat,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return train_df, val_df


def label_counts(df_pat: pd.DataFrame) -> Dict[str, int]:
    counts = df_pat["label"].value_counts().to_dict()
    return {
        "n_patients": int(len(df_pat)),
        "n_pos": int(counts.get(1, 0)),
        "n_neg": int(counts.get(0, 0)),
    }


def main() -> None:
    args = parse_args()

    if args.n_splits < 2:
        print("ERROR: n_splits must be >= 2.", file=sys.stderr)
        sys.exit(1)
    if args.n_repeats < 1:
        print("ERROR: n_repeats must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.val_size < 0 or args.val_size >= 1.0:
        print("ERROR: val_size must be in [0, 1).", file=sys.stderr)
        sys.exit(1)

    df_meta = load_metadata(args.metadata_csv)
    df_pat = make_patient_table(df_meta)

    label_counts_overall = df_pat["label"].value_counts()
    if label_counts_overall.min() < args.n_splits:
        print(
            "Warning: Some folds may have no examples of a class "
            "because a class has fewer patients than n_splits."
        )

    index_rows = []
    os.makedirs(args.output_dir, exist_ok=True)

    for repeat in range(args.n_repeats):
        folds = make_stratified_folds(df_pat, args.n_splits, args.seed + repeat)

        for fold_idx in range(args.n_splits):
            test_pids = set(folds[fold_idx])
            trainval_pids = set(df_pat["patient_id"]) - test_pids

            df_test_pat = df_pat[df_pat["patient_id"].isin(test_pids)].reset_index(
                drop=True
            )
            df_trainval_pat = df_pat[
                df_pat["patient_id"].isin(trainval_pids)
            ].reset_index(drop=True)

            df_train_pat, df_val_pat = safe_train_val_split(
                df_trainval_pat, args.val_size, args.seed + repeat + fold_idx
            )

            fold_dir = os.path.join(
                args.output_dir, f"repeat_{repeat:02d}", f"fold_{fold_idx:02d}"
            )
            os.makedirs(fold_dir, exist_ok=True)

            df_train = df_meta[df_meta["patient_id"].isin(df_train_pat["patient_id"])]
            df_val = df_meta[df_meta["patient_id"].isin(df_val_pat["patient_id"])]
            df_test = df_meta[df_meta["patient_id"].isin(df_test_pat["patient_id"])]

            train_csv = os.path.join(fold_dir, "metadata_train.csv")
            val_csv = os.path.join(fold_dir, "metadata_val.csv")
            test_csv = os.path.join(fold_dir, "metadata_test.csv")

            df_train.to_csv(train_csv, index=False)
            df_val.to_csv(val_csv, index=False)
            df_test.to_csv(test_csv, index=False)

            df_pat_splits = pd.concat(
                [
                    df_train_pat.assign(split="train"),
                    df_val_pat.assign(split="val"),
                    df_test_pat.assign(split="test"),
                ],
                axis=0,
            ).reset_index(drop=True)
            df_pat_splits.to_csv(
                os.path.join(fold_dir, "patient_splits.csv"), index=False
            )

            df_meta_splits = pd.concat(
                [
                    df_train.assign(split="train"),
                    df_val.assign(split="val"),
                    df_test.assign(split="test"),
                ],
                axis=0,
            ).reset_index(drop=True)
            df_meta_splits.to_csv(
                os.path.join(fold_dir, "metadata_with_splits.csv"), index=False
            )

            counts_train = label_counts(df_train_pat)
            counts_val = label_counts(df_val_pat)
            counts_test = label_counts(df_test_pat)

            index_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold_idx,
                    "train_csv": train_csv,
                    "val_csv": val_csv,
                    "test_csv": test_csv,
                    "train_n_patients": counts_train["n_patients"],
                    "train_n_pos": counts_train["n_pos"],
                    "train_n_neg": counts_train["n_neg"],
                    "val_n_patients": counts_val["n_patients"],
                    "val_n_pos": counts_val["n_pos"],
                    "val_n_neg": counts_val["n_neg"],
                    "test_n_patients": counts_test["n_patients"],
                    "test_n_pos": counts_test["n_pos"],
                    "test_n_neg": counts_test["n_neg"],
                }
            )

            print(
                f"Repeat {repeat:02d}, Fold {fold_idx:02d} -> "
                f"train {counts_train['n_patients']} / "
                f"val {counts_val['n_patients']} / "
                f"test {counts_test['n_patients']}"
            )

    index_path = os.path.join(args.output_dir, "index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    print(f"Wrote CV index to {index_path}")


if __name__ == "__main__":
    main()
