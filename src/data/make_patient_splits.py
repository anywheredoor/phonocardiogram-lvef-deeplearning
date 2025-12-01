#!/usr/bin/env python3
"""
Create patient-level stratified train/val/test splits from metadata.csv.

- Splits are done at PATIENT level (no leakage).
- Stratified on the binary label (EF <= 40 -> 1, else 0).
- Default split: ~65% train, 15% val, 20% test (by patients).

Outputs:
    - splits/patient_splits.csv        (one row per patient with split assignment)
    - splits/metadata_with_splits.csv  (metadata.csv + split column)
    - splits/metadata_train.csv
    - splits/metadata_val.csv
    - splits/metadata_test.csv

Example:
    python -m src.data.make_patient_splits \
        --metadata_csv metadata.csv \
        --output_dir splits \
        --test_size 0.20 \
        --val_size 0.15 \
        --seed 42
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make patient-level stratified train/val/test splits from metadata.csv"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="metadata.csv",
        help="Path to metadata CSV (default: metadata.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits",
        help="Directory to write split CSVs (default: splits)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Proportion of patients to assign to TEST (default: 0.20)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help=(
            "Proportion of patients to assign to VAL (of the TOTAL dataset). "
            "The rest goes to TRAIN. Default: 0.15 (i.e. ~65/15/20 train/val/test)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_metadata(metadata_csv: str) -> pd.DataFrame:
    if not os.path.isfile(metadata_csv):
        print(f"ERROR: metadata CSV not found at {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv)

    required_cols = {"patient_id", "ef", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: metadata CSV is missing required columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    return df


def make_patient_table(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse metadata to one row per patient with:
    - patient_id
    - ef (first)
    - label (first)
    """
    df_pat = (
        df_meta.groupby("patient_id")
        .agg({"ef": "first", "label": "first"})
        .reset_index()
    )

    # Sanity check: no conflicting labels per patient
    label_counts = df_meta.groupby("patient_id")["label"].nunique()
    bad = label_counts[label_counts > 1]
    if len(bad) > 0:
        print(
            "ERROR: Some patients have multiple labels in metadata. "
            "This should not happen.",
            file=sys.stderr,
        )
        print(bad.head(), file=sys.stderr)
        sys.exit(1)

    return df_pat


def stratified_patient_split(
    df_pat: pd.DataFrame, test_size: float, val_size: float, seed: int
):
    """
    Perform patient-level stratified split into train/val/test.

    test_size, val_size are proportions of TOTAL patients.
    The remaining goes to train.

    Strategy:
    1) Split into train_val vs test with stratification.
    2) Split train_val into train vs val with stratification.
    """
    if test_size <= 0 or val_size < 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            "Require: test_size > 0, val_size >= 0, and test_size + val_size < 1.0"
        )

    labels = df_pat["label"]

    # First: train_val vs test
    df_train_val, df_test = train_test_split(
        df_pat,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    # Now split train_val into train vs val.
    # val proportion relative to the remaining (train+val) set:
    rel_val_size = val_size / (1.0 - test_size)

    if rel_val_size > 0:
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=rel_val_size,
            stratify=df_train_val["label"],
            random_state=seed,
            shuffle=True,
        )
    else:
        df_train = df_train_val
        df_val = df_train_val.iloc[0:0].copy()  # empty

    return df_train, df_val, df_test


def main():
    args = parse_args()

    df_meta = load_metadata(args.metadata_csv)
    df_pat = make_patient_table(df_meta)

    print(f"Total patients: {len(df_pat)}")
    print("Label counts (patients):")
    print(df_pat["label"].value_counts())

    df_train, df_val, df_test = stratified_patient_split(
        df_pat, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    print(
        f"Train patients: {len(df_train)}, "
        f"Val patients: {len(df_val)}, "
        f"Test patients: {len(df_test)}"
    )

    # Add a 'split' column and combine
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_pat_splits = pd.concat([df_train, df_val, df_test], axis=0).reset_index(
        drop=True
    )

    # Sanity check: no overlap in patient_ids
    assert df_pat_splits["patient_id"].nunique() == len(df_pat), \
        "Some patients appear in multiple splits!"

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Save patient-level splits
    pat_split_path = os.path.join(args.output_dir, "patient_splits.csv")
    df_pat_splits.to_csv(pat_split_path, index=False)
    print(f"Wrote patient splits to: {pat_split_path}")

    # 2) Merge back to metadata to make per-example splits
    df_meta_splits = df_meta.merge(
        df_pat_splits[["patient_id", "split"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )

    # Sanity check: no missing splits
    assert df_meta_splits["split"].isna().sum() == 0

    # Save full metadata with split column (handy for debugging)
    full_meta_path = os.path.join(args.output_dir, "metadata_with_splits.csv")
    df_meta_splits.to_csv(full_meta_path, index=False)
    print(f"Wrote metadata with split column to: {full_meta_path}")

    # Save split-specific metadata CSVs
    for split in ["train", "val", "test"]:
        split_df = df_meta_splits[df_meta_splits["split"] == split].reset_index(
            drop=True
        )
        out_path = os.path.join(args.output_dir, f"metadata_{split}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"{split.capitalize()} examples: {len(split_df)} -> {out_path}")


if __name__ == "__main__":
    main()
