#!/usr/bin/env python3
"""
Precompute MFCC or gammatone spectrograms and cache them as .pt tensors.

For each input split CSV, this script:
    - Uses PCGDataset to build the final [3, H, W] tensor (normalised).
    - Saves tensors to cache/<representation>/<split>/<patientid_basename>.pt.
    - Writes a new CSV next to the input with an extra 'cache_path' column.
    - Optional device/position filters help align cached data with specific experiments.

Run from repo root, e.g.:
    python -m src.data.precompute_cache --representation mfcc
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from src.datasets.pcg_dataset import PCGDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute TF representations and cache as .pt tensors."
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["mfcc", "gammatone"],
        required=True,
        help="Time-frequency representation to precompute.",
    )
    parser.add_argument(
        "--tf_stats_json",
        type=str,
        default="tf_stats.json",
        help="Path to TF stats JSON (default: tf_stats.json)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (H=W) used in PCGDataset (default: 224)",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="cache",
        help="Root directory to store cached tensors (default: cache)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=[
            "splits/metadata_train.csv",
            "splits/metadata_val.csv",
            "splits/metadata_test.csv",
        ],
        help="List of split CSVs to process.",
    )
    parser.add_argument(
        "--device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to include when caching.",
    )
    parser.add_argument(
        "--position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to include when caching.",
    )
    return parser.parse_args()


def precompute_for_split(
    csv_path,
    representation,
    mean,
    std,
    image_size,
    cache_root,
    device_filter=None,
    position_filter=None,
):
    """
    Precompute cached tensors for a single split CSV and write a new CSV with 'cache_path'.
    """
    # Split name like "train", "val", "test"
    base = os.path.basename(csv_path)  # e.g. metadata_train.csv
    split_name = base.replace("metadata_", "").replace(".csv", "")  # train

    print(f"\n=== Precomputing for split: {split_name}, representation: {representation} ===")

    # Use your existing PCGDataset to compute the final [3, H, W] tensors
    ds = PCGDataset(
        csv_path=csv_path,
        representation=representation,
        mean=mean,
        std=std,
        image_size=image_size,
        device_filter=device_filter,
        position_filter=position_filter,
    )

    df = ds.df.copy()  # keep same rows / order
    cache_dir = os.path.join(cache_root, representation, split_name)
    os.makedirs(cache_dir, exist_ok=True)

    cache_paths = []

    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc=f"{representation}-{split_name}"):
            img, label, meta = ds[idx]  # img: [3, H, W] tensor

            patient_id = meta["patient_id"]
            wav_path = meta["path"]
            wav_base = os.path.basename(wav_path).replace(".wav", "")

            fname = f"{patient_id}_{wav_base}.pt"
            out_path = os.path.join(cache_dir, fname)

            torch.save(img, out_path)
            cache_paths.append(out_path)

    df["cache_path"] = cache_paths

    out_csv = os.path.join(
        os.path.dirname(csv_path),
        f"cached_{base}",  # e.g. cached_metadata_train.csv
    )
    df.to_csv(out_csv, index=False)

    print(f"Saved cached split CSV to: {out_csv}")
    print(f"Cached tensors in: {cache_dir}")


def main():
    args = parse_args()

    # Load TF stats
    with open(args.tf_stats_json, "r") as f:
        stats = json.load(f)

    if args.representation not in stats:
        raise ValueError(
            f"Representation '{args.representation}' not found in {args.tf_stats_json}. "
            f"Available: {list(stats.keys())}"
        )

    rep_stats = stats[args.representation]
    mean = rep_stats["mean"]
    std = rep_stats["std"]

    print(f"Using stats for {args.representation}: mean={mean:.6f}, std={std:.6f}")

    for csv_path in args.splits:
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} does not exist, skipping.")
            continue
        precompute_for_split(
            csv_path=csv_path,
            representation=args.representation,
            mean=mean,
            std=std,
            image_size=args.image_size,
            cache_root=args.cache_root,
            device_filter=args.device_filter,
            position_filter=args.position_filter,
        )


if __name__ == "__main__":
    main()
