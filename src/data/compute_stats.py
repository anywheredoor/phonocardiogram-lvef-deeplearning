#!/usr/bin/env python3
"""
Compute dataset-level mean/std for time-frequency images.

This walks through the training split with PCGDataset (on-the-fly features) and
records a single global mean and std per representation. These stats are used
for z-score normalisation during training and during cache precomputation.
Use --device_filter / --position_filter to restrict the computation to a
subset of devices or auscultation positions.
Use --per_device to also compute per-device stats.

Run from repo root, e.g.:
    python -m src.data.compute_stats --representations mfcc gammatone
"""

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.pcg_dataset import PCGDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean/std for TF images from PCGDataset."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="splits/metadata_train.csv",
        help="Path to training split CSV (default: splits/metadata_train.csv)",
    )
    parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        default=["mfcc", "gammatone"],
        help=(
            "List of representations to compute stats for. "
            "Choices: 'mfcc', 'gammatone'. Default: both."
        ),
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=2000,
        help="Target sample rate used in PCGDataset (default: 2000)",
    )
    parser.add_argument(
        "--fixed_duration",
        type=float,
        default=4.0,
        help="Fixed duration in seconds used in PCGDataset (default: 4.0)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (H=W) used in PCGDataset (default: 224)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for iterating the dataset (default: 16)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="tf_stats.json",
        help="Output JSON file to save stats (default: tf_stats.json)",
    )
    parser.add_argument(
        "--device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to include when computing stats.",
    )
    parser.add_argument(
        "--position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to include when computing stats.",
    )
    parser.add_argument(
        "--per_device",
        action="store_true",
        help="Also compute per-device mean/std from the training split.",
    )
    return parser.parse_args()


def compute_mean_std_for_rep(
    train_csv: str,
    representation: str,
    sample_rate: int,
    fixed_duration: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device_filter=None,
    position_filter=None,
):
    """
    Compute global mean and std for a single representation
    using PCGDataset on the training split.
    """
    print(f"\n=== Computing stats for representation: {representation} ===")

    ds = PCGDataset(
        csv_path=train_csv,
        representation=representation,
        sample_rate=sample_rate,
        fixed_duration=fixed_duration,
        image_size=image_size,
        mean=None,  # very important: no normalisation here
        std=None,
        device_filter=device_filter,
        position_filter=position_filter,
        clamp=False,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    n_pixels = 0
    sum_ = 0.0
    sum_sq = 0.0

    for imgs, labels, meta in tqdm(loader, desc=f"{representation}"):
        # imgs: [B, 3, H, W]
        imgs = imgs.to(torch.float32)
        # Flatten all dimensions except batch
        imgs = imgs.reshape(imgs.size(0), -1)  # [B, C*H*W]
        sum_ += imgs.sum().item()
        sum_sq += (imgs ** 2).sum().item()
        n_pixels += imgs.numel()

    mean = sum_ / n_pixels
    var = (sum_sq / n_pixels) - (mean ** 2)
    std = var ** 0.5

    print(f"{representation} -> mean = {mean:.6f}, std = {std:.6f}")
    return float(mean), float(std)


def main():
    args = parse_args()

    stats = {}
    df_devices = None
    if args.per_device:
        df_devices = pd.read_csv(args.train_csv, dtype={"patient_id": str})
        if args.device_filter is not None:
            df_devices = df_devices[df_devices["device"].isin(args.device_filter)]
        if args.position_filter is not None and "position" in df_devices.columns:
            df_devices = df_devices[df_devices["position"].isin(args.position_filter)]
        df_devices = df_devices.reset_index(drop=True)

    for rep in args.representations:
        if rep not in ("mfcc", "gammatone"):
            print(f"Skipping unknown representation: {rep}")
            continue

        mean, std = compute_mean_std_for_rep(
            train_csv=args.train_csv,
            representation=rep,
            sample_rate=args.sample_rate,
            fixed_duration=args.fixed_duration,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_filter=args.device_filter,
            position_filter=args.position_filter,
        )
        stats[rep] = {"mean": mean, "std": std}

        if args.per_device:
            if df_devices is None or len(df_devices) == 0:
                print("No devices available for per-device stats; skipping.")
                continue
            stats[rep]["per_device"] = {}
            devices = sorted(df_devices["device"].unique().tolist())
            for device in devices:
                print(f"\n--- Per-device stats: {rep} / {device} ---")
                d_mean, d_std = compute_mean_std_for_rep(
                    train_csv=args.train_csv,
                    representation=rep,
                    sample_rate=args.sample_rate,
                    fixed_duration=args.fixed_duration,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device_filter=[device],
                    position_filter=args.position_filter,
                )
                stats[rep]["per_device"][device] = {"mean": d_mean, "std": d_std}

    if not stats:
        print("No stats computed. Nothing to save.")
        return

    with open(args.output_json, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats to {args.output_json}")


if __name__ == "__main__":
    main()
