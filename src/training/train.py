#!/usr/bin/env python3
"""
Unified training entrypoint for PCG-based low LVEF detection.

Supports:
    - On-the-fly spectrogram computation (PCGDataset).
    - Cached spectrogram tensors (CachedPCGDataset).
    - ImageNet-pretrained backbones from timm.
    - BCEWithLogitsLoss with optional positive-class weighting.
    - Primary metric: F1 for the positive class (EF <= 40, label=1).
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.pcg_dataset import CachedPCGDataset, PCGDataset, ef_to_label
from src.models.models import BACKBONE_CONFIGS, create_model
from src.utils.metrics import compute_binary_metrics, format_metrics, tune_threshold


# --------------------------------------------------------------------------- #
# Argument parsing and setup
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCG-based low LVEF detector."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="splits/metadata_train.csv",
        help="Path to training split CSV.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="splits/metadata_val.csv",
        help="Path to validation split CSV.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="splits/metadata_test.csv",
        help="Path to test split CSV.",
    )
    parser.add_argument(
        "--tf_stats_json",
        type=str,
        default="tf_stats.json",
        help="Path to JSON with mean/std for each representation.",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="mfcc",
        choices=["mfcc", "gammatone"],
        help="Time-frequency representation to use.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone architecture (timm name is resolved internally).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size (H=W).",
    )
    parser.add_argument(
        "--device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to keep (e.g. --device_filter iphone android_phone).",
    )
    parser.add_argument(
        "--position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to keep (e.g. --position_filter M A).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (AMP) when using CUDA.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use CachedPCGDataset instead of on-the-fly spectrograms.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Optional positive-class weight for BCEWithLogitsLoss.",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for metric computation.",
    )
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Tune decision threshold on the validation set to maximise F1_pos.",
    )
    parser.add_argument(
        "--threshold_grid",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of thresholds to search when tuning. "
        "If omitted, uses 0.05..0.95 step 0.05.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save the best checkpoint.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store per-run metrics and summary.csv.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name; if omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # macOS GPU
        return torch.device("mps")
    return torch.device("cpu")


def load_tf_stats(tf_stats_json: str, representation: str) -> Tuple[float, float]:
    with open(tf_stats_json, "r") as f:
        stats = json.load(f)

    if representation not in stats:
        raise ValueError(
            f"Representation '{representation}' not found in {tf_stats_json}. "
            f"Available: {list(stats.keys())}"
        )
    rep_stats = stats[representation]
    return float(rep_stats["mean"]), float(rep_stats["std"])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def describe_labels(name: str, df) -> None:
    labels = [ef_to_label(float(row["ef"])) for _, row in df.iterrows()]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"{name} label distribution (EF <= 40 -> 1):")
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c}")


def build_datasets(args, mean: float = None, std: float = None):
    device_filter = args.device_filter if args.device_filter else None
    position_filter = args.position_filter if args.position_filter else None

    if args.use_cache:
        train_ds = CachedPCGDataset(
            args.train_csv, device_filter=device_filter, position_filter=position_filter
        )
        val_ds = CachedPCGDataset(
            args.val_csv, device_filter=device_filter, position_filter=position_filter
        )
        test_ds = CachedPCGDataset(
            args.test_csv, device_filter=device_filter, position_filter=position_filter
        )
    else:
        if mean is None or std is None:
            raise ValueError("Mean/std must be provided for on-the-fly features.")
        train_ds = PCGDataset(
            csv_path=args.train_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=device_filter,
            position_filter=position_filter,
            image_size=args.image_size,
        )
        val_ds = PCGDataset(
            csv_path=args.val_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=device_filter,
            position_filter=position_filter,
            image_size=args.image_size,
        )
        test_ds = PCGDataset(
            csv_path=args.test_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=device_filter,
            position_filter=position_filter,
            image_size=args.image_size,
        )

    describe_labels("Train", train_ds.df)
    describe_labels("Val", val_ds.df)
    describe_labels("Test", test_ds.df)
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
):
    model.train()
    running_loss = 0.0
    n_examples = 0

    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            print("Warning: encountered non-finite loss; skipping batch.")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_examples += batch_size

    return running_loss / max(n_examples, 1)


def evaluate(model, loader, device, threshold: float, use_amp: bool, return_arrays: bool = False):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    if len(all_logits) == 0:
        return {
            "f1_pos": float("nan"),
            "accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    metrics = compute_binary_metrics(logits_np, labels_np, threshold=threshold)
    if return_arrays:
        return metrics, logits_np, labels_np
    return metrics


def save_run_outputs(
    run_dir: str,
    summary_path: str,
    run_name: str,
    args: argparse.Namespace,
    tuned_threshold: float,
    val_metrics: dict,
    test_metrics: dict,
    ckpt_path: str,
):
    os.makedirs(run_dir, exist_ok=True)

    payload = {
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat(),
        "args": vars(args),
        "tuned_threshold": tuned_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": ckpt_path,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    def _flatten(prefix: str, d: dict):
        return {f"{prefix}_{k}": v for k, v in d.items()}

    row = {
        "run_name": run_name,
        "timestamp": payload["timestamp"],
        "representation": args.representation,
        "backbone": args.backbone,
        "use_cache": args.use_cache,
        "device_filter": ",".join(args.device_filter) if args.device_filter else "",
        "position_filter": ",".join(args.position_filter) if args.position_filter else "",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "pos_weight": args.pos_weight if args.pos_weight is not None else "",
        "amp": args.amp,
        "tuned_threshold": tuned_threshold,
        "checkpoint_path": ckpt_path,
    }
    row.update(_flatten("val", val_metrics))
    row.update(_flatten("test", test_metrics))

    pd.DataFrame([row]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    summary_df = pd.DataFrame([row])
    if os.path.exists(summary_path):
        summary_df = pd.concat([pd.read_csv(summary_path), summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    print(f"Using device: {device} (AMP={'on' if use_amp else 'off'})")

    run_name = args.run_name or f"{args.backbone}_{args.representation}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.results_dir, run_name)
    summary_path = os.path.join(args.results_dir, "summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)

    mean = std = None
    if not args.use_cache:
        mean, std = load_tf_stats(args.tf_stats_json, args.representation)
        print(f"Loaded TF stats for {args.representation}: mean={mean:.6f}, std={std:.6f}")

    train_ds, val_ds, test_ds = build_datasets(args, mean, std)

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    model = create_model(backbone=args.backbone, pretrained=True, num_classes=1)
    model.to(device)

    pos_weight_tensor = None
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [args.pos_weight], device=device, dtype=torch.float32
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_suffix = "cached" if args.use_cache else "on_the_fly"
    ckpt_name = f"{args.backbone}_{args.representation}_{ckpt_suffix}_best.pth"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    best_val_f1 = -float("inf")
    tuned_threshold = args.eval_threshold

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        print(f"Train loss: {train_loss:.4f}")

        if args.tune_threshold:
            val_metrics, val_logits, val_labels = evaluate(
                model,
                val_loader,
                device,
                threshold=args.eval_threshold,
                use_amp=use_amp,
                return_arrays=True,
            )
            tuned_threshold, val_metrics = tune_threshold(
                val_logits, val_labels, thresholds=args.threshold_grid
            )
            print(f"Tuned threshold on val: {tuned_threshold:.3f}")
        else:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                threshold=args.eval_threshold,
                use_amp=use_amp,
                return_arrays=False,
            )
        print("Val metrics:", format_metrics(val_metrics))

        if val_metrics["f1_pos"] > best_val_f1:
            best_val_f1 = val_metrics["f1_pos"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_threshold": tuned_threshold,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"New best model saved to {ckpt_path}")

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    tuned_threshold = checkpoint.get("best_threshold", tuned_threshold)
    best_val_metrics = checkpoint.get("val_metrics", val_metrics)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=tuned_threshold,
        use_amp=use_amp,
        return_arrays=False,
    )
    print(f"Test metrics (threshold={tuned_threshold:.3f}):", format_metrics(test_metrics))

    save_run_outputs(
        run_dir=run_dir,
        summary_path=summary_path,
        run_name=run_name,
        args=args,
        tuned_threshold=tuned_threshold,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        ckpt_path=ckpt_path,
    )
    print(f"Saved run artifacts to {run_dir}")
    print(f"Summary table updated at {summary_path}")


if __name__ == "__main__":
    main()
