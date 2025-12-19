#!/usr/bin/env python3
"""
Unified training entrypoint for PCG-based low LVEF detection.

Supports:
    - On-the-fly spectrogram computation (PCGDataset).
    - Cached spectrogram tensors (CachedPCGDataset).
    - ImageNet-pretrained backbones from timm.
    - BCEWithLogitsLoss with optional (auto) positive-class weighting.
    - Split-specific device/position filters (train/val/test).
    - Optional per-device test evaluation.
    - Early stopping on validation F1_pos.
    - Gradient accumulation and LR scheduling.
    - Eval-only checkpoint evaluation (no retraining).
    - Primary metric: F1 for the positive class (EF <= 40, label=1).
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (AdamW recommended).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine schedule.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs with linear LR ramp.",
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
        "--sample_rate",
        type=int,
        default=2000,
        help="Target sample rate in Hz for on-the-fly features.",
    )
    parser.add_argument(
        "--fixed_duration",
        type=float,
        default=4.0,
        help="Fixed waveform duration in seconds for on-the-fly features.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="global",
        choices=["global", "per_device", "none"],
        help="Normalization strategy for on-the-fly features.",
    )
    parser.add_argument(
        "--device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to keep (e.g. --device_filter iphone android_phone).",
    )
    parser.add_argument(
        "--train_device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to keep for TRAIN only.",
    )
    parser.add_argument(
        "--val_device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to keep for VAL only.",
    )
    parser.add_argument(
        "--test_device_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of devices to keep for TEST only.",
    )
    parser.add_argument(
        "--position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to keep (e.g. --position_filter M A).",
    )
    parser.add_argument(
        "--train_position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to keep for TRAIN only.",
    )
    parser.add_argument(
        "--val_position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to keep for VAL only.",
    )
    parser.add_argument(
        "--test_position_filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of positions to keep for TEST only.",
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
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * steps).",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Optional positive-class weight for BCEWithLogitsLoss.",
    )
    parser.add_argument(
        "--auto_pos_weight",
        action="store_true",
        help="Compute pos_weight from the training split (neg/pos).",
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
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without F1_pos improvement).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum F1_pos improvement to reset early stopping.",
    )
    parser.add_argument(
        "--per_device_eval",
        action="store_true",
        help="Evaluate test metrics per device (uses the filtered test set).",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-example predictions for val/test to CSV.",
    )
    parser.add_argument(
        "--save_history",
        action="store_true",
        help="Save per-epoch training/validation history to CSV.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate a saved checkpoint on the splits.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for --eval_only.",
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (may reduce performance).",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"Warning: deterministic algorithms not fully enabled: {exc}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # macOS GPU
        return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_input_paths(args: argparse.Namespace) -> None:
    for key in ("train_csv", "val_csv", "test_csv"):
        path = getattr(args, key, None)
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{key} not found: {path}")
    if not args.use_cache and args.normalization != "none":
        if not os.path.isfile(args.tf_stats_json):
            raise FileNotFoundError(
                f"tf_stats_json not found: {args.tf_stats_json}"
            )


def get_autocast(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def apply_checkpoint_args(args: argparse.Namespace, ckpt_args: dict) -> None:
    if not ckpt_args:
        return
    override_fields = [
        "representation",
        "backbone",
        "use_cache",
        "image_size",
        "sample_rate",
        "fixed_duration",
        "normalization",
    ]
    for field in override_fields:
        if field not in ckpt_args:
            continue
        ckpt_value = ckpt_args[field]
        current_value = getattr(args, field, None)
        if current_value != ckpt_value:
            print(
                f"Eval-only: overriding --{field}={current_value} "
                f"with checkpoint value {ckpt_value}"
            )
        setattr(args, field, ckpt_value)


def blank_metrics() -> dict:
    return {
        "f1_pos": float("nan"),
        "accuracy": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }


def load_tf_stats(
    tf_stats_json: str, representation: str, normalization: str
) -> Tuple[object, object]:
    with open(tf_stats_json, "r") as f:
        stats = json.load(f)

    if representation not in stats:
        raise ValueError(
            f"Representation '{representation}' not found in {tf_stats_json}. "
            f"Available: {list(stats.keys())}"
        )
    rep_stats = stats[representation]
    if normalization == "none":
        return None, None
    if normalization == "global":
        return float(rep_stats["mean"]), float(rep_stats["std"])
    if normalization == "per_device":
        per_device = rep_stats.get("per_device")
        if per_device is None:
            raise ValueError(
                f"No per_device stats found for {representation} in {tf_stats_json}. "
                "Recompute stats with --per_device."
            )
        mean_map = {"__global__": float(rep_stats["mean"])}
        std_map = {"__global__": float(rep_stats["std"])}
        for device, vals in per_device.items():
            mean_map[device] = float(vals["mean"])
            std_map[device] = float(vals["std"])
        return mean_map, std_map
    raise ValueError(f"Unknown normalization strategy: {normalization}")


def _normalize_filter(values):
    if values is None:
        return None
    if isinstance(values, (list, tuple)) and len(values) == 0:
        return None
    return values


def _resolve_filter(shared, specific):
    specific = _normalize_filter(specific)
    if specific is not None:
        return specific
    return _normalize_filter(shared)


def compute_pos_weight_from_df(df) -> Tuple[float, int, int]:
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
    elif "ef" in df.columns:
        labels = pd.to_numeric(df["ef"], errors="coerce").apply(
            lambda v: ef_to_label(v) if pd.notna(v) else np.nan
        )
    else:
        return None, 0, 0

    labels = labels.dropna().astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg
    return float(n_neg / n_pos), n_pos, n_neg


def build_optimizer(args, model: torch.nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        t_max = max(1, args.epochs - max(0, args.warmup_epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def apply_warmup_lr(optimizer, base_lr: float, warmup_epochs: int, epoch: int) -> float:
    if warmup_epochs <= 0:
        return optimizer.param_groups[0]["lr"]
    if epoch <= warmup_epochs:
        scale = float(epoch) / float(warmup_epochs)
        new_lr = base_lr * scale
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr
    return optimizer.param_groups[0]["lr"]


def resolve_cached_csv_path(path: str, representation: str) -> str:
    base = os.path.basename(path)
    directory = os.path.dirname(path)
    if base.startswith(f"cached_{representation}_"):
        return path
    if base.startswith("cached_"):
        if base.startswith("cached_metadata_"):
            candidate = os.path.join(directory, f"cached_{representation}_{base[len('cached_'):]}")
            if os.path.exists(candidate):
                return candidate
            raise FileNotFoundError(
                f"Cached CSV not found for representation '{representation}': {candidate}"
            )
        raise ValueError(
            f"Cached CSV appears to be for a different representation: {path}"
        )
    if base.startswith("metadata_"):
        candidate = os.path.join(directory, f"cached_{representation}_{base}")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"Cached CSV not found for representation '{representation}': {candidate}"
        )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"CSV not found: {path}")


def _unbatch_meta(meta):
    if meta is None:
        return []
    if isinstance(meta, list):
        if len(meta) == 0:
            return []
        if isinstance(meta[0], dict):
            return meta
        return []
    if isinstance(meta, dict):
        first_val = next(iter(meta.values()))
        if torch.is_tensor(first_val):
            batch_size = first_val.shape[0]
        else:
            batch_size = len(first_val)
        items = []
        for i in range(batch_size):
            row = {}
            for key, value in meta.items():
                if torch.is_tensor(value):
                    row[key] = value[i].item()
                else:
                    row[key] = value[i]
            items.append(row)
        return items
    return []


def save_predictions_csv(
    out_path: str,
    logits: np.ndarray,
    labels: np.ndarray,
    meta: list,
    args: argparse.Namespace,
    split: str,
    threshold: float,
):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    rows = []
    for i in range(len(labels)):
        row = {}
        if i < len(meta):
            row.update(meta[i])
        row.update(
            {
                "run_name": args.run_name,
                "representation": args.representation,
                "backbone": args.backbone,
                "split": split,
                "label": int(labels[i]),
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "pred": int(preds[i]),
                "correct": int(preds[i] == labels[i]),
                "threshold": float(threshold),
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


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
    if args.use_cache:
        train_base = os.path.basename(args.train_csv)
        val_base = os.path.basename(args.val_csv)
        test_base = os.path.basename(args.test_csv)
        rep_prefix = f"cached_{args.representation}_metadata_"
        if (
            train_base.startswith("cached_")
            and not train_base.startswith(rep_prefix)
        ):
            raise ValueError(
                f"--use_cache expects cached CSVs for representation '{args.representation}'. "
                f"Got train_csv={args.train_csv}"
            )
        if (
            val_base.startswith("cached_")
            and not val_base.startswith(rep_prefix)
        ):
            raise ValueError(
                f"--use_cache expects cached CSVs for representation '{args.representation}'. "
                f"Got val_csv={args.val_csv}"
            )
        if (
            test_base.startswith("cached_")
            and not test_base.startswith(rep_prefix)
        ):
            raise ValueError(
                f"--use_cache expects cached CSVs for representation '{args.representation}'. "
                f"Got test_csv={args.test_csv}"
            )

    if args.train_device_filter and not (args.val_device_filter or args.device_filter):
        print(
            "Warning: --train_device_filter is set but no validation device filter "
            "was provided; validation will include all devices."
        )
    if args.train_position_filter and not (args.val_position_filter or args.position_filter):
        print(
            "Warning: --train_position_filter is set but no validation position filter "
            "was provided; validation will include all positions."
        )

    train_device_filter = _resolve_filter(args.device_filter, args.train_device_filter)
    val_device_filter = _resolve_filter(args.device_filter, args.val_device_filter)
    test_device_filter = _resolve_filter(args.device_filter, args.test_device_filter)

    train_position_filter = _resolve_filter(
        args.position_filter, args.train_position_filter
    )
    val_position_filter = _resolve_filter(
        args.position_filter, args.val_position_filter
    )
    test_position_filter = _resolve_filter(
        args.position_filter, args.test_position_filter
    )

    if args.use_cache:
        train_ds = CachedPCGDataset(
            args.train_csv,
            device_filter=train_device_filter,
            position_filter=train_position_filter,
        )
        val_ds = CachedPCGDataset(
            args.val_csv,
            device_filter=val_device_filter,
            position_filter=val_position_filter,
        )
        test_ds = CachedPCGDataset(
            args.test_csv,
            device_filter=test_device_filter,
            position_filter=test_position_filter,
        )
    else:
        if args.normalization != "none" and (mean is None or std is None):
            raise ValueError("Mean/std must be provided for on-the-fly features.")
        train_ds = PCGDataset(
            csv_path=args.train_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=train_device_filter,
            position_filter=train_position_filter,
            image_size=args.image_size,
            sample_rate=args.sample_rate,
            fixed_duration=args.fixed_duration,
        )
        val_ds = PCGDataset(
            csv_path=args.val_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=val_device_filter,
            position_filter=val_position_filter,
            image_size=args.image_size,
            sample_rate=args.sample_rate,
            fixed_duration=args.fixed_duration,
        )
        test_ds = PCGDataset(
            csv_path=args.test_csv,
            representation=args.representation,
            mean=mean,
            std=std,
            device_filter=test_device_filter,
            position_filter=test_position_filter,
            image_size=args.image_size,
            sample_rate=args.sample_rate,
            fixed_duration=args.fixed_duration,
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
    scaler,
    use_amp: bool,
    device_type: str,
    grad_accum_steps: int = 1,
):
    model.train()
    running_loss = 0.0
    n_examples = 0

    accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels, _) in enumerate(tqdm(loader, desc="Train", leave=False)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        with get_autocast(device_type, use_amp):
            logits = model(imgs).squeeze(-1)
            raw_loss = criterion(logits, labels)

        if not torch.isfinite(raw_loss):
            print("Warning: encountered non-finite loss; skipping batch.")
            continue

        loss = raw_loss / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        do_step = (step + 1) % accum_steps == 0 or (step + 1) == len(loader)
        if do_step:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        running_loss += raw_loss.item() * batch_size
        n_examples += batch_size

    return running_loss / max(n_examples, 1)


def evaluate(
    model,
    loader,
    device,
    threshold: float,
    use_amp: bool,
    device_type: str,
    return_arrays: bool = False,
):
    model.eval()
    all_logits = []
    all_labels = []

    all_meta = []
    with torch.no_grad():
        for imgs, labels, meta in tqdm(loader, desc="Eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with get_autocast(device_type, use_amp):
                logits = model(imgs).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if return_arrays:
                all_meta.extend(_unbatch_meta(meta))

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
        return metrics, logits_np, labels_np, all_meta
    return metrics


def evaluate_by_device(
    model,
    test_ds,
    device,
    threshold: float,
    use_amp: bool,
    device_type: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    if "device" not in test_ds.df.columns:
        return {}

    device_metrics = {}
    devices = sorted(test_ds.df["device"].unique().tolist())
    for device_name in devices:
        idxs = test_ds.df.index[test_ds.df["device"] == device_name].tolist()
        if not idxs:
            continue
        subset = Subset(test_ds, idxs)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        metrics = evaluate(
            model,
            loader,
            device,
            threshold=threshold,
            use_amp=use_amp,
            device_type=device_type,
            return_arrays=False,
        )
        device_metrics[device_name] = metrics

    return device_metrics


def save_run_outputs(
    run_dir: str,
    summary_path: str,
    run_name: str,
    args: argparse.Namespace,
    tuned_threshold: float,
    val_metrics: dict,
    test_metrics: dict,
    test_metrics_by_device: dict,
    ckpt_path: str,
    history_rows: list,
):
    os.makedirs(run_dir, exist_ok=True)

    payload = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "tuned_threshold": tuned_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_device": test_metrics_by_device,
        "checkpoint_path": ckpt_path,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    if history_rows:
        pd.DataFrame(history_rows).to_csv(
            os.path.join(run_dir, "history.csv"), index=False
        )

    def _flatten(prefix: str, d: dict):
        return {f"{prefix}_{k}": v for k, v in d.items()}

    row = {
        "run_name": run_name,
        "timestamp": payload["timestamp"],
        "representation": args.representation,
        "backbone": args.backbone,
        "eval_only": args.eval_only,
        "use_cache": args.use_cache,
        "device_filter": ",".join(args.device_filter) if args.device_filter else "",
        "train_device_filter": ",".join(args.train_device_filter) if args.train_device_filter else "",
        "val_device_filter": ",".join(args.val_device_filter) if args.val_device_filter else "",
        "test_device_filter": ",".join(args.test_device_filter) if args.test_device_filter else "",
        "position_filter": ",".join(args.position_filter) if args.position_filter else "",
        "train_position_filter": ",".join(args.train_position_filter) if args.train_position_filter else "",
        "val_position_filter": ",".join(args.val_position_filter) if args.val_position_filter else "",
        "test_position_filter": ",".join(args.test_position_filter) if args.test_position_filter else "",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "grad_accum_steps": args.grad_accum_steps,
        "sample_rate": args.sample_rate,
        "fixed_duration": args.fixed_duration,
        "image_size": args.image_size,
        "normalization": args.normalization,
        "pos_weight": args.pos_weight if args.pos_weight is not None else np.nan,
        "auto_pos_weight": args.auto_pos_weight,
        "amp": args.amp,
        "tuned_threshold": tuned_threshold,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "checkpoint_path": ckpt_path,
    }
    row.update(_flatten("val", val_metrics))
    row.update(_flatten("test", test_metrics))

    pd.DataFrame([row]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    summary_rows = []
    overall_row = dict(row)
    overall_row["metric_scope"] = "overall"
    overall_row["device"] = ""
    summary_rows.append(overall_row)

    if test_metrics_by_device:
        for device_name, metrics in test_metrics_by_device.items():
            device_row = dict(row)
            for key in list(device_row.keys()):
                if key.startswith("val_"):
                    device_row[key] = np.nan
            device_row.update(_flatten("test", metrics))
            device_row["metric_scope"] = "test_device"
            device_row["device"] = device_name
            summary_rows.append(device_row)

    summary_df = pd.DataFrame(summary_rows)
    if os.path.exists(summary_path):
        summary_df = pd.concat([pd.read_csv(summary_path), summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    checkpoint = None
    if args.eval_only:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required with --eval_only")
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(
                f"--checkpoint_path not found: {args.checkpoint_path}"
            )
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        apply_checkpoint_args(args, checkpoint.get("args", {}))
    set_seed(args.seed, deterministic=args.deterministic)

    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    amp_device_type = "cuda" if use_amp else "cpu"
    print(f"Using device: {device} (AMP={'on' if use_amp else 'off'})")
    if args.deterministic:
        print("Deterministic mode enabled (may reduce performance).")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup_epochs must be >= 0")
    if args.scheduler != "none" and args.warmup_epochs >= args.epochs:
        print(
            "Warning: warmup_epochs >= epochs; cosine schedule will not decay.",
        )
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.grad_accum_steps > 1:
        eff_bs = args.batch_size * args.grad_accum_steps
        print(f"Gradient accumulation: {args.grad_accum_steps} step(s) (effective batch={eff_bs})")

    if args.use_cache:
        orig_train = args.train_csv
        orig_val = args.val_csv
        orig_test = args.test_csv
        args.train_csv = resolve_cached_csv_path(args.train_csv, args.representation)
        args.val_csv = resolve_cached_csv_path(args.val_csv, args.representation)
        args.test_csv = resolve_cached_csv_path(args.test_csv, args.representation)
        if args.train_csv != orig_train:
            print(f"Resolved train_csv -> {args.train_csv}")
        if args.val_csv != orig_val:
            print(f"Resolved val_csv -> {args.val_csv}")
        if args.test_csv != orig_test:
            print(f"Resolved test_csv -> {args.test_csv}")

    validate_input_paths(args)

    if args.run_name:
        run_name = args.run_name
    elif args.eval_only:
        ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        run_name = f"eval_{ckpt_tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = f"{args.backbone}_{args.representation}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    args.run_name = run_name
    run_dir = os.path.join(args.results_dir, run_name)
    summary_path = os.path.join(args.results_dir, "summary.csv")
    os.makedirs(run_dir, exist_ok=True)

    mean = std = None
    if not args.use_cache:
        if args.normalization != "none":
            mean, std = load_tf_stats(
                args.tf_stats_json, args.representation, args.normalization
            )
            if args.normalization == "global" and mean is not None:
                print(
                    f"Loaded TF stats for {args.representation}: "
                    f"mean={mean:.6f}, std={std:.6f}"
                )
            elif args.normalization == "per_device":
                device_keys = [k for k in mean.keys() if k != "__global__"]
                print(
                    f"Loaded per-device stats for {args.representation}: "
                    f"{len(device_keys)} device(s)"
                )
        else:
            print("Normalisation disabled for on-the-fly features.")
    else:
        if args.sample_rate != 2000 or args.fixed_duration != 4.0:
            print(
                "Warning: --sample_rate/--fixed_duration are ignored with --use_cache. "
                "Ensure cached tensors were built with your desired settings."
            )
        if args.image_size != 224:
            print(
                "Warning: --image_size is ignored with --use_cache. "
                "Ensure cached tensors were built with your desired image_size."
            )
        if args.normalization != "none":
            print(
                "Note: --normalization is ignored with --use_cache. "
                "Ensure cached tensors were built with the intended normalization."
            )

    train_ds, val_ds, test_ds = build_datasets(args, mean, std)

    use_pin_memory = device.type == "cuda"
    worker_init_fn = seed_worker if args.deterministic else None
    generator = None
    if args.deterministic:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    model = create_model(backbone=args.backbone, pretrained=True, num_classes=1)
    model.to(device)

    if args.eval_only:
        if args.tune_threshold:
            print(
                "Warning: --tune_threshold ignored in eval_only; "
                "using checkpoint threshold."
            )
        if args.auto_pos_weight or args.pos_weight is not None:
            print("Note: class weighting flags are ignored in eval_only.")

        if checkpoint is None:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        tuned_threshold = checkpoint.get("best_threshold", args.eval_threshold)

        if args.save_predictions:
            val_metrics, val_logits, val_labels, val_meta = evaluate(
                model,
                val_loader,
                device,
                threshold=tuned_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
                return_arrays=True,
            )
            save_predictions_csv(
                os.path.join(run_dir, "predictions_val.csv"),
                val_logits,
                val_labels,
                val_meta,
                args,
                split="val",
                threshold=tuned_threshold,
            )

            test_metrics, test_logits, test_labels, test_meta = evaluate(
                model,
                test_loader,
                device,
                threshold=tuned_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
                return_arrays=True,
            )
            save_predictions_csv(
                os.path.join(run_dir, "predictions_test.csv"),
                test_logits,
                test_labels,
                test_meta,
                args,
                split="test",
                threshold=tuned_threshold,
            )
        else:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                threshold=tuned_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
                return_arrays=False,
            )
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                threshold=tuned_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
                return_arrays=False,
            )

        test_metrics_by_device = {}
        if args.per_device_eval:
            test_metrics_by_device = evaluate_by_device(
                model=model,
                test_ds=test_ds,
                device=device,
                threshold=tuned_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=use_pin_memory,
            )
            for device_name, metrics in test_metrics_by_device.items():
                print(f"Test metrics [{device_name}]:", format_metrics(metrics))

        print(f"Val metrics (threshold={tuned_threshold:.3f}):", format_metrics(val_metrics))
        print(f"Test metrics (threshold={tuned_threshold:.3f}):", format_metrics(test_metrics))

        save_run_outputs(
            run_dir=run_dir,
            summary_path=summary_path,
            run_name=run_name,
            args=args,
            tuned_threshold=tuned_threshold,
            val_metrics=val_metrics if val_metrics else blank_metrics(),
            test_metrics=test_metrics,
            test_metrics_by_device=test_metrics_by_device,
            ckpt_path=args.checkpoint_path,
            history_rows=[],
        )
        print(f"Saved eval-only artifacts to {run_dir}")
        print(f"Summary table updated at {summary_path}")
        return

    pos_weight_value = args.pos_weight
    if args.auto_pos_weight:
        if args.pos_weight is not None:
            print("Warning: --auto_pos_weight ignored because --pos_weight is set.")
        else:
            computed, n_pos, n_neg = compute_pos_weight_from_df(train_ds.df)
            if computed is None:
                print(
                    f"Warning: cannot compute pos_weight (pos={n_pos}, neg={n_neg}). "
                    "Proceeding without class weighting."
                )
            else:
                pos_weight_value = computed
                print(
                    f"Auto pos_weight: {pos_weight_value:.4f} "
                    f"(pos={n_pos}, neg={n_neg})"
                )

    args.pos_weight = pos_weight_value
    pos_weight_tensor = None
    if pos_weight_value is not None:
        pos_weight_tensor = torch.tensor(
            [pos_weight_value], device=device, dtype=torch.float32
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = get_grad_scaler(use_amp)

    ckpt_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pth")

    best_val_f1 = -float("inf")
    tuned_threshold = args.eval_threshold
    patience = max(0, args.early_stopping_patience)
    patience_counter = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        lr = apply_warmup_lr(optimizer, args.lr, args.warmup_epochs, epoch)
        print(f"LR: {lr:.6g}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            device_type=amp_device_type,
            grad_accum_steps=args.grad_accum_steps,
        )
        print(f"Train loss: {train_loss:.4f}")

        if args.tune_threshold:
            val_metrics, val_logits, val_labels, _ = evaluate(
                model,
                val_loader,
                device,
                threshold=args.eval_threshold,
                use_amp=use_amp,
                device_type=amp_device_type,
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
                device_type=amp_device_type,
                return_arrays=False,
            )
        print("Val metrics:", format_metrics(val_metrics))

        if args.save_history:
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_f1_pos": float(val_metrics.get("f1_pos", float("nan"))),
                    "val_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "val_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
                    "val_sensitivity": float(val_metrics.get("sensitivity", float("nan"))),
                    "val_specificity": float(val_metrics.get("specificity", float("nan"))),
                    "threshold": float(tuned_threshold),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )

        current_f1 = val_metrics["f1_pos"]
        improved = (
            np.isfinite(current_f1)
            and (current_f1 - best_val_f1) > args.early_stopping_min_delta
        )

        if improved:
            best_val_f1 = current_f1
            patience_counter = 0
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
        elif patience > 0:
            patience_counter += 1
            print(
                f"No F1_pos improvement for {patience_counter}/{patience} epoch(s)."
            )
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tuned_threshold = checkpoint.get("best_threshold", tuned_threshold)
    best_val_metrics = checkpoint.get("val_metrics", val_metrics)

    if args.save_predictions:
        val_metrics_eval, val_logits, val_labels, val_meta = evaluate(
            model,
            val_loader,
            device,
            threshold=tuned_threshold,
            use_amp=use_amp,
            device_type=amp_device_type,
            return_arrays=True,
        )
        save_predictions_csv(
            os.path.join(run_dir, "predictions_val.csv"),
            val_logits,
            val_labels,
            val_meta,
            args,
            split="val",
            threshold=tuned_threshold,
        )

        test_metrics, test_logits, test_labels, test_meta = evaluate(
            model,
            test_loader,
            device,
            threshold=tuned_threshold,
            use_amp=use_amp,
            device_type=amp_device_type,
            return_arrays=True,
        )
        save_predictions_csv(
            os.path.join(run_dir, "predictions_test.csv"),
            test_logits,
            test_labels,
            test_meta,
            args,
            split="test",
            threshold=tuned_threshold,
        )
    else:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            threshold=tuned_threshold,
            use_amp=use_amp,
            device_type=amp_device_type,
            return_arrays=False,
        )
    print(f"Test metrics (threshold={tuned_threshold:.3f}):", format_metrics(test_metrics))

    test_metrics_by_device = {}
    if args.per_device_eval:
        test_metrics_by_device = evaluate_by_device(
            model=model,
            test_ds=test_ds,
            device=device,
            threshold=tuned_threshold,
            use_amp=use_amp,
            device_type=amp_device_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
        )
        for device_name, metrics in test_metrics_by_device.items():
            print(f"Test metrics [{device_name}]:", format_metrics(metrics))

    save_run_outputs(
        run_dir=run_dir,
        summary_path=summary_path,
        run_name=run_name,
        args=args,
        tuned_threshold=tuned_threshold,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        test_metrics_by_device=test_metrics_by_device,
        ckpt_path=ckpt_path,
        history_rows=history_rows,
    )
    print(f"Saved run artifacts to {run_dir}")
    print(f"Summary table updated at {summary_path}")


if __name__ == "__main__":
    main()
