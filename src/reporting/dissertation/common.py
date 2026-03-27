#!/usr/bin/env python3
"""Shared helpers for dissertation figures and tables."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if "MPLCONFIGDIR" not in os.environ:
    _MPL_CACHE = Path(tempfile.gettempdir()) / "pcg_matplotlib_cache"
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_USE_SHM"] = "FALSE"
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.models import BACKBONE_CONFIGS, create_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

METRIC_COLS = [
    "test_f1_pos",
    "test_accuracy",
    "test_sensitivity",
    "test_specificity",
    "test_auroc",
    "test_auprc",
]

CV_GROUP_COLS = [
    "train_device_filter",
    "representation",
    "backbone",
    "image_size",
    "normalization",
]

DEVICE_ORDER = ["iphone", "android_phone", "digital_stethoscope"]
REPRESENTATION_ORDER = ["mfcc", "gammatone"]
BACKBONE_ORDER = [
    "mobilenetv2",
    "mobilenetv3_large",
    "efficientnet_b0",
    "efficientnetv2_s",
    "swinv2_tiny",
    "swinv2_small",
]

BACKBONE_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2_s": "EfficientNetV2-S",
    "swinv2_tiny": "SwinV2-Tiny",
    "swinv2_small": "SwinV2-Small",
}

DEVICE_LABELS = {
    "android_phone": "Android phone",
    "digital_stethoscope": "Digital stethoscope",
    "iphone": "iPhone",
}

DEVICE_ABBREV = {
    "android_phone": "A",
    "digital_stethoscope": "D",
    "iphone": "I",
}

AUSCULTATION_SITE_ORDER = ["A", "M", "P", "T"]
AUSCULTATION_SITE_LABELS = {
    "A": "Aortic",
    "M": "Mitral",
    "P": "Pulmonic",
    "T": "Tricuspid",
}

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_SEED = 42

EXPERIMENT_LABELS = {
    "final_within": "Final within-device",
    "cross_pairwise": "Cross-device (pairwise target)",
    "cross_overall": "Cross-device (overall aggregate)",
    "pool_overall": "Pooled-device (overall aggregate)",
    "pool_per_device": "Pooled-device (per target device)",
}

def _pretty_device(device: str) -> str:
    return DEVICE_LABELS.get(device, device)


def _pretty_device_after_trained_on(device: str) -> str:
    if device == "digital_stethoscope":
        return "digital stethoscope"
    return _pretty_device(device)

def _device_categorical(series: pd.Series) -> pd.Categorical:
    return pd.Categorical(series.astype(str), categories=DEVICE_ORDER, ordered=True)

def _device_or_all_categorical(series: pd.Series) -> pd.Categorical:
    return pd.Categorical(
        series.astype(str),
        categories=["all_devices"] + DEVICE_ORDER,
        ordered=True,
    )

def _sort_by_device_columns(df: pd.DataFrame, device_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    tmp_cols: List[str] = []
    for i, col in enumerate(device_cols):
        if col not in out.columns:
            continue
        tmp = f"__device_sort_{i}"
        if col == "target_device":
            out[tmp] = _device_or_all_categorical(out[col].fillna("").replace("", pd.NA).fillna("all_devices"))
        else:
            out[tmp] = _device_categorical(out[col].fillna(""))
        tmp_cols.append(tmp)
    if tmp_cols:
        out = out.sort_values(tmp_cols, kind="mergesort")
        out = out.drop(columns=tmp_cols)
    return out

def _bool_from_mixed(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {missing}")

def load_and_normalize_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _ensure_columns(
        df,
        [
            "run_name",
            "eval_only",
            "metric_scope",
            "checkpoint_path",
            "representation",
            "backbone",
            "test_f1_pos",
            "test_accuracy",
            "test_sensitivity",
            "test_specificity",
            "test_auroc",
            "test_auprc",
        ],
    )
    for col in [
        "train_device_filter",
        "val_device_filter",
        "test_device_filter",
        "metric_scope",
        "device",
        "checkpoint_path",
        "representation",
        "backbone",
        "run_name",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["eval_only_bool"] = _bool_from_mixed(df["eval_only"])
    return df

def classify_runs(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for run_name, g in df.groupby("run_name", dropna=False):
        n_rows = len(g)
        eval_only = bool(g["eval_only_bool"].iloc[0])
        train_device = str(g["train_device_filter"].iloc[0]) if "train_device_filter" in g else ""
        val_device = str(g["val_device_filter"].iloc[0]) if "val_device_filter" in g else ""
        test_device = str(g["test_device_filter"].iloc[0]) if "test_device_filter" in g else ""
        metric_scopes = sorted(g["metric_scope"].astype(str).unique().tolist())

        if run_name.startswith("cv_") and n_rows == 1:
            run_kind = "cv"
        elif eval_only and n_rows == 3 and set(metric_scopes) == {"overall", "test_device"}:
            run_kind = "cross_eval_group"
        elif (
            (not eval_only)
            and n_rows == 2
            and train_device
            and (train_device == val_device == test_device)
            and set(metric_scopes) == {"overall", "test_device"}
        ):
            run_kind = "final_within"
        elif (
            (not eval_only)
            and n_rows == 4
            and (not train_device)
            and set(metric_scopes) == {"overall", "test_device"}
        ):
            run_kind = "pool"
        else:
            run_kind = "other"

        records.append(
            {
                "run_name": run_name,
                "run_kind": run_kind,
                "n_rows": n_rows,
                "eval_only": eval_only,
                "representation": g["representation"].iloc[0],
                "backbone": g["backbone"].iloc[0],
                "image_size": g["image_size"].iloc[0] if "image_size" in g else np.nan,
                "train_device_filter": train_device,
                "val_device_filter": val_device,
                "test_device_filter": test_device,
                "metric_scopes": "|".join(metric_scopes),
                "checkpoint_path": g["checkpoint_path"].iloc[0],
            }
        )
    out = pd.DataFrame.from_records(records)
    return out.sort_values(["run_kind", "run_name"]).reset_index(drop=True)

def extract_clean_views(df: pd.DataFrame, run_catalog: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    run_kind_map = run_catalog.set_index("run_name")["run_kind"].to_dict()
    df = df.copy()
    df["run_kind"] = df["run_name"].map(run_kind_map).fillna("other")

    cv_rows = df[df["run_kind"] == "cv"].copy()

    final_rows = df[df["run_kind"] == "final_within"].copy()
    final_overall = final_rows[final_rows["metric_scope"] == "overall"].copy()
    final_per_device = final_rows[final_rows["metric_scope"] == "test_device"].copy()
    if not final_per_device.empty:
        final_per_device["target_device"] = final_per_device["device"]

    cross_rows = df[df["run_kind"] == "cross_eval_group"].copy()
    cross_overall = cross_rows[cross_rows["metric_scope"] == "overall"].copy()
    cross_pairwise = cross_rows[cross_rows["metric_scope"] == "test_device"].copy()
    if not cross_pairwise.empty:
        cross_pairwise["target_device"] = cross_pairwise["device"]

    pool_rows = df[df["run_kind"] == "pool"].copy()
    pool_overall = pool_rows[pool_rows["metric_scope"] == "overall"].copy()
    pool_per_device = pool_rows[pool_rows["metric_scope"] == "test_device"].copy()
    if not pool_per_device.empty:
        pool_per_device["target_device"] = pool_per_device["device"]

    # Infer source device/model for cross-device eval rows by matching checkpoint_path
    if not cross_rows.empty and not final_overall.empty:
        final_map = final_overall[
            ["checkpoint_path", "train_device_filter", "representation", "backbone", "run_name"]
        ].copy()
        final_map = final_map.rename(
            columns={
                "train_device_filter": "source_device",
                "representation": "source_representation",
                "backbone": "source_backbone",
                "run_name": "source_run_name",
            }
        )
        cross_overall = cross_overall.merge(final_map, on="checkpoint_path", how="left")
        cross_pairwise = cross_pairwise.merge(final_map, on="checkpoint_path", how="left")
        cross_pairwise["source_to_target"] = (
            cross_pairwise["source_device"].fillna("")
            + " -> "
            + cross_pairwise["target_device"].fillna("")
        )

    return {
        "all_rows": df,
        "cv_rows": cv_rows,
        "final_rows": final_rows,
        "final_overall": final_overall,
        "final_per_device": final_per_device,
        "cross_rows": cross_rows,
        "cross_overall": cross_overall,
        "cross_pairwise": cross_pairwise,
        "pool_rows": pool_rows,
        "pool_overall": pool_overall,
        "pool_per_device": pool_per_device,
    }

def aggregate_cv(cv_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_columns(cv_rows, CV_GROUP_COLS + METRIC_COLS)
    agg = (
        cv_rows.groupby(CV_GROUP_COLS, as_index=False)
        .agg(
            folds=("run_name", "count"),
            mean_test_f1_pos=("test_f1_pos", "mean"),
            sd_test_f1_pos=("test_f1_pos", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_sensitivity=("test_sensitivity", "mean"),
            mean_test_specificity=("test_specificity", "mean"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
        )
        .sort_values(
            ["train_device_filter", "mean_test_f1_pos", "mean_test_auprc", "mean_test_auroc"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )
    agg = _sort_by_device_columns(agg, ["train_device_filter"]).reset_index(drop=True)
    best = (
        agg.sort_values(
            ["train_device_filter", "mean_test_f1_pos", "mean_test_auprc", "mean_test_auroc"],
            ascending=[True, False, False, False],
        )
        .groupby("train_device_filter", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best = _sort_by_device_columns(best, ["train_device_filter"]).reset_index(drop=True)
    return agg, best

def _round_numeric(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float32", "float64"]).columns
    out[float_cols] = out[float_cols].round(decimals)
    return out

def _save_csv(df: pd.DataFrame, path: Path, decimals: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _round_numeric(df, decimals=decimals).to_csv(path, index=False)

def _save_fig(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _coerce_binary_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["prob"] = pd.to_numeric(out["prob"], errors="coerce")
    out = out.dropna(subset=["label", "prob"]).copy()
    out["label"] = out["label"].astype(int)
    out["prob"] = out["prob"].astype(float).clip(0.0, 1.0)
    return out

def _load_predictions_file(results_run_dir: Path, run_name: str, split: str = "test") -> pd.DataFrame | None:
    path = results_run_dir / run_name / f"predictions_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"label", "prob", "device"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "run_name" not in df.columns:
        df["run_name"] = run_name
    return _coerce_binary_prediction_df(df)

def _load_history_file(results_run_dir: Path, run_name: str) -> pd.DataFrame | None:
    path = results_run_dir / run_name / "history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"epoch", "train_loss", "val_f1_pos"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df

def _curve_inputs_from_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    if df is None or df.empty:
        return None
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    return y_true, y_prob


def _assemble_pooled_test_predictions_for_best_within_model(
    views: Dict[str, pd.DataFrame], results_run_dir: Path, within_run_name: str
) -> pd.DataFrame | None:
    parts: List[pd.DataFrame] = []

    within_pred = _load_predictions_file(results_run_dir, within_run_name, split="test")
    if within_pred is not None and not within_pred.empty:
        parts.append(within_pred.copy())

    cross_pairwise = views.get("cross_pairwise", pd.DataFrame()).copy()
    if not cross_pairwise.empty and "source_run_name" in cross_pairwise.columns:
        eval_runs = (
            cross_pairwise.loc[
                cross_pairwise["source_run_name"].astype(str) == within_run_name, "run_name"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        for eval_run_name in eval_runs:
            pred = _load_predictions_file(results_run_dir, eval_run_name, split="test")
            if pred is None or pred.empty:
                continue
            parts.append(pred.copy())

    if not parts:
        return None

    merged = pd.concat(parts, ignore_index=True)
    dedup_cols = [c for c in ["patient_id", "path", "device", "label"] if c in merged.columns]
    if dedup_cols:
        merged = merged.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    else:
        merged = merged.drop_duplicates().reset_index(drop=True)

    return _coerce_binary_prediction_df(merged)


def _compute_binary_metrics_from_prob(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    labels = np.asarray(y_true).astype(int)
    probs = np.asarray(y_prob, dtype=float).clip(0.0, 1.0)
    preds = (probs >= float(threshold)).astype(int)

    f1_pos = f1_score(labels, preds, pos_label=1, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "f1_pos": float(f1_pos),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }

def _patient_cluster_bootstrap_auroc(
    df: pd.DataFrame,
    cluster_col: str = "patient_id",
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> Dict[str, object]:
    required = {cluster_col, "label", "prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bootstrap AUROC requires columns: {missing}")

    sub = df.dropna(subset=[cluster_col, "label", "prob"]).copy()
    if sub.empty or sub["label"].nunique() < 2:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
        }

    groups = [
        (
            g["label"].to_numpy(dtype=int),
            g["prob"].to_numpy(dtype=float),
        )
        for _, g in sub.groupby(cluster_col, sort=False)
    ]
    if not groups:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
        }

    rng = np.random.default_rng(seed)
    boot_vals: List[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 10, n_bootstrap)
    n_groups = len(groups)

    while len(boot_vals) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.integers(0, n_groups, size=n_groups)
        y_boot = np.concatenate([groups[i][0] for i in sample_idx])
        if np.unique(y_boot).size < 2:
            continue
        p_boot = np.concatenate([groups[i][1] for i in sample_idx])
        boot_vals.append(float(roc_auc_score(y_boot, p_boot)))

    if not boot_vals:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": attempts,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
        }

    ci_low, ci_high = np.quantile(np.asarray(boot_vals, dtype=float), [0.025, 0.975])
    return {
        "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
        "bootstrap_replicates_requested": n_bootstrap,
        "bootstrap_replicates_used": len(boot_vals),
        "bootstrap_attempts": attempts,
        "auroc_ci95_low": float(ci_low),
        "auroc_ci95_high": float(ci_high),
        "auroc_above_random_95ci": bool(ci_low > 0.5),
    }


def _patient_cluster_bootstrap_threshold_metrics(
    df: pd.DataFrame,
    threshold: float,
    cluster_col: str = "patient_id",
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> Dict[str, object]:
    required = {cluster_col, "label", "prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bootstrap threshold metrics require columns: {missing}")

    sub = df.dropna(subset=[cluster_col, "label", "prob"]).copy()
    if sub.empty or sub["label"].nunique() < 2:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "f1_pos_ci95_low": np.nan,
            "f1_pos_ci95_high": np.nan,
            "sensitivity_ci95_low": np.nan,
            "sensitivity_ci95_high": np.nan,
            "specificity_ci95_low": np.nan,
            "specificity_ci95_high": np.nan,
        }

    groups = [
        (
            g["label"].to_numpy(dtype=int),
            g["prob"].to_numpy(dtype=float),
        )
        for _, g in sub.groupby(cluster_col, sort=False)
    ]
    if not groups:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "f1_pos_ci95_low": np.nan,
            "f1_pos_ci95_high": np.nan,
            "sensitivity_ci95_low": np.nan,
            "sensitivity_ci95_high": np.nan,
            "specificity_ci95_low": np.nan,
            "specificity_ci95_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    boot_f1: List[float] = []
    boot_sens: List[float] = []
    boot_spec: List[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 10, n_bootstrap)
    n_groups = len(groups)

    while len(boot_f1) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.integers(0, n_groups, size=n_groups)
        y_boot = np.concatenate([groups[i][0] for i in sample_idx])
        if np.unique(y_boot).size < 2:
            continue
        p_boot = np.concatenate([groups[i][1] for i in sample_idx])
        metrics = _compute_binary_metrics_from_prob(y_boot, p_boot, threshold=threshold)
        boot_f1.append(metrics["f1_pos"])
        boot_sens.append(metrics["sensitivity"])
        boot_spec.append(metrics["specificity"])

    if not boot_f1:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": attempts,
            "f1_pos_ci95_low": np.nan,
            "f1_pos_ci95_high": np.nan,
            "sensitivity_ci95_low": np.nan,
            "sensitivity_ci95_high": np.nan,
            "specificity_ci95_low": np.nan,
            "specificity_ci95_high": np.nan,
        }

    f1_low, f1_high = np.quantile(np.asarray(boot_f1, dtype=float), [0.025, 0.975])
    sens_low, sens_high = np.quantile(np.asarray(boot_sens, dtype=float), [0.025, 0.975])
    spec_low, spec_high = np.quantile(np.asarray(boot_spec, dtype=float), [0.025, 0.975])
    return {
        "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
        "bootstrap_replicates_requested": n_bootstrap,
        "bootstrap_replicates_used": len(boot_f1),
        "bootstrap_attempts": attempts,
        "f1_pos_ci95_low": float(f1_low),
        "f1_pos_ci95_high": float(f1_high),
        "sensitivity_ci95_low": float(sens_low),
        "sensitivity_ci95_high": float(sens_high),
        "specificity_ci95_low": float(spec_low),
        "specificity_ci95_high": float(spec_high),
    }


def _patient_cluster_bootstrap_discrimination_vs_random(
    df: pd.DataFrame,
    cluster_col: str = "patient_id",
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> Dict[str, object]:
    required = {cluster_col, "label", "prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bootstrap discrimination metrics require columns: {missing}")

    sub = df.dropna(subset=[cluster_col, "label", "prob"]).copy()
    if sub.empty or sub["label"].nunique() < 2:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auprc_ci95_low": np.nan,
            "auprc_ci95_high": np.nan,
            "prevalence_records_ci95_low": np.nan,
            "prevalence_records_ci95_high": np.nan,
            "delta_auroc_vs_random_ci95_low": np.nan,
            "delta_auroc_vs_random_ci95_high": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_low": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
            "auprc_above_random_95ci": False,
        }

    groups = [
        (
            g["label"].to_numpy(dtype=int),
            g["prob"].to_numpy(dtype=float),
        )
        for _, g in sub.groupby(cluster_col, sort=False)
    ]
    if not groups:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": 0,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auprc_ci95_low": np.nan,
            "auprc_ci95_high": np.nan,
            "prevalence_records_ci95_low": np.nan,
            "prevalence_records_ci95_high": np.nan,
            "delta_auroc_vs_random_ci95_low": np.nan,
            "delta_auroc_vs_random_ci95_high": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_low": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
            "auprc_above_random_95ci": False,
        }

    rng = np.random.default_rng(seed)
    boot_auroc: List[float] = []
    boot_auprc: List[float] = []
    boot_prevalence: List[float] = []
    boot_delta_auroc: List[float] = []
    boot_delta_auprc: List[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 10, n_bootstrap)
    n_groups = len(groups)

    while len(boot_auroc) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.integers(0, n_groups, size=n_groups)
        y_boot = np.concatenate([groups[i][0] for i in sample_idx])
        if np.unique(y_boot).size < 2:
            continue
        p_boot = np.concatenate([groups[i][1] for i in sample_idx])
        prevalence_boot = float(np.mean(y_boot))
        auroc_boot = float(roc_auc_score(y_boot, p_boot))
        auprc_boot = float(average_precision_score(y_boot, p_boot))
        boot_auroc.append(auroc_boot)
        boot_auprc.append(auprc_boot)
        boot_prevalence.append(prevalence_boot)
        boot_delta_auroc.append(auroc_boot - 0.5)
        boot_delta_auprc.append(auprc_boot - prevalence_boot)

    if not boot_auroc:
        return {
            "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
            "bootstrap_replicates_requested": n_bootstrap,
            "bootstrap_replicates_used": 0,
            "bootstrap_attempts": attempts,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auprc_ci95_low": np.nan,
            "auprc_ci95_high": np.nan,
            "prevalence_records_ci95_low": np.nan,
            "prevalence_records_ci95_high": np.nan,
            "delta_auroc_vs_random_ci95_low": np.nan,
            "delta_auroc_vs_random_ci95_high": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_low": np.nan,
            "delta_auprc_vs_random_prevalence_ci95_high": np.nan,
            "auroc_above_random_95ci": False,
            "auprc_above_random_95ci": False,
        }

    auroc_low, auroc_high = np.quantile(np.asarray(boot_auroc, dtype=float), [0.025, 0.975])
    auprc_low, auprc_high = np.quantile(np.asarray(boot_auprc, dtype=float), [0.025, 0.975])
    prev_low, prev_high = np.quantile(np.asarray(boot_prevalence, dtype=float), [0.025, 0.975])
    delta_auroc_low, delta_auroc_high = np.quantile(np.asarray(boot_delta_auroc, dtype=float), [0.025, 0.975])
    delta_auprc_low, delta_auprc_high = np.quantile(np.asarray(boot_delta_auprc, dtype=float), [0.025, 0.975])
    return {
        "bootstrap_unit": f"{cluster_col} (cluster bootstrap)",
        "bootstrap_replicates_requested": n_bootstrap,
        "bootstrap_replicates_used": len(boot_auroc),
        "bootstrap_attempts": attempts,
        "auroc_ci95_low": float(auroc_low),
        "auroc_ci95_high": float(auroc_high),
        "auprc_ci95_low": float(auprc_low),
        "auprc_ci95_high": float(auprc_high),
        "prevalence_records_ci95_low": float(prev_low),
        "prevalence_records_ci95_high": float(prev_high),
        "delta_auroc_vs_random_ci95_low": float(delta_auroc_low),
        "delta_auroc_vs_random_ci95_high": float(delta_auroc_high),
        "delta_auprc_vs_random_prevalence_ci95_low": float(delta_auprc_low),
        "delta_auprc_vs_random_prevalence_ci95_high": float(delta_auprc_high),
        "auroc_above_random_95ci": bool(delta_auroc_low > 0.0),
        "auprc_above_random_95ci": bool(delta_auprc_low > 0.0),
    }


def compute_test_discrimination_vs_random_baseline(
    views: Dict[str, pd.DataFrame], results_run_dir: Path
) -> pd.DataFrame:
    if not results_run_dir.exists():
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    final_overall = views.get("final_overall", pd.DataFrame()).copy()
    cross_pairwise = views.get("cross_pairwise", pd.DataFrame()).copy()
    pool_overall = views.get("pool_overall", pd.DataFrame()).copy()

    for device in DEVICE_ORDER:
        row = final_overall[final_overall["train_device_filter"] == device]
        if row.empty:
            continue
        result_row = row.iloc[0]
        run_name = str(result_row["run_name"])
        pred = _load_predictions_file(results_run_dir, run_name, split="test")
        if pred is None or pred.empty:
            continue
        pred = pred.copy()
        pred["patient_id"] = pred["patient_id"].astype(str)
        rows.append(
            {
                "evaluation_group": "final_within",
                "evaluation_label": (
                    f"Best-config within-device model trained on {_pretty_device_after_trained_on(device)} "
                    "(same-device test set)"
                ),
                "run_name": run_name,
                "source_run_name": "",
                "source_device": device,
                "target_device": device,
                "n_records": int(len(pred)),
                "n_patients": int(pred["patient_id"].nunique()),
                "n_positive_records": int(pred["label"].sum()),
                "prevalence_records": float(pred["label"].mean()),
                "auroc": float(roc_auc_score(pred["label"].astype(int), pred["prob"].astype(float))),
                "auprc": float(average_precision_score(pred["label"].astype(int), pred["prob"].astype(float))),
                "random_auroc_baseline": 0.5,
                "random_auprc_baseline": float(pred["label"].mean()),
                **_patient_cluster_bootstrap_discrimination_vs_random(pred, cluster_col="patient_id"),
                }
            )

    for device in DEVICE_ORDER:
        row = final_overall[final_overall["train_device_filter"] == device]
        if row.empty:
            continue
        result_row = row.iloc[0]
        run_name = str(result_row["run_name"])
        pred = _assemble_pooled_test_predictions_for_best_within_model(views, results_run_dir, run_name)
        if pred is None or pred.empty:
            continue
        pred = pred.copy()
        pred["patient_id"] = pred["patient_id"].astype(str)
        rows.append(
            {
                "evaluation_group": "final_within_pooled_test",
                "evaluation_label": (
                    f"Best-config within-device model trained on {_pretty_device_after_trained_on(device)} "
                    "(pooled test set)"
                ),
                "run_name": run_name,
                "source_run_name": "",
                "source_device": device,
                "target_device": "all_devices",
                "n_records": int(len(pred)),
                "n_patients": int(pred["patient_id"].nunique()),
                "n_positive_records": int(pred["label"].sum()),
                "prevalence_records": float(pred["label"].mean()),
                "auroc": float(roc_auc_score(pred["label"].astype(int), pred["prob"].astype(float))),
                "auprc": float(average_precision_score(pred["label"].astype(int), pred["prob"].astype(float))),
                "random_auroc_baseline": 0.5,
                "random_auprc_baseline": float(pred["label"].mean()),
                **_patient_cluster_bootstrap_discrimination_vs_random(pred, cluster_col="patient_id"),
            }
        )

    for source_device in DEVICE_ORDER:
        for target_device in DEVICE_ORDER:
            if source_device == target_device:
                continue
            row = cross_pairwise[
                (cross_pairwise["source_device"] == source_device)
                & (cross_pairwise["target_device"] == target_device)
            ]
            if row.empty:
                continue
            result_row = row.iloc[0]
            run_name = str(result_row["run_name"])
            pred = _load_predictions_file(results_run_dir, run_name, split="test")
            if pred is None or pred.empty:
                continue
            pred = pred.copy()
            pred = pred[pred["device"].astype(str) == target_device].copy()
            if pred.empty or pred["label"].nunique() < 2:
                continue
            pred["patient_id"] = pred["patient_id"].astype(str)
            rows.append(
                {
                    "evaluation_group": "cross_pairwise",
                    "evaluation_label": f"Cross-device: {_pretty_device(source_device)} -> {_pretty_device(target_device)}",
                    "run_name": run_name,
                    "source_run_name": str(result_row["source_run_name"]),
                    "source_device": source_device,
                    "target_device": target_device,
                    "n_records": int(len(pred)),
                    "n_patients": int(pred["patient_id"].nunique()),
                    "n_positive_records": int(pred["label"].sum()),
                    "prevalence_records": float(pred["label"].mean()),
                    "auroc": float(roc_auc_score(pred["label"].astype(int), pred["prob"].astype(float))),
                    "auprc": float(average_precision_score(pred["label"].astype(int), pred["prob"].astype(float))),
                    "random_auroc_baseline": 0.5,
                    "random_auprc_baseline": float(pred["label"].mean()),
                    **_patient_cluster_bootstrap_discrimination_vs_random(pred, cluster_col="patient_id"),
                }
            )

    if not pool_overall.empty:
        result_row = pool_overall.iloc[0]
        run_name = str(result_row["run_name"])
        pred = _load_predictions_file(results_run_dir, run_name, split="test")
        if pred is not None and not pred.empty:
            pred = pred.copy()
            pred["patient_id"] = pred["patient_id"].astype(str)
            rows.append(
                {
                    "evaluation_group": "pool_overall",
                    "evaluation_label": "Pooled-device model trained on all devices (pooled test set)",
                    "run_name": run_name,
                    "source_run_name": "",
                    "source_device": "pooled_all_devices",
                    "target_device": "all_devices",
                    "n_records": int(len(pred)),
                    "n_patients": int(pred["patient_id"].nunique()),
                    "n_positive_records": int(pred["label"].sum()),
                    "prevalence_records": float(pred["label"].mean()),
                    "auroc": float(roc_auc_score(pred["label"].astype(int), pred["prob"].astype(float))),
                    "auprc": float(average_precision_score(pred["label"].astype(int), pred["prob"].astype(float))),
                    "random_auroc_baseline": 0.5,
                    "random_auprc_baseline": float(pred["label"].mean()),
                    **_patient_cluster_bootstrap_discrimination_vs_random(pred, cluster_col="patient_id"),
                }
            )

    return pd.DataFrame(rows)

def compute_pooled_auroc_by_auscultation_site(
    views: Dict[str, pd.DataFrame], results_run_dir: Path
) -> pd.DataFrame:
    pool_overall = views.get("pool_overall", pd.DataFrame()).copy()
    if pool_overall.empty or not results_run_dir.exists():
        return pd.DataFrame()

    pool_row = pool_overall.iloc[0]
    run_name = str(pool_row["run_name"])
    pred = _load_predictions_file(results_run_dir, run_name, split="test")
    if pred is None or pred.empty:
        return pd.DataFrame()

    required_cols = {"patient_id", "position"}
    missing = sorted(required_cols - set(pred.columns))
    if missing:
        raise ValueError(
            f"{results_run_dir / run_name / 'predictions_test.csv'} is missing required columns: {missing}"
        )

    pred = pred.copy()
    pred["patient_id"] = pred["patient_id"].astype(str)
    pred["position"] = pred["position"].astype(str).str.strip().str.upper()

    rows: List[Dict[str, object]] = []
    for site_code in AUSCULTATION_SITE_ORDER:
        sub = pred[pred["position"] == site_code].copy()
        if sub.empty or sub["label"].nunique() < 2:
            continue
        boot = _patient_cluster_bootstrap_auroc(
            sub,
            cluster_col="patient_id",
            n_bootstrap=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_RANDOM_SEED,
        )
        rows.append(
            {
                "run_name": run_name,
                "representation": str(pool_row["representation"]),
                "backbone": str(pool_row["backbone"]),
                "tuned_threshold": float(pool_row["tuned_threshold"]),
                "site_code": site_code,
                "auscultation_site": AUSCULTATION_SITE_LABELS.get(site_code, site_code),
                "n_records": int(len(sub)),
                "n_patients": int(sub["patient_id"].nunique()),
                "n_positive_records": int(sub["label"].sum()),
                "prevalence_records": float(sub["label"].mean()),
                "auroc": float(roc_auc_score(sub["label"].astype(int), sub["prob"].astype(float))),
                "random_auroc_baseline": 0.5,
                **boot,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["site_code"] = pd.Categorical(out["site_code"], categories=AUSCULTATION_SITE_ORDER, ordered=True)
    out = out.sort_values("site_code").reset_index(drop=True)
    return out

def _build_results_prediction_bundle(
    views: Dict[str, pd.DataFrame], results_run_dir: Path
) -> Dict[str, object]:
    bundle: Dict[str, object] = {
        "available": False,
        "within_vs_pooled_test_by_device": {},
        "training_histories": {},
    }
    if not results_run_dir.exists():
        return bundle

    final_overall = views.get("final_overall", pd.DataFrame()).copy()
    pool_overall = views.get("pool_overall", pd.DataFrame()).copy()
    if final_overall.empty or pool_overall.empty:
        return bundle

    pool_run_name = str(pool_overall.iloc[0]["run_name"])
    pool_test = _load_predictions_file(results_run_dir, pool_run_name, split="test")
    if pool_test is None or pool_test.empty:
        return bundle

    within_vs_pooled: Dict[str, Dict[str, object]] = {}
    for device in DEVICE_ORDER:
        row = final_overall[final_overall["train_device_filter"] == device]
        if row.empty:
            continue
        within_run_name = str(row.iloc[0]["run_name"])
        within_test = _load_predictions_file(results_run_dir, within_run_name, split="test")
        if within_test is None or within_test.empty:
            continue
        within_device_df = within_test[within_test["device"].astype(str) == device].copy()
        pooled_device_df = pool_test[pool_test["device"].astype(str) == device].copy()
        if within_device_df.empty or pooled_device_df.empty:
            continue
        within_vs_pooled[device] = {
            "within_run_name": within_run_name,
            "pooled_run_name": pool_run_name,
            "within": within_device_df,
            "pooled": pooled_device_df,
        }
    if not within_vs_pooled:
        return bundle

    # Training histories for 3 final within-device runs + pooled run (if present)
    history_records: Dict[str, Dict[str, object]] = {}
    for _, r in final_overall.iterrows():
        run_name = str(r["run_name"])
        hist = _load_history_file(results_run_dir, run_name)
        if hist is None or hist.empty:
            continue
        history_records[run_name] = {
            "panel_key": str(r["train_device_filter"]),
            "panel_label": _pretty_device(str(r["train_device_filter"])),
            "run_name": run_name,
            "kind": "final_within",
            "history": hist.copy(),
        }
    pool_hist = _load_history_file(results_run_dir, pool_run_name)
    if pool_hist is not None and not pool_hist.empty:
        history_records[pool_run_name] = {
            "panel_key": "pooled_all_devices",
            "panel_label": "Pooled (all devices)",
            "run_name": pool_run_name,
            "kind": "pooled",
            "history": pool_hist.copy(),
        }

    bundle["available"] = True
    bundle["within_vs_pooled_test_by_device"] = within_vs_pooled
    bundle["training_histories"] = history_records
    return bundle

def compute_backbone_parameter_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for backbone in BACKBONE_ORDER:
        if backbone not in BACKBONE_CONFIGS:
            continue
        model = create_model(backbone=backbone, pretrained=False, num_classes=1)
        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        rows.append(
            {
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS.get(backbone, backbone),
                "timm_model_name": BACKBONE_CONFIGS[backbone],
                "model_family": "transformer" if "swin" in backbone else "cnn",
                "num_classes": 1,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "total_params_millions": total_params / 1e6,
                "trainable_params_millions": trainable_params / 1e6,
            }
        )
        del model
    return pd.DataFrame(rows)
