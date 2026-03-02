#!/usr/bin/env python3
"""Shared helpers for dissertation figures and tables."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MPL_CACHE = _SCRIPT_DIR.parent / "reports" / ".matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
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
from sklearn.metrics import roc_auc_score

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

def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        p_bin = float(np.mean(y_prob[mask]))
        y_bin = float(np.mean(y_true[mask]))
        ece += abs(y_bin - p_bin) * (np.sum(mask) / n)
    return float(ece)

def _curve_inputs_from_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    if df is None or df.empty:
        return None
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    return y_true, y_prob

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
