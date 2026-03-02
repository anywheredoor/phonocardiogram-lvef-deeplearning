#!/usr/bin/env python3
"""Table builders for dissertation summary outputs."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from dissertation_common import (
    EXPERIMENT_LABELS,
    METRIC_COLS,
    _device_or_all_categorical,
    _sort_by_device_columns,
)

def build_summary_tables(
    df: pd.DataFrame,
    run_catalog: pd.DataFrame,
    views: Dict[str, pd.DataFrame],
    cv_agg: pd.DataFrame,
    cv_best: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    # Row and run inventory to document summary.csv structure
    row_counts = (
        df.groupby(["run_kind", "metric_scope"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["run_kind", "metric_scope"])
        .reset_index(drop=True)
    )
    run_counts = (
        run_catalog.groupby("run_kind", dropna=False)
        .agg(
            run_count=("run_name", "count"),
            total_rows=("n_rows", "sum"),
            row_pattern_examples=("n_rows", lambda s: ",".join(map(str, sorted(s.unique())))),
        )
        .reset_index()
        .sort_values("run_kind")
        .reset_index(drop=True)
    )
    tables["table_01a_summary_csv_run_inventory_by_kind.csv"] = run_counts
    tables["table_01b_summary_csv_row_counts_by_kind_and_metric_scope.csv"] = row_counts

    tables["table_02_cv_aggregated_config_metrics_by_training_device.csv"] = _sort_by_device_columns(
        cv_agg, ["train_device_filter"]
    )
    tables["table_03_cv_best_config_per_device.csv"] = _sort_by_device_columns(
        cv_best, ["train_device_filter"]
    )

    final_overall = views["final_overall"].copy()
    if not final_overall.empty:
        cols = [
            "run_name",
            "train_device_filter",
            "representation",
            "backbone",
            "image_size",
            "tuned_threshold",
        ] + METRIC_COLS
        tables["table_04_final_within_device_overall_results.csv"] = _sort_by_device_columns(
            final_overall[cols], ["train_device_filter"]
        )

    cross_pairwise = views["cross_pairwise"].copy()
    if not cross_pairwise.empty:
        cols = [
            "run_name",
            "source_run_name",
            "source_device",
            "target_device",
            "source_representation",
            "source_backbone",
            "tuned_threshold",
        ] + METRIC_COLS
        tables["table_05_cross_device_pairwise_results_source_to_target.csv"] = _sort_by_device_columns(
            cross_pairwise[cols], ["source_device", "target_device"]
        )

    cross_overall = views["cross_overall"].copy()
    if not cross_overall.empty:
        cols = [
            "run_name",
            "source_run_name",
            "source_device",
            "source_representation",
            "source_backbone",
            "test_device_filter",
            "tuned_threshold",
        ] + METRIC_COLS
        tables["table_06_cross_device_overall_results_by_source_model.csv"] = _sort_by_device_columns(
            cross_overall[cols], ["source_device"]
        )

    pool_rows = views["pool_rows"].copy()
    if not pool_rows.empty:
        pool_rows = pool_rows.copy()
        pool_rows["result_scope"] = np.where(
            pool_rows["metric_scope"] == "overall", "overall_aggregate", "per_target_device"
        )
        pool_rows["target_device"] = pool_rows["device"].replace("", pd.NA)
        pool_rows.loc[pool_rows["metric_scope"] == "overall", "target_device"] = "all_devices"
        cols = [
            "run_name",
            "result_scope",
            "target_device",
            "representation",
            "backbone",
            "image_size",
            "tuned_threshold",
        ] + METRIC_COLS
        pool_table = pool_rows[cols].copy()
        pool_table["__result_scope_sort"] = pd.Categorical(
            pool_table["result_scope"],
            categories=["overall_aggregate", "per_target_device"],
            ordered=True,
        )
        pool_table["__target_sort"] = _device_or_all_categorical(pool_table["target_device"])
        pool_table = (
            pool_table.sort_values(["__result_scope_sort", "__target_sort"])
            .drop(columns=["__result_scope_sort", "__target_sort"])
            .reset_index(drop=True)
        )
        tables["table_07_pooled_device_results_overall_and_per_device.csv"] = pool_table

    # Pooled vs within deltas per target device
    final_per = views["final_per_device"].copy()
    pool_per = views["pool_per_device"].copy()
    if not final_per.empty and not pool_per.empty:
        fw = final_per[["target_device"] + METRIC_COLS].rename(
            columns={m: f"{m}_within" for m in METRIC_COLS}
        )
        pp = pool_per[["target_device"] + METRIC_COLS].rename(
            columns={m: f"{m}_pool" for m in METRIC_COLS}
        )
        delta = fw.merge(pp, on="target_device", how="inner")
        for m in METRIC_COLS:
            delta[f"{m}_delta_pool_minus_within"] = delta[f"{m}_pool"] - delta[f"{m}_within"]
        tables["table_08_pooled_vs_final_within_device_metric_deltas.csv"] = _sort_by_device_columns(
            delta, ["target_device"]
        )

    # Cross-device gaps relative to target within-device and pooled target
    if not cross_pairwise.empty and not final_per.empty:
        within_map = final_per.set_index("target_device")
        pool_map = pool_per.set_index("target_device") if not pool_per.empty else None
        gap_rows: List[Dict[str, object]] = []
        for _, r in cross_pairwise.iterrows():
            out: Dict[str, object] = {
                "source_device": r.get("source_device", ""),
                "target_device": r.get("target_device", ""),
                "source_representation": r.get("source_representation", ""),
                "source_backbone": r.get("source_backbone", ""),
                "source_to_target": r.get("source_to_target", ""),
            }
            tgt = r.get("target_device", "")
            for m in METRIC_COLS:
                out[m] = r[m]
                if tgt in within_map.index:
                    out[f"{m}_gap_vs_target_within"] = r[m] - within_map.loc[tgt, m]
                if pool_map is not None and tgt in pool_map.index:
                    out[f"{m}_gap_vs_pooled_target"] = r[m] - pool_map.loc[tgt, m]
            gap_rows.append(out)
        tables["table_09_cross_device_pairwise_gaps_vs_within_and_pooled_target.csv"] = _sort_by_device_columns(
            pd.DataFrame(gap_rows), ["source_device", "target_device"]
        )

    # Combined headline results (long format) for thesis manuscript table drafting
    long_rows: List[pd.DataFrame] = []
    if not final_overall.empty:
        tmp = final_overall.copy()
        tmp["experiment_group"] = "final_within"
        tmp["experiment_label"] = EXPERIMENT_LABELS["final_within"]
        tmp["source_device"] = tmp["train_device_filter"]
        tmp["target_device"] = tmp["train_device_filter"]
        long_rows.append(
            tmp[
                [
                    "experiment_group",
                    "experiment_label",
                    "run_name",
                    "source_device",
                    "target_device",
                    "representation",
                    "backbone",
                    "metric_scope",
                ]
                + METRIC_COLS
            ]
        )
    if not views["cross_pairwise"].empty:
        tmp = views["cross_pairwise"].copy()
        tmp["experiment_group"] = "cross_pairwise"
        tmp["experiment_label"] = EXPERIMENT_LABELS["cross_pairwise"]
        tmp["representation"] = tmp["source_representation"]
        tmp["backbone"] = tmp["source_backbone"]
        long_rows.append(
            tmp[
                [
                    "experiment_group",
                    "experiment_label",
                    "run_name",
                    "source_device",
                    "target_device",
                    "representation",
                    "backbone",
                    "metric_scope",
                ]
                + METRIC_COLS
            ]
        )
    if not views["cross_overall"].empty:
        tmp = views["cross_overall"].copy()
        tmp["experiment_group"] = "cross_overall"
        tmp["experiment_label"] = EXPERIMENT_LABELS["cross_overall"]
        tmp["representation"] = tmp["source_representation"]
        tmp["backbone"] = tmp["source_backbone"]
        tmp["source_device"] = tmp["source_device"]
        tmp["target_device"] = tmp["test_device_filter"]
        long_rows.append(
            tmp[
                [
                    "experiment_group",
                    "experiment_label",
                    "run_name",
                    "source_device",
                    "target_device",
                    "representation",
                    "backbone",
                    "metric_scope",
                ]
                + METRIC_COLS
            ]
        )
    if not views["pool_overall"].empty:
        tmp = views["pool_overall"].copy()
        tmp["experiment_group"] = "pool_overall"
        tmp["experiment_label"] = EXPERIMENT_LABELS["pool_overall"]
        tmp["source_device"] = "pooled_all_devices"
        tmp["target_device"] = "all_devices"
        long_rows.append(
            tmp[
                [
                    "experiment_group",
                    "experiment_label",
                    "run_name",
                    "source_device",
                    "target_device",
                    "representation",
                    "backbone",
                    "metric_scope",
                ]
                + METRIC_COLS
            ]
        )
    if not views["pool_per_device"].empty:
        tmp = views["pool_per_device"].copy()
        tmp["experiment_group"] = "pool_per_device"
        tmp["experiment_label"] = EXPERIMENT_LABELS["pool_per_device"]
        tmp["source_device"] = "pooled_all_devices"
        tmp["target_device"] = tmp["target_device"]
        long_rows.append(
            tmp[
                [
                    "experiment_group",
                    "experiment_label",
                    "run_name",
                    "source_device",
                    "target_device",
                    "representation",
                    "backbone",
                    "metric_scope",
                ]
                + METRIC_COLS
            ]
        )
    if long_rows:
        headline = pd.concat(long_rows, ignore_index=True)
        headline = _sort_by_device_columns(headline, ["source_device", "target_device"])
        headline = headline.sort_values(["experiment_group"]).reset_index(drop=True)
        tables["table_10_headline_results_long_format_for_dissertation.csv"] = headline

    return tables
