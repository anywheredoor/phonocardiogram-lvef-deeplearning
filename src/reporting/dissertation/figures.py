#!/usr/bin/env python3
"""Figure builders for dissertation summary outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import roc_auc_score, roc_curve

from src.reporting.dissertation.common import (
    AUSCULTATION_SITE_ORDER,
    BACKBONE_LABELS,
    BACKBONE_ORDER,
    DEVICE_ORDER,
    REPRESENTATION_ORDER,
    _assemble_pooled_test_predictions_for_best_within_model,
    _curve_inputs_from_predictions,
    _load_predictions_file,
    _pretty_device,
    _pretty_device_after_trained_on,
    _save_fig,
)

def _annot_matrix_from_mean_sd(
    mean_mat: pd.DataFrame,
    sd_mat: pd.DataFrame | None = None,
    decimals: int = 3,
    sep: str = "\n",
) -> np.ndarray:
    annot = np.empty(mean_mat.shape, dtype=object)
    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            v = mean_mat.iloc[i, j]
            if pd.isna(v):
                annot[i, j] = ""
                continue
            if sd_mat is None or sd_mat.empty:
                annot[i, j] = f"{v:.{decimals}f}"
            else:
                s = sd_mat.iloc[i, j]
                annot[i, j] = f"{v:.{decimals}f}{sep}± {s:.{decimals}f}" if pd.notna(s) else f"{v:.{decimals}f}"
    return annot

def plot_cv_heatmap_mean_f1(
    cv_agg: pd.DataFrame, cv_best: pd.DataFrame, figures_dir: Path, dpi: int
) -> None:
    heatmap_backbone_order = [
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_large",
        "efficientnetv2_s",
        "swinv2_tiny",
        "swinv2_small",
    ]
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7.0), constrained_layout=True)
    vmin = cv_agg["mean_test_f1_pos"].min()
    vmax = cv_agg["mean_test_f1_pos"].max()

    for ax, device in zip(axes, DEVICE_ORDER):
        sub = cv_agg[cv_agg["train_device_filter"] == device].copy()
        sub["backbone"] = pd.Categorical(sub["backbone"], heatmap_backbone_order, ordered=True)
        sub["representation"] = pd.Categorical(
            sub["representation"], REPRESENTATION_ORDER, ordered=True
        )
        sub = sub.sort_values(["backbone", "representation"])

        mean_mat = sub.pivot(index="backbone", columns="representation", values="mean_test_f1_pos")
        sd_mat = sub.pivot(index="backbone", columns="representation", values="sd_test_f1_pos")
        mean_mat = mean_mat.reindex(index=heatmap_backbone_order, columns=REPRESENTATION_ORDER)
        sd_mat = sd_mat.reindex(index=heatmap_backbone_order, columns=REPRESENTATION_ORDER)
        annot = _annot_matrix_from_mean_sd(mean_mat, sd_mat, sep=" ")

        hm = sns.heatmap(
            mean_mat,
            ax=ax,
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt="",
            linewidths=0.8,
            linecolor="white",
            cbar=(device == DEVICE_ORDER[-1]),
            cbar_kws={"label": "Mean cross-validation test F1"},
            annot_kws={"fontsize": 10.2, "linespacing": 1.08},
        )
        ax.set_title(_pretty_device(device))
        ax.set_xlabel("Representation", labelpad=10)
        if ax is axes[0]:
            ax.set_ylabel("Backbone", labelpad=14)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_xticklabels(["MFCC", "Gammatone"], rotation=0)
        ax.set_yticklabels(
            [BACKBONE_LABELS.get(t.get_text(), t.get_text().replace("_", " ")) for t in ax.get_yticklabels()],
            rotation=0,
        )

        if device == DEVICE_ORDER[-1] and hm.collections:
            cbar = hm.collections[0].colorbar
            if cbar is not None:
                cbar.set_label("Mean cross-validation test F1", labelpad=14)

        # Denote top-3 configurations per device by mean cross-validation test F1.
        ranked = (
            sub[["backbone", "representation", "mean_test_f1_pos"]]
            .dropna()
            .sort_values(["mean_test_f1_pos"], ascending=[False])
            .head(3)
            .reset_index(drop=True)
        )
        for rank_idx, row in ranked.iterrows():
            bb = str(row["backbone"])
            rep = str(row["representation"])
            if bb not in mean_mat.index or rep not in mean_mat.columns:
                continue
            r = list(mean_mat.index).index(bb)
            c = list(mean_mat.columns).index(rep)
            ax.text(
                c + 0.06,
                r + 0.09,
                f"#{rank_idx + 1}",
                ha="left",
                va="top",
                fontsize=8.3,
                color="#111111",
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
                zorder=4.6,
            )

        # Highlight selected best config with a subtle inset red border.
        best_row = cv_best[cv_best["train_device_filter"] == device]
        if not best_row.empty:
            best_backbone = str(best_row.iloc[0]["backbone"])
            best_repr = str(best_row.iloc[0]["representation"])
            try:
                r = list(mean_mat.index).index(best_backbone)
                c = list(mean_mat.columns).index(best_repr)
                inset = 0.025
                ax.add_patch(
                    Rectangle(
                        (c + inset, r + inset),
                        1 - 2 * inset,
                        1 - 2 * inset,
                        fill=False,
                        edgecolor="#FF3B30",
                        linewidth=2.0,
                        joinstyle="round",
                        zorder=10,
                    )
                )
            except ValueError:
                pass

    _save_fig(
        fig,
        figures_dir / "figure_02_cv_model_selection_heatmap_mean_test_f1_by_training_device",
        dpi=dpi,
    )

def plot_backbone_parameter_sizes(
    param_df: pd.DataFrame, figures_dir: Path, dpi: int
) -> None:
    if param_df.empty:
        return

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Smallest-to-largest for consistency with backbone progression in other figures.
    plot_df = param_df.copy().sort_values("total_params_millions", ascending=True).reset_index(drop=True)
    family_colors = {"cnn": "#FF7F0E", "transformer": "#2E86DE"}
    bar_colors = [family_colors.get(f, "#666666") for f in plot_df["model_family"]]

    fig, ax = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    bars = ax.barh(
        plot_df["backbone_label"],
        plot_df["total_params_millions"],
        color=bar_colors,
        edgecolor="#FFFFFF",
        linewidth=1.0,
        height=0.48,
    )
    ax.invert_yaxis()

    x_max = float(plot_df["total_params_millions"].max())
    ax.set_xlim(0, x_max * 1.25)
    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Backbone", labelpad=18)
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False, bottom=False)

    # Annotate with exact values in millions.
    for bar, v in zip(bars, plot_df["total_params_millions"]):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}M",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold",
            color="#2F2F2F",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.95,
            ),
            zorder=5,
        )

    # Legend by model family
    handles = [
        plt.Line2D([0], [0], color=family_colors["cnn"], lw=8),
        plt.Line2D([0], [0], color=family_colors["transformer"], lw=8),
    ]
    labels = ["CNN", "Transformer"]
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.06),
        borderaxespad=0.0,
    )

    _save_fig(
        fig,
        figures_dir / "figure_01_backbone_parameter_sizes_used_in_experiments",
        dpi=dpi,
    )

def plot_cv_performance_vs_parameter_tradeoff(
    cv_agg: pd.DataFrame,
    cv_best: pd.DataFrame,
    param_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    if cv_agg.empty or param_df.empty:
        return

    plot_df = cv_agg.merge(
        param_df[["backbone", "total_params_millions"]],
        on="backbone",
        how="left",
    )
    plot_df = plot_df.dropna(subset=["total_params_millions", "mean_test_f1_pos"]).copy()
    if plot_df.empty:
        return
    plot_df["representation"] = pd.Categorical(
        plot_df["representation"], REPRESENTATION_ORDER, ordered=True
    )

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rep_colors = {"mfcc": "#FF7F0E", "gammatone": "#2E86DE"}
    uniform_point_size = 66
    uniform_legend_size = 6.3
    smaller_point_size = 46
    smaller_legend_size = 4.9
    backbone_point_style = {
        "mobilenetv2": {"marker": "D", "size": smaller_point_size, "edgewidth": 0.95},
        "mobilenetv3_large": {"marker": "s", "size": smaller_point_size, "edgewidth": 0.95},
        "efficientnet_b0": {"marker": "v", "size": uniform_point_size, "edgewidth": 0.95},
        "efficientnetv2_s": {"marker": "^", "size": uniform_point_size, "edgewidth": 0.95},
        "swinv2_tiny": {"marker": "P", "size": uniform_point_size, "edgewidth": 0.95},
        "swinv2_small": {"marker": "X", "size": uniform_point_size, "edgewidth": 0.95},
    }
    backbone_legend_sizes = {b: uniform_legend_size for b in BACKBONE_ORDER}
    backbone_legend_sizes["mobilenetv2"] = smaller_legend_size
    backbone_legend_sizes["mobilenetv3_large"] = smaller_legend_size
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), sharey=True, constrained_layout=False)
    y_min = float(plot_df["mean_test_f1_pos"].min())
    y_max = float(plot_df["mean_test_f1_pos"].max())
    y_pad = max(0.012, (y_max - y_min) * 0.12)
    tick_vals = np.array([2, 5, 10, 20, 50], dtype=float)

    for ax, device in zip(axes, DEVICE_ORDER):
        sub = plot_df[plot_df["train_device_filter"] == device].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["backbone", "representation"])

        # Connect MFCC and Gammatone results for the same backbone (same parameter count).
        for backbone, g in sub.groupby("backbone"):
            if len(g) < 2:
                continue
            x = float(g["total_params_millions"].iloc[0])
            ys = g["mean_test_f1_pos"].astype(float).tolist()
            ax.plot(
                [x, x],
                [min(ys), max(ys)],
                color="#1E1E1E",
                linewidth=1.15,
                alpha=0.75,
                zorder=1,
            )

        for rep in REPRESENTATION_ORDER:
            draw_backbones = list(BACKBONE_ORDER)
            # Ensure Android MFCC upside-down triangle (EfficientNet-B0) is visible on top.
            if device == "android_phone" and rep == "mfcc" and "efficientnet_b0" in draw_backbones:
                draw_backbones = [b for b in draw_backbones if b != "efficientnet_b0"] + ["efficientnet_b0"]
            for backbone in draw_backbones:
                g = sub[(sub["representation"] == rep) & (sub["backbone"] == backbone)]
                if g.empty:
                    continue
                style = backbone_point_style.get(
                    backbone, {"marker": "o", "size": 66, "edgewidth": 0.95}
                )
                point_zorder = 3.2
                # In Android panel, keep MFCC upside-down triangle above Gammatone square.
                if device == "android_phone" and rep == "gammatone" and backbone == "mobilenetv3_large":
                    point_zorder = 3.0
                if device == "android_phone" and rep == "mfcc" and backbone == "efficientnet_b0":
                    point_zorder = 3.7
                ax.scatter(
                    g["total_params_millions"],
                    g["mean_test_f1_pos"],
                    s=style["size"],
                    marker=style["marker"],
                    c=rep_colors.get(rep, "#666666"),
                    edgecolors="#1E1E1E",
                    linewidths=0.45,
                    alpha=0.98,
                    zorder=point_zorder,
                )

        # Highlight the selected best config used throughout the within-device reporting.
        best_row = cv_best[cv_best["train_device_filter"] == device]
        if not best_row.empty:
            best_backbone = str(best_row.iloc[0]["backbone"])
            best_repr = str(best_row.iloc[0]["representation"])
            g = sub[(sub["backbone"] == best_backbone) & (sub["representation"] == best_repr)]
            if not g.empty:
                ax.scatter(
                    g["total_params_millions"],
                    g["mean_test_f1_pos"],
                    s=162,
                    facecolors="none",
                    edgecolors="#FF3B30",
                    linewidths=2.0,
                    zorder=4.2,
                )

        ax.set_title(_pretty_device(device))
        ax.set_xscale("log")
        ax.set_xlim(1.7, 60)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(int(v)) for v in tick_vals])
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Model parameters (millions, log scale)")
        ax.set_ylabel("Mean cross-validation test F1", labelpad=10)
        ax.grid(axis="x", color="#E4E4E4", linewidth=0.8)
        ax.grid(axis="y", color="#EFEFEF", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.minorticks_off()
        sns.despine(ax=ax)

    legend_handles = [
        Patch(facecolor=rep_colors["mfcc"], edgecolor="none", label="MFCC (orange color filled in symbols)"),
        Patch(facecolor=rep_colors["gammatone"], edgecolor="none", label="Gammatone (blue color filled in symbols)"),
        plt.Line2D([0], [0], color="#1E1E1E", alpha=0.75, lw=1.2, label="Paired Gammatone/MFCC (same backbone)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#FF3B30",
            markeredgewidth=1.8,
            markersize=7.8,
            label="best-config within-device (selected)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.852, 0.665),
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.5,
        handlelength=1.35,
        prop={"size": 9},
    )
    legend_backbone_order = [
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_large",
        "efficientnetv2_s",
        "swinv2_tiny",
        "swinv2_small",
    ]
    backbone_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=backbone_point_style[b]["marker"],
            linestyle="None",
            markerfacecolor="#6A6A6A",
            markeredgecolor="#1E1E1E",
            markeredgewidth=0.8,
            markersize=backbone_legend_sizes.get(b, 6.3),
            label=BACKBONE_LABELS.get(b, b),
        )
        for b in legend_backbone_order
    ]
    fig.legend(
        handles=backbone_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.852, 0.46),
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.30,
        handlelength=1.35,
        prop={"size": 8.5},
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.08, right=0.82, wspace=0.33)

    _save_fig(
        fig,
        figures_dir / "figure_03_cross_validation_performance_vs_parameter_size_tradeoff_by_device",
        dpi=dpi,
    )

def plot_representation_effect_gammatone_minus_mfcc(
    cv_agg: pd.DataFrame, figures_dir: Path, dpi: int
) -> None:
    if cv_agg.empty:
        return
    rep_effect_backbone_order = [
        "mobilenetv2",
        "efficientnet_b0",
        "mobilenetv3_large",
        "efficientnetv2_s",
        "swinv2_tiny",
        "swinv2_small",
    ]

    pivot = cv_agg.pivot_table(
        index=["train_device_filter", "backbone"],
        columns="representation",
        values="mean_test_f1_pos",
        aggfunc="first",
        observed=False,
    ).reset_index()
    if "gammatone" not in pivot.columns or "mfcc" not in pivot.columns:
        return
    pivot["delta_f1_gammatone_minus_mfcc"] = pivot["gammatone"] - pivot["mfcc"]
    pivot["backbone"] = pd.Categorical(pivot["backbone"], rep_effect_backbone_order, ordered=True)
    pivot["train_device_filter"] = pd.Categorical(
        pivot["train_device_filter"], DEVICE_ORDER, ordered=True
    )
    pivot = pivot.sort_values(["backbone", "train_device_filter"]).reset_index(drop=True)

    heat = pivot.pivot_table(
        index="backbone",
        columns="train_device_filter",
        values="delta_f1_gammatone_minus_mfcc",
        aggfunc="first",
        observed=False,
    ).reindex(index=rep_effect_backbone_order, columns=DEVICE_ORDER)
    if heat.dropna(how="all").empty:
        return
    heat.index = [BACKBONE_LABELS.get(b, b) for b in heat.index]
    heat.columns = [_pretty_device(d) for d in heat.columns]

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    delta_max = float(np.nanmax(np.abs(heat.values)))
    lim = max(0.03, delta_max)
    annot = np.array(
        [[("" if pd.isna(v) else f"{v:+.3f}") for v in row] for row in heat.to_numpy()],
        dtype=object,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=False)
    cmap = sns.diverging_palette(20, 240, s=95, l=50, center="light", as_cmap=True)
    hm = sns.heatmap(
        heat,
        ax=ax,
        cmap=cmap,
        center=0.0,
        vmin=-lim,
        vmax=lim,
        linewidths=0.8,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar=True,
        cbar_kws={"label": "Δ mean cross-validation test F1 (Gammatone − MFCC)"},
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Training device", labelpad=10)
    ax.set_ylabel("Backbone", labelpad=12)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    if hm.collections:
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Δ mean cross-validation test F1 (Gammatone − MFCC)", labelpad=14)
    # Improve text contrast on strong positive/negative cells.
    threshold = lim * 0.55
    for txt in ax.texts:
        try:
            val = float(txt.get_text())
        except Exception:
            continue
        txt.set_color("white" if abs(val) >= threshold else "#222222")
        txt.set_fontweight("semibold" if abs(val) >= threshold else "normal")
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.18, right=0.95)

    _save_fig(
        fig,
        figures_dir / "figure_04_representation_effect_gammatone_minus_mfcc_cross_validation_test_f1",
        dpi=dpi,
    )

def plot_source_model_transfer_roc_curves(
    views: Dict[str, pd.DataFrame], results_run_dir: Path, figures_dir: Path, dpi: int
) -> None:
    """Figure 5: per-source ROC curves (within-device + two cross-device targets)."""
    if not results_run_dir.exists():
        return

    final_overall = views.get("final_overall", pd.DataFrame()).copy()
    cross_pairwise = views.get("cross_pairwise", pd.DataFrame()).copy()
    if final_overall.empty or cross_pairwise.empty:
        return

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    target_colors = {
        "iphone": "#FF7F0E",
        "android_phone": "#2E86DE",
        "digital_stethoscope": "#27AE60",
    }
    pred_cache: Dict[str, pd.DataFrame] = {}

    def get_pred(run_name: str) -> pd.DataFrame | None:
        if run_name in pred_cache:
            return pred_cache[run_name]
        df = _load_predictions_file(results_run_dir, run_name, split="test")
        pred_cache[run_name] = df
        return df

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.6), sharex=True, sharey=True, constrained_layout=False)

    for ax, source_device in zip(axes, DEVICE_ORDER):
        legend_handles: List[plt.Line2D] = []
        legend_labels: List[str] = []

        source_label = _pretty_device_after_trained_on(source_device)
        ax.set_title(f"Best-config within-device model\ntrained on {source_label}")

        # Within-device curve from final run.
        final_row = final_overall[final_overall["train_device_filter"] == source_device]
        if final_row.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#666666")
            continue
        within_run = str(final_row.iloc[0]["run_name"])
        within_pred = get_pred(within_run)
        if within_pred is not None and not within_pred.empty:
            within_df = within_pred[within_pred["device"].astype(str) == source_device].copy()
            xy = _curve_inputs_from_predictions(within_df)
            if xy is not None:
                y_true, y_prob = xy
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                line, = ax.plot(
                    fpr,
                    tpr,
                    color=target_colors.get(source_device, "#2E86DE"),
                    linewidth=2.4,
                    linestyle="-",
                    alpha=1.0,
                    zorder=3,
                )
                legend_handles.append(line)
                legend_labels.append(
                    f"{_pretty_device(source_device)} (best-config within-device), AUROC = {auc:.3f}"
                )

        # Cross-device curves for this source model.
        cross_rows = cross_pairwise[cross_pairwise["source_device"] == source_device].copy()
        cross_rows["target_device"] = pd.Categorical(
            cross_rows["target_device"], categories=DEVICE_ORDER, ordered=True
        )
        cross_rows = cross_rows.sort_values("target_device")
        for _, r in cross_rows.iterrows():
            target_device = str(r.get("target_device", ""))
            if not target_device or target_device == source_device:
                continue
            run_name = str(r.get("run_name", ""))
            pred_df = get_pred(run_name)
            if pred_df is None or pred_df.empty:
                continue
            sub = pred_df[pred_df["device"].astype(str) == target_device].copy()
            xy = _curve_inputs_from_predictions(sub)
            if xy is None:
                continue
            y_true, y_prob = xy
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            line, = ax.plot(
                fpr,
                tpr,
                color=target_colors.get(target_device, "#777777"),
                linewidth=1.15,
                alpha=0.92,
                zorder=2.5,
            )
            legend_handles.append(line)
            legend_labels.append(
                f"{_pretty_device(target_device)} (cross-device), AUROC = {auc:.3f}"
            )

        # Chance diagonal.
        ax.plot([0, 1], [0, 1], color="#B8B8B8", linewidth=1.1, linestyle="--", zorder=1)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.tick_params(axis="y", labelleft=True)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.grid(axis="both", color="#ECECEC", linewidth=0.8)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="lower right",
                frameon=True,
                framealpha=0.88,
                facecolor="white",
                edgecolor="#DDDDDD",
                fontsize=7.6,
                handlelength=2.0,
                labelspacing=0.35,
                borderpad=0.2,
            )

    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.07, right=0.99, wspace=0.22)

    _save_fig(
        fig,
        figures_dir / "figure_05_source_model_transfer_roc_curves_within_and_cross_device",
        dpi=dpi,
    )

def plot_pooled_test_roc_comparison_pooled_vs_best_within_models(
    views: Dict[str, pd.DataFrame], results_run_dir: Path, figures_dir: Path, dpi: int
) -> None:
    """Figure 6: pooled-test ROC comparison (pooled model vs three within-device best configs)."""
    if not results_run_dir.exists():
        return

    final_overall = views.get("final_overall", pd.DataFrame()).copy()
    pool_overall = views.get("pool_overall", pd.DataFrame()).copy()
    if final_overall.empty or pool_overall.empty:
        return

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13.2,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.7, 7.0), constrained_layout=False)

    model_colors = {
        "pooled": "#1A1A1A",
        "iphone": "#FF7F0E",
        "android_phone": "#2E86DE",
        "digital_stethoscope": "#27AE60",
    }

    curves: List[Dict[str, object]] = []

    pool_run_name = str(pool_overall.iloc[0]["run_name"])
    pool_pred = _load_predictions_file(results_run_dir, pool_run_name, split="test")
    pool_xy = _curve_inputs_from_predictions(pool_pred)
    if pool_xy is not None:
        y_true, y_prob = pool_xy
        curves.append(
            {
                "key": "pooled",
                "label_base": "Pooled-device model trained on all devices",
                "y_true": y_true,
                "y_prob": y_prob,
                "color": model_colors["pooled"],
                "lw": 2.4,
                "zorder": 3.6,
            }
        )

    for device in DEVICE_ORDER:
        row = final_overall[final_overall["train_device_filter"] == device]
        if row.empty:
            continue
        within_run_name = str(row.iloc[0]["run_name"])
        merged_pred = _assemble_pooled_test_predictions_for_best_within_model(
            views=views,
            results_run_dir=results_run_dir,
            within_run_name=within_run_name,
        )
        xy = _curve_inputs_from_predictions(merged_pred)
        if xy is None:
            continue
        y_true, y_prob = xy
        curves.append(
            {
                "key": device,
                "label_base": (
                    f"Best-config within-device model trained on "
                    f"{_pretty_device_after_trained_on(device)}"
                ),
                "y_true": y_true,
                "y_prob": y_prob,
                "color": model_colors.get(device, "#666666"),
                "lw": 1.15,
                "zorder": 3.0,
            }
        )

    if not curves:
        plt.close(fig)
        return

    legend_handles: List[plt.Line2D] = []
    legend_labels: List[str] = []
    for c in curves:
        y_true = c["y_true"]
        y_prob = c["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        line, = ax.plot(
            fpr,
            tpr,
            color=str(c["color"]),
            linewidth=float(c["lw"]),
            alpha=1.0,
            zorder=float(c["zorder"]),
        )
        legend_handles.append(line)
        legend_labels.append(f"{c['label_base']}, AUROC = {auc:.3f}")

    ax.plot([0, 1], [0, 1], color="#B8B8B8", linewidth=1.2, linestyle="--", zorder=1.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Models evaluated on pooled test set")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.grid(axis="both", color="#ECECEC", linewidth=0.85)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8.0,
        handlelength=2.2,
        labelspacing=0.35,
        borderpad=0.25,
    )

    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.1, right=0.97)

    _save_fig(
        fig,
        figures_dir / "figure_06_pooled_test_roc_comparison_pooled_vs_best_within_device_models",
        dpi=dpi,
    )

def plot_pooled_auroc_by_auscultation_site(
    site_auroc_df: pd.DataFrame, figures_dir: Path, dpi: int
) -> None:
    if site_auroc_df.empty:
        return

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_df = site_auroc_df.copy()
    plot_df["site_code"] = pd.Categorical(
        plot_df["site_code"], categories=AUSCULTATION_SITE_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values("site_code").reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.2), constrained_layout=True)
    point_color = "#111111"
    line_color = "#111111"
    x = np.array([1.18, 1.27, 1.36, 1.45], dtype=float)
    x_gap = float(x[1] - x[0])

    lower_err = plot_df["auroc"].to_numpy(dtype=float) - plot_df["auroc_ci95_low"].to_numpy(dtype=float)
    upper_err = plot_df["auroc_ci95_high"].to_numpy(dtype=float) - plot_df["auroc"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        plot_df["auroc"].to_numpy(dtype=float),
        yerr=[lower_err, upper_err],
        fmt="o",
        color=point_color,
        ecolor=line_color,
        elinewidth=2.0,
        capsize=3.5,
        capthick=1.4,
        markersize=7.4,
        markeredgewidth=0.9,
        markeredgecolor="#111111",
        zorder=3.2,
    )

    ax.axhline(0.5, color="#BDBDBD", linewidth=1.1, linestyle="--", zorder=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["auscultation_site"].tolist(), rotation=0)
    ax.set_xlabel("Auscultation site", labelpad=10)
    ax.set_ylabel("AUROC", labelpad=12)
    ax.set_title("Pooled-device model on pooled test set", pad=12)

    y_min = 0.495
    y_max = 0.855
    ax.set_xlim(float(x[0] - x_gap / 2.0), float(x[-1] + x_gap / 2.0))
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(0.50, 0.86, 0.05))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.85)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False, bottom=False)

    for xi, row in zip(x, plot_df.itertuples(index=False)):
        ax.annotate(
            f"{float(row.auroc):.3f}",
            (xi, float(row.auroc)),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9.4,
            fontweight="semibold",
            color="#2B2B2B",
        )

    _save_fig(
        fig,
        figures_dir / "figure_07_pooled_model_recording_level_auroc_by_auscultation_site",
        dpi=dpi,
    )
