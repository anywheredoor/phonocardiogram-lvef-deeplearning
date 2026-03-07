#!/usr/bin/env python3
"""Generate dissertation-ready figures and tables from summary.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.reporting.dissertation.common import (
    _save_csv,
    _pretty_device,
    _sort_by_device_columns,
    aggregate_cv,
    classify_runs,
    compute_backbone_parameter_table,
    compute_pooled_auroc_by_auscultation_site,
    compute_test_discrimination_vs_random_baseline,
    extract_clean_views,
    load_and_normalize_summary,
)
from src.reporting.dissertation.figures import (
    plot_backbone_parameter_sizes,
    plot_cv_heatmap_mean_f1,
    plot_cv_performance_vs_parameter_tradeoff,
    plot_pooled_auroc_by_auscultation_site,
    plot_pooled_test_roc_comparison_pooled_vs_best_within_models,
    plot_representation_effect_gammatone_minus_mfcc,
    plot_source_model_transfer_roc_curves,
)
from src.reporting.dissertation.tables import (
    build_discrimination_vs_random_baseline_table,
    build_pooled_test_performance_table,
    build_raw_dataset_summary_table,
    build_shared_training_settings_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dissertation figures/tables from summary.csv."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="summary.csv",
        help="Path to summary CSV (default: ./summary.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/dissertation_summary_outputs",
        help="Output directory for generated tables and figures.",
    )
    parser.add_argument(
        "--results_run_dir",
        type=str,
        default="results/first run",
        help=(
            "Optional directory containing saved run folders (run_name-matched) "
            "with predictions/history for ROC/PR/calibration/learning-curve figures."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for PNG outputs (default: 300).",
    )
    return parser.parse_args()


def write_readme(
    output_dir: Path,
    tables: Dict[str, object],
    run_catalog,
    cv_best,
    views,
) -> None:
    lines: List[str] = []
    lines.append("# Dissertation Summary Outputs")
    lines.append("")
    lines.append("These dissertation tables and figures were generated automatically from the project data files.")
    lines.append("Table 1 uses `lvef.csv` and `heart_sounds` to summarize the study cohort before signal preprocessing.")
    lines.append("Table 2 summarizes the shared training settings used across experiments.")
    lines.append("Table 3 reports pooled-test F1, sensitivity, and specificity for the pooled model and the best-config within-device models.")
    lines.append("Table 4 summarizes whether AUROC and AUPRC remained above their random baselines under 95% patient-cluster bootstrap confidence intervals.")
    lines.append("Figures 1-4 use aggregate metrics from the summary CSV.")
    lines.append("Figures 5-7 use saved test predictions from `results/first run` (run folders named by `run_name`).")
    lines.append("")
    lines.append("## Summary CSV structure detected")
    lines.append("")
    lines.append(f"- Total rows: {len(views['all_rows'])}")
    lines.append(f"- Distinct run names: {run_catalog['run_name'].nunique()}")
    lines.append("")
    run_counts = run_catalog.groupby('run_kind', dropna=False)['run_name'].count().sort_index()
    for run_kind, count in run_counts.items():
        lines.append(f"- `{run_kind}`: {count} runs, {int(run_catalog.loc[run_catalog['run_kind'] == run_kind, 'n_rows'].sum())} rows")
    lines.append("")
    lines.append("## Best CV config per training device (by mean test F1_pos)")
    lines.append("")
    for _, row in _sort_by_device_columns(cv_best, ["train_device_filter"]).iterrows():
        lines.append(
            "- "
            f"{_pretty_device(row['train_device_filter'])}: "
            f"{row['representation']} + {row['backbone']} "
            f"(image_size={int(row['image_size'])}, "
            f"mean F1={row['mean_test_f1_pos']:.3f} ± {row['sd_test_f1_pos']:.3f})"
        )
    lines.append("")
    lines.append("## Notes for dissertation use")
    lines.append("")
    lines.append("- Final within-device runs appear twice in `summary.csv` (`overall` and same-device `test_device`); use the `overall` row for headline tables.")
    lines.append("- Cross-device eval runs contain one `overall` row plus one row per target device; use the metadata CSVs if you need the six pairwise source→target results later.")
    lines.append("- Pooled-device run contains one `overall` row plus three per-device rows; use the metadata CSVs if you need aggregate or per-target reporting later.")
    lines.append("- `overall` rows are aggregate evaluations and are not guaranteed to equal the unweighted arithmetic mean of the per-device rows.")
    lines.append("")
    if tables:
        lines.append("## Generated tables")
        lines.append("")
        for name in sorted(tables):
            lines.append(f"- `{name}`")
        lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    figure_names = sorted(p.name for p in (output_dir / 'figures').glob('*.png'))
    for name in figure_names:
        lines.append(f"- `{name}` (PNG) and `{name.replace('.png', '.pdf')}` (PDF)")
    (output_dir / 'README.md').write_text("\n".join(lines) + "\n", encoding='utf-8')


def generate_outputs(summary_csv: str, output_dir: str, dpi: int, results_run_dir: str | None = None) -> None:
    out_dir = Path(output_dir)
    tables_dir = out_dir / 'tables'
    figures_dir = out_dir / 'figures'
    metadata_dir = out_dir / 'metadata'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize_summary(summary_csv)
    run_catalog = classify_runs(df)
    views = extract_clean_views(df, run_catalog)
    cv_agg, cv_best = aggregate_cv(views['cv_rows'])
    backbone_param_df = compute_backbone_parameter_table()
    results_run_path = Path(results_run_dir) if results_run_dir else Path("__missing__")
    bootstrap_vs_random_df = compute_test_discrimination_vs_random_baseline(
        views=views,
        results_run_dir=results_run_path,
    )

    tables = {
        "table_01_dataset_summary_before_preprocessing.csv": build_raw_dataset_summary_table(
            lvef_csv_path=_REPO_ROOT / "lvef.csv",
            heart_sounds_dir=_REPO_ROOT / "heart_sounds",
        ),
        "table_02_shared_training_settings_used_across_experiments.csv": build_shared_training_settings_table(),
        "table_03_pooled_test_performance_pooled_vs_best_within_models.csv": build_pooled_test_performance_table(
            views=views,
            results_run_dir=results_run_path,
        ),
        "table_04_discrimination_vs_random_baseline_under_bootstrap_ci.csv": build_discrimination_vs_random_baseline_table(
            bootstrap_df=bootstrap_vs_random_df,
        ),
    }

    _save_csv(run_catalog, metadata_dir / 'run_catalog_classified_from_summary.csv')
    _save_csv(views['all_rows'], metadata_dir / 'summary_rows_with_run_kind.csv')
    _save_csv(backbone_param_df, metadata_dir / 'backbone_parameter_counts_used_in_experiments.csv')
    if not bootstrap_vs_random_df.empty:
        _save_csv(
            bootstrap_vs_random_df,
            metadata_dir / 'bootstrap_ci_vs_random_baseline_patient_cluster_test_predictions.csv',
        )
    pooled_site_auroc_df = compute_pooled_auroc_by_auscultation_site(
        views=views,
        results_run_dir=results_run_path,
    )
    if not pooled_site_auroc_df.empty:
        _save_csv(
            pooled_site_auroc_df,
            metadata_dir / 'pooled_model_recording_level_auroc_by_auscultation_site_patient_cluster_bootstrap.csv',
        )

    for filename, table_df in tables.items():
        _save_csv(table_df, tables_dir / filename)

    plot_cv_heatmap_mean_f1(cv_agg=cv_agg, cv_best=cv_best, figures_dir=figures_dir, dpi=dpi)
    plot_backbone_parameter_sizes(param_df=backbone_param_df, figures_dir=figures_dir, dpi=dpi)
    plot_cv_performance_vs_parameter_tradeoff(
        cv_agg=cv_agg,
        cv_best=cv_best,
        param_df=backbone_param_df,
        figures_dir=figures_dir,
        dpi=dpi,
    )
    plot_representation_effect_gammatone_minus_mfcc(
        cv_agg=cv_agg,
        figures_dir=figures_dir,
        dpi=dpi,
    )
    plot_source_model_transfer_roc_curves(
        views=views,
        results_run_dir=results_run_path,
        figures_dir=figures_dir,
        dpi=dpi,
    )
    plot_pooled_test_roc_comparison_pooled_vs_best_within_models(
        views=views,
        results_run_dir=results_run_path,
        figures_dir=figures_dir,
        dpi=dpi,
    )
    plot_pooled_auroc_by_auscultation_site(
        site_auroc_df=pooled_site_auroc_df,
        figures_dir=figures_dir,
        dpi=dpi,
    )

    write_readme(
        output_dir=out_dir,
        tables=tables,
        run_catalog=run_catalog,
        cv_best=cv_best,
        views=views,
    )


def main() -> None:
    args = parse_args()
    generate_outputs(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        dpi=args.dpi,
        results_run_dir=args.results_run_dir,
    )
    print(f"Generated dissertation figures/tables in: {args.output_dir}")


if __name__ == '__main__':
    main()
