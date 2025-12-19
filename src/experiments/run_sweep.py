#!/usr/bin/env python3
"""
Run a grid of training jobs defined in a JSON config.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a training sweep from JSON.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep config JSON.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on number of runs.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue sweep even if a run fails.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def expand_grid(grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [_normalize_list(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _list_to_slug(value: Any) -> str:
    if value is None:
        return "all"
    if isinstance(value, list):
        return "-".join(str(v) for v in value) if value else "all"
    return str(value)


def _sanitize_name(name: str) -> str:
    safe = name.replace(os.sep, "-").replace(" ", "")
    for ch in [",", "[", "]", "(", ")", "'", '"', ":", ";"]:
        safe = safe.replace(ch, "")
    return safe


def build_run_name(
    fmt: str, params: Dict[str, Any], run_tag: str
) -> str:
    fields = dict(params)
    fields["run_tag"] = run_tag
    fields["device_filter"] = _list_to_slug(params.get("device_filter"))
    fields["train_devices"] = _list_to_slug(params.get("train_device_filter"))
    fields["val_devices"] = _list_to_slug(params.get("val_device_filter"))
    fields["test_devices"] = _list_to_slug(params.get("test_device_filter"))
    fields["position_filter"] = _list_to_slug(params.get("position_filter"))
    fields["train_positions"] = _list_to_slug(params.get("train_position_filter"))
    fields["val_positions"] = _list_to_slug(params.get("val_position_filter"))
    fields["test_positions"] = _list_to_slug(params.get("test_position_filter"))
    name = fmt.format(**fields)
    return _sanitize_name(name)


def params_to_cli(params: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            if len(value) == 0:
                continue
            args.append(flag)
            args.extend([str(v) for v in value])
            continue
        args.append(flag)
        args.append(str(value))
    return args


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    python_exec = config.get("python", sys.executable)
    module = config.get("module", "src.training.train")
    defaults = config.get("defaults", {})
    grid = config.get("grid", {})
    runs = config.get("runs", [{}])
    if not runs:
        runs = [{}]
    combine = bool(config.get("combine_grid_with_runs", True))
    run_name_format = config.get(
        "run_name_format", "{backbone}_{representation}_{run_tag}"
    )
    skip_if_exists = bool(config.get("skip_if_exists", False))

    if "results_dir" in config and "results_dir" not in defaults:
        defaults["results_dir"] = config["results_dir"]
    if "output_dir" in config and "output_dir" not in defaults:
        defaults["output_dir"] = config["output_dir"]

    extra_args_global = _normalize_list(config.get("extra_args"))

    run_count = 0
    for run_idx, run_override in enumerate(runs):
        run_override = dict(run_override)
        run_tag = run_override.pop("name", None) or run_override.pop("run_tag", None)
        if run_tag is None:
            run_tag = f"run{run_idx:03d}"

        run_extra_args = _normalize_list(run_override.pop("extra_args", None))

        grid_iter = expand_grid(grid) if combine else [{}]
        for grid_params in grid_iter:
            params = dict(defaults)
            params.update(grid_params)
            params.update(run_override)
            params.pop("extra_args", None)

            grad_steps = _to_int(params.get("grad_accum_steps"))
            batch_size = _to_int(params.get("batch_size"))
            if grad_steps is not None and grad_steps > 1:
                if batch_size is not None:
                    eff_bs = batch_size * grad_steps
                    print(
                        f"Note: grad_accum_steps={grad_steps} (effective batch={eff_bs})"
                    )
                else:
                    print(f"Note: grad_accum_steps={grad_steps} (batch_size not set)")

            if "run_name" not in params:
                params["run_name"] = build_run_name(run_name_format, params, run_tag)

            results_dir = params.get("results_dir")
            run_name = params.get("run_name")
            if skip_if_exists and results_dir and run_name:
                metrics_path = os.path.join(results_dir, run_name, "metrics.json")
                if os.path.exists(metrics_path):
                    print(f"Skipping existing run: {run_name}")
                    continue

            cli_args = params_to_cli(params)
            cmd = [python_exec, "-m", module] + cli_args + extra_args_global + run_extra_args
            cmd_str = " ".join(cmd)
            print(f"[{run_count:03d}] {cmd_str}")

            if not args.dry_run:
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"Run failed: {exc}")
                    if not args.continue_on_error:
                        sys.exit(exc.returncode)

            run_count += 1
            if args.max_runs is not None and run_count >= args.max_runs:
                print("Reached --max_runs limit.")
                return


if __name__ == "__main__":
    main()
