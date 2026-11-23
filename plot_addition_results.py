#!/usr/bin/env python3
"""Plot saved Addition SDAS/GDAS evaluations (no model, no GPU, no retraining).

Reads either the CURRENT layout:
  results_addition_gradual_selected_seed0/evaluation_gradual/summary.csv
  results_addition_standard_all_layers_seed0/evaluation_standard/summary.csv
or the MCQA-style layout:
  <result-root>/eval_configs_<hash>/config_summary.csv

Produces annotated C-layer x T-layer heatmaps, overview figures, and (if both
methods are provided) GDAS minus SDAS heatmaps in percentage points.

Examples:
  python3 plot_addition_results.py
  python3 plot_addition_results.py --standard results_addition_standard_all_layers_seed0
  python3 plot_addition_results.py --gradual-eval results_addition_gradual_selected_seed0/eval_configs_abc1234567
  python3 plot_addition_results.py --overview-only
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


CORE_METRICS = [
    "C_only_Y_after_C",
    "C_only_T_after_C",
    "C_only_Y_via_T_replay",
    "C_only_Y_restored_via_T",
    "C_only_Y_via_T_replay_recipient_different",
    "C_T_Y_via_T_replay_recipient_different",
]
RECIPIENT_METRICS = [
    "T_only_Y_via_T_replay_recipient_same",
    "T_only_Y_via_T_replay_recipient_different",
    "C_only_Y_via_T_replay_recipient_same",
    "C_only_Y_via_T_replay_recipient_different",
    "C_T_Y_via_T_replay_recipient_same",
    "C_T_Y_via_T_replay_recipient_different",
    "C_only_T_after_C_recipient_different",
    "C_only_Y_after_C_recipient_different",
]
EXCLUDED_FIELDS = {"C_layer", "T_layer", "method", "recipient_seed", "recipient_version"}


def flatten_payload(payload):
    row = {"C_layer": payload["C_layer"], "T_layer": payload["T_layer"]}
    for name, value in payload.get("calibration", {}).items():
        row[name] = value
    for name, value in payload.get("clean", {}).items():
        row["clean_" + name] = value
    for family in ("T_only", "C_only", "C_T"):
        for section in ("metrics", "recipient"):
            for name, value in payload.get(section, {}).get(family, {}).items():
                row[family + "_" + name] = value
    return row


def read_rows(source):
    source = Path(source)
    if source.is_file():
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("Empty CSV: " + str(source))
        return rows

    for filename in ("config_summary.csv", "summary.csv"):
        summary = source / filename
        if summary.is_file():
            return read_rows(summary)

    configs = sorted(source.glob("*/metrics.json"))
    if not configs:
        raise FileNotFoundError("No summary CSV or per-config metrics.json in " + str(source))
    return [flatten_payload(json.loads(path.read_text())) for path in configs]


def resolve_source(root, method, explicit_eval=None):
    if explicit_eval:
        source = Path(explicit_eval)
        if not source.exists():
            raise FileNotFoundError(source)
        return source

    root = Path(root)
    if root.is_file() or (root / "config_summary.csv").is_file():
        return root
    if (root / "summary.csv").is_file() and (
        root.name.startswith("evaluation_") or root.name.startswith("eval_configs")
    ):
        return root

    profiles = sorted(path for path in root.glob("eval_configs_*") if path.is_dir())
    if len(profiles) > 1:
        options = "\n  ".join(str(path) for path in profiles)
        raise ValueError(
            f"{method}: multiple evaluation datasets found. Select exactly one "
            f"with --{method}-eval:\n  {options}"
        )
    if len(profiles) == 1:
        return profiles[0]

    # Older seed-0 runs have this directory even if root/summary.csv also exists.
    legacy = root / ("evaluation_" + method)
    if legacy.is_dir():
        return legacy
    unhashed = root / "eval_configs"
    if unhashed.is_dir():
        return unhashed
    raise FileNotFoundError(
        f"No evaluation results for {method} under {root}. "
        "Expected evaluation_<method> or eval_configs_<hash>."
    )


def parse_table(source):
    table = {}
    for row in read_rows(source):
        c = int(row["C_layer"])
        t = int(row["T_layer"])
        pair = (c, t)
        if pair in table:
            raise ValueError(f"Duplicate C={c}, T={t} in {source}")
        values = {}
        for name, raw in row.items():
            if name in EXCLUDED_FIELDS or raw is None or raw == "":
                continue
            try:
                number = float(raw)
            except (ValueError, TypeError):
                continue
            if math.isfinite(number):
                values[name] = number
        table[pair] = values
    if not table:
        raise ValueError("No valid configurations in " + str(source))
    return table


def metric_names(table):
    result = set()
    for row in table.values():
        result.update(row)
    return sorted(result)


def matrix(table, metric):
    c_layers = sorted({c for c, _ in table})
    t_layers = sorted({t for _, t in table})
    values = np.full((len(c_layers), len(t_layers)), np.nan)
    for (c, t), row in table.items():
        value = row.get(metric)
        if value is not None:
            values[c_layers.index(c), t_layers.index(t)] = 100.0 * value
    return c_layers, t_layers, values


def delta_table(gradual, standard):
    matched = sorted(set(gradual) & set(standard))
    if not matched:
        raise ValueError("SDAS and GDAS share no (C,T) configurations")
    output = {}
    for pair in matched:
        row = {}
        common = set(gradual[pair]) & set(standard[pair])
        for name in common:
            row[name] = gradual[pair][name] - standard[pair][name]
        output[pair] = row
    return output


def draw_heatmap(ax, table, metric, difference=False):
    c_layers, t_layers, values = matrix(table, metric)
    finite = values[np.isfinite(values)]
    if difference:
        magnitude = max(10.0, math.ceil(max(abs(finite), default=0.0) / 10.0) * 10.0)
        cmap = plt.get_cmap("RdBu_r").copy()
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=cmap,
            norm=TwoSlopeNorm(vmin=-magnitude, vcenter=0, vmax=magnitude),
            interpolation="nearest", aspect="auto",
        )
    else:
        cmap = plt.get_cmap("viridis").copy()
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=cmap,
            vmin=0, vmax=100, interpolation="nearest", aspect="auto",
        )
    cmap.set_bad("#e8ebef")
    ax.set_xticks(range(len(t_layers)), labels=t_layers)
    ax.set_yticks(range(len(c_layers)), labels=c_layers)
    ax.set_xlabel("T layer")
    ax.set_ylabel("C layer")
    ax.set_title(metric.replace("_recipient_", "  recipient ").replace("_", " "), fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(t_layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(c_layers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            label = f"{value:+.1f}" if difference else f"{value:.1f}"
            ink = "white" if (abs(value) > 35 if difference else value < 45) else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=ink)
    return image


def save_single(table, metric, destination, difference=False):
    c_count = len({c for c, _ in table})
    t_count = len({t for _, t in table})
    fig, ax = plt.subplots(figsize=(max(6, 1.15 * t_count), max(4.3, 0.75 * c_count)))
    image = draw_heatmap(ax, table, metric, difference=difference)
    fig.colorbar(image, ax=ax, label="Difference (percentage points)" if difference else "Accuracy (%)")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overview(table, names, destination, heading, difference=False):
    names = [name for name in names if name in metric_names(table)]
    if not names:
        return False
    columns = 2 if len(names) < 5 else 3
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.1 * rows), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // columns, index % columns]
        draw_heatmap(ax, table, name, difference=difference)
    for index in range(len(names), rows * columns):
        axes[index // columns, index % columns].axis("off")
    fig.suptitle(heading, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def export_method(table, label, out_dir, selected_metrics, overview_only=False):
    folder = out_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    save_overview(table, CORE_METRICS, folder / "overview_core.png", label.upper() + " — causal chain")
    save_overview(table, RECIPIENT_METRICS, folder / "overview_recipient.png", label.upper() + " — recipient transfer")
    metrics = selected_metrics or metric_names(table)
    written = 0
    if not overview_only:
        for name in metrics:
            if name not in metric_names(table):
                continue
            save_single(table, name, folder / "all_metrics" / (name + ".png"))
            written += 1
    return written


def write_comparison_summary(standard, gradual, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "n_pairs", "standard_mean_percent", "gradual_mean_percent", "delta_pp"]
    with (directory / "comparison_means.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in sorted(set(metric_names(standard)) & set(metric_names(gradual))):
            pairs = [pair for pair in gradual if pair in standard and
                     name in gradual[pair] and name in standard[pair]]
            if not pairs:
                continue
            sv = float(np.mean([standard[p][name] for p in pairs])) * 100
            gv = float(np.mean([gradual[p][name] for p in pairs])) * 100
            writer.writerow({
                "metric": name, "n_pairs": len(pairs),
                "standard_mean_percent": f"{sv:.4f}",
                "gradual_mean_percent": f"{gv:.4f}",
                "delta_pp": f"{gv - sv:+.4f}",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradual", default="results_addition_gradual_selected_seed0",
                        help="GDAS run root or direct evaluation directory/CSV")
    parser.add_argument("--standard", help="Optional SDAS run root or evaluation directory/CSV")
    parser.add_argument("--gradual-eval", help="Explicit GDAS evaluation folder for multiple datasets")
    parser.add_argument("--standard-eval", help="Explicit SDAS evaluation folder for multiple datasets")
    parser.add_argument("--out", default="addition_plots_seed0", help="Output plot directory")
    parser.add_argument("--metrics", help="Comma-separated metric columns for individual heatmaps; default all")
    parser.add_argument("--overview-only", action="store_true", help="Only overview PNGs, no individual heatmaps")
    args = parser.parse_args()

    out = Path(args.out)
    gradual_source = resolve_source(args.gradual, "gradual", args.gradual_eval)
    gradual = parse_table(gradual_source)
    print(f"[GDAS] {gradual_source} — {len(gradual)} configurations")
    chosen = [x.strip() for x in args.metrics.split(",") if x.strip()] if args.metrics else None
    n = export_method(gradual, "gradual", out, chosen, args.overview_only)
    print(f"[saved] {out / 'gradual'} (2 overviews, {n} individual heatmaps)")

    if args.standard or args.standard_eval:
        standard_root = args.standard or "."
        standard_source = resolve_source(standard_root, "standard", args.standard_eval)
        standard = parse_table(standard_source)
        print(f"[SDAS] {standard_source} — {len(standard)} configurations")
        if ("eval_configs_" in str(gradual_source) and "eval_configs_" in str(standard_source)):
            first = next((p.name for p in Path(gradual_source).parents if p.name.startswith("eval_configs_")), None)
            second = next((p.name for p in Path(standard_source).parents if p.name.startswith("eval_configs_")), None)
            if Path(gradual_source).name.startswith("eval_configs_"):
                first = Path(gradual_source).name
            if Path(standard_source).name.startswith("eval_configs_"):
                second = Path(standard_source).name
            if first != second:
                raise ValueError(f"Different evaluation profiles: GDAS={first}, SDAS={second}")
        n = export_method(standard, "standard", out, chosen, args.overview_only)
        print(f"[saved] {out / 'standard'} (2 overviews, {n} individual heatmaps)")
        delta = delta_table(gradual, standard)
        directory = out / "gradual_minus_standard"
        save_overview(delta, CORE_METRICS, directory / "overview_core_delta.png",
                      "GDAS − SDAS (percentage points)", difference=True)
        save_overview(delta, RECIPIENT_METRICS, directory / "overview_recipient_delta.png",
                      "GDAS − SDAS recipient transfer (percentage points)", difference=True)
        common_names = sorted(set(metric_names(gradual)) & set(metric_names(standard)))
        names = chosen or common_names
        n = 0
        if not args.overview_only:
            for name in names:
                if name not in common_names:
                    continue
                save_single(delta, name, directory / "all_metrics" / (name + ".png"), difference=True)
                n += 1
        write_comparison_summary(standard, gradual, directory)
        print(f"[saved] {directory} ({len(delta)} matched configurations, {n} delta heatmaps)")
    print("Done. No GPU/model/training/evaluation required.")


if __name__ == "__main__":
    main()
