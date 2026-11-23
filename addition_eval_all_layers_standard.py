"""Evaluate saved Standard DAS handles on CAL, select promising C/T layers.

Order:
  1. addition_train_all_layers_standard.py --layers all
  2. addition_eval_all_layers_standard.py --threshold-pct 5
  3. addition_train_gradual_selected.py

This file evaluates every trained C/T layer on the VALIDATION split
using free-autoregressive numeric IIA. It never reads TEST labels.
After all requested layer evaluations finish, it selects layers within
5 percentage points of the best per-variable CAL IIA and saves
selected_layers_standard.json, which GDAS consumes automatically.
"""

import argparse
from pathlib import Path

import torch

from addition_das import collect_layer_states
from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_pipeline_utils import (
    check_artifacts,
    count_layers,
    dataset_signature,
    handle_paths,
    load_checkpoint,
    parse_layers,
    read_json,
    write_csv,
    write_json,
)
from addition_select_promising_layers import select_standard
from addition_train_standard_grid import (
    build_factual_c_bank,
    clean_readout,
    free_y_accuracy,
)


def make_signature(variable, layer, ckpt_path, readout_path, data_signature, generation_cap):
    return {
        "split": "validation",
        "variable": variable,
        "layer": layer,
        "metric": "free_autoregressive_numeric_v2",
        "generation_cap": generation_cap,
        "data": data_signature,
        "artifacts": check_artifacts([ckpt_path, readout_path]),
    }


def summarize_saved_results(cal_dir, variables, layers, expected_signatures):
    summary = []
    for variable in variables:
        for layer in layers:
            path = cal_dir / f"{variable}_layer_{layer:02d}.json"
            if not path.exists():
                continue
            record = read_json(path)
            if record.get("signature") != expected_signatures[(variable, layer)]:
                continue
            summary.append({
                "variable": variable,
                "layer": layer,
                "cal_iia": record["cal_iia"],
                "readout_cal_base": record["readout_cal_base"],
                "readout_cal_source": record["readout_cal_source"],
                "checkpoint_best_cal_iia": record["checkpoint_best_cal_iia"],
            })
    summary.sort(key=lambda row: (row["variable"], row["layer"]))
    fields = [
        "variable", "layer", "cal_iia",
        "readout_cal_base", "readout_cal_source", "checkpoint_best_cal_iia",
    ]
    write_csv(cal_dir / "summary.csv", summary, fields)
    return summary


@torch.no_grad()
def evaluate_layer(model, tokenizer, variable, layer, banks, rows, checkpoint,
                   readout, batch_size, generation_cap):
    device = next(model.parameters()).device
    Q = checkpoint["basis"].float().to(device)
    if variable == "T":
        bank = banks["T"]["validation"]
        factual = banks["T_factual"]["validation"].copy()
        factual["target"] = factual["target_t"]
        W = readout["W_T"].float().to(device)
        b = readout["b_T"].float().to(device)
        train_cal_iia = float(checkpoint["validation_iia"])
    else:
        bank = banks["C"]["validation"]
        factual = build_factual_c_bank(
            tokenizer, rows["validation"]["C_only"], banks["prefix_ids"]
        )
        W = readout["W_C"].float().to(device)
        b = readout["b_C"].float().to(device)
        train_cal_iia = float(checkpoint["validation_direct_iia"])

    donor = collect_layer_states(
        model, bank["source_input_ids"], bank["source_attention_mask"],
        bank["source_positions"], layer, batch_size,
    )
    cal_iia = free_y_accuracy(
        model, tokenizer, bank, donor, Q, layer, batch_size, generation_cap
    )
    base_acc, source_acc = clean_readout(
        model, factual, Q, layer, W, b, batch_size
    )
    return {
        "variable": variable,
        "layer": layer,
        "cal_iia": float(cal_iia),
        "readout_cal_base": float(base_acc),
        "readout_cal_source": float(source_acc),
        "checkpoint_best_cal_iia": train_cal_iia,
    }


def save_selection(root, cal_dir, summary, threshold_pct, signatures, variables, layers):
    if set(variables) != {"C", "T"}:
        print("Skipping selection because both C and T must be evaluated")
        return

    expected = len(layers) * 2
    if len(summary) != expected:
        raise RuntimeError(f"Expected {expected} CAL results, got {len(summary)}")

    chosen = select_standard(summary, threshold_pct)
    if not chosen["pairs"]:
        raise RuntimeError(
            "No valid C<T combination among selected layers. "
            "Inspect CAL IIA or use a wider threshold."
        )
    output = root / "selected_layers_standard.json"
    selection = {
        "method": "standard",
        "selection_source": "addition_eval_all_layers_standard.py",
        "calibration_split": "validation",
        "calibration_summary": str(cal_dir / "summary.csv"),
        "threshold_pct": threshold_pct,
        "evaluated_layers": layers,
        "calibration_signatures": {
            f"{v}_L{layer}": signatures[(v, layer)]
            for v in variables
            for layer in layers
        },
        **chosen,
    }
    write_json(output, selection)
    write_csv(root / "selected_layers_standard.csv", chosen["pairs"],
              ["C_layer", "T_layer"])
    print("\n[Standard DAS promising candidates]")
    for variable in ("C", "T"):
        best = max(row["cal_iia"] for row in summary if row["variable"] == variable)
        print(f"  {variable}: best CAL IIA={best:.4f}, selected {chosen['layers'][variable]}")
    print(f"  Configurations with C<T: {len(chosen['pairs'])}")
    print(f"[saved] {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results_addition_standard_all_layers_seed0")
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--layers", default="all", help="all, 0,1,2, or 0:26")
    parser.add_argument("--variables", default="T,C")
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Recompute saved CAL metrics")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold_pct <= 100:
        parser.error("--threshold-pct must be in [0, 100]")

    variables = [v.strip().upper() for v in args.variables.split(",") if v.strip()]
    if len(set(variables)) != len(variables) or any(v not in ("T", "C") for v in variables):
        parser.error("--variables must be unique and drawn from T,C")

    root = Path(args.result_dir)
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("method") != "standard":
        raise RuntimeError(f"Expected Standard DAS run manifest in {root}")
    if manifest.get("generation_cap") != args.generation_cap:
        raise RuntimeError(
            "CAL generation cap differs from training; "
            "use the original --generation-cap"
        )

    model, tokenizer = load_gemma_model(args.device)
    layers = parse_layers(args.layers, count_layers(model))
    rows = load_addition_rows(
        model, tokenizer, args.data_dir, args.cache_dir,
        args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache,
    )
    banks = build_all_banks(tokenizer, rows)
    data_sig = dataset_signature(model, args.data_dir, args.cache_dir,
                                 args.max_new_tokens)
    if data_sig != manifest["data"]:
        raise RuntimeError("Dataset/cache differs from the Standard DAS training run")

    cal_dir = root / "calibration_standard"
    cal_dir.mkdir(parents=True, exist_ok=True)
    signatures = {}
    for variable in variables:
        for layer in layers:
            ckpt_path, readout_path = handle_paths(root, variable, layer)
            signatures[(variable, layer)] = make_signature(
                variable, layer, ckpt_path, readout_path,
                data_sig, args.generation_cap,
            )

    for index, (variable, layer) in enumerate(signatures, 1):
        output = cal_dir / f"{variable}_layer_{layer:02d}.json"
        signature = signatures[(variable, layer)]
        if output.exists() and not args.force:
            old = read_json(output)
            if old.get("signature") == signature:
                print(f"[cached {index}/{len(signatures)}] {variable} L{layer}")
                continue
        print(f"[CAL {index}/{len(signatures)}] {variable} L{layer}", flush=True)
        ckpt_path, readout_path = handle_paths(root, variable, layer)
        checkpoint = load_checkpoint(ckpt_path, variable, layer, "standard")
        readout = torch.load(readout_path, map_location="cpu", weights_only=False)
        if int(readout["layer"]) != layer:
            raise RuntimeError(f"Readout checkpoint layer mismatch: {readout_path}")
        metrics = evaluate_layer(
            model, tokenizer, variable, layer, banks, rows, checkpoint,
            readout, args.eval_batch_size, args.generation_cap,
        )
        metrics["signature"] = signature
        write_json(output, metrics)
        summarize_saved_results(cal_dir, variables, layers, signatures)
        print(f"  CAL IIA={metrics['cal_iia']:.4f}, "
              f"readout base/source={metrics['readout_cal_base']:.4f}/"
              f"{metrics['readout_cal_source']:.4f}")

    summary = summarize_saved_results(cal_dir, variables, layers, signatures)
    save_selection(root, cal_dir, summary, args.threshold_pct,
                   signatures, variables, layers)


if __name__ == "__main__":
    main()
