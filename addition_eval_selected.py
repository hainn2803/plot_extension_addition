"""Evaluate saved Addition Standard or Gradual DAS C -> T -> Y configurations.

Uses held-out test pairs only AFTER calibration-based layer selection.
Both SDAS and GDAS use MCQA-style eval_configs_<hash>/T_L{t}__C_L{c}/
with metrics.json, eval_faithfulness.txt, eval_faithfulness.csv;
cumulative config_summary.csv and all_config_metrics.csv.
Old evaluation_<method> results are imported without rerunning inference.
"""

import argparse
import re
from pathlib import Path

import torch

from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_recipient_schedule import RECIPIENT_VERSION, prepare_schedules
from addition_eval_recipient_transfer import eval_recipient
from addition_gradual_snapshot import (
    TRAINING_SELECTION_NAME,
    artifact_hashes_by_name,
    validate_gradual_pair_snapshot,
)
from addition_eval_reports import config_dir, import_legacy, rebuild_summary
from addition_eval_profile import profile_directory, profile_settings
from addition_train_standard_grid import (
    build_factual_c_bank,
    clean_readout,
    eval_c_full,
    eval_t_full,
)
from addition_pipeline_utils import (
    check_artifacts,
    dataset_signature,
    evaluation_signature,
    gradual_paths,
    handle_paths,
    load_checkpoint,
    read_json,
    write_json,
)


def compatible_signature(saved, current):
    """Reuse metrics after relocating the same exact T checkpoint bytes."""
    if saved == current:
        return True
    if not isinstance(saved, dict):
        return False
    for field in set(current).union(saved):
        if field != "artifacts" and saved.get(field) != current.get(field):
            return False
    try:
        return (artifact_hashes_by_name(saved["artifacts"])
                == artifact_hashes_by_name(current["artifacts"]))
    except (KeyError, TypeError, ValueError):
        return False


def get_paths(args, c_layer, t_layer):
    # Frozen T is copied into every GDAS run: no SDAS directory is needed.
    t_paths = handle_paths(args.result_dir, "T", t_layer)
    if args.method == "standard":
        c_paths = handle_paths(args.result_dir, "C", c_layer)
    else:
        c_paths = gradual_paths(args.result_dir, c_layer, t_layer)
    return c_paths, t_paths


def load_pair(args, c_layer, t_layer):
    (c_path, c_ro_path), (t_path, t_ro_path) = get_paths(args, c_layer, t_layer)
    check_artifacts([c_path, c_ro_path, t_path, t_ro_path])

    if args.method == "gradual":
        validate_gradual_pair_snapshot(args.result_dir, c_layer, t_layer)

    c_cp = load_checkpoint(c_path, "C", c_layer, args.method,
                           t_layer if args.method == "gradual" else None)
    t_cp = load_checkpoint(t_path, "T", t_layer, "standard")
    c_ro = torch.load(c_ro_path, map_location="cpu", weights_only=False)
    t_ro = torch.load(t_ro_path, map_location="cpu", weights_only=False)
    if int(c_ro["layer"]) != c_layer or int(t_ro["layer"]) != t_layer:
        raise RuntimeError(f"Readout and DAS layer mismatch for C{c_layer}->T{t_layer}")
    return c_cp, c_ro, t_cp, t_ro


@torch.no_grad()
def evaluate_one(args, model, tokenizer, rows, banks, c_layer, t_layer,
                 c_cp, c_ro, t_cp, t_ro):
    device = next(model.parameters()).device
    Q_c = c_cp["basis"].float().to(device)
    Q_t = t_cp["basis"].float().to(device)
    W_c = c_ro["W_C"].float().to(device)
    b_c = c_ro["b_C"].float().to(device)
    W_t = t_ro["W_T"].float().to(device)
    b_t = t_ro["b_T"].float().to(device)

    c_factual = build_factual_c_bank(tokenizer, rows["test"]["C_only"], banks["prefix_ids"])
    t_factual = {
        "input_ids": banks["T_factual"]["test"]["input_ids"],
        "attention_mask": banks["T_factual"]["test"]["attention_mask"],
        "positions": banks["T_factual"]["test"]["positions"],
        "target": banks["T_factual"]["test"]["target_t"],
    }
    c_base, c_source = clean_readout(
        model, c_factual, Q_c, c_layer, W_c, b_c, args.eval_batch_size
    )
    t_base, t_source = clean_readout(
        model, t_factual, Q_t, t_layer, W_t, b_t, args.eval_batch_size
    )
    t_metrics = eval_t_full(
        model, tokenizer, banks["T"]["test"], Q_t, t_layer,
        W_t, b_t, args.eval_batch_size, args.generation_cap,
    )
    c_metrics = {}
    for family in ("C", "C_T"):
        family_rows = rows["test"]["C_only" if family == "C" else "C_T"]
        c_metrics[family] = eval_c_full(
            model, tokenizer, banks[family]["test"], family_rows,
            Q_c, c_layer, W_c, b_c, Q_t, t_layer, W_t, b_t,
            args.eval_batch_size, args.generation_cap,
        )

    return {
        "method": args.method,
        "C_layer": c_layer,
        "T_layer": t_layer,
        "calibration": {
            "C_cal_iia": float(c_cp["validation_direct_iia"]),
            "T_cal_iia": float(t_cp["validation_iia"]),
        },
        "clean": {
            "C_base": c_base, "C_source": c_source,
            "T_base": t_base, "T_source": t_source,
        },
        "metrics": {
            "T_only": t_metrics,
            "C_only": c_metrics["C"],
            "C_T": c_metrics["C_T"],
        },
    }


@torch.no_grad()
def evaluate_recipient_only(args, model, tokenizer, banks, schedules,
                            c_layer, t_layer, c_cp, c_ro, t_cp, t_ro):
    device = next(model.parameters()).device
    Q_c = c_cp["basis"].float().to(device)
    Q_t = t_cp["basis"].float().to(device)
    W_t = t_ro["W_T"].float().to(device)
    b_t = t_ro["b_T"].float().to(device)
    return eval_recipient(
        model, tokenizer, banks, schedules, banks["prefix_ids"],
        Q_c, c_layer, Q_t, t_layer, W_t, b_t,
        args.eval_batch_size, args.generation_cap,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("standard", "gradual"), required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--standard-root", default=None,
                        help="Legacy accepted flag; no longer needed after T snapshot")
    parser.add_argument(
        "--selection",
        help="Optional selection JSON. GDAS defaults to its frozen training_pairs.json; "
             "SDAS defaults to selected_layers_standard.json.",
    )
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic cross-question recipient assignments")
    parser.add_argument("--skip-recipient", action="store_true",
                        help="Skip same/different recipient transfer experiments")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--migrate-only", action="store_true",
        help="Import old evaluation results to MCQA eval_configs and export reports "
             "without model loading or GPU",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_dir)
    if args.selection:
        selection_path = Path(args.selection)
    elif args.method == "gradual":
        # This records exactly the grid chosen at train time.
        selection_path = root / TRAINING_SELECTION_NAME
    else:
        selection_path = root / "selected_layers_standard.json"

    selection = read_json(selection_path)
    selection_method = selection.get("method")
    if selection_method not in ("standard", "gradual"):
        raise ValueError(f"Unrecognized selection method in {selection_path}")
    if args.method == "standard" and selection_method != "standard":
        raise ValueError(f"Standard DAS evaluation requires Standard selection: {selection_path}")
    print(f"[selection] {selection_path} ({selection_method})", flush=True)
    pairs = sorted({(int(item["C_layer"]), int(item["T_layer"]))
                    for item in selection["pairs"]}, key=lambda x: (x[1], x[0]))
    if not pairs or any(c >= t or c < 0 for c, t in pairs):
        raise ValueError("Selection has no valid C<T configurations")
    if args.list_only:
        for c, t in pairs:
            print(f"C{c} -> T{t}")
        print(f"{len(pairs)} configurations")
        return

    manifest = read_json(root / "run_manifest.json")
    if manifest["method"] != args.method:
        raise RuntimeError(f"Wrong method for evaluation directory: {root}")

    def evaluation_settings(data_sig):
        return profile_settings(
            data_sig, seed=args.seed, eval_batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            generation_cap=args.generation_cap,
            recipient_version=RECIPIENT_VERSION,
        )

    if args.migrate_only:
        # For old runs, the historical evaluation used the training dataset
        # and default seed 0. This route never loads model or reads new data.
        data_sig = manifest["data"]
        eval_dir = profile_directory(root, evaluation_settings(data_sig))
        import_legacy(root, args.method, pairs, data_sig, eval_dir, args.seed)
        existing = rebuild_summary(eval_dir, data_sig)
        print(f"[saved] {eval_dir / 'config_summary.csv'} "
              f"({len(existing)} cached configurations; no model loaded)")
        return

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir,
                              args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)
    data_sig = dataset_signature(model, args.data_dir, args.cache_dir, args.max_new_tokens)
    manifest = read_json(root / "run_manifest.json")
    if manifest["generation_cap"] != args.generation_cap:
        raise RuntimeError("CAL and TEST must use the same generation cap")
    if manifest.get("model") != str(model.config._name_or_path):
        raise RuntimeError("Evaluation model differs from trained checkpoints")
    if manifest["data"] != data_sig:
        print("[evaluation] Different dataset from training; using its independent "
              "hashed profile. Ensure it is a held-out test dataset.", flush=True)

    eval_dir = profile_directory(root, evaluation_settings(data_sig))
    if data_sig == manifest["data"]:
        import_legacy(root, args.method, pairs, data_sig, eval_dir, args.seed)
    rebuild_summary(eval_dir, data_sig)
    recipient_schedules = None
    for index, (c, t) in enumerate(pairs, 1):
        (c_path, c_ro_path), (t_path, t_ro_path) = get_paths(args, c, t)
        signature = evaluation_signature(
            args.method, c, t, [c_path, c_ro_path, t_path, t_ro_path],
            data_sig, args.generation_cap,
        )
        output = config_dir(eval_dir, c, t) / "metrics.json"
        saved = None
        if output.exists() and not args.force:
            previous = read_json(output)
            if compatible_signature(previous.get("signature"), signature):
                saved = previous

        recipient_done = False
        if saved is not None:
            recipient_done = (
                saved.get("recipient_version") == RECIPIENT_VERSION
                and saved.get("recipient_seed") == args.seed
                and all(
                    family in saved.get("recipient", {})
                    for family in ("T_only", "C_only", "C_T")
                )
            )
        if saved is not None and (args.skip_recipient or recipient_done):
            if saved["signature"] != signature:
                saved["signature"] = signature
                write_json(output, saved)
            print(f"[cached {index}/{len(pairs)}] C{c}->T{t}")
            continue

        c_cp, c_ro, t_cp, t_ro = load_pair(args, c, t)
        if saved is not None:
            # Upgrade old completed configurations without rerunning expensive
            # direct, replay and restoration tests.
            print(f"[recipient upgrade {index}/{len(pairs)}] C{c}->T{t}", flush=True)
            payload = saved
        else:
            print(f"[evaluate {index}/{len(pairs)}] {args.method} C{c}->T{t}", flush=True)
            payload = evaluate_one(args, model, tokenizer, rows, banks, c, t,
                                   c_cp, c_ro, t_cp, t_ro)

        if not args.skip_recipient:
            if recipient_schedules is None:
                recipient_schedules = prepare_schedules(rows["test"], args.seed)
            payload["recipient"] = evaluate_recipient_only(
                args, model, tokenizer, banks, recipient_schedules,
                c, t, c_cp, c_ro, t_cp, t_ro,
            )
            payload["recipient_version"] = RECIPIENT_VERSION
            payload["recipient_seed"] = args.seed
            print(f"  different-recipient T replay (C_only) = "
                  f"{payload['recipient']['C_only']['Y_via_T_replay_recipient_different']:.4f}")

        payload["signature"] = signature
        write_json(output, payload)
        rebuild_summary(eval_dir, data_sig)
        print(f"  C_only: Y={payload['metrics']['C_only']['Y_after_C']:.4f} "
              f"T={payload['metrics']['C_only']['T_after_C']:.4f} "
              f"replay={payload['metrics']['C_only']['Y_via_T_replay']:.4f} "
              f"restore={payload['metrics']['C_only']['Y_restored_via_T']:.4f}")

    summary = rebuild_summary(eval_dir, data_sig)
    print(f"[saved] {eval_dir / 'config_summary.csv'} ({len(summary)} evaluations)")


if __name__ == "__main__":
    main()
