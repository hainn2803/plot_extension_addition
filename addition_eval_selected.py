"""Evaluate saved Addition Standard or Gradual DAS C -> T -> Y configurations.

Uses held-out test pairs only AFTER calibration-based layer selection.
Saves one metrics.json per configuration and a flattened summary.csv.
Existing results are reused only when checkpoint/data/generation signatures match.
"""

import argparse
import re
from pathlib import Path

import torch

from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_recipient_schedule import RECIPIENT_VERSION, prepare_schedules
from addition_eval_recipient_transfer import eval_recipient
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
    write_csv,
    write_json,
)


def flatten(payload):
    row = {
        "method": payload["method"],
        "C_layer": payload["C_layer"],
        "T_layer": payload["T_layer"],
        "C_cal_iia": payload["calibration"]["C_cal_iia"],
        "T_cal_iia": payload["calibration"]["T_cal_iia"],
    }
    for name, value in payload["clean"].items():
        row[f"clean_{name}"] = value
    for family in ("T_only", "C_only", "C_T"):
        for metric, value in payload["metrics"][family].items():
            row[f"{family}_{metric}"] = value
        for metric, value in payload.get("recipient", {}).get(family, {}).items():
            row[f"{family}_{metric}"] = value
    return row


def rebuild_summary(directory, data_sig):
    rows = []
    for path in sorted(directory.glob("C_*_T_*/metrics.json")):
        payload = read_json(path)
        if payload.get("signature", {}).get("data") != data_sig:
            continue
        rows.append(flatten(payload))
    rows.sort(key=lambda row: (row["T_layer"], row["C_layer"]))
    if rows:
        fields = list(rows[0])
        for row in rows[1:]:
            for field in row:
                if field not in fields:
                    fields.append(field)
        write_csv(directory / "summary.csv", rows, fields)
    return rows


def get_paths(args, c_layer, t_layer):
    t_root = args.standard_root if args.method == "gradual" else args.result_dir
    t_paths = handle_paths(t_root, "T", t_layer)
    if args.method == "standard":
        c_paths = handle_paths(args.result_dir, "C", c_layer)
    else:
        c_paths = gradual_paths(args.result_dir, c_layer, t_layer)
    return c_paths, t_paths


def load_pair(args, c_layer, t_layer):
    (c_path, c_ro_path), (t_path, t_ro_path) = get_paths(args, c_layer, t_layer)
    check_artifacts([c_path, c_ro_path, t_path, t_ro_path])

    if args.method == "gradual":
        metadata = read_json(c_path.parent / "train_signature.json")
        if metadata["upstream_artifacts"] != check_artifacts([t_path, t_ro_path]):
            raise RuntimeError(f"Frozen T artifacts changed since GDAS C training: {c_path}")

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
    parser.add_argument("--standard-root", default="results_addition_standard_all_layers_seed0",
                        help="Required source of frozen T handles for gradual")
    parser.add_argument(
        "--selection",
        help="Optional selection JSON. By default both methods use the Standard "
             "CAL-selected pairs, so SDAS/GDAS are evaluated on the same grid.",
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
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_dir)
    if args.selection:
        selection_path = Path(args.selection)
    elif args.method == "gradual":
        # GDAS was trained using SDAS promising candidates. Evaluate the same
        # 20 (C,T) pairs unless the caller explicitly supplies another selection.
        selection_path = Path(args.standard_root) / "selected_layers_standard.json"
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

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir,
                              args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)
    data_sig = dataset_signature(model, args.data_dir, args.cache_dir, args.max_new_tokens)
    manifest = read_json(root / "run_manifest.json")
    if manifest["method"] != args.method or manifest["data"] != data_sig:
        raise RuntimeError("Evaluation dataset does not match trained checkpoints")
    if manifest["generation_cap"] != args.generation_cap:
        raise RuntimeError("CAL and TEST must use the same generation cap")
    if args.method == "gradual":
        if manifest["standard_root"] != str(Path(args.standard_root).resolve()):
            raise RuntimeError("Wrong Standard T root for this GDAS run")

    eval_dir = root / f"evaluation_{args.method}"
    recipient_schedules = None
    for index, (c, t) in enumerate(pairs, 1):
        (c_path, c_ro_path), (t_path, t_ro_path) = get_paths(args, c, t)
        signature = evaluation_signature(
            args.method, c, t, [c_path, c_ro_path, t_path, t_ro_path],
            data_sig, args.generation_cap,
        )
        output = eval_dir / f"C_{c:02d}_T_{t:02d}" / "metrics.json"
        saved = None
        if output.exists() and not args.force:
            previous = read_json(output)
            if previous.get("signature") == signature:
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
    print(f"[saved] {eval_dir / 'summary.csv'} ({len(summary)} evaluations)")


if __name__ == "__main__":
    main()
