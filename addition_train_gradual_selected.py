"""Train-only Addition GDAS grid: frozen Standard T -> conditional GDAS C.

Default pairs come from the Standard DAS CAL-selected layers. For an exhaustive
GDAS sweep, pass --t-layers all --c-layers all (requires trained Standard T
handles at every layer). Every GDAS C checkpoint is keyed by BOTH (C,T).
"""

import argparse
import re
from pathlib import Path

import torch

from addition_data_load import build_all_banks, load_addition_rows
from addition_das import set_seed
from addition_neural_network import load_gemma_model
from addition_train_standard_grid import build_factual_c_bank, fit_c_readout
from addition_train_gradual_grid import train_c_gradual
from addition_pipeline_utils import (
    assert_run_manifest,
    check_artifacts,
    count_layers,
    dataset_signature,
    gradual_paths,
    handle_paths,
    load_checkpoint,
    parse_layers,
    read_json,
    write_csv,
    write_json,
)


def rebuild_summary(root, standard_root):
    rows = []
    for directory in sorted((root / "pairs").glob("C_*_T_*")):
        match = re.fullmatch(r"C_(\d+)_T_(\d+)", directory.name)
        if not match:
            continue
        c, t = map(int, match.groups())
        ckpt_path, readout_path = gradual_paths(root, c, t)
        t_ckpt_path, t_readout_path = handle_paths(standard_root, "T", t)
        if not all(p.exists() for p in (ckpt_path, readout_path, t_ckpt_path, t_readout_path)):
            continue
        cp = load_checkpoint(ckpt_path, "C", c, "gradual", t)
        t_cp = load_checkpoint(t_ckpt_path, "T", t, "standard")
        c_readout = torch.load(readout_path, map_location="cpu", weights_only=False)
        rows.append({
            "C_layer": c,
            "T_layer": t,
            "C_cal_iia": float(cp["validation_direct_iia"]),
            "C_cal_score": float(cp["validation_score"]),
            "C_cal_T_replay_iia": float(cp["validation_T_mediator_iia"]),
            "C_cal_T_readout_acc": float(cp["validation_T_readout_acc"]),
            "T_cal_iia": float(t_cp["validation_iia"]),
            "C_readout_cal_acc": float(c_readout["validation_accuracy"]),
        })
    rows.sort(key=lambda r: (r["T_layer"], r["C_layer"]))
    fields = [
        "C_layer", "T_layer", "C_cal_iia", "C_cal_score",
        "C_cal_T_replay_iia", "C_cal_T_readout_acc", "T_cal_iia",
        "C_readout_cal_acc",
    ]
    write_csv(root / "summary.csv", rows, fields)
    return rows


def requested_pairs(args, root, model):
    if args.t_layers is None and args.c_layers is None:
        selection = Path(args.selection) if args.selection else (
            Path(args.standard_root) / "selected_layers_standard.json"
        )
        chosen = read_json(selection)
        if chosen.get("method") != "standard":
            raise ValueError(f"Expected Standard CAL selection: {selection}")
        if chosen.get("selection_source") != "addition_eval_all_layers_standard.py":
            raise RuntimeError(
                "Run addition_eval_all_layers_standard.py after the full-layer "
                "Standard sweep, before starting GDAS."
            )
        # Reject stale selection if any CAL-selected C/T checkpoint changed.
        signatures = chosen.get("calibration_signatures", {})
        for pair in chosen["pairs"]:
            for variable, layer in (("C", pair["C_layer"]), ("T", pair["T_layer"])):
                expected = signatures.get(f"{variable}_L{layer}")
                if expected is None:
                    raise RuntimeError(f"Missing CAL signature for {variable} L{layer}")
                ckpt_path, readout_path = handle_paths(args.standard_root, variable, layer)
                if expected["artifacts"] != check_artifacts([ckpt_path, readout_path]):
                    raise RuntimeError(
                        f"Stale CAL selection for {variable} L{layer}: "
                        "rerun addition_eval_all_layers_standard.py"
                    )
        pairs = chosen["pairs"]
    else:
        selection = Path(args.selection) if args.selection else (
            Path(args.standard_root) / "selected_layers_standard.json"
        )
        base = read_json(selection) if selection.exists() else {"layers": {}}
        num_layers = count_layers(model)
        t_layers = (parse_layers(args.t_layers, num_layers) if args.t_layers is not None
                    else base["layers"].get("T", []))
        c_layers = (parse_layers(args.c_layers, num_layers) if args.c_layers is not None
                    else base["layers"].get("C", []))
        pairs = [{"C_layer": c, "T_layer": t} for t in t_layers for c in c_layers if c < t]
    pairs = sorted({(int(p["C_layer"]), int(p["T_layer"])) for p in pairs},
                   key=lambda x: (x[1], x[0]))
    if not pairs or any(c < 0 or c >= t or t >= count_layers(model) for c, t in pairs):
        raise ValueError("Empty or invalid C<T GDAS pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", default="results_addition_standard_all_layers_seed0")
    parser.add_argument("--result-dir", default="results_addition_gradual_selected_seed0")
    parser.add_argument("--selection", help="Standard selected_layers_standard.json")
    parser.add_argument("--c-layers", help="Override selection with all, list or 0:26")
    parser.add_argument("--t-layers", help="Override selection with all, list or 0:26")
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=2.0)
    parser.add_argument("--lambda-readout", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(args.result_dir)
    standard_root = Path(args.standard_root)
    root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    data_rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir,
                                   args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, data_rows)
    sig = dataset_signature(model, args.data_dir, args.cache_dir, args.max_new_tokens)
    standard_manifest = read_json(standard_root / "run_manifest.json")
    if standard_manifest["data"] != sig or standard_manifest["seed"] != args.seed:
        raise RuntimeError("Standard T handles and GDAS must use the same data and seed")
    if standard_manifest["generation_cap"] != args.generation_cap:
        raise RuntimeError("Generation cap differs from the Standard CAL sweep")

    assert_run_manifest(root, {
        "method": "gradual", "data": sig, "model": str(model.config._name_or_path),
        "standard_root": str(standard_root.resolve()), "seed": args.seed,
        "c_dim": args.c_dim, "c_epochs": args.c_epochs, "lr": args.lr,
        "lambda_med": args.lambda_med, "lambda_readout": args.lambda_readout,
        "train_batch_size": args.train_batch_size, "eval_batch_size": args.eval_batch_size,
        "generation_cap": args.generation_cap,
    })

    pairs = requested_pairs(args, root, model)
    prefix_ids = banks["prefix_ids"]
    c_factual_train = build_factual_c_bank(tokenizer, data_rows["train"]["C_only"], prefix_ids)
    c_factual_val = build_factual_c_bank(tokenizer, data_rows["validation"]["C_only"], prefix_ids)

    t_cache = {}
    for index, (c_layer, t_layer) in enumerate(pairs, 1):
        if t_layer not in t_cache:
            t_path, t_ro_path = handle_paths(standard_root, "T", t_layer)
            check_artifacts([t_path, t_ro_path])
            t_cache[t_layer] = (
                load_checkpoint(t_path, "T", t_layer, "standard"),
                torch.load(t_ro_path, map_location="cpu", weights_only=False),
            )
        t_cp, t_readout = t_cache[t_layer]
        t_path, t_ro_path = handle_paths(standard_root, "T", t_layer)
        c_path, c_ro_path = gradual_paths(root, c_layer, t_layer)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = c_path.parent / "train_signature.json"
        identity = {
            "upstream_artifacts": check_artifacts([t_path, t_ro_path]),
            "run_manifest": read_json(root / "run_manifest.json"),
            "C_layer": c_layer, "T_layer": t_layer,
        }
        if meta_path.exists():
            if read_json(meta_path) != identity:
                raise RuntimeError(f"Stale GDAS dependency at {meta_path}; use a new result-dir")
        elif c_path.exists() or c_ro_path.exists():
            raise RuntimeError(f"Legacy pair without dependency signature at {c_path.parent}")
        else:
            write_json(meta_path, identity)

        print(f"[gradual {index}/{len(pairs)}] C{c_layer} -> T{t_layer}", flush=True)
        if c_path.exists():
            c_cp = load_checkpoint(c_path, "C", c_layer, "gradual", t_layer)
        else:
            c_cp = train_c_gradual(
                model, tokenizer, banks["C"]["train"], banks["C"]["validation"],
                t_cp, t_readout, c_layer, args.c_dim, args.c_epochs, args.lr,
                args.lambda_med, args.lambda_readout, args.train_batch_size,
                args.eval_batch_size, args.generation_cap, args.seed, c_path,
            )
        if not c_ro_path.exists():
            c_readout = fit_c_readout(
                model, c_factual_train, c_factual_val, c_cp["basis"], c_layer,
                args.eval_batch_size, args.seed,
            )
            torch.save(c_readout, c_ro_path)
        rebuild_summary(root, standard_root)

    summary = rebuild_summary(root, standard_root)
    print(f"[saved] {root / 'summary.csv'} ({len(summary)} completed GDAS pairs)")


if __name__ == "__main__":
    main()
