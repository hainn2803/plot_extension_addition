"""Train independent Standard DAS C/T handles at every requested layer; no chain evaluation.

Example:
  python3 addition_train_all_layers_standard.py --layers all --seed 0
  python3 addition_train_all_layers_standard.py --variables C --layers 0:12 --seed 0
Rerunning skips completed handles; summary.csv is regenerated from checkpoints.
"""

import argparse
import re
from pathlib import Path

import torch

from addition_data_load import build_all_banks, load_addition_rows
from addition_das import set_seed
from addition_neural_network import load_gemma_model
from addition_train_standard_grid import (
    build_factual_c_bank,
    fit_c_readout,
    train_c_standard,
    train_t_handle,
)
from addition_train_t import fit_t_readout
from addition_pipeline_utils import (
    assert_run_manifest,
    count_layers,
    dataset_signature,
    handle_paths,
    load_checkpoint,
    parse_layers,
    write_csv,
)


def rebuild_summary(root):
    rows = []
    for variable in ("T", "C"):
        folder = root / f"{variable}_handles"
        for directory in sorted(folder.glob(f"{variable}_layer_*")):
            match = re.fullmatch(rf"{variable}_layer_(\d+)", directory.name)
            if not match:
                continue
            layer = int(match.group(1))
            ckpt_path, readout_path = handle_paths(root, variable, layer)
            if not ckpt_path.exists() or not readout_path.exists():
                continue
            ckpt = load_checkpoint(ckpt_path, variable, layer, "standard")
            readout = torch.load(readout_path, map_location="cpu", weights_only=False)
            cal_iia = ckpt["validation_iia"] if variable == "T" else ckpt["validation_direct_iia"]
            rows.append({
                "variable": variable,
                "layer": layer,
                "cal_iia": float(cal_iia),
                "readout_cal_acc": float(readout["validation_accuracy"]),
                "best_epoch": int(ckpt["best_epoch"]),
                "seed": int(ckpt["seed"]),
            })
    rows.sort(key=lambda row: (row["variable"], row["layer"]))
    write_csv(root / "summary.csv", rows, ["variable", "layer", "cal_iia", "readout_cal_acc", "best_epoch", "seed"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results_addition_standard_all_layers_seed0")
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--variables", default="T,C", help="T,C or a subset")
    parser.add_argument("--layers", default="all", help="all; 0,1,2; or half-open range 0:26")
    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--t-epochs", type=int, default=20)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    variables = [v.strip().upper() for v in args.variables.split(",") if v.strip()]
    if not variables or any(v not in ("T", "C") for v in variables):
        parser.error("--variables must be T,C or a subset")

    set_seed(args.seed)
    root = Path(args.result_dir)
    root.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_gemma_model(args.device)
    layers = parse_layers(args.layers, count_layers(model))
    data_rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir,
                                   args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, data_rows)
    sig = dataset_signature(model, args.data_dir, args.cache_dir, args.max_new_tokens)
    assert_run_manifest(root, {
        "method": "standard", "data": sig, "model": str(model.config._name_or_path),
        "t_dim": args.t_dim, "c_dim": args.c_dim,
        "t_epochs": args.t_epochs, "c_epochs": args.c_epochs,
        "lr": args.lr, "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size, "generation_cap": args.generation_cap,
        "seed": args.seed,
    })

    for variable in variables:
        for layer in layers:
            ckpt_path, readout_path = handle_paths(root, variable, layer)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[standard] {variable} layer {layer}/{count_layers(model)-1}", flush=True)

            if ckpt_path.exists():
                checkpoint = load_checkpoint(ckpt_path, variable, layer, "standard")
            elif variable == "T":
                checkpoint = train_t_handle(
                    model, tokenizer, banks["T"]["train"], banks["T"]["validation"],
                    layer, args.t_dim, args.t_epochs, args.lr,
                    args.train_batch_size, args.eval_batch_size, args.generation_cap,
                    args.seed, ckpt_path,
                )
            else:
                checkpoint = train_c_standard(
                    model, tokenizer, banks["C"]["train"], banks["C"]["validation"],
                    layer, args.c_dim, args.c_epochs, args.lr,
                    args.train_batch_size, args.eval_batch_size, args.generation_cap,
                    args.seed, ckpt_path,
                )

            if not readout_path.exists():
                if variable == "T":
                    fit_t_readout(
                        model, banks["T_factual"]["train"], banks["T_factual"]["validation"],
                        checkpoint["basis"], layer, args.eval_batch_size, args.seed,
                        readout_path,
                    )
                else:
                    prefix_ids = banks["prefix_ids"]
                    train_bank = build_factual_c_bank(tokenizer, data_rows["train"]["C_only"], prefix_ids)
                    val_bank = build_factual_c_bank(tokenizer, data_rows["validation"]["C_only"], prefix_ids)
                    readout = fit_c_readout(
                        model, train_bank, val_bank, checkpoint["basis"], layer,
                        args.eval_batch_size, args.seed,
                    )
                    torch.save(readout, readout_path)

            rebuild_summary(root)
    summary = rebuild_summary(root)
    print(f"[saved] {root / 'summary.csv'} ({len(summary)} trained variable/layer entries)")


if __name__ == "__main__":
    main()
