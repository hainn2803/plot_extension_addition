import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from addition_das import collect_layer_states, set_seed
from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_train_c import evaluate_c, train_c
from addition_train_t import evaluate_t, fit_t_readout, train_t


def parse_layers(value, num_layers):
    if value == "all":
        return list(range(num_layers))

    layers = []

    for item in value.split(","):
        item = item.strip()
        if item:
            layers.append(int(item))

    return layers


def save_csv(rows, path):
    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_t_test_metrics(model, tokenizer, bank, checkpoint, batch_size):
    layer = int(checkpoint["layer"])
    device = next(model.parameters()).device
    Q = checkpoint["basis"].float().to(device)
    source_states = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], layer, batch_size)
    metrics = evaluate_t(model, tokenizer, bank, source_states, Q, layer, batch_size)
    checkpoint["test_iia"] = float(metrics["numeric_iia"])
    checkpoint["test_token_iia"] = float(metrics["token_iia"])
    checkpoint["test_teacher_forced_iia"] = float(metrics["teacher_forced_iia"])
    return checkpoint


def add_c_test_metrics(model, tokenizer, test_bank, propagation_bank, checkpoint, t_checkpoint, t_readout, batch_size):
    device = next(model.parameters()).device
    c_layer = int(checkpoint["layer"])
    t_layer = int(t_checkpoint["layer"])
    Q_c = checkpoint["basis"].float().to(device)
    Q_t = t_checkpoint["basis"].float().to(device)
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_positions"], c_layer, batch_size)
    test_metrics = evaluate_c(model, tokenizer, test_bank, test_source, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size)

    propagation_source = collect_layer_states(model, propagation_bank["source_input_ids"], propagation_bank["source_attention_mask"], propagation_bank["source_positions"], c_layer, batch_size)
    propagation_metrics = evaluate_c(model, tokenizer, propagation_bank, propagation_source, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size)

    checkpoint["test_direct_iia"] = float(test_metrics["direct_numeric_iia"])
    checkpoint["test_direct_token_iia"] = float(test_metrics["direct_token_iia"])
    checkpoint["test_direct_teacher_forced_iia"] = float(test_metrics["direct_teacher_forced_iia"])
    checkpoint["test_T_mediator_iia"] = float(test_metrics["mediator_numeric_iia"])
    checkpoint["test_T_mediator_token_iia"] = float(test_metrics["mediator_token_iia"])
    checkpoint["test_T_mediator_teacher_forced_iia"] = float(test_metrics["mediator_teacher_forced_iia"])
    checkpoint["test_T_readout_acc"] = float(test_metrics["t_readout_acc"])

    checkpoint["propagation_direct_iia"] = float(propagation_metrics["direct_numeric_iia"])
    checkpoint["propagation_direct_token_iia"] = float(propagation_metrics["direct_token_iia"])
    checkpoint["propagation_direct_teacher_forced_iia"] = float(propagation_metrics["direct_teacher_forced_iia"])
    checkpoint["propagation_T_mediator_iia"] = float(propagation_metrics["mediator_numeric_iia"])
    checkpoint["propagation_T_mediator_token_iia"] = float(propagation_metrics["mediator_token_iia"])
    checkpoint["propagation_T_mediator_teacher_forced_iia"] = float(propagation_metrics["mediator_teacher_forced_iia"])
    checkpoint["propagation_T_readout_acc"] = float(propagation_metrics["t_readout_acc"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_gradual_2.0")
    parser.add_argument("--t-layers", default="18,19")
    parser.add_argument("--c-layers", default="16,17")
    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--t-epochs", type=int, default=20)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--c-lambda-med", type=float, default=2.0)
    parser.add_argument("--c-lambda-readout", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    result_dir = Path(args.result_dir)
    sweep_dir = result_dir / "sweep"
    result_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)

    t_layers = parse_layers(args.t_layers, model.config.num_hidden_layers)
    c_layers_requested = parse_layers(args.c_layers, model.config.num_hidden_layers)

    if len(t_layers) == 0:
        raise ValueError("No T layers to sweep")

    t_summary = []
    print("\n=== SWEEP T LAYERS WITH LEARNED DAS ===")

    for t_layer in t_layers:
        layer_dir = sweep_dir / f"T_layer_{t_layer:02d}"
        checkpoint_path = layer_dir / "das_t.pt"

        if args.resume and checkpoint_path.exists():
            result = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if result.get("metric_version") != "autoregressive_numeric_v1":
                raise RuntimeError(f"Old/incompatible T checkpoint at {checkpoint_path}. Remove it or run without --resume.")
            print(f"[resume T layer {t_layer:02d}] validation_iia={result['validation_iia']:.4f}")
        else:
            result = train_t(model, tokenizer, banks["T"]["train"], banks["T"]["validation"], None, t_layer, args.t_dim, args.t_epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.seed, checkpoint_path)

        t_summary.append({
            "layer": int(t_layer),
            "validation_iia": float(result["validation_iia"]),
            "validation_token_iia": float(result["validation_token_iia"]),
            "validation_teacher_forced_iia": float(result["validation_teacher_forced_iia"]),
            "best_epoch": int(result["best_epoch"]),
        })

    save_csv(t_summary, result_dir / "t_layer_sweep.csv")
    best_t_row = max(t_summary, key=lambda row: row["validation_iia"])
    best_t_layer = int(best_t_row["layer"])
    best_t_sweep_path = sweep_dir / f"T_layer_{best_t_layer:02d}" / "das_t.pt"
    root_t_path = result_dir / "das_t.pt"
    shutil.copy2(best_t_sweep_path, root_t_path)
    t_checkpoint = torch.load(root_t_path, map_location="cpu", weights_only=False)
    t_checkpoint = add_t_test_metrics(model, tokenizer, banks["T"]["test"], t_checkpoint, args.eval_batch_size)
    torch.save(t_checkpoint, root_t_path)

    print(
        f"\n[T SELECTED] layer={best_t_layer} "
        f"validation_iia={best_t_row['validation_iia']:.4f} "
        f"test_iia={t_checkpoint['test_iia']:.4f}"
    )

    print("\n=== FIT T READOUT AT SELECTED T LAYER ===")
    t_readout_path = result_dir / "t_readout_LogisticRegression.pt"
    t_readout = fit_t_readout(model, banks["T_factual"]["train"], banks["T_factual"]["validation"], t_checkpoint["basis"], best_t_layer, args.eval_batch_size, args.seed, t_readout_path)

    c_layers = []

    for layer in c_layers_requested:
        if layer < best_t_layer:
            c_layers.append(layer)

    if len(c_layers) == 0:
        raise ValueError(f"No C layers before selected T layer {best_t_layer}")

    c_summary = []
    print("\n=== SWEEP C LAYERS WITH LEARNED DAS ===")

    for c_layer in c_layers:
        layer_dir = sweep_dir / f"C_layer_{c_layer:02d}_T_layer_{best_t_layer:02d}"
        checkpoint_path = layer_dir / "das_c.pt"

        if args.resume and checkpoint_path.exists():
            result = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if result.get("metric_version") != "autoregressive_numeric_v1":
                raise RuntimeError(f"Old/incompatible C checkpoint at {checkpoint_path}. Remove it or run without --resume.")
            print(f"[resume C layer {c_layer:02d}] validation_score={result['validation_score']:.4f}")
        else:
            result = train_c(model, tokenizer, banks["C"]["train"], banks["C"]["validation"], None, None, t_checkpoint, t_readout, c_layer, args.c_dim, args.c_epochs, args.lr, args.c_lambda_med, args.c_lambda_readout, args.train_batch_size, args.eval_batch_size, args.seed, checkpoint_path)

        c_summary.append({
            "layer": int(c_layer),
            "T_layer": int(best_t_layer),
            "validation_score": float(result["validation_score"]),
            "validation_direct_iia": float(result["validation_direct_iia"]),
            "validation_direct_token_iia": float(result["validation_direct_token_iia"]),
            "validation_direct_teacher_forced_iia": float(result["validation_direct_teacher_forced_iia"]),
            "validation_T_mediator_iia": float(result["validation_T_mediator_iia"]),
            "validation_T_mediator_token_iia": float(result["validation_T_mediator_token_iia"]),
            "validation_T_readout_acc": float(result["validation_T_readout_acc"]),
            "best_epoch": int(result["best_epoch"]),
        })

    save_csv(c_summary, result_dir / "c_layer_sweep.csv")
    best_c_row = max(c_summary, key=lambda row: row["validation_score"])
    best_c_layer = int(best_c_row["layer"])
    best_c_sweep_path = sweep_dir / f"C_layer_{best_c_layer:02d}_T_layer_{best_t_layer:02d}" / "das_c.pt"
    root_c_path = result_dir / "das_c.pt"
    shutil.copy2(best_c_sweep_path, root_c_path)
    c_checkpoint = torch.load(root_c_path, map_location="cpu", weights_only=False)
    c_checkpoint = add_c_test_metrics(model, tokenizer, banks["C"]["test"], banks["C_T"]["test"], c_checkpoint, t_checkpoint, t_readout, args.eval_batch_size)
    torch.save(c_checkpoint, root_c_path)

    selected = {
        "T_layer": best_t_layer,
        "T_validation_iia": best_t_row["validation_iia"],
        "T_test_iia": t_checkpoint["test_iia"],
        "C_layer": best_c_layer,
        "C_validation_score": best_c_row["validation_score"],
        "C_test_direct_iia": c_checkpoint["test_direct_iia"],
        "C_test_T_mediator_iia": c_checkpoint["test_T_mediator_iia"],
        "C_test_T_readout_acc": c_checkpoint["test_T_readout_acc"],
        "C_lambda_med": args.c_lambda_med,
        "C_lambda_readout": args.c_lambda_readout,
        "seed": args.seed,
    }
    (result_dir / "selected_layers.json").write_text(json.dumps(selected, indent=2) + "\n")

    print(
        f"\n[C SELECTED] layer={best_c_layer} validation_score={best_c_row['validation_score']:.4f} "
        f"test_direct_iia={c_checkpoint['test_direct_iia']:.4f}"
    )
    print(f"[FINAL] C layer={best_c_layer} -> T layer={best_t_layer} -> Y")
    print(f"[saved] {root_c_path}")
    print(f"[saved] {root_t_path}")
    print(f"[saved] {t_readout_path}")


if __name__ == "__main__":
    main()
