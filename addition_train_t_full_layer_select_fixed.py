"""Select the T layer by full-vector IIA, then train DAS only at that layer.

Flow:
1. Build the controlled addition banks.
2. On the validation T-only bank, sweep full-vector interchange across layers.
3. Select the layer with the highest autoregressive numeric IIA.
4. Train T DAS only at the selected layer.
5. Evaluate the selected DAS checkpoint on test once.
6. Fit the T readout for later C -> T -> Y experiments.

The layer-selection metric is autoregressive numeric IIA, not teacher-forced IIA,
so it remains valid when an answer contains more than one token.
"""

import argparse
from pathlib import Path

from addition_das import collect_layer_states, set_seed
from addition_data_load import build_all_banks, load_addition_rows
from addition_full_layer_iia import evaluate_layer, parse_layers, save_csv
from addition_neural_network import load_gemma_model
from addition_train_t import fit_t_readout, train_t


def sweep_full_layer_iia(model, tokenizer, validation_bank, layers, batch_size):
    results = []

    print("\n=== FULL-LAYER T IIA SWEEP ON VALIDATION ===")

    for layer in layers:
        source_states = collect_layer_states(model, validation_bank["source_input_ids"], validation_bank["source_attention_mask"], validation_bank["source_positions"], layer, batch_size)
        result = evaluate_layer(model, tokenizer, validation_bank, source_states, layer, batch_size)
        results.append(result)

        print(
            f"[T full layer {layer:02d}] "
            f"numeric_iia={result['autoregressive_numeric_iia']:.4f} "
            f"token_iia={result['autoregressive_token_iia']:.4f} "
            f"first_token_iia={result['first_token_iia']:.4f} "
            f"teacher_iia={result['teacher_forced_token_iia']:.4f}"
        )

    best_result = max(results, key=lambda result: result["autoregressive_numeric_iia"])
    best_layer = int(best_result["layer"])

    print(
        f"\n[T FULL-LAYER BEST] "
        f"layer={best_layer} "
        f"numeric_iia={best_result['autoregressive_numeric_iia']:.4f} "
        f"token_iia={best_result['autoregressive_token_iia']:.4f} "
        f"first_token_iia={best_result['first_token_iia']:.4f} "
        f"teacher_iia={best_result['teacher_forced_token_iia']:.4f}"
    )

    return best_layer, best_result, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_t_full_select")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)

    layers = parse_layers(args.layers, model.config.num_hidden_layers)

    if len(layers) == 0:
        raise ValueError("No layers to sweep.")

    max_target_tokens = int(banks["T"]["validation"]["target_answer_mask"].sum(dim=1).max())

    print("T candidate layers =", layers)
    print("validation rows =", len(banks["T"]["validation"]["target_y"]))
    print("max validation target tokens =", max_target_tokens)

    best_layer, best_full_result, full_results = sweep_full_layer_iia(model, tokenizer, banks["T"]["validation"], layers, args.eval_batch_size)

    sweep_path = result_dir / "full_layer_iia_T_validation.csv"
    save_csv(full_results, sweep_path)
    print("saved full-layer sweep =", sweep_path)

    print(f"\n=== TRAIN T DAS AT SELECTED LAYER {best_layer} ===")

    checkpoint_path = result_dir / "das_t.pt"
    das_result = train_t(
        model,
        tokenizer,
        banks["T"]["train"],
        banks["T"]["validation"],
        banks["T"]["test"],
        best_layer,
        args.dim,
        args.epochs,
        args.lr,
        args.train_batch_size,
        args.eval_batch_size,
        args.seed,
        checkpoint_path,
    )

    print("\n=== FIT T READOUT ===")

    readout_path = result_dir / "t_readout_LogisticRegression.pt"
    readout_result = fit_t_readout(
        model,
        banks["T_factual"]["train"],
        banks["T_factual"]["validation"],
        das_result["basis"],
        best_layer,
        args.eval_batch_size,
        args.seed,
        readout_path,
    )

    print("\n=== SUMMARY ===")
    print(f"selected_layer = {best_layer}")
    print(f"full_layer_validation_numeric_iia = {best_full_result['autoregressive_numeric_iia']:.4f}")
    print(f"das_validation_numeric_iia = {das_result['validation_iia']:.4f}")
    print(f"das_test_numeric_iia = {das_result['test_iia']:.4f}")
    print(f"t_readout_validation_accuracy = {readout_result['validation_accuracy']:.4f}")
    print("DAS checkpoint =", checkpoint_path)
    print("T readout =", readout_path)


if __name__ == "__main__":
    main()
