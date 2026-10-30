"""Sweep full-vector interchange intervention across layers for addition.

For each layer:
1. Collect the source hidden state at answer_start.
2. Replace the full base hidden vector at answer_start with the source vector.
3. Measure counterfactual output accuracy.

Primary metric:
    autoregressive_numeric_iia

This metric generates the answer autoregressively, so it remains valid when the
numeric answer is represented by more than one token.

A teacher-forced token-exact IIA is also reported only as a diagnostic.
"""

import argparse
import csv
from pathlib import Path

import torch

from addition_das import autoregressive_numeric_accuracy, autoregressive_token_iia, collect_layer_states, sequence_iia, sequence_logits, hidden_to_logits, set_seed
from addition_data_load import build_pair_bank, get_answer_prefix_ids, load_addition_rows
from addition_neural_network import load_gemma_model


def parse_layers(value, num_layers):
    if value == "all":
        return list(range(num_layers - 1, -1, -1))

    layers = []

    for item in value.split(","):
        item = item.strip()
        if item:
            layers.append(int(item))

    return sorted(layers, reverse=True)


@torch.no_grad()
def run_full_layer_teacher_forced(model, bank, source_states, layer, batch_size=32):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["teacher_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["teacher_input_ids"]))
        ids = bank["teacher_input_ids"][start:end].to(device)
        mask = bank["teacher_attention_mask"][start:end].to(device)
        donor = source_states[start:end].to(device)
        intervention_positions = bank["intervention_positions"][start:end].to(device)
        label_positions = bank["label_positions"][start:end]
        rows = torch.arange(len(ids), device=device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            new_hidden[rows, intervention_positions, :] = donor.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            logits = sequence_logits(model, outputs, label_positions)
            all_logits.append(logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def run_full_layer_autoregressive(model, bank, source_states, layer, batch_size=32):
    device = next(model.parameters()).device
    max_target_length = int(bank["target_answer_mask"].sum(dim=1).max())
    all_generated = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_states[start:end].to(device)
        intervention_positions = bank["base_positions"][start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        generated = []

        for _ in range(max_target_length):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                new_hidden[rows, intervention_positions, :] = donor.to(hidden.dtype)

                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]

                return new_hidden

            handle = model.model.layers[layer].register_forward_hook(hook)

            try:
                position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
                next_logits = hidden_to_logits(model, outputs.last_hidden_state[:, -1, :])
                next_ids = next_logits.argmax(dim=-1)
            finally:
                handle.remove()

            generated.append(next_ids.detach().cpu())
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            new_mask = torch.ones((len(ids), 1), dtype=mask.dtype, device=device)
            mask = torch.cat([mask, new_mask], dim=1)

        all_generated.append(torch.stack(generated, dim=1))

    return torch.cat(all_generated, dim=0)


def select_rows(rows, split, target):
    if target == "T":
        return rows[split]["T_only"], "T"

    if target == "C":
        return rows[split]["C_only"], "C"

    if target == "C_T":
        return rows[split]["C_T"], "C"

    raise ValueError(f"Unknown target: {target}")


def evaluate_layer(model, tokenizer, bank, source_states, layer, batch_size):
    generated = run_full_layer_autoregressive(model, bank, source_states, layer, batch_size)
    ar_numeric_iia = autoregressive_numeric_accuracy(generated, bank["target_y"], tokenizer, bank["target_answer_mask"])
    ar_token_iia = autoregressive_token_iia(generated, bank["target_answer_ids"], bank["target_answer_mask"])

    teacher_logits = run_full_layer_teacher_forced(model, bank, source_states, layer, batch_size)
    teacher_iia = sequence_iia(teacher_logits, bank["target_answer_ids"], bank["target_answer_mask"])

    first_token_prediction = generated[:, 0]
    first_token_target = bank["target_answer_ids"][:, 0]
    first_token_iia = float((first_token_prediction.cpu() == first_token_target.cpu()).float().mean())

    return {
        "layer": layer,
        "autoregressive_numeric_iia": ar_numeric_iia,
        "autoregressive_token_iia": ar_token_iia,
        "first_token_iia": first_token_iia,
        "teacher_forced_token_iia": teacher_iia,
    }


def save_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer",
        "autoregressive_numeric_iia",
        "autoregressive_token_iia",
        "first_token_iia",
        "teacher_forced_token_iia",
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--target", choices=["T", "C", "C_T"], default="C")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--filter-max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_gemma_model(args.device)

    rows = load_addition_rows(
        model,
        tokenizer,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.filter_max_new_tokens,
        force_rebuild=args.force_rebuild_cache,
    )

    selected_rows, intervention_name = select_rows(rows, args.split, args.target)

    if len(selected_rows) == 0:
        raise ValueError(f"No rows for split={args.split}, target={args.target}")

    prefix_ids = get_answer_prefix_ids(tokenizer)
    bank = build_pair_bank(tokenizer, selected_rows, intervention_name, prefix_ids)
    layers = parse_layers(args.layers, model.config.num_hidden_layers)

    max_target_tokens = int(bank["target_answer_mask"].sum(dim=1).max())
    print(f"split={args.split} target={args.target} rows={len(selected_rows)}")
    print(f"layers={layers}")
    print(f"max_target_tokens={max_target_tokens}")

    results = []

    for layer in layers:
        source_states = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], layer, args.batch_size)
        result = evaluate_layer(model, tokenizer, bank, source_states, layer, args.batch_size)
        results.append(result)

        print(
            f"[layer {layer:02d}] "
            f"AR_numeric={result['autoregressive_numeric_iia']:.4f} "
            f"AR_token={result['autoregressive_token_iia']:.4f} "
            f"first={result['first_token_iia']:.4f} "
            f"teacher={result['teacher_forced_token_iia']:.4f}"
        )

    best = max(results, key=lambda item: item["autoregressive_numeric_iia"])
    print(
        f"\n[BEST] layer={best['layer']} "
        f"AR_numeric={best['autoregressive_numeric_iia']:.4f} "
        f"AR_token={best['autoregressive_token_iia']:.4f} "
        f"first={best['first_token_iia']:.4f} "
        f"teacher={best['teacher_forced_token_iia']:.4f}"
    )

    output = args.output

    if output is None:
        output = f"full_layer_iia_{args.target}_{args.split}.csv"

    save_csv(results, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
