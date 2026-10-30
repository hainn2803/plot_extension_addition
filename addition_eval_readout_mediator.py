import argparse
import json
import random
import re
from pathlib import Path

import torch

from addition_das import autoregressive_numeric_accuracy, autoregressive_token_iia, classification_accuracy, collect_layer_states, first_token_accuracy, run_das_autoregressive, run_das_intervention, sequence_iia
from addition_data_load import build_all_banks, build_state_inputs, build_teacher_forcing_batch, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_train_c import run_c_intervention_and_capture, run_frozen_t_mediator


def clean_t_readout(model, bank, Q_t, t_layer, W_t, b_t, batch_size):
    states = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["positions"], t_layer, batch_size)
    logits = (states.to(Q_t.device) @ Q_t) @ W_t.T + b_t
    base_indices = torch.arange(0, len(bank["target_t"]), 2)
    source_indices = torch.arange(1, len(bank["target_t"]), 2)
    base_acc = classification_accuracy(logits[base_indices.to(logits.device)], bank["target_t"][base_indices])
    source_acc = classification_accuracy(logits[source_indices.to(logits.device)], bank["target_t"][source_indices])
    return base_acc, source_acc


def numeric_correctness(generated_ids, target_y, tokenizer, target_answer_mask):
    correct = []

    for i in range(len(generated_ids)):
        target_length = int(target_answer_mask[i].sum().item())
        output_ids = generated_ids[i, :target_length].detach().cpu().tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        numbers = re.findall(r"-?\d+", output_text)

        if len(numbers) == 0:
            correct.append(False)
            continue

        predicted_y = int(numbers[0])
        expected_y = int(target_y[i].item())
        correct.append(predicted_y == expected_y)

    return torch.tensor(correct, dtype=torch.bool)


@torch.no_grad()
def eval_t(model, tokenizer, bank, Q_t, t_layer, batch_size):
    source_t = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], t_layer, batch_size)
    teacher_logits = run_das_intervention(model, bank, source_t, Q_t, t_layer, batch_size, False)
    generated_ids = run_das_autoregressive(model, bank, source_t, Q_t, t_layer, batch_size)

    return {
        "Y_after_T": autoregressive_numeric_accuracy(generated_ids, bank["target_y"], tokenizer, bank["target_answer_mask"]),
        "Y_after_T_token_exact": autoregressive_token_iia(generated_ids, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_T_teacher_forced": sequence_iia(teacher_logits, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_T_first_token": first_token_accuracy(teacher_logits, bank["target_answer_ids"]),
    }


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size):
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], c_layer, batch_size)
    direct_teacher, generated_t = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size, False)
    replay_teacher = run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size, False)
    t_logits = (generated_t.to(Q_t.device) @ Q_t) @ W_t.T + b_t

    direct_generated = run_das_autoregressive(model, bank, source_c, Q_c, c_layer, batch_size)
    replay_generated = run_das_autoregressive(model, bank, generated_t, Q_t, t_layer, batch_size)

    t_prediction = t_logits.argmax(-1).detach().cpu()
    t_correct = t_prediction == bank["target_t"].detach().cpu()
    y_correct = numeric_correctness(direct_generated, bank["target_y"], tokenizer, bank["target_answer_mask"])
    joint_ty_correct = t_correct & y_correct

    results = {
        "Y_after_C": float(y_correct.float().mean()),
        "Y_after_C_token_exact": autoregressive_token_iia(direct_generated, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_C_teacher_forced": sequence_iia(direct_teacher, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_C_first_token": first_token_accuracy(direct_teacher, bank["target_answer_ids"]),
        "T_after_C": float(t_correct.float().mean()),
        "T_and_Y_after_C": float(joint_ty_correct.float().mean()),
        "Y_via_T_replay": autoregressive_numeric_accuracy(replay_generated, bank["target_y"], tokenizer, bank["target_answer_mask"]),
        "Y_via_T_replay_token_exact": autoregressive_token_iia(replay_generated, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_via_T_replay_teacher_forced": sequence_iia(replay_teacher, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_via_T_replay_first_token": first_token_accuracy(replay_teacher, bank["target_answer_ids"]),
    }
    return results, generated_t


def build_recipient_examples(c_rows, seed):
    rng = random.Random(seed)
    recipients = []

    for donor_index, donor_row in enumerate(c_rows):
        transferred_t = donor_row["T_after_C_intervene"]
        candidate_indices = []

        for candidate_index, candidate in enumerate(c_rows):
            if candidate_index == donor_index:
                continue

            if candidate["base_input"] == donor_row["base_input"]:
                continue

            if candidate["base_T"] == transferred_t:
                continue

            candidate_indices.append(candidate_index)

        if len(candidate_indices) == 0:
            raise RuntimeError(f"Could not find recipient for donor {donor_index}")

        chosen_index = rng.choice(candidate_indices)
        recipient = c_rows[chosen_index]
        target_y = 100 * recipient["base_hundreds_out"] + 10 * transferred_t + recipient["base_ones_out"]

        recipients.append({
            "donor_index": donor_index,
            "recipient_index": chosen_index,
            "donor_base_input": donor_row["base_input"],
            "donor_source_input": donor_row["source_input"],
            "donor_base_T": donor_row["base_T"],
            "transferred_T": transferred_t,
            "recipient_input": recipient["base_input"],
            "recipient_natural_T": recipient["base_T"],
            "recipient_natural_Y": recipient["base_Y"],
            "recipient_ones_out": recipient["base_ones_out"],
            "recipient_hundreds_out": recipient["base_hundreds_out"],
            "recipient_target_Y": target_y,
        })

    return recipients


def build_clean_t_donor_pool(t_rows):
    pool = {}

    for row in t_rows:
        candidates = [(row["base_input"], row["base_T"]), (row["source_input"], row["source_T"])]

        for prompt, t_value in candidates:
            if t_value not in pool:
                pool[t_value] = []
            if prompt not in pool[t_value]:
                pool[t_value].append(prompt)

    return pool


def choose_clean_t_donors(recipient_examples, donor_pool, seed):
    rng = random.Random(seed)
    prompts = []

    for example in recipient_examples:
        t_value = example["transferred_T"]
        if t_value not in donor_pool:
            raise RuntimeError(f"No clean T donor for T={t_value}")

        candidates = []

        for prompt in donor_pool[t_value]:
            if prompt == example["recipient_input"]:
                continue
            candidates.append(prompt)

        if len(candidates) == 0:
            raise RuntimeError(f"No distinct clean T donor for T={t_value}")

        prompts.append(rng.choice(candidates))

    return prompts


def build_recipient_bank(tokenizer, recipient_examples, prefix_ids):
    prompts = []
    target_y = []

    for example in recipient_examples:
        prompts.append(example["recipient_input"])
        target_y.append(example["recipient_target_Y"])

    base_input_ids, base_attention_mask, base_positions = build_state_inputs(tokenizer, prompts, prefix_ids)
    teacher = build_teacher_forcing_batch(tokenizer, prompts, target_y, prefix_ids)
    bank = {
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "base_positions": base_positions,
        "target_y": torch.tensor(target_y, dtype=torch.long),
    }
    bank.update(teacher)
    return bank


@torch.no_grad()
def eval_recipients(model, tokenizer, c_rows, t_rows, prefix_ids, generated_t, Q_t, t_layer, batch_size, seed):
    examples = build_recipient_examples(c_rows, seed)
    recipient_bank = build_recipient_bank(tokenizer, examples, prefix_ids)

    if len(examples) != len(generated_t):
        raise RuntimeError("Recipient count does not match generated T states")

    induced_teacher = run_frozen_t_mediator(model, recipient_bank, generated_t, Q_t, t_layer, batch_size, False)
    induced_generated = run_das_autoregressive(model, recipient_bank, generated_t, Q_t, t_layer, batch_size)

    donor_pool = build_clean_t_donor_pool(t_rows)
    clean_prompts = choose_clean_t_donors(examples, donor_pool, seed + 1)
    clean_input_ids, clean_attention_mask, clean_positions = build_state_inputs(tokenizer, clean_prompts, prefix_ids)
    clean_states = collect_layer_states(model, clean_input_ids, clean_attention_mask, clean_positions, t_layer, batch_size)
    clean_teacher = run_frozen_t_mediator(model, recipient_bank, clean_states, Q_t, t_layer, batch_size, False)
    clean_generated = run_das_autoregressive(model, recipient_bank, clean_states, Q_t, t_layer, batch_size)

    return {
        "induced_T_recipient_Y": autoregressive_numeric_accuracy(induced_generated, recipient_bank["target_y"], tokenizer, recipient_bank["target_answer_mask"]),
        "induced_T_recipient_token_exact": autoregressive_token_iia(induced_generated, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"]),
        "induced_T_recipient_teacher_forced": sequence_iia(induced_teacher, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"]),
        "induced_T_recipient_first_token": first_token_accuracy(induced_teacher, recipient_bank["target_answer_ids"]),
        "clean_T_recipient_Y": autoregressive_numeric_accuracy(clean_generated, recipient_bank["target_y"], tokenizer, recipient_bank["target_answer_mask"]),
        "clean_T_recipient_token_exact": autoregressive_token_iia(clean_generated, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"]),
        "clean_T_recipient_teacher_forced": sequence_iia(clean_teacher, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"]),
        "clean_T_recipient_first_token": first_token_accuracy(clean_teacher, recipient_bank["target_answer_ids"]),
        "num_recipients": len(examples),
        "num_unique_recipients": len(set(example["recipient_index"] for example in examples)),
    }, examples


def save_jsonl(rows, path):
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def format_metrics(metrics):
    parts = []

    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_standard")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model(args.device)
    device = next(model.parameters()).device
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.batch_size, args.max_new_tokens, False)
    banks = build_all_banks(tokenizer, rows)
    prefix_ids = banks["prefix_ids"]

    result_dir = Path(args.result_dir)
    t_checkpoint = torch.load(result_dir / "das_t.pt", map_location="cpu", weights_only=False)
    c_checkpoint = torch.load(result_dir / "das_c.pt", map_location="cpu", weights_only=False)
    t_readout = torch.load(result_dir / "t_readout_LogisticRegression.pt", map_location="cpu", weights_only=False)

    Q_t = t_checkpoint["basis"].float().to(device)
    t_layer = int(t_checkpoint["layer"])
    Q_c = c_checkpoint["basis"].float().to(device)
    c_layer = int(c_checkpoint["layer"])
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    print("\n=== CLEAN T READOUT ===")
    t_base, t_source = clean_t_readout(model, banks["T_factual"]["test"], Q_t, t_layer, W_t, b_t, args.batch_size)
    print(f"T base/source: {t_base:.4f} / {t_source:.4f}")

    print("\n=== T -> Y ===")
    t_results = eval_t(model, tokenizer, banks["T"]["test"], Q_t, t_layer, args.batch_size)
    print(format_metrics(t_results))

    print("\n=== C -> T -> Y [C_only] ===")
    c_results, generated_t = eval_c(model, tokenizer, banks["C"]["test"], Q_c, c_layer, Q_t, t_layer, W_t, b_t, args.batch_size)
    print(format_metrics(c_results))

    print("\n=== C -> T -> Y [C_T] ===")
    c_t_results, generated_t_c_t = eval_c(model, tokenizer, banks["C_T"]["test"], Q_c, c_layer, Q_t, t_layer, W_t, b_t, args.batch_size)
    print(format_metrics(c_t_results))

    print("\n=== OTHER RECIPIENTS [C_only] ===")
    recipient_results, recipient_examples = eval_recipients(model, tokenizer, rows["test"]["C_only"], rows["test"]["T_only"], prefix_ids, generated_t, Q_t, t_layer, args.batch_size, args.seed + 100)
    print(format_metrics(recipient_results))

    print("\n=== OTHER RECIPIENTS [C_T] ===")
    c_t_recipient_results, c_t_recipient_examples = eval_recipients(model, tokenizer, rows["test"]["C_T"], rows["test"]["T_only"], prefix_ids, generated_t_c_t, Q_t, t_layer, args.batch_size, args.seed + 200)
    print(format_metrics(c_t_recipient_results))

    lines = [
        "=== CHECKPOINT ===",
        f"T layer: {t_layer}",
        f"C layer: {c_layer}",
        f"T validation_iia: {float(t_checkpoint['validation_iia']):.4f}",
        f"C validation_score: {float(c_checkpoint['validation_score']):.4f}",
        "",
        "=== CLEAN T READOUT ===",
        f"T base/source: {t_base:.4f} / {t_source:.4f}",
        "",
        "=== T -> Y ===",
        format_metrics(t_results),
        "",
        "=== C -> T -> Y [C_only] ===",
        format_metrics(c_results),
        "",
        "=== C -> T -> Y [C_T] ===",
        format_metrics(c_t_results),
        "",
        "=== OTHER RECIPIENTS [C_only] ===",
        format_metrics(recipient_results),
        "",
        "=== OTHER RECIPIENTS [C_T] ===",
        format_metrics(c_t_recipient_results),
    ]

    (result_dir / "eval_results.txt").write_text("\n".join(lines) + "\n")
    save_jsonl(recipient_examples, result_dir / "recipient_pairs_C_only.jsonl")
    save_jsonl(c_t_recipient_examples, result_dir / "recipient_pairs_C_T.jsonl")
    print(f"\n[saved] {result_dir / 'eval_results.txt'}")


if __name__ == "__main__":
    main()
