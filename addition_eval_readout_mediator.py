"""Evaluate C -> T -> Y on controlled two-digit addition.

Mirrors mcqa_eval_readout_mediator.py and adds an other-recipient replay test.

Metrics:
1. Clean T readout on natural T-only base/source states.
2. T-only direct DAS IIA.
3. C-only and C_T:
   - Y_after_C direct IIA
   - T_after_C readout accuracy
   - Y_after_C via frozen T replay
4. Other-recipient replay:
   - replay C-induced T state into a different addition problem
   - compare against the recipient output implied by transferred T
   - clean natural T donor control on the same recipients
   - report exact-Y and first numeric token accuracy

Run:
    python3 addition_eval_readout_mediator.py --result-dir results_addition_standard
    python3 addition_eval_readout_mediator.py --result-dir results_addition_gradual
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from addition_gradual_das_c import run_c_intervention_and_capture, run_frozen_t_mediator
from addition_gradual_das_t import build_factual_t_bank, build_pair_bank, build_state_inputs, build_teacher_forcing_batch, collect_layer_states, get_answer_prefix_ids, load_gemma_model, read_jsonl, run_das_intervention, sequence_iia, set_seed


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    Q = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    return Q, layer, checkpoint


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    W = checkpoint[f"W_{name}"].float().to(device)
    b = checkpoint[f"b_{name}"].float().to(device)
    return W, b, checkpoint


def accuracy(logits, target):
    prediction = logits.argmax(-1).detach().cpu()
    target = target.detach().cpu()
    return float((prediction == target).float().mean())


def first_token_accuracy(logits, target_ids):
    prediction = logits[:, 0, :].argmax(-1).detach().cpu()
    target = target_ids[:, 0].detach().cpu()
    return float((prediction == target).float().mean())


def parse_problem(prompt):
    numbers = re.findall(r"\d+", prompt)

    if len(numbers) < 2:
        raise ValueError(f"Could not parse operands from prompt: {prompt}")

    return int(numbers[0]), int(numbers[1])


def clean_t_readout(model, tokenizer, rows, prefix_ids, Q_t, t_layer, W_t, b_t, batch_size):
    bank = build_factual_t_bank(tokenizer, rows, prefix_ids)
    states = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["positions"], t_layer, batch_size)
    logits = (states.to(Q_t.device) @ Q_t) @ W_t.T + b_t

    base_indices = torch.arange(0, len(rows) * 2, 2)
    source_indices = torch.arange(1, len(rows) * 2, 2)

    base_acc = accuracy(logits[base_indices.to(logits.device)], bank["target_t"][base_indices])
    source_acc = accuracy(logits[source_indices.to(logits.device)], bank["target_t"][source_indices])

    return base_acc, source_acc


@torch.no_grad()
def eval_t(model, tokenizer, rows, prefix_ids, Q_t, t_layer, batch_size):
    bank = build_pair_bank(tokenizer, rows, "T", prefix_ids)
    source_t = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], t_layer, batch_size)
    logits = run_das_intervention(model, bank, source_t, Q_t, t_layer, batch_size, False)

    return {
        "Y_after_T": sequence_iia(logits, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_T_first_token": first_token_accuracy(logits, bank["target_answer_ids"]),
    }


@torch.no_grad()
def eval_c(model, tokenizer, rows, prefix_ids, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size):
    bank = build_pair_bank(tokenizer, rows, "C", prefix_ids)

    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], c_layer, batch_size)

    direct_y, generated_t = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size, False)
    replay_y = run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size, False)

    t_logits = (generated_t.to(Q_t.device) @ Q_t) @ W_t.T + b_t

    results = {
        "Y_after_C": sequence_iia(direct_y, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_after_C_first_token": first_token_accuracy(direct_y, bank["target_answer_ids"]),
        "T_after_C": accuracy(t_logits, bank["target_t"]),
        "Y_via_T_replay": sequence_iia(replay_y, bank["target_answer_ids"], bank["target_answer_mask"]),
        "Y_via_T_replay_first_token": first_token_accuracy(replay_y, bank["target_answer_ids"]),
    }

    return results, generated_t


def build_recipient_examples(c_rows, seed):
    rng = random.Random(seed)
    indices = list(range(len(c_rows)))
    rng.shuffle(indices)

    recipients = []
    used_recipient_indices = set()

    for donor_index, donor_row in enumerate(c_rows):
        transferred_t = donor_row["T_after_C_intervene"]
        donor_ones = donor_row["base_ones_out"]

        chosen_index = None

        for candidate_index in indices:
            if candidate_index == donor_index:
                continue

            if candidate_index in used_recipient_indices:
                continue

            candidate = c_rows[candidate_index]

            if candidate["base_input"] == donor_row["base_input"]:
                continue

            if candidate["base_T"] == transferred_t:
                continue

            if candidate["base_ones_out"] == donor_ones:
                continue

            chosen_index = candidate_index
            break

        if chosen_index is None:
            for candidate_index in indices:
                if candidate_index == donor_index:
                    continue

                candidate = c_rows[candidate_index]

                if candidate["base_input"] == donor_row["base_input"]:
                    continue

                if candidate["base_T"] == transferred_t:
                    continue

                chosen_index = candidate_index
                break

        if chosen_index is None:
            raise RuntimeError(f"Could not find recipient for donor row {donor_index}")

        used_recipient_indices.add(chosen_index)
        recipient_row = c_rows[chosen_index]

        recipient_target_y = 100 * recipient_row["base_hundreds_out"] + 10 * transferred_t + recipient_row["base_ones_out"]

        recipients.append({
            "donor_index": donor_index,
            "recipient_index": chosen_index,
            "donor_base_input": donor_row["base_input"],
            "donor_source_input": donor_row["source_input"],
            "transferred_T": transferred_t,
            "donor_base_T": donor_row["base_T"],
            "donor_source_T": donor_row["source_T"],
            "recipient_input": recipient_row["base_input"],
            "recipient_natural_T": recipient_row["base_T"],
            "recipient_natural_Y": recipient_row["base_Y"],
            "recipient_ones_out": recipient_row["base_ones_out"],
            "recipient_hundreds_out": recipient_row["base_hundreds_out"],
            "recipient_target_Y": recipient_target_y,
        })

    return recipients


def build_clean_t_donor_pool(t_rows):
    pool = {}

    for row in t_rows:
        candidates = [
            (row["base_input"], row["base_T"]),
            (row["source_input"], row["source_T"]),
        ]

        for prompt, T in candidates:
            if T not in pool:
                pool[T] = []

            if prompt not in pool[T]:
                pool[T].append(prompt)

    return pool


def choose_clean_t_donors(recipient_examples, donor_pool):
    prompts = []

    for example in recipient_examples:
        transferred_t = example["transferred_T"]

        if transferred_t not in donor_pool:
            raise RuntimeError(f"No clean natural T donor for T={transferred_t}")

        chosen_prompt = None

        for prompt in donor_pool[transferred_t]:
            if prompt != example["recipient_input"]:
                chosen_prompt = prompt
                break

        if chosen_prompt is None:
            raise RuntimeError(f"No distinct clean T donor for T={transferred_t}")

        prompts.append(chosen_prompt)

    return prompts


def build_recipient_bank(tokenizer, recipient_examples, prefix_ids):
    prompts = []
    target_y = []

    for example in recipient_examples:
        prompts.append(example["recipient_input"])
        target_y.append(example["recipient_target_Y"])

    return build_teacher_forcing_batch(tokenizer, prompts, target_y, prefix_ids)


@torch.no_grad()
def eval_recipients(model, tokenizer, c_rows, t_rows, prefix_ids, generated_t, Q_t, t_layer, batch_size, seed):
    recipient_examples = build_recipient_examples(c_rows, seed)
    recipient_bank = build_recipient_bank(tokenizer, recipient_examples, prefix_ids)

    if len(recipient_examples) != len(generated_t):
        raise RuntimeError("Recipient count does not match generated T state count.")

    induced_replay_y = run_frozen_t_mediator(model, recipient_bank, generated_t, Q_t, t_layer, batch_size, False)

    donor_pool = build_clean_t_donor_pool(t_rows)
    clean_donor_prompts = choose_clean_t_donors(recipient_examples, donor_pool)
    clean_input_ids, clean_attention_mask, clean_positions = build_state_inputs(tokenizer, clean_donor_prompts, prefix_ids)
    clean_t_states = collect_layer_states(model, clean_input_ids, clean_attention_mask, clean_positions, t_layer, batch_size)

    clean_replay_y = run_frozen_t_mediator(model, recipient_bank, clean_t_states, Q_t, t_layer, batch_size, False)

    induced_exact = sequence_iia(induced_replay_y, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"])
    induced_first = first_token_accuracy(induced_replay_y, recipient_bank["target_answer_ids"])

    clean_exact = sequence_iia(clean_replay_y, recipient_bank["target_answer_ids"], recipient_bank["target_answer_mask"])
    clean_first = first_token_accuracy(clean_replay_y, recipient_bank["target_answer_ids"])

    return {
        "induced_T_recipient_Y": induced_exact,
        "induced_T_recipient_first_token": induced_first,
        "clean_T_recipient_Y": clean_exact,
        "clean_T_recipient_first_token": clean_first,
        "num_recipients": len(recipient_examples),
    }, recipient_examples


def save_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_filtered_pairs")
    parser.add_argument("--result-dir", default="results_addition_joint_gradual_all2.0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    model, tokenizer = load_gemma_model(args.device)
    device = next(model.parameters()).device
    prefix_ids = get_answer_prefix_ids(tokenizer)

    Q_t, t_layer, t_info = load_ckpt(result_dir / "das_t.pt", device)
    Q_c, c_layer, c_info = load_ckpt(result_dir / "das_c.pt", device)
    W_t, b_t, _ = load_readout(result_dir / "t_readout_LogisticRegression.pt", "T", device)

    t_rows = read_jsonl(data_dir / "test_T_only.jsonl")
    c_only_rows = read_jsonl(data_dir / "test_C_only.jsonl")
    c_t_rows = read_jsonl(data_dir / "test_C_T.jsonl")

    print("\n=== CLEAN T READOUT ===")
    t_base, t_source = clean_t_readout(model, tokenizer, t_rows, prefix_ids, Q_t, t_layer, W_t, b_t, args.batch_size)
    print(f"T base/source: {t_base:.4f} / {t_source:.4f}")

    print("\n=== T -> Y ===")
    t_results = eval_t(model, tokenizer, t_rows, prefix_ids, Q_t, t_layer, args.batch_size)
    print(" | ".join(f"{key}: {value:.4f}" for key, value in t_results.items()))

    print("\n=== C -> T -> Y [C_only] ===")
    c_only_results, c_only_generated_t = eval_c(model, tokenizer, c_only_rows, prefix_ids, Q_c, c_layer, Q_t, t_layer, W_t, b_t, args.batch_size)
    print(" | ".join(f"{key}: {value:.4f}" for key, value in c_only_results.items()))

    print("\n=== C -> T -> Y [C_T] ===")
    c_t_results, _ = eval_c(model, tokenizer, c_t_rows, prefix_ids, Q_c, c_layer, Q_t, t_layer, W_t, b_t, args.batch_size)
    print(" | ".join(f"{key}: {value:.4f}" for key, value in c_t_results.items()))

    print("\n=== OTHER RECIPIENTS [C_only induced T] ===")
    recipient_results, recipient_examples = eval_recipients(model, tokenizer, c_only_rows, t_rows, prefix_ids, c_only_generated_t, Q_t, t_layer, args.batch_size, args.seed + 100)
    print(" | ".join(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in recipient_results.items()))

    lines = []

    lines.append("=== CHECKPOINT ===")
    lines.append(f"T layer: {t_layer}")
    lines.append(f"C layer: {c_layer}")
    lines.append(f"T checkpoint test_iia: {float(t_info['test_iia']):.4f}")

    lines.append("\n=== CLEAN T READOUT ===")
    lines.append(f"T base/source: {t_base:.4f} / {t_source:.4f}")

    lines.append("\n=== T -> Y ===")
    lines.append(" | ".join(f"{key}: {value:.4f}" for key, value in t_results.items()))

    lines.append("\n=== C -> T -> Y [C_only] ===")
    lines.append(" | ".join(f"{key}: {value:.4f}" for key, value in c_only_results.items()))

    lines.append("\n=== C -> T -> Y [C_T] ===")
    lines.append(" | ".join(f"{key}: {value:.4f}" for key, value in c_t_results.items()))

    lines.append("\n=== OTHER RECIPIENTS [C_only induced T] ===")
    lines.append(" | ".join(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in recipient_results.items()))

    text = "\n".join(lines)

    output_path = result_dir / "eval_results.txt"
    output_path.write_text(text + "\n")

    recipient_path = result_dir / "recipient_pairs.jsonl"
    save_jsonl(recipient_examples, recipient_path)

    print(f"\nsaved results = {output_path}")
    print(f"saved recipient pairs = {recipient_path}")


if __name__ == "__main__":
    main()
