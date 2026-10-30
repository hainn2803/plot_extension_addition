"""Train T DAS on two-digit addition.

The model naturally emits a shared leading-space token before the numeric answer.
We treat that shared token as a fixed output prefix and localize T at the
answer-start state after that prefix, so the intervened state directly predicts
the first numeric answer token.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from addition_constants import HF_TOKEN, MODEL_NAME

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN


def load_gemma_model(device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR)
    model.eval()
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def parse_layers(value):
    if value == "all":
        return None

    layers = []

    for layer in value.split(","):
        layers.append(int(layer))

    return layers


def longest_common_prefix(sequences):
    if len(sequences) == 0:
        return []

    prefix = list(sequences[0])

    for sequence in sequences[1:]:
        length = min(len(prefix), len(sequence))
        new_length = 0

        for index in range(length):
            if prefix[index] != sequence[index]:
                break
            new_length += 1

        prefix = prefix[:new_length]

        if len(prefix) == 0:
            break

    return prefix


def get_answer_prefix_ids(tokenizer):
    sequences = []

    for value in range(10, 100):
        ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]
        sequences.append(ids)

    prefix_ids = longest_common_prefix(sequences)

    if len(prefix_ids) == 0:
        raise ValueError("No shared answer prefix token found for two-digit answers.")

    print("[answer prefix] ids =", prefix_ids)
    print("[answer prefix] text =", repr(tokenizer.decode(prefix_ids)))
    return prefix_ids


def answer_content_ids(tokenizer, value, prefix_ids):
    ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]

    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f"Answer {value} does not begin with the shared prefix.")

    content_ids = ids[len(prefix_ids):]

    if len(content_ids) == 0:
        raise ValueError(f"Answer {value} has no numeric tokens after the shared prefix.")

    return content_ids


def pad_sequences(sequences, tokenizer):
    max_length = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    last_positions = torch.zeros(len(sequences), dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        pad = max_length - len(sequence)
        input_ids[row_index, pad:] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, pad:] = 1
        last_positions[row_index] = max_length - 1

    return input_ids, attention_mask, last_positions


def build_state_inputs(tokenizer, prompts, prefix_ids):
    sequences = []

    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        sequence = prompt_ids + prefix_ids
        sequences.append(sequence)

    return pad_sequences(sequences, tokenizer)


def build_teacher_forcing_batch(tokenizer, prompts, target_values, prefix_ids):
    sequences = []
    intervention_positions = []
    label_positions = []
    target_ids = []

    for prompt, target_value in zip(prompts, target_values):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = answer_content_ids(tokenizer, target_value, prefix_ids)

        sequence = prompt_ids + prefix_ids + answer_ids[:-1]
        intervention_position = len(prompt_ids) + len(prefix_ids) - 1
        positions = []

        for answer_index in range(len(answer_ids)):
            positions.append(intervention_position + answer_index)

        sequences.append(sequence)
        intervention_positions.append(intervention_position)
        label_positions.append(positions)
        target_ids.append(answer_ids)

    max_seq_len = max(len(sequence) for sequence in sequences)
    max_answer_len = max(len(ids) for ids in target_ids)
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(sequences), max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_seq_len), dtype=torch.long)
    padded_intervention_positions = torch.zeros(len(sequences), dtype=torch.long)
    padded_label_positions = torch.zeros((len(sequences), max_answer_len), dtype=torch.long)
    padded_target_ids = torch.zeros((len(sequences), max_answer_len), dtype=torch.long)
    target_mask = torch.zeros((len(sequences), max_answer_len), dtype=torch.bool)

    for row_index, sequence in enumerate(sequences):
        pad = max_seq_len - len(sequence)

        input_ids[row_index, pad:] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, pad:] = 1
        padded_intervention_positions[row_index] = pad + intervention_positions[row_index]

        for answer_index in range(len(target_ids[row_index])):
            padded_label_positions[row_index, answer_index] = pad + label_positions[row_index][answer_index]
            padded_target_ids[row_index, answer_index] = target_ids[row_index][answer_index]
            target_mask[row_index, answer_index] = True

    return {
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attention_mask,
        "intervention_positions": padded_intervention_positions,
        "label_positions": padded_label_positions,
        "target_answer_ids": padded_target_ids,
        "target_answer_mask": target_mask,
    }


def build_pair_bank(tokenizer, rows, intervention_name, prefix_ids):
    base_texts = []
    source_texts = []
    target_y = []
    target_t = []

    for row in rows:
        base_texts.append(row["base_input"])
        source_texts.append(row["source_input"])

        if intervention_name == "T":
            target_y.append(row["Y_after_T_intervene"])
            target_t.append(row["T_after_T_intervene"])
        elif intervention_name == "C":
            target_y.append(row["Y_after_C_intervene"])
            target_t.append(row["T_after_C_intervene"])
        else:
            raise ValueError(intervention_name)

    source_input_ids, source_attention_mask, source_positions = build_state_inputs(tokenizer, source_texts, prefix_ids)
    teacher = build_teacher_forcing_batch(tokenizer, base_texts, target_y, prefix_ids)

    bank = {
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "source_positions": source_positions,
        "target_y": torch.tensor(target_y, dtype=torch.long),
        "target_t": torch.tensor(target_t, dtype=torch.long),
    }

    bank.update(teacher)
    return bank


def build_factual_t_bank(tokenizer, rows, prefix_ids):
    texts = []
    labels = []

    for row in rows:
        texts.append(row["base_input"])
        labels.append(row["base_T"])

        texts.append(row["source_input"])
        labels.append(row["source_T"])

    input_ids, attention_mask, positions = build_state_inputs(tokenizer, texts, prefix_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "target_t": torch.tensor(labels, dtype=torch.long),
    }


class LearnedSubspace(nn.Module):
    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        initial = torch.randn(hidden_size, subspace_dim, dtype=torch.float32)
        self.raw = nn.Parameter(torch.linalg.qr(initial, mode="reduced").Q)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


def sequence_logits(model, outputs, label_positions):
    hidden = outputs.last_hidden_state
    device = hidden.device
    rows = torch.arange(hidden.shape[0], device=device).unsqueeze(1)
    positions = label_positions.to(device)
    selected_hidden = hidden[rows, positions, :]
    logits = model.lm_head(selected_hidden).float()

    softcap = getattr(model.config, "final_logit_softcapping", None)

    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits


def sequence_cross_entropy(logits, target_ids, target_mask):
    device = logits.device
    targets = target_ids.to(device)
    mask = target_mask.to(device)

    return F.cross_entropy(logits[mask], targets[mask])


def sequence_iia(logits, target_ids, target_mask):
    predictions = logits.argmax(-1).detach().cpu()
    targets = target_ids.detach().cpu()
    mask = target_mask.detach().cpu()

    correct = predictions == targets
    correct = correct | (~mask)
    return float(correct.all(dim=1).float().mean())


@torch.no_grad()
def collect_layer_states(model, input_ids, attention_mask, positions, layer, batch_size):
    device = next(model.parameters()).device
    states = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        position = positions[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        captured = {}

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            captured["x"] = hidden[rows, position, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()

        states.append(captured["x"])

    return torch.cat(states, dim=0)


@torch.no_grad()
def collect_all_layer_states(model, input_ids, attention_mask, positions, layers, batch_size):
    result = {}

    for layer in layers:
        result[layer] = collect_layer_states(model, input_ids, attention_mask, positions, layer, batch_size)

    return result


def slice_teacher_bank(bank, idx):
    return {
        "teacher_input_ids": bank["teacher_input_ids"][idx],
        "teacher_attention_mask": bank["teacher_attention_mask"][idx],
        "intervention_positions": bank["intervention_positions"][idx],
        "label_positions": bank["label_positions"][idx],
        "target_answer_ids": bank["target_answer_ids"][idx],
        "target_answer_mask": bank["target_answer_mask"][idx],
    }


@torch.no_grad()
def clean_sequence_iia(model, tokenizer, prompts, target_values, prefix_ids, batch_size):
    bank = build_teacher_forcing_batch(tokenizer, prompts, target_values, prefix_ids)
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["teacher_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["teacher_input_ids"]))
        ids = bank["teacher_input_ids"][start:end].to(device)
        mask = bank["teacher_attention_mask"][start:end].to(device)
        label_positions = bank["label_positions"][start:end]

        position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
        outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        all_logits.append(sequence_logits(model, outputs, label_positions).cpu())

    logits = torch.cat(all_logits, dim=0)
    return sequence_iia(logits, bank["target_answer_ids"], bank["target_answer_mask"])


@torch.no_grad()
def run_full_layer_intervention(model, bank, source_states, layer, batch_size):
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
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            new_hidden[rows, intervention_positions, :] = donor.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            all_logits.append(sequence_logits(model, outputs, label_positions).cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


def run_das_intervention(model, bank, source_states, Q, layer, batch_size, require_grad):
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
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q) @ Q.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            logits = sequence_logits(model, outputs, label_positions)

            if require_grad:
                all_logits.append(logits)
            else:
                all_logits.append(logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def evaluate_t(model, bank, source_states, Q, layer, batch_size):
    logits = run_das_intervention(model, bank, source_states, Q, layer, batch_size, False)
    return sequence_iia(logits, bank["target_answer_ids"], bank["target_answer_mask"])


def fit_t_readout(model, checkpoint_path, train_rows, val_rows, tokenizer, prefix_ids, batch_size, save_path, seed):
    set_seed(seed)
    device = next(model.parameters()).device

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    Q = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])

    train_bank = build_factual_t_bank(tokenizer, train_rows, prefix_ids)
    val_bank = build_factual_t_bank(tokenizer, val_rows, prefix_ids)

    train_states = collect_layer_states(model, train_bank["input_ids"], train_bank["attention_mask"], train_bank["positions"], layer, batch_size).to(device)
    val_states = collect_layer_states(model, val_bank["input_ids"], val_bank["attention_mask"], val_bank["positions"], layer, batch_size).to(device)

    x_train = (train_states @ Q).detach()
    x_val = (val_states @ Q).detach()
    y_train = train_bank["target_t"].to(device)
    y_val = val_bank["target_t"].to(device)

    mean = x_train.mean(0)
    std = x_train.std(0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    best_acc = -1.0
    best = None

    for penalty in (1e-4, 1e-3, 1e-2):
        readout = torch.nn.Linear(Q.shape[1], 10).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)

        for _ in range(200):
            optimizer.zero_grad()
            logits = readout(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss = loss + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()

            prediction = readout(x_val).argmax(-1)
            acc = float((prediction == y_val).float().mean())

            if acc > best_acc:
                weights = readout.weight.detach() / std.unsqueeze(0)
                bias = readout.bias.detach() - weights @ mean
                best_acc = acc
                best = {
                    "W_T": weights.cpu(),
                    "b_T": bias.cpu(),
                    "layer": layer,
                    "basis": checkpoint["basis"],
                }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, save_path)
    print(f"[T readout] val_acc={best_acc:.4f}")


def train_t(model, tokenizer, train_rows, val_rows, test_rows, prefix_ids, layers=None, subspace_dim=128, epochs=20, lr=1e-2, train_batch_size=32, eval_batch_size=32, seed=0, save_path="results/das_t.pt"):
    set_seed(seed)
    device = next(model.parameters()).device

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_bank = build_pair_bank(tokenizer, train_rows, "T", prefix_ids)
    val_bank = build_pair_bank(tokenizer, val_rows, "T", prefix_ids)
    test_bank = build_pair_bank(tokenizer, test_rows, "T", prefix_ids)

    val_source_prompts = []
    val_source_targets = []

    for row in val_rows:
        val_source_prompts.append(row["source_input"])
        val_source_targets.append(row["source_Y"])

    clean_iia = clean_sequence_iia(model, tokenizer, val_source_prompts, val_source_targets, prefix_ids, eval_batch_size)
    print(f"[sanity] clean source exact answer accuracy={clean_iia:.4f}")

    if clean_iia < 0.95:
        raise RuntimeError("Clean teacher-forced answer accuracy is too low. Stop before DAS training.")

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    val_source_by_layer = collect_all_layer_states(model, val_bank["source_input_ids"], val_bank["source_attention_mask"], val_bank["source_positions"], layers, eval_batch_size)

    layer_scores = []

    for layer in layers:
        logits = run_full_layer_intervention(model, val_bank, val_source_by_layer[layer], layer, eval_batch_size)
        iia = sequence_iia(logits, val_bank["target_answer_ids"], val_bank["target_answer_mask"])
        layer_scores.append((layer, iia))
        print(f"[T layer {layer}] full_iia={iia:.4f}")

    layer = max(layer_scores, key=lambda x: x[1])[0]
    print(f"[T] selected layer={layer}")

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_positions"], layer, eval_batch_size)
    val_source = val_source_by_layer[layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_positions"], layer, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_iia = -1.0
    best_basis = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["teacher_input_ids"]))
        total_loss = 0.0

        for start in range(0, len(permutation), train_batch_size):
            idx = permutation[start:start + train_batch_size]
            mini = slice_teacher_bank(train_bank, idx)

            optimizer.zero_grad()
            Q = alignment.basis()

            logits = run_das_intervention(model, mini, train_source[idx], Q, layer, len(idx), True)
            loss = sequence_cross_entropy(logits, mini["target_answer_ids"], mini["target_answer_mask"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        val_iia = evaluate_t(model, val_bank, val_source, alignment.basis(), layer, eval_batch_size)
        print(f"[T epoch {epoch:02d}] loss={total_loss / len(permutation):.4f} val_iia={val_iia:.4f}")

        if val_iia > best_iia:
            best_iia = val_iia
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch

    test_iia = evaluate_t(model, test_bank, test_source, best_basis.to(device), layer, eval_batch_size)

    result = {
        "variable": "T",
        "layer": layer,
        "basis": best_basis,
        "token_position": "answer_start",
        "answer_prefix_ids": prefix_ids,
        "subspace_dim": subspace_dim,
        "val_iia": best_iia,
        "test_iia": test_iia,
        "best_epoch": best_epoch,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, save_path)

    print(f"[T BEST] layer={layer} epoch={best_epoch} val={best_iia:.4f} test={test_iia:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_filtered_pairs")
    parser.add_argument("--result-dir", default="results_addition_t")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    train_rows = read_jsonl(Path(args.data_dir) / "train_T_only.jsonl")
    val_rows = read_jsonl(Path(args.data_dir) / "val_T_only.jsonl")
    test_rows = read_jsonl(Path(args.data_dir) / "test_T_only.jsonl")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    prefix_ids = get_answer_prefix_ids(tokenizer)

    checkpoint_path = result_dir / "das_t.pt"
    readout_path = result_dir / "t_readout_LogisticRegression.pt"

    train_t(model, tokenizer, train_rows, val_rows, test_rows, prefix_ids, parse_layers(args.layers), args.dim, args.epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.seed, str(checkpoint_path))
    fit_t_readout(model, checkpoint_path, train_rows, val_rows, tokenizer, prefix_ids, args.eval_batch_size, str(readout_path), args.seed)


if __name__ == "__main__":
    main()
