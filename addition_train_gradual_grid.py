
import argparse
import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from addition_das import LearnedSubspace, collect_layer_states, hidden_to_logits, run_das_intervention, sequence_cross_entropy, set_seed, slice_teacher_bank
from addition_data_load import build_all_banks, build_state_inputs, load_addition_rows
from addition_neural_network import load_gemma_model
from addition_train_t import fit_t_readout

METRIC_VERSION = "free_autoregressive_numeric_v2"


def parse_layers(value):
    layers = []
    for item in value.split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    return layers


def save_csv(rows, path):
    if len(rows) == 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_integer(text):
    numbers = re.findall(r"-?\d+", text)
    if len(numbers) == 0:
        return None
    return int(numbers[0])


def eos_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


@torch.no_grad()
def run_free_generation(model, bank, donor_states, Q, layer, tokenizer, batch_size, generation_cap):
    device = next(model.parameters()).device
    stop_ids = eos_ids(tokenizer)
    all_sequences = []

    if len(stop_ids) == 0:
        raise ValueError("Tokenizer has no EOS token")

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = donor_states[start:end].to(device)
        positions = bank["base_positions"][start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        finished = torch.zeros(len(ids), dtype=torch.bool, device=device)
        generated = [[] for _ in range(len(ids))]

        for _ in range(generation_cap):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                current = hidden[rows, positions, :].float()
                delta = ((donor.float() - current) @ Q) @ Q.T
                new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
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

            append_ids = next_ids.clone()
            for i in range(len(ids)):
                if finished[i]:
                    append_ids[i] = tokenizer.eos_token_id
                    continue
                token_id = int(next_ids[i].item())
                generated[i].append(token_id)
                if token_id in stop_ids:
                    finished[i] = True

            ids = torch.cat([ids, append_ids.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones((len(ids), 1), dtype=mask.dtype, device=device)], dim=1)
            if bool(finished.all()):
                break

        all_sequences.extend(generated)

    return all_sequences


def score_sequences(sequences, target_y, tokenizer):
    target_y = target_y.detach().cpu() if torch.is_tensor(target_y) else torch.tensor(target_y)
    correct = []
    predictions = []
    for i, token_ids in enumerate(sequences):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prediction = first_integer(text)
        predictions.append(prediction)
        correct.append(prediction == int(target_y[i].item()))
    return torch.tensor(correct, dtype=torch.bool), predictions


@torch.no_grad()
def free_y_accuracy(model, tokenizer, bank, donor_states, Q, layer, batch_size, generation_cap):
    sequences = run_free_generation(model, bank, donor_states, Q, layer, tokenizer, batch_size, generation_cap)
    correct, _ = score_sequences(sequences, bank["target_y"], tokenizer)
    return float(correct.float().mean())


def projected_swap_state(base_state, source_state, Q):
    base = base_state.reshape(base_state.shape[0], -1).to(Q.device).float()
    source = source_state.reshape(source_state.shape[0], -1).to(Q.device).float()
    swapped = base + ((source - base) @ Q) @ Q.T
    return swapped.reshape(base_state.shape)


def readout_logits(states, Q, W, b):
    flat = states.reshape(states.shape[0], -1).to(Q.device).float()
    return (flat @ Q) @ W.T + b


def build_factual_c_bank(tokenizer, pair_rows, prefix_ids):
    prompts = []
    labels = []
    for row in pair_rows:
        prompts.append(row["base_input"])
        labels.append(row["base_C"])
        prompts.append(row["source_input"])
        labels.append(row["source_C"])
    input_ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "target": torch.tensor(labels, dtype=torch.long),
    }


def fit_c_readout(model, train_bank, validation_bank, Q, layer, batch_size, seed):
    set_seed(seed)
    device = next(model.parameters()).device
    Q = Q.float().to(device)

    train_states = collect_layer_states(model, train_bank["input_ids"], train_bank["attention_mask"], train_bank["positions"], layer, batch_size).to(device)
    val_states = collect_layer_states(model, validation_bank["input_ids"], validation_bank["attention_mask"], validation_bank["positions"], layer, batch_size).to(device)
    x_train = (train_states @ Q).detach()
    x_val = (val_states @ Q).detach()
    y_train = train_bank["target"].to(device)
    y_val = validation_bank["target"].to(device)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    best_accuracy = -1.0
    best = None

    for penalty in [1e-4, 1e-3, 1e-2]:
        readout = torch.nn.Linear(Q.shape[1], 2).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)

        for _ in range(200):
            optimizer.zero_grad()
            logits = readout(x_train)
            loss = F.cross_entropy(logits, y_train) + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = float((readout(x_val).argmax(dim=-1) == y_val).float().mean())

            if accuracy > best_accuracy:
                weights = readout.weight.detach() / std.unsqueeze(0)
                bias = readout.bias.detach() - weights @ mean
                best_accuracy = accuracy
                best = {
                    "W_C": weights.cpu(),
                    "b_C": bias.cpu(),
                    "layer": int(layer),
                    "basis": Q.detach().cpu(),
                    "validation_accuracy": accuracy,
                    "penalty": float(penalty),
                    "seed": int(seed),
                }

    return best


@torch.no_grad()
def clean_readout(model, bank, Q, layer, W, b, batch_size):
    states = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["positions"], layer, batch_size)
    prediction = readout_logits(states, Q, W, b).argmax(dim=-1).detach().cpu()
    target = bank["target"].detach().cpu()
    base_idx = torch.arange(0, len(target), 2)
    source_idx = torch.arange(1, len(target), 2)
    base_acc = float((prediction[base_idx] == target[base_idx]).float().mean())
    source_acc = float((prediction[source_idx] == target[source_idx]).float().mean())
    return base_acc, source_acc


@torch.no_grad()
def capture_t_after_c(model, bank, source_c, Q_c, c_layer, t_layer, batch_size):
    device = next(model.parameters()).device
    all_t = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_c[start:end].to(device)
        positions = bank["base_positions"][start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        captured = {}

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current = hidden[rows, positions, :].float()
            delta = ((donor.float() - current) @ Q_c) @ Q_c.T
            new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        def t_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["t"] = hidden[rows, positions, :].detach().float().cpu()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)
        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            c_handle.remove()
            t_handle.remove()

        all_t.append(captured["t"])

    return torch.cat(all_t, dim=0)


@torch.no_grad()
def run_c_restoration(model, bank, source_c, Q_c, c_layer, clean_base_t, Q_t, t_layer, tokenizer, batch_size, generation_cap):
    device = next(model.parameters()).device
    stop_ids = eos_ids(tokenizer)
    all_sequences = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        c_donor = source_c[start:end].to(device)
        clean_t = clean_base_t[start:end].to(device)
        positions = bank["base_positions"][start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        finished = torch.zeros(len(ids), dtype=torch.bool, device=device)
        generated = [[] for _ in range(len(ids))]

        for _ in range(generation_cap):
            def c_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                current = hidden[rows, positions, :].float()
                delta = ((c_donor.float() - current) @ Q_c) @ Q_c.T
                new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden

            def t_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                current = hidden[rows, positions, :].float()
                delta = ((clean_t.float() - current) @ Q_t) @ Q_t.T
                new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden

            c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
            t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)
            try:
                position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
                next_logits = hidden_to_logits(model, outputs.last_hidden_state[:, -1, :])
                next_ids = next_logits.argmax(dim=-1)
            finally:
                c_handle.remove()
                t_handle.remove()

            append_ids = next_ids.clone()
            for i in range(len(ids)):
                if finished[i]:
                    append_ids[i] = tokenizer.eos_token_id
                    continue
                token_id = int(next_ids[i].item())
                generated[i].append(token_id)
                if token_id in stop_ids:
                    finished[i] = True

            ids = torch.cat([ids, append_ids.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones((len(ids), 1), dtype=mask.dtype, device=device)], dim=1)
            if bool(finished.all()):
                break

        all_sequences.extend(generated)

    return all_sequences


@torch.no_grad()
def eval_t_full(model, tokenizer, bank, Q_t, t_layer, W_t, b_t, batch_size, generation_cap):
    base_t = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], t_layer, batch_size)
    source_t = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], t_layer, batch_size)
    swapped_t = projected_swap_state(base_t, source_t, Q_t)
    t_prediction = readout_logits(swapped_t, Q_t, W_t, b_t).argmax(dim=-1).detach().cpu()
    t_ok = t_prediction == bank["target_t"].detach().cpu()

    sequences = run_free_generation(model, bank, source_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    y_ok, _ = score_sequences(sequences, bank["target_y"], tokenizer)

    return {
        "Joint_T_Y_after_T": float((t_ok & y_ok).float().mean()),
        "T_after_T": float(t_ok.float().mean()),
        "Y_after_T": float(y_ok.float().mean()),
    }


@torch.no_grad()
def eval_c_full(model, tokenizer, bank, pair_rows, Q_c, c_layer, W_c, b_c, Q_t, t_layer, W_t, b_t, batch_size, generation_cap):
    base_c = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], c_layer, batch_size)
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], c_layer, batch_size)
    swapped_c = projected_swap_state(base_c, source_c, Q_c)
    c_prediction = readout_logits(swapped_c, Q_c, W_c, b_c).argmax(dim=-1).detach().cpu()
    target_c = torch.tensor([row["C_after_C_intervene"] for row in pair_rows], dtype=torch.long)
    c_ok = c_prediction == target_c

    generated_t = capture_t_after_c(model, bank, source_c, Q_c, c_layer, t_layer, batch_size)
    t_prediction = readout_logits(generated_t, Q_t, W_t, b_t).argmax(dim=-1).detach().cpu()
    t_ok = t_prediction == bank["target_t"].detach().cpu()

    direct_sequences = run_free_generation(model, bank, source_c, Q_c, c_layer, tokenizer, batch_size, generation_cap)
    y_ok, _ = score_sequences(direct_sequences, bank["target_y"], tokenizer)

    replay_sequences = run_free_generation(model, bank, generated_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    replay_ok, _ = score_sequences(replay_sequences, bank["target_y"], tokenizer)

    clean_base_t = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], t_layer, batch_size)
    restoration_sequences = run_c_restoration(model, bank, source_c, Q_c, c_layer, clean_base_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    base_y = torch.tensor([row["base_Y"] for row in pair_rows], dtype=torch.long)
    restoration_ok, _ = score_sequences(restoration_sequences, base_y, tokenizer)

    return {
        "Joint_C_T_Y_after_C": float((c_ok & t_ok & y_ok).float().mean()),
        "C_after_C": float(c_ok.float().mean()),
        "T_after_C": float(t_ok.float().mean()),
        "Y_after_C": float(y_ok.float().mean()),
        "Y_via_T_replay": float(replay_ok.float().mean()),
        "Y_restored_via_T": float(restoration_ok.float().mean()),
    }

def train_t_handle(model, tokenizer, train_bank, validation_bank, layer, subspace_dim, epochs, lr, train_batch_size, eval_batch_size, generation_cap, seed, save_path):
    set_seed(seed)
    device = next(model.parameters()).device
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_positions"], layer, eval_batch_size)
    validation_source = collect_layer_states(model, validation_bank["source_input_ids"], validation_bank["source_attention_mask"], validation_bank["source_positions"], layer, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    best_validation_iia = -1.0
    best_basis = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["teacher_input_ids"]))
        total_loss = 0.0

        for start in range(0, len(permutation), train_batch_size):
            indices = permutation[start:start + train_batch_size]
            mini_bank = slice_teacher_bank(train_bank, indices)
            optimizer.zero_grad()
            Q = alignment.basis()
            logits = run_das_intervention(model, mini_bank, train_source[indices], Q, layer, len(indices), True)
            loss = sequence_cross_entropy(logits, mini_bank["target_answer_ids"], mini_bank["target_answer_mask"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        validation_iia = free_y_accuracy(model, tokenizer, validation_bank, validation_source, alignment.basis(), layer, eval_batch_size, generation_cap)
        print(f"[T layer {layer:02d} epoch {epoch:02d}] loss={total_loss / len(permutation):.4f} validation_free_Y={validation_iia:.4f}")

        if validation_iia > best_validation_iia:
            best_validation_iia = validation_iia
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch

    result = {
        "variable": "T",
        "metric_version": METRIC_VERSION,
        "layer": int(layer),
        "basis": best_basis,
        "token_position": "answer_start",
        "subspace_dim": int(subspace_dim),
        "best_epoch": int(best_epoch),
        "validation_iia": float(best_validation_iia),
        "seed": int(seed),
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, save_path)
    return result


from addition_train_c import run_c_intervention_and_capture, run_frozen_t_mediator


@torch.no_grad()
def evaluate_c_validation_free(model, tokenizer, bank, source_c, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size, generation_cap):
    generated_t = capture_t_after_c(model, bank, source_c, Q_c, c_layer, t_layer, batch_size)

    direct_sequences = run_free_generation(model, bank, source_c, Q_c, c_layer, tokenizer, batch_size, generation_cap)
    direct_ok, _ = score_sequences(direct_sequences, bank["target_y"], tokenizer)

    replay_sequences = run_free_generation(model, bank, generated_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    replay_ok, _ = score_sequences(replay_sequences, bank["target_y"], tokenizer)

    t_logits = readout_logits(generated_t, Q_t, W_t, b_t)
    t_prediction = t_logits.argmax(dim=-1).detach().cpu()
    t_ok = t_prediction == bank["target_t"].detach().cpu()

    return {
        "direct_numeric_iia": float(direct_ok.float().mean()),
        "mediator_numeric_iia": float(replay_ok.float().mean()),
        "t_readout_acc": float(t_ok.float().mean()),
    }


def train_c_gradual(model, tokenizer, train_bank, validation_bank, t_checkpoint, t_readout, c_layer, subspace_dim, epochs, lr, lambda_med, lambda_readout, train_batch_size, eval_batch_size, generation_cap, seed, save_path):
    set_seed(seed)
    device = next(model.parameters()).device

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    Q_t = t_checkpoint["basis"].float().to(device)
    t_layer = int(t_checkpoint["layer"])
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    if c_layer >= t_layer:
        raise ValueError(f"C layer must be before T layer. Got C={c_layer}, T={t_layer}")

    train_source = collect_layer_states(
        model,
        train_bank["source_input_ids"],
        train_bank["source_attention_mask"],
        train_bank["source_positions"],
        c_layer,
        eval_batch_size,
    )

    validation_source = collect_layer_states(
        model,
        validation_bank["source_input_ids"],
        validation_bank["source_attention_mask"],
        validation_bank["source_positions"],
        c_layer,
        eval_batch_size,
    )

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_validation = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["teacher_input_ids"]))
        total_direct = 0.0
        total_mediator = 0.0
        total_readout = 0.0

        for start in range(0, len(permutation), train_batch_size):
            indices = permutation[start:start + train_batch_size]
            mini_bank = slice_teacher_bank(train_bank, indices)

            optimizer.zero_grad()
            Q_c = alignment.basis()

            direct_y, generated_t = run_c_intervention_and_capture(
                model,
                mini_bank,
                train_source[indices],
                Q_c,
                c_layer,
                t_layer,
                len(indices),
                True,
            )

            mediator_y = run_frozen_t_mediator(
                model,
                mini_bank,
                generated_t,
                Q_t,
                t_layer,
                len(indices),
                True,
            )

            t_logits = (generated_t @ Q_t) @ W_t.T + b_t

            direct_loss = sequence_cross_entropy(
                direct_y,
                mini_bank["target_answer_ids"],
                mini_bank["target_answer_mask"],
            )

            mediator_loss = sequence_cross_entropy(
                mediator_y,
                mini_bank["target_answer_ids"],
                mini_bank["target_answer_mask"],
            )

            readout_loss = F.cross_entropy(
                t_logits,
                mini_bank["target_t"].to(device),
            )

            loss = direct_loss
            loss = loss + lambda_med * mediator_loss
            loss = loss + lambda_readout * readout_loss
            loss.backward()
            optimizer.step()

            n = len(indices)
            total_direct += direct_loss.item() * n
            total_mediator += mediator_loss.item() * n
            total_readout += readout_loss.item() * n

        validation = evaluate_c_validation_free(
            model,
            tokenizer,
            validation_bank,
            validation_source,
            alignment.basis(),
            c_layer,
            Q_t,
            t_layer,
            W_t,
            b_t,
            eval_batch_size,
            generation_cap,
        )

        score = min(
            validation["direct_numeric_iia"],
            validation["mediator_numeric_iia"],
            validation["t_readout_acc"],
        )

        n = len(permutation)
        print(
            f"[C{c_layer:02d}->T{t_layer:02d} epoch {epoch:02d}] "
            f"direct_loss={total_direct / n:.4f} "
            f"med_loss={total_mediator / n:.4f} "
            f"readout_loss={total_readout / n:.4f} | "
            f"direct_free_Y={validation['direct_numeric_iia']:.4f} "
            f"replay_free_Y={validation['mediator_numeric_iia']:.4f} "
            f"T_readout={validation['t_readout_acc']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_validation = dict(validation)

    result = {
        "variable": "C",
        "method": "gradual",
        "metric_version": METRIC_VERSION,
        "layer": int(c_layer),
        "basis": best_basis,
        "token_position": "answer_start",
        "subspace_dim": int(subspace_dim),
        "best_epoch": int(best_epoch),
        "lambda_med": float(lambda_med),
        "lambda_readout": float(lambda_readout),
        "T_layer": int(t_layer),
        "validation_score": float(best_score),
        "validation_direct_iia": float(best_validation["direct_numeric_iia"]),
        "validation_T_mediator_iia": float(best_validation["mediator_numeric_iia"]),
        "validation_T_readout_acc": float(best_validation["t_readout_acc"]),
        "seed": int(seed),
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, save_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-layers", default="15,16,17")
    parser.add_argument("--t-layers", default="18,19,20")
    parser.add_argument("--result-dir", default="results_addition_gradual_grid")
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--t-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=2.0)
    parser.add_argument("--lambda-readout", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    c_layers = parse_layers(args.c_layers)
    t_layers = parse_layers(args.t_layers)

    for c_layer in c_layers:
        for t_layer in t_layers:
            if c_layer >= t_layer:
                raise ValueError(f"Invalid pair C={c_layer}, T={t_layer}")

    set_seed(args.seed)
    root = Path(args.result_dir)
    root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(
        model,
        tokenizer,
        args.data_dir,
        args.cache_dir,
        args.eval_batch_size,
        args.max_new_tokens,
        args.force_rebuild_cache,
    )

    banks = build_all_banks(tokenizer, rows)
    prefix_ids = banks["prefix_ids"]

    t_objects = {}

    for t_layer in t_layers:
        layer_dir = root / "T_handles" / f"T_layer_{t_layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = layer_dir / "das_t.pt"
        readout_path = layer_dir / "t_readout_LogisticRegression.pt"

        if args.resume and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            checkpoint = train_t_handle(
                model,
                tokenizer,
                banks["T"]["train"],
                banks["T"]["validation"],
                t_layer,
                args.t_dim,
                args.t_epochs,
                args.lr,
                args.train_batch_size,
                args.eval_batch_size,
                args.generation_cap,
                args.seed,
                ckpt_path,
            )

        if args.resume and readout_path.exists():
            readout = torch.load(readout_path, map_location="cpu", weights_only=False)
        else:
            readout = fit_t_readout(
                model,
                banks["T_factual"]["train"],
                banks["T_factual"]["validation"],
                checkpoint["basis"],
                t_layer,
                args.eval_batch_size,
                args.seed,
                readout_path,
            )

        t_objects[t_layer] = {"checkpoint": checkpoint, "readout": readout}

    t_factual_test = {
        "input_ids": banks["T_factual"]["test"]["input_ids"],
        "attention_mask": banks["T_factual"]["test"]["attention_mask"],
        "positions": banks["T_factual"]["test"]["positions"],
        "target": banks["T_factual"]["test"]["target_t"],
    }

    c_factual_train = build_factual_c_bank(tokenizer, rows["train"]["C_only"], prefix_ids)
    c_factual_val = build_factual_c_bank(tokenizer, rows["validation"]["C_only"], prefix_ids)
    c_factual_test = build_factual_c_bank(tokenizer, rows["test"]["C_only"], prefix_ids)

    output_rows = []
    csv_path = root / "pair_grid.csv"

    for t_layer in t_layers:
        t_checkpoint = t_objects[t_layer]["checkpoint"]
        t_readout = t_objects[t_layer]["readout"]

        for c_layer in c_layers:
            pair_dir = root / "pairs" / f"C_{c_layer:02d}_T_{t_layer:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            c_ckpt_path = pair_dir / "das_c.pt"
            c_readout_path = pair_dir / "c_readout_LogisticRegression.pt"

            if args.resume and c_ckpt_path.exists():
                c_checkpoint = torch.load(c_ckpt_path, map_location="cpu", weights_only=False)
                if int(c_checkpoint["T_layer"]) != t_layer:
                    raise RuntimeError(f"{c_ckpt_path} was trained with T={c_checkpoint['T_layer']}, expected T={t_layer}")
            else:
                c_checkpoint = train_c_gradual(
                    model,
                    tokenizer,
                    banks["C"]["train"],
                    banks["C"]["validation"],
                    t_checkpoint,
                    t_readout,
                    c_layer,
                    args.c_dim,
                    args.c_epochs,
                    args.lr,
                    args.lambda_med,
                    args.lambda_readout,
                    args.train_batch_size,
                    args.eval_batch_size,
                    args.generation_cap,
                    args.seed,
                    c_ckpt_path,
                )

            device = next(model.parameters()).device
            Q_c = c_checkpoint["basis"].float().to(device)
            Q_t = t_checkpoint["basis"].float().to(device)

            if args.resume and c_readout_path.exists():
                c_readout = torch.load(c_readout_path, map_location="cpu", weights_only=False)
            else:
                c_readout = fit_c_readout(
                    model,
                    c_factual_train,
                    c_factual_val,
                    Q_c,
                    c_layer,
                    args.eval_batch_size,
                    args.seed,
                )
                torch.save(c_readout, c_readout_path)

            W_c = c_readout["W_C"].float().to(device)
            b_c = c_readout["b_C"].float().to(device)
            W_t = t_readout["W_T"].float().to(device)
            b_t = t_readout["b_T"].float().to(device)

            c_base, c_source = clean_readout(model, c_factual_test, Q_c, c_layer, W_c, b_c, args.eval_batch_size)
            t_base, t_source = clean_readout(model, t_factual_test, Q_t, t_layer, W_t, b_t, args.eval_batch_size)

            t_metrics = eval_t_full(
                model,
                tokenizer,
                banks["T"]["test"],
                Q_t,
                t_layer,
                W_t,
                b_t,
                args.eval_batch_size,
                args.generation_cap,
            )

            c_only = eval_c_full(
                model,
                tokenizer,
                banks["C"]["test"],
                rows["test"]["C_only"],
                Q_c,
                c_layer,
                W_c,
                b_c,
                Q_t,
                t_layer,
                W_t,
                b_t,
                args.eval_batch_size,
                args.generation_cap,
            )

            c_t = eval_c_full(
                model,
                tokenizer,
                banks["C_T"]["test"],
                rows["test"]["C_T"],
                Q_c,
                c_layer,
                W_c,
                b_c,
                Q_t,
                t_layer,
                W_t,
                b_t,
                args.eval_batch_size,
                args.generation_cap,
            )

            row = {
                "method": "gradual",
                "C_layer": c_layer,
                "T_layer": t_layer,
                "C_validation_score": c_checkpoint["validation_score"],
                "C_validation_direct_iia": c_checkpoint["validation_direct_iia"],
                "C_validation_T_mediator_iia": c_checkpoint["validation_T_mediator_iia"],
                "C_validation_T_readout_acc": c_checkpoint["validation_T_readout_acc"],
                "T_validation_iia": t_checkpoint["validation_iia"],
                "clean_C_base": c_base,
                "clean_C_source": c_source,
                "clean_T_base": t_base,
                "clean_T_source": t_source,
                "T_only_Joint_T_Y_after_T": t_metrics["Joint_T_Y_after_T"],
                "T_only_T_after_T": t_metrics["T_after_T"],
                "T_only_Y_after_T": t_metrics["Y_after_T"],
                "C_only_Joint_C_T_Y_after_C": c_only["Joint_C_T_Y_after_C"],
                "C_only_C_after_C": c_only["C_after_C"],
                "C_only_T_after_C": c_only["T_after_C"],
                "C_only_Y_after_C": c_only["Y_after_C"],
                "C_only_Y_via_T_replay": c_only["Y_via_T_replay"],
                "C_only_Y_restored_via_T": c_only["Y_restored_via_T"],
                "C_T_Joint_C_T_Y_after_C": c_t["Joint_C_T_Y_after_C"],
                "C_T_C_after_C": c_t["C_after_C"],
                "C_T_T_after_C": c_t["T_after_C"],
                "C_T_Y_after_C": c_t["Y_after_C"],
                "C_T_Y_via_T_replay": c_t["Y_via_T_replay"],
                "C_T_Y_restored_via_T": c_t["Y_restored_via_T"],
            }

            output_rows.append(row)
            save_csv(output_rows, csv_path)

            payload = {
                "method": "gradual",
                "C_layer": c_layer,
                "T_layer": t_layer,
                "lambda_med": args.lambda_med,
                "lambda_readout": args.lambda_readout,
                "generation_cap": args.generation_cap,
                "metrics": {
                    "T_only": t_metrics,
                    "C_only": c_only,
                    "C_T": c_t,
                },
            }

            (pair_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")

            print(
                f"[GRADUAL C{c_layer}->T{t_layer}] "
                f"C_only T={c_only['T_after_C']:.4f} "
                f"Y={c_only['Y_after_C']:.4f} "
                f"replay={c_only['Y_via_T_replay']:.4f} "
                f"restore={c_only['Y_restored_via_T']:.4f} | "
                f"C_T T={c_t['T_after_C']:.4f} "
                f"Y={c_t['Y_after_C']:.4f}"
            )

    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
