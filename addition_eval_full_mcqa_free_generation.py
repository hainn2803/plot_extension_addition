"""MCQA-style evaluator for addition C -> T -> Y.

All Y metrics use free autoregressive generation.
Ground-truth answer length is NEVER used to stop generation.
Ground-truth answer tokens are NEVER fed during evaluation.
Generation stops only when the model emits EOS or --generation-cap is reached.
The first integer decoded from the full completion is compared with the target Y.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from addition_das import collect_layer_states, hidden_to_logits, set_seed
from addition_data_load import build_all_banks, build_state_inputs, load_addition_rows
from addition_neural_network import load_gemma_model


def projected_swap(base_state, donor_state, Q):
    base = base_state.reshape(base_state.shape[0], -1).to(Q.device).float()
    donor = donor_state.reshape(donor_state.shape[0], -1).to(Q.device).float()
    out = base + ((donor - base) @ Q) @ Q.T
    return out.reshape(base_state.shape)


def readout_logits(states, Q, W, b):
    x = states.reshape(states.shape[0], -1).to(Q.device).float()
    return (x @ Q) @ W.T + b


def first_integer(text):
    values = re.findall(r"-?\d+", text)
    return None if len(values) == 0 else int(values[0])


def score_sequences(sequences, targets, tokenizer):
    correct = []
    predictions = []
    texts = []

    for i, token_ids in enumerate(sequences):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prediction = first_integer(text)
        target = int(targets[i].item())
        texts.append(text)
        predictions.append(prediction)
        correct.append(prediction == target)

    return torch.tensor(correct, dtype=torch.bool), predictions, texts


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def eos_id_set(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


@torch.no_grad()
def free_generate_single_intervention(model, bank, donor_states, Q, layer, tokenizer, batch_size, generation_cap):
    device = next(model.parameters()).device
    eos_ids = eos_id_set(tokenizer)
    eos_fill = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else list(eos_ids)[0]
    all_sequences = []

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
                return (new_hidden,) + output[1:] if isinstance(output, tuple) else new_hidden

            handle = model.model.layers[layer].register_forward_hook(hook)
            try:
                position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
                next_ids = hidden_to_logits(model, outputs.last_hidden_state[:, -1, :]).argmax(dim=-1)
            finally:
                handle.remove()

            append_ids = next_ids.clone()
            for i in range(len(ids)):
                if finished[i]:
                    append_ids[i] = eos_fill
                    continue
                token_id = int(next_ids[i].item())
                generated[i].append(token_id)
                if token_id in eos_ids:
                    finished[i] = True

            ids = torch.cat([ids, append_ids.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones((len(ids), 1), dtype=mask.dtype, device=device)], dim=1)
            if bool(finished.all()):
                break

        all_sequences.extend(generated)

    return all_sequences


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
            return (new_hidden,) + output[1:] if isinstance(output, tuple) else new_hidden

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
def free_generate_c_with_t_restoration(model, bank, source_c, Q_c, c_layer, clean_base_t, Q_t, t_layer, tokenizer, batch_size, generation_cap):
    device = next(model.parameters()).device
    eos_ids = eos_id_set(tokenizer)
    eos_fill = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else list(eos_ids)[0]
    all_sequences = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source_c_batch = source_c[start:end].to(device)
        clean_t_batch = clean_base_t[start:end].to(device)
        positions = bank["base_positions"][start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        finished = torch.zeros(len(ids), dtype=torch.bool, device=device)
        generated = [[] for _ in range(len(ids))]

        for _ in range(generation_cap):
            def c_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                current = hidden[rows, positions, :].float()
                delta = ((source_c_batch.float() - current) @ Q_c) @ Q_c.T
                new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
                return (new_hidden,) + output[1:] if isinstance(output, tuple) else new_hidden

            def t_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                new_hidden = hidden.clone()
                current = hidden[rows, positions, :].float()
                delta = ((clean_t_batch.float() - current) @ Q_t) @ Q_t.T
                new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)
                return (new_hidden,) + output[1:] if isinstance(output, tuple) else new_hidden

            c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
            t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)
            try:
                position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
                next_ids = hidden_to_logits(model, outputs.last_hidden_state[:, -1, :]).argmax(dim=-1)
            finally:
                c_handle.remove()
                t_handle.remove()

            append_ids = next_ids.clone()
            for i in range(len(ids)):
                if finished[i]:
                    append_ids[i] = eos_fill
                    continue
                token_id = int(next_ids[i].item())
                generated[i].append(token_id)
                if token_id in eos_ids:
                    finished[i] = True

            ids = torch.cat([ids, append_ids.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones((len(ids), 1), dtype=mask.dtype, device=device)], dim=1)
            if bool(finished.all()):
                break

        all_sequences.extend(generated)

    return all_sequences


def build_factual_bank(tokenizer, rows, prefix_ids, variable):
    prompts = []
    labels = []
    for row in rows:
        prompts.extend([row["base_input"], row["source_input"]])
        labels.extend([row[f"base_{variable}"], row[f"source_{variable}"]])
    input_ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "positions": positions, "target": torch.tensor(labels, dtype=torch.long)}


def fit_readout(model, train_bank, validation_bank, Q, layer, num_classes, batch_size, seed):
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
    best_acc = -1.0
    best = None

    for penalty in [1e-4, 1e-3, 1e-2]:
        readout = torch.nn.Linear(Q.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)
        for _ in range(200):
            optimizer.zero_grad()
            logits = readout(x_train)
            loss = F.cross_entropy(logits, y_train) + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc = float((readout(x_val).argmax(-1) == y_val).float().mean())
            if acc > best_acc:
                W = readout.weight.detach() / std.unsqueeze(0)
                b = readout.bias.detach() - W @ mean
                best_acc = acc
                best = {"W": W.cpu(), "b": b.cpu(), "validation_accuracy": acc, "penalty": penalty}

    return best


@torch.no_grad()
def clean_readout(model, bank, Q, layer, W, b, batch_size):
    states = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["positions"], layer, batch_size)
    prediction = readout_logits(states, Q, W, b).argmax(-1).detach().cpu()
    base_idx = torch.arange(0, len(bank["target"]), 2)
    source_idx = torch.arange(1, len(bank["target"]), 2)
    base_acc = float((prediction[base_idx] == bank["target"][base_idx]).float().mean())
    source_acc = float((prediction[source_idx] == bank["target"][source_idx]).float().mean())
    return base_acc, source_acc


@torch.no_grad()
def eval_t(model, tokenizer, bank, Q_t, t_layer, W_t, b_t, batch_size, generation_cap):
    base_t = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], t_layer, batch_size)
    source_t = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], t_layer, batch_size)
    direct_t = projected_swap(base_t, source_t, Q_t)
    t_ok = readout_logits(direct_t, Q_t, W_t, b_t).argmax(-1).detach().cpu() == bank["target_t"].cpu()
    generated = free_generate_single_intervention(model, bank, source_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    y_ok, _, _ = score_sequences(generated, bank["target_y"], tokenizer)
    return {"Joint_T_Y_after_T": float((t_ok & y_ok).float().mean()), "T_after_T": float(t_ok.float().mean()), "Y_after_T": float(y_ok.float().mean())}


@torch.no_grad()
def eval_c(model, tokenizer, bank, pair_rows, Q_c, c_layer, W_c, b_c, Q_t, t_layer, W_t, b_t, batch_size, generation_cap):
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_positions"], c_layer, batch_size)
    base_c = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], c_layer, batch_size)
    direct_c = projected_swap(base_c, source_c, Q_c)
    target_c = torch.tensor([row["C_after_C_intervene"] for row in pair_rows], dtype=torch.long)
    c_ok = readout_logits(direct_c, Q_c, W_c, b_c).argmax(-1).detach().cpu() == target_c

    generated_t = capture_t_after_c(model, bank, source_c, Q_c, c_layer, t_layer, batch_size)
    t_ok = readout_logits(generated_t, Q_t, W_t, b_t).argmax(-1).detach().cpu() == bank["target_t"].cpu()

    direct_generated = free_generate_single_intervention(model, bank, source_c, Q_c, c_layer, tokenizer, batch_size, generation_cap)
    y_ok, _, _ = score_sequences(direct_generated, bank["target_y"], tokenizer)

    replay_generated = free_generate_single_intervention(model, bank, generated_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    replay_ok, _, _ = score_sequences(replay_generated, bank["target_y"], tokenizer)

    clean_base_t = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_positions"], t_layer, batch_size)
    restored_generated = free_generate_c_with_t_restoration(model, bank, source_c, Q_c, c_layer, clean_base_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    base_y = torch.tensor([row["base_Y"] for row in pair_rows], dtype=torch.long)
    restored_ok, _, _ = score_sequences(restored_generated, base_y, tokenizer)

    metrics = {
        "Joint_C_T_Y_after_C": float((c_ok & t_ok & y_ok).float().mean()),
        "C_after_C": float(c_ok.float().mean()),
        "T_after_C": float(t_ok.float().mean()),
        "Y_after_C": float(y_ok.float().mean()),
        "Y_via_T_replay": float(replay_ok.float().mean()),
        "Y_restored_via_T": float(restored_ok.float().mean()),
    }
    return metrics, generated_t


def build_recipient_examples(c_rows, seed):
    rng = random.Random(seed)
    result = []
    for donor_index, donor in enumerate(c_rows):
        transferred_t = donor["T_after_C_intervene"]
        candidates = []
        for i, recipient in enumerate(c_rows):
            if i == donor_index:
                continue
            if recipient["base_input"] == donor["base_input"]:
                continue
            if recipient["base_T"] == transferred_t:
                continue
            candidates.append(i)
        if len(candidates) == 0:
            raise RuntimeError(f"No recipient for donor {donor_index}")
        recipient_index = rng.choice(candidates)
        recipient = c_rows[recipient_index]
        target_y = 100 * recipient["base_hundreds_out"] + 10 * transferred_t + recipient["base_ones_out"]
        result.append({"donor_index": donor_index, "recipient_index": recipient_index, "transferred_T": transferred_t, "recipient_input": recipient["base_input"], "recipient_target_Y": target_y})
    return result


def build_recipient_bank(tokenizer, examples, prefix_ids):
    prompts = [x["recipient_input"] for x in examples]
    targets = [x["recipient_target_Y"] for x in examples]
    input_ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)
    return {"base_input_ids": input_ids, "base_attention_mask": attention_mask, "base_positions": positions, "target_y": torch.tensor(targets, dtype=torch.long)}


def build_clean_t_pool(t_rows):
    pool = {}
    for row in t_rows:
        for prompt, t_value in [(row["base_input"], row["base_T"]), (row["source_input"], row["source_T"])]:
            pool.setdefault(t_value, [])
            if prompt not in pool[t_value]:
                pool[t_value].append(prompt)
    return pool


@torch.no_grad()
def eval_recipients(model, tokenizer, c_rows, t_rows, prefix_ids, generated_t, Q_t, t_layer, batch_size, generation_cap, seed):
    examples = build_recipient_examples(c_rows, seed)
    bank = build_recipient_bank(tokenizer, examples, prefix_ids)
    induced = free_generate_single_intervention(model, bank, generated_t, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    induced_ok, _, _ = score_sequences(induced, bank["target_y"], tokenizer)

    rng = random.Random(seed + 1)
    pool = build_clean_t_pool(t_rows)
    clean_prompts = []
    for example in examples:
        candidates = [p for p in pool[example["transferred_T"]] if p != example["recipient_input"]]
        if len(candidates) == 0:
            raise RuntimeError(f"No clean T donor for T={example['transferred_T']}")
        clean_prompts.append(rng.choice(candidates))

    clean_ids, clean_mask, clean_positions = build_state_inputs(tokenizer, clean_prompts, prefix_ids)
    clean_states = collect_layer_states(model, clean_ids, clean_mask, clean_positions, t_layer, batch_size)
    clean_generated = free_generate_single_intervention(model, bank, clean_states, Q_t, t_layer, tokenizer, batch_size, generation_cap)
    clean_ok, _, _ = score_sequences(clean_generated, bank["target_y"], tokenizer)
    metrics = {"induced_T_recipient_Y": float(induced_ok.float().mean()), "clean_T_recipient_Y": float(clean_ok.float().mean()), "num_recipients": len(examples), "num_unique_recipients": len(set(x["recipient_index"] for x in examples))}
    return metrics, examples


def validate_checkpoints(model, t_ckpt, c_ckpt, t_readout):
    hidden_size = model.config.hidden_size
    Q_t = t_ckpt["basis"]
    Q_c = c_ckpt["basis"]
    t_layer = int(t_ckpt["layer"])
    c_layer = int(c_ckpt["layer"])
    if Q_t.shape[0] != hidden_size or Q_c.shape[0] != hidden_size:
        raise ValueError("Checkpoint basis width does not match model hidden size")
    if c_layer >= t_layer:
        raise ValueError(f"Expected C layer < T layer, got C={c_layer}, T={t_layer}")
    if "T_layer" in c_ckpt and int(c_ckpt["T_layer"]) != t_layer:
        raise ValueError("C checkpoint was trained with a different T layer")
    if int(t_readout["layer"]) != t_layer:
        raise ValueError("T readout layer does not match T checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_gradual_2.0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Only for natural-model filtering/cache; NOT evaluation stopping")
    parser.add_argument("--generation-cap", type=int, default=8, help="Safety cap; evaluation otherwise stops only at EOS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-recipients", action="store_true")
    args = parser.parse_args()

    if args.generation_cap < 1:
        raise ValueError("--generation-cap must be positive")

    set_seed(args.seed)
    model, tokenizer = load_gemma_model(args.device)
    device = next(model.parameters()).device
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.batch_size, args.max_new_tokens, False)
    banks = build_all_banks(tokenizer, rows)
    prefix_ids = banks["prefix_ids"]
    result_dir = Path(args.result_dir)

    t_ckpt = torch.load(result_dir / "das_t.pt", map_location="cpu", weights_only=False)
    c_ckpt = torch.load(result_dir / "das_c.pt", map_location="cpu", weights_only=False)
    t_readout = torch.load(result_dir / "t_readout_LogisticRegression.pt", map_location="cpu", weights_only=False)
    validate_checkpoints(model, t_ckpt, c_ckpt, t_readout)

    Q_t = t_ckpt["basis"].float().to(device)
    Q_c = c_ckpt["basis"].float().to(device)
    t_layer = int(t_ckpt["layer"])
    c_layer = int(c_ckpt["layer"])
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    c_factual = {}
    for split in ["train", "validation", "test"]:
        c_factual[split] = build_factual_bank(tokenizer, rows[split]["C_only"], prefix_ids, "C")

    c_readout = fit_readout(model, c_factual["train"], c_factual["validation"], Q_c, c_layer, 2, args.batch_size, args.seed)
    W_c = c_readout["W"].float().to(device)
    b_c = c_readout["b"].float().to(device)
    torch.save({"W_C": c_readout["W"], "b_C": c_readout["b"], "layer": c_layer, "basis": Q_c.cpu(), "validation_accuracy": c_readout["validation_accuracy"], "penalty": c_readout["penalty"]}, result_dir / "c_readout_LogisticRegression.pt")

    t_factual_test = {"input_ids": banks["T_factual"]["test"]["input_ids"], "attention_mask": banks["T_factual"]["test"]["attention_mask"], "positions": banks["T_factual"]["test"]["positions"], "target": banks["T_factual"]["test"]["target_t"]}
    c_base, c_source = clean_readout(model, c_factual["test"], Q_c, c_layer, W_c, b_c, args.batch_size)
    t_base, t_source = clean_readout(model, t_factual_test, Q_t, t_layer, W_t, b_t, args.batch_size)

    print("\n=== EVALUATION SEMANTICS ===")
    print(f"Free autoregressive generation until EOS or generation_cap={args.generation_cap}")
    print("Ground-truth answer length is NOT used for stopping")
    print("Ground-truth answer tokens are NEVER fed during Y evaluation")
    print("\n=== CLEAN READOUT ===")
    print(f"C base/source: {c_base:.4f} / {c_source:.4f}")
    print(f"T base/source: {t_base:.4f} / {t_source:.4f}")

    t_metrics = eval_t(model, tokenizer, banks["T"]["test"], Q_t, t_layer, W_t, b_t, args.batch_size, args.generation_cap)
    print("\n=== NODE CORRECTNESS — T intervention [T_only] ===")
    print(" | ".join(f"{k}: {v:.4f}" for k, v in t_metrics.items()))

    c_metrics = {}
    generated_t = {}
    for name, bank_key, pair_type in [("C_only", "C", "C_only"), ("C_T", "C_T", "C_T")]:
        metrics, states = eval_c(model, tokenizer, banks[bank_key]["test"], rows["test"][pair_type], Q_c, c_layer, W_c, b_c, Q_t, t_layer, W_t, b_t, args.batch_size, args.generation_cap)
        c_metrics[name] = metrics
        generated_t[name] = states
        print(f"\n=== NODE CORRECTNESS — C intervention [{name}] ===")
        print(f"Joint_C_T_Y_after_C: {metrics['Joint_C_T_Y_after_C']:.4f} | C_after_C: {metrics['C_after_C']:.4f} | T_after_C: {metrics['T_after_C']:.4f} | Y_after_C: {metrics['Y_after_C']:.4f}")
        print(f"\n=== MEDIATOR — C intervention [{name}] ===")
        print(f"Y_via_T_replay: {metrics['Y_via_T_replay']:.4f}")
        print(f"\n=== RESTORATION — C intervention [{name}] ===")
        print(f"Y_restored_via_T: {metrics['Y_restored_via_T']:.4f}")

    recipient_metrics = {}
    recipient_examples = {}
    if not args.skip_recipients:
        for name, pair_type, offset in [("C_only", "C_only", 100), ("C_T", "C_T", 200)]:
            metrics, examples = eval_recipients(model, tokenizer, rows["test"][pair_type], rows["test"]["T_only"], prefix_ids, generated_t[name], Q_t, t_layer, args.batch_size, args.generation_cap, args.seed + offset)
            recipient_metrics[name] = metrics
            recipient_examples[name] = examples
            print(f"\n=== OTHER RECIPIENTS [{name}] ===")
            print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()))

    output = {
        "evaluation_semantics": {"generation": "free_autoregressive_until_eos_or_cap", "generation_cap": args.generation_cap, "ground_truth_length_used_for_stopping": False, "teacher_forcing_used_for_Y_evaluation": False, "score": "first_integer_in_full_completion_equals_target_integer"},
        "layers": {"C": c_layer, "T": t_layer},
        "clean_readout": {"C_base": c_base, "C_source": c_source, "T_base": t_base, "T_source": t_source},
        "T_only": t_metrics,
        "C_only": c_metrics["C_only"],
        "C_T": c_metrics["C_T"],
        "recipients": recipient_metrics,
    }
    (result_dir / "eval_full_mcqa_free_generation.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "=== EVALUATION SEMANTICS ===",
        f"Free autoregressive generation until EOS or generation_cap={args.generation_cap}",
        "Ground-truth answer length is NOT used for stopping",
        "Ground-truth answer tokens are NEVER fed during Y evaluation",
        "",
        "=== CLEAN READOUT ===",
        f"C base/source: {c_base:.4f} / {c_source:.4f}",
        f"T base/source: {t_base:.4f} / {t_source:.4f}",
        "",
        "=== NODE CORRECTNESS — T intervention [T_only] ===",
        " | ".join(f"{k}: {v:.4f}" for k, v in t_metrics.items()),
    ]
    for name in ["C_only", "C_T"]:
        m = c_metrics[name]
        lines += ["", f"=== NODE CORRECTNESS — C intervention [{name}] ===", f"Joint_C_T_Y_after_C: {m['Joint_C_T_Y_after_C']:.4f} | C_after_C: {m['C_after_C']:.4f} | T_after_C: {m['T_after_C']:.4f} | Y_after_C: {m['Y_after_C']:.4f}", "", f"=== MEDIATOR — C intervention [{name}] ===", f"Y_via_T_replay: {m['Y_via_T_replay']:.4f}", "", f"=== RESTORATION — C intervention [{name}] ===", f"Y_restored_via_T: {m['Y_restored_via_T']:.4f}"]
    if not args.skip_recipients:
        for name in ["C_only", "C_T"]:
            lines += ["", f"=== OTHER RECIPIENTS [{name}] ===", " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in recipient_metrics[name].items())]
            save_jsonl(recipient_examples[name], result_dir / f"recipient_pairs_{name}.jsonl")
    (result_dir / "eval_full_mcqa_free_generation.txt").write_text("\n".join(lines) + "\n")

    print(f"\n[saved] {result_dir / 'eval_full_mcqa_free_generation.txt'}")
    print(f"[saved] {result_dir / 'eval_full_mcqa_free_generation.json'}")


if __name__ == "__main__":
    main()
