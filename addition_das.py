import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedSubspace(nn.Module):
    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        initial = torch.randn(hidden_size, subspace_dim, dtype=torch.float32)
        self.raw = nn.Parameter(torch.linalg.qr(initial, mode="reduced").Q)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hidden_to_logits(model, hidden):
    logits = model.lm_head(hidden).float()
    softcap = getattr(model.config, "final_logit_softcapping", None)

    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits


def sequence_logits(model, outputs, label_positions):
    hidden = outputs.last_hidden_state
    device = hidden.device
    rows = torch.arange(hidden.shape[0], device=device).unsqueeze(1)
    positions = label_positions.to(device)
    selected_hidden = hidden[rows, positions, :]
    return hidden_to_logits(model, selected_hidden)


def sequence_cross_entropy(logits, target_ids, target_mask):
    targets = target_ids.to(logits.device)
    mask = target_mask.to(logits.device)
    batch_size, sequence_length, vocab_size = logits.shape
    token_losses = F.cross_entropy(logits.reshape(batch_size * sequence_length, vocab_size), targets.reshape(-1), reduction="none")
    token_losses = token_losses.reshape(batch_size, sequence_length)
    token_losses = token_losses * mask.float()
    example_lengths = mask.sum(dim=1).clamp_min(1)
    example_losses = token_losses.sum(dim=1) / example_lengths
    return example_losses.mean()


def sequence_iia(logits, target_ids, target_mask):
    predictions = logits.argmax(dim=-1).detach().cpu()
    targets = target_ids.detach().cpu()
    mask = target_mask.detach().cpu()
    correct = predictions == targets
    correct = correct | (~mask)
    return float(correct.all(dim=1).float().mean())


def first_token_accuracy(logits, target_ids):
    predictions = logits[:, 0, :].argmax(dim=-1).detach().cpu()
    targets = target_ids[:, 0].detach().cpu()
    return float((predictions == targets).float().mean())


def classification_accuracy(logits, targets):
    predictions = logits.argmax(dim=-1).detach().cpu()
    targets = targets.detach().cpu()
    return float((predictions == targets).float().mean())


def autoregressive_token_iia(generated_ids, target_ids, target_mask):
    generated_ids = generated_ids.detach().cpu()
    target_ids = target_ids.detach().cpu()
    target_mask = target_mask.detach().cpu()
    max_target_length = target_ids.shape[1]

    if generated_ids.shape[1] < max_target_length:
        return 0.0

    generated_ids = generated_ids[:, :max_target_length]
    correct = generated_ids == target_ids
    correct = correct | (~target_mask)
    return float(correct.all(dim=1).float().mean())


def autoregressive_numeric_accuracy(generated_ids, target_values, tokenizer, target_mask):
    generated_ids = generated_ids.detach().cpu()
    target_values = target_values.detach().cpu()
    target_mask = target_mask.detach().cpu()
    correct = 0

    for row_index in range(len(generated_ids)):
        target_length = int(target_mask[row_index].sum())
        ids = generated_ids[row_index, :target_length].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        numbers = re.findall(r"\d+", text)
        prediction = None

        if len(numbers) > 0:
            prediction = int(numbers[0])

        if prediction == int(target_values[row_index]):
            correct += 1

    return correct / len(generated_ids)


@torch.no_grad()
def collect_layer_states(model, input_ids, attention_mask, positions, layer, batch_size=32):
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
            hidden = output[0] if isinstance(output, tuple) else output
            captured["state"] = hidden[rows, position, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()

        states.append(captured["state"])

    return torch.cat(states, dim=0)


def slice_teacher_bank(bank, indices):
    mini = {}

    for key in ["teacher_input_ids", "teacher_attention_mask", "intervention_positions", "label_positions", "target_answer_ids", "target_answer_mask"]:
        mini[key] = bank[key][indices]

    if "target_t" in bank:
        mini["target_t"] = bank["target_t"][indices]

    return mini


def run_das_intervention(model, bank, source_states, Q, layer, batch_size=32, require_grad=False):
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
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q) @ Q.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
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
def run_das_autoregressive(model, bank, source_states, Q, layer, batch_size=32):
    device = next(model.parameters()).device
    all_generated = []
    max_target_length = int(bank["target_answer_mask"].sum(dim=1).max())

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
                base_state = hidden[rows, intervention_positions, :].float()
                delta = ((donor.float() - base_state) @ Q) @ Q.T
                new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

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


def run_state_replay(model, bank, donor_states, Q, layer, batch_size=32, require_grad=False):
    return run_das_intervention(model, bank, donor_states, Q, layer, batch_size, require_grad)
