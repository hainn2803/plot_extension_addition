import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from addition_das import LearnedSubspace, autoregressive_numeric_accuracy, autoregressive_token_iia, classification_accuracy, collect_layer_states, run_das_autoregressive, sequence_cross_entropy, sequence_iia, sequence_logits, set_seed, slice_teacher_bank
from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model


def run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size=32, require_grad=False):
    device = next(model.parameters()).device
    all_y = []
    all_t = []

    for start in range(0, len(bank["teacher_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["teacher_input_ids"]))
        ids = bank["teacher_input_ids"][start:end].to(device)
        mask = bank["teacher_attention_mask"][start:end].to(device)
        donor = source_c[start:end].to(device)
        intervention_positions = bank["intervention_positions"][start:end].to(device)
        label_positions = bank["label_positions"][start:end]
        rows = torch.arange(len(ids), device=device)
        captured = {}

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q_c) @ Q_c.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def t_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["t"] = hidden[rows, intervention_positions, :].float()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y_logits = sequence_logits(model, outputs, label_positions)
            t_state = captured["t"]

            if require_grad:
                all_y.append(y_logits)
                all_t.append(t_state)
            else:
                all_y.append(y_logits.detach().cpu())
                all_t.append(t_state.detach().cpu())
        finally:
            c_handle.remove()
            t_handle.remove()

    return torch.cat(all_y, dim=0), torch.cat(all_t, dim=0)


def run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size=32, require_grad=False):
    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["teacher_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["teacher_input_ids"]))
        ids = bank["teacher_input_ids"][start:end].to(device)
        mask = bank["teacher_attention_mask"][start:end].to(device)
        donor = generated_t[start:end].to(device)
        intervention_positions = bank["intervention_positions"][start:end].to(device)
        label_positions = bank["label_positions"][start:end]
        rows = torch.arange(len(ids), device=device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q_t) @ Q_t.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[t_layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y_logits = sequence_logits(model, outputs, label_positions)

            if require_grad:
                all_y.append(y_logits)
            else:
                all_y.append(y_logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_y, dim=0)


@torch.no_grad()
def evaluate_c(model, tokenizer, bank, source_c, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size):
    direct_teacher_y, generated_t = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size, False)
    mediator_teacher_y = run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size, False)
    t_logits = (generated_t.to(Q_t.device) @ Q_t) @ W_t.T + b_t

    direct_generated = run_das_autoregressive(model, bank, source_c, Q_c, c_layer, batch_size)
    mediator_generated = run_das_autoregressive(model, bank, generated_t, Q_t, t_layer, batch_size)

    return {
        "direct_numeric_iia": autoregressive_numeric_accuracy(direct_generated, bank["target_y"], tokenizer, bank["target_answer_mask"]),
        "direct_token_iia": autoregressive_token_iia(direct_generated, bank["target_answer_ids"], bank["target_answer_mask"]),
        "direct_teacher_forced_iia": sequence_iia(direct_teacher_y, bank["target_answer_ids"], bank["target_answer_mask"]),
        "mediator_numeric_iia": autoregressive_numeric_accuracy(mediator_generated, bank["target_y"], tokenizer, bank["target_answer_mask"]),
        "mediator_token_iia": autoregressive_token_iia(mediator_generated, bank["target_answer_ids"], bank["target_answer_mask"]),
        "mediator_teacher_forced_iia": sequence_iia(mediator_teacher_y, bank["target_answer_ids"], bank["target_answer_mask"]),
        "t_readout_acc": classification_accuracy(t_logits, bank["target_t"]),
    }


def train_c(model, tokenizer, train_bank, validation_bank, test_bank, propagation_bank, t_checkpoint, t_readout, layer, subspace_dim=128, epochs=20, lr=1e-2, lambda_med=0.0, lambda_readout=0.0, train_batch_size=32, eval_batch_size=32, seed=0, save_path=None):
    set_seed(seed)
    device = next(model.parameters()).device

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    Q_t = t_checkpoint["basis"].float().to(device)
    t_layer = int(t_checkpoint["layer"])
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    if layer >= t_layer:
        raise ValueError(f"C layer must be before T layer. Got C={layer}, T={t_layer}")

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_positions"], layer, eval_batch_size)
    validation_source = collect_layer_states(model, validation_bank["source_input_ids"], validation_bank["source_attention_mask"], validation_bank["source_positions"], layer, eval_batch_size)
    test_source = None
    propagation_source = None

    if test_bank is not None:
        test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_positions"], layer, eval_batch_size)

    if propagation_bank is not None:
        propagation_source = collect_layer_states(model, propagation_bank["source_input_ids"], propagation_bank["source_attention_mask"], propagation_bank["source_positions"], layer, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_validation_metrics = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["teacher_input_ids"]))
        total_direct_loss = 0.0
        total_mediator_loss = 0.0
        total_readout_loss = 0.0

        for start in range(0, len(permutation), train_batch_size):
            indices = permutation[start:start + train_batch_size]
            mini_bank = slice_teacher_bank(train_bank, indices)

            optimizer.zero_grad()
            Q_c = alignment.basis()
            direct_y, generated_t = run_c_intervention_and_capture(model, mini_bank, train_source[indices], Q_c, layer, t_layer, len(indices), True)
            mediator_y = run_frozen_t_mediator(model, mini_bank, generated_t, Q_t, t_layer, len(indices), True)
            t_logits = (generated_t @ Q_t) @ W_t.T + b_t

            direct_loss = sequence_cross_entropy(direct_y, mini_bank["target_answer_ids"], mini_bank["target_answer_mask"])
            mediator_loss = sequence_cross_entropy(mediator_y, mini_bank["target_answer_ids"], mini_bank["target_answer_mask"])
            readout_loss = F.cross_entropy(t_logits, mini_bank["target_t"].to(device))

            loss = direct_loss
            loss = loss + lambda_med * mediator_loss
            loss = loss + lambda_readout * readout_loss
            loss.backward()
            optimizer.step()

            batch_size_actual = len(indices)
            total_direct_loss += direct_loss.item() * batch_size_actual
            total_mediator_loss += mediator_loss.item() * batch_size_actual
            total_readout_loss += readout_loss.item() * batch_size_actual

        validation_metrics = evaluate_c(model, tokenizer, validation_bank, validation_source, alignment.basis(), layer, Q_t, t_layer, W_t, b_t, eval_batch_size)

        active_metrics = [validation_metrics["direct_numeric_iia"]]
        if lambda_med > 0:
            active_metrics.append(validation_metrics["mediator_numeric_iia"])
        if lambda_readout > 0:
            active_metrics.append(validation_metrics["t_readout_acc"])

        score = min(active_metrics)
        n = len(permutation)

        print(
            f"[C layer {layer:02d} epoch {epoch:02d}] "
            f"direct_loss={total_direct_loss / n:.4f} "
            f"med_loss={total_mediator_loss / n:.4f} "
            f"readout_loss={total_readout_loss / n:.4f} | "
            f"direct_numeric={validation_metrics['direct_numeric_iia']:.4f} "
            f"T_med_numeric={validation_metrics['mediator_numeric_iia']:.4f} "
            f"T_readout={validation_metrics['t_readout_acc']:.4f} "
            f"direct_teacher={validation_metrics['direct_teacher_forced_iia']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_validation_metrics = validation_metrics

    Q_best = best_basis.to(device)
    test_metrics = None
    propagation_metrics = None

    if test_bank is not None:
        test_metrics = evaluate_c(model, tokenizer, test_bank, test_source, Q_best, layer, Q_t, t_layer, W_t, b_t, eval_batch_size)

    if propagation_bank is not None:
        propagation_metrics = evaluate_c(model, tokenizer, propagation_bank, propagation_source, Q_best, layer, Q_t, t_layer, W_t, b_t, eval_batch_size)

    result = {
        "variable": "C",
        "metric_version": "autoregressive_numeric_v1",
        "layer": int(layer),
        "basis": best_basis,
        "token_position": "answer_start",
        "subspace_dim": int(subspace_dim),
        "best_epoch": int(best_epoch),
        "lambda_med": float(lambda_med),
        "lambda_readout": float(lambda_readout),
        "T_layer": int(t_layer),
        "validation_score": float(best_score),
        "validation_direct_iia": float(best_validation_metrics["direct_numeric_iia"]),
        "validation_direct_token_iia": float(best_validation_metrics["direct_token_iia"]),
        "validation_direct_teacher_forced_iia": float(best_validation_metrics["direct_teacher_forced_iia"]),
        "validation_T_mediator_iia": float(best_validation_metrics["mediator_numeric_iia"]),
        "validation_T_mediator_token_iia": float(best_validation_metrics["mediator_token_iia"]),
        "validation_T_mediator_teacher_forced_iia": float(best_validation_metrics["mediator_teacher_forced_iia"]),
        "validation_T_readout_acc": float(best_validation_metrics["t_readout_acc"]),
        "seed": int(seed),
    }

    if test_metrics is not None:
        result["test_direct_iia"] = float(test_metrics["direct_numeric_iia"])
        result["test_direct_token_iia"] = float(test_metrics["direct_token_iia"])
        result["test_direct_teacher_forced_iia"] = float(test_metrics["direct_teacher_forced_iia"])
        result["test_T_mediator_iia"] = float(test_metrics["mediator_numeric_iia"])
        result["test_T_mediator_token_iia"] = float(test_metrics["mediator_token_iia"])
        result["test_T_mediator_teacher_forced_iia"] = float(test_metrics["mediator_teacher_forced_iia"])
        result["test_T_readout_acc"] = float(test_metrics["t_readout_acc"])

    if propagation_metrics is not None:
        result["propagation_direct_iia"] = float(propagation_metrics["direct_numeric_iia"])
        result["propagation_direct_token_iia"] = float(propagation_metrics["direct_token_iia"])
        result["propagation_direct_teacher_forced_iia"] = float(propagation_metrics["direct_teacher_forced_iia"])
        result["propagation_T_mediator_iia"] = float(propagation_metrics["mediator_numeric_iia"])
        result["propagation_T_mediator_token_iia"] = float(propagation_metrics["mediator_token_iia"])
        result["propagation_T_mediator_teacher_forced_iia"] = float(propagation_metrics["mediator_teacher_forced_iia"])
        result["propagation_T_readout_acc"] = float(propagation_metrics["t_readout_acc"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)

    if test_metrics is None:
        print(f"[C BEST layer {layer:02d}] epoch={best_epoch} validation_score={best_score:.4f}")
    else:
        print(
            f"[C BEST layer {layer:02d}] epoch={best_epoch} validation_score={best_score:.4f} "
            f"test_direct_numeric={test_metrics['direct_numeric_iia']:.4f} "
            f"test_T_med_numeric={test_metrics['mediator_numeric_iia']:.4f} "
            f"test_T_readout={test_metrics['t_readout_acc']:.4f}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_c")
    parser.add_argument("--t-checkpoint", required=True)
    parser.add_argument("--t-readout", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=0.0)
    parser.add_argument("--lambda-readout", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    set_seed(seed=args.seed)
    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)
    t_checkpoint = torch.load(args.t_checkpoint, map_location="cpu", weights_only=False)
    t_readout = torch.load(args.t_readout, map_location="cpu", weights_only=False)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_c(model, tokenizer, banks["C"]["train"], banks["C"]["validation"], banks["C"]["test"], banks["C_T"]["test"], t_checkpoint, t_readout, args.layer, args.dim, args.epochs, args.lr, args.lambda_med, args.lambda_readout, args.train_batch_size, args.eval_batch_size, args.seed, result_dir / "das_c.pt")


if __name__ == "__main__":
    main()
