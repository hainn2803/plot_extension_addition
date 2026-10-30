"""Train C DAS on two-digit addition using the answer-start state."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from addition_gradual_das_t import LearnedSubspace, build_pair_bank, collect_all_layer_states, collect_layer_states, get_answer_prefix_ids, load_gemma_model, parse_layers, read_jsonl, sequence_cross_entropy, sequence_iia, sequence_logits, set_seed, slice_teacher_bank


def run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size, require_grad):
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
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q_c) @ Q_c.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def t_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            captured["t"] = hidden[rows, intervention_positions, :].float()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
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


def run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size, require_grad):
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
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_state = hidden[rows, intervention_positions, :].float()
            delta = ((donor.float() - base_state) @ Q_t) @ Q_t.T
            new_hidden[rows, intervention_positions, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[t_layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
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


@torch.no_grad()
def evaluate_c(model, bank, source_c, Q_c, c_layer, Q_t, t_layer, W_t, b_t, batch_size):
    direct_y, generated_t = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, t_layer, batch_size, False)
    mediator_y = run_frozen_t_mediator(model, bank, generated_t, Q_t, t_layer, batch_size, False)

    direct_iia = sequence_iia(direct_y, bank["target_answer_ids"], bank["target_answer_mask"])
    mediator_iia = sequence_iia(mediator_y, bank["target_answer_ids"], bank["target_answer_mask"])

    t_logits = (generated_t.to(Q_t.device) @ Q_t) @ W_t.T + b_t
    t_prediction = t_logits.argmax(-1).cpu()
    t_readout_acc = float((t_prediction == bank["target_t"].cpu()).float().mean())

    return direct_iia, mediator_iia, t_readout_acc


def train_c(model, tokenizer, train_rows, val_rows, test_rows, propagation_rows, prefix_ids, t_checkpoint_path, t_readout_path, layers=None, subspace_dim=128, epochs=20, lr=1e-2, lambda_med=0.0, lambda_readout=0.0, train_batch_size=32, eval_batch_size=32, seed=0, save_path="results/das_c.pt"):
    set_seed(seed)
    device = next(model.parameters()).device

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    t_checkpoint = torch.load(t_checkpoint_path, map_location="cpu", weights_only=False)
    Q_t = t_checkpoint["basis"].float().to(device)
    t_layer = int(t_checkpoint["layer"])

    t_readout = torch.load(t_readout_path, map_location="cpu", weights_only=False)
    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    train_bank = build_pair_bank(tokenizer, train_rows, "C", prefix_ids)
    val_bank = build_pair_bank(tokenizer, val_rows, "C", prefix_ids)
    test_bank = build_pair_bank(tokenizer, test_rows, "C", prefix_ids)
    propagation_bank = build_pair_bank(tokenizer, propagation_rows, "C", prefix_ids)

    if layers is None:
        layers = list(range(t_layer))
    else:
        valid_layers = []

        for layer in layers:
            if layer < t_layer:
                valid_layers.append(layer)

        layers = valid_layers

    if len(layers) == 0:
        raise ValueError(f"No valid C layers before T layer {t_layer}")

    val_source_by_layer = collect_all_layer_states(model, val_bank["source_input_ids"], val_bank["source_attention_mask"], val_bank["source_positions"], layers, eval_batch_size)

    layer_scores = []

    for layer in layers:
        logits = run_full_layer_intervention(model, val_bank, val_source_by_layer[layer], layer, eval_batch_size)
        iia = sequence_iia(logits, val_bank["target_answer_ids"], val_bank["target_answer_mask"])
        layer_scores.append((layer, iia))
        print(f"[C layer {layer}] full_iia={iia:.4f}")

    c_layer = max(layer_scores, key=lambda x: x[1])[0]
    print(f"[C] selected layer={c_layer}, T layer={t_layer}")

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_positions"], c_layer, eval_batch_size)
    val_source = val_source_by_layer[c_layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_positions"], c_layer, eval_batch_size)
    propagation_source = collect_layer_states(model, propagation_bank["source_input_ids"], propagation_bank["source_attention_mask"], propagation_bank["source_positions"], c_layer, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["teacher_input_ids"]))

        total_direct_loss = 0.0
        total_mediator_loss = 0.0
        total_readout_loss = 0.0

        for start in range(0, len(permutation), train_batch_size):
            idx = permutation[start:start + train_batch_size]
            mini = slice_teacher_bank(train_bank, idx)

            optimizer.zero_grad()
            Q_c = alignment.basis()

            direct_y, generated_t = run_c_intervention_and_capture(model, mini, train_source[idx], Q_c, c_layer, t_layer, len(idx), True)
            mediator_y = run_frozen_t_mediator(model, mini, generated_t, Q_t, t_layer, len(idx), True)

            t_target = train_bank["target_t"][idx].to(device)
            t_logits = (generated_t @ Q_t) @ W_t.T + b_t

            direct_loss = sequence_cross_entropy(direct_y, mini["target_answer_ids"], mini["target_answer_mask"])
            mediator_loss = sequence_cross_entropy(mediator_y, mini["target_answer_ids"], mini["target_answer_mask"])
            readout_loss = F.cross_entropy(t_logits, t_target)

            loss = direct_loss
            loss = loss + lambda_med * mediator_loss
            loss = loss + lambda_readout * readout_loss

            loss.backward()
            optimizer.step()

            batch_n = len(idx)
            total_direct_loss += direct_loss.item() * batch_n
            total_mediator_loss += mediator_loss.item() * batch_n
            total_readout_loss += readout_loss.item() * batch_n

        metrics = evaluate_c(model, val_bank, val_source, alignment.basis(), c_layer, Q_t, t_layer, W_t, b_t, eval_batch_size)
        direct_iia, mediator_iia, t_readout_acc = metrics

        active_metrics = [direct_iia]

        if lambda_med > 0:
            active_metrics.append(mediator_iia)

        if lambda_readout > 0:
            active_metrics.append(t_readout_acc)

        score = min(active_metrics)
        n = len(permutation)

        print(
            f"[C epoch {epoch:02d}] "
            f"direct_loss={total_direct_loss / n:.4f} "
            f"med_loss={total_mediator_loss / n:.4f} "
            f"readout_loss={total_readout_loss / n:.4f} | "
            f"direct={direct_iia:.4f} "
            f"T_med={mediator_iia:.4f} "
            f"T_readout={t_readout_acc:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_metrics = metrics

    Q_best = best_basis.to(device)

    test_metrics = evaluate_c(model, test_bank, test_source, Q_best, c_layer, Q_t, t_layer, W_t, b_t, eval_batch_size)
    propagation_metrics = evaluate_c(model, propagation_bank, propagation_source, Q_best, c_layer, Q_t, t_layer, W_t, b_t, eval_batch_size)

    result = {
        "variable": "C",
        "layer": c_layer,
        "basis": best_basis,
        "token_position": "answer_start",
        "answer_prefix_ids": prefix_ids,
        "subspace_dim": subspace_dim,
        "best_epoch": best_epoch,
        "lambda_med": lambda_med,
        "lambda_readout": lambda_readout,
        "T_layer": t_layer,
        "val_direct_iia": best_metrics[0],
        "val_T_mediator_iia": best_metrics[1],
        "val_T_readout_acc": best_metrics[2],
        "test_direct_iia": test_metrics[0],
        "test_T_mediator_iia": test_metrics[1],
        "test_T_readout_acc": test_metrics[2],
        "propagation_direct_iia": propagation_metrics[0],
        "propagation_T_mediator_iia": propagation_metrics[1],
        "propagation_T_readout_acc": propagation_metrics[2],
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, save_path)

    print(
        f"[C BEST] layer={c_layer} epoch={best_epoch} "
        f"test_direct={test_metrics[0]:.4f} "
        f"test_T_med={test_metrics[1]:.4f} "
        f"test_T_readout={test_metrics[2]:.4f} | "
        f"C_T_direct={propagation_metrics[0]:.4f} "
        f"C_T_Tmed={propagation_metrics[1]:.4f} "
        f"C_T_Treadout={propagation_metrics[2]:.4f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_filtered_pairs")
    parser.add_argument("--result-dir", default="results_addition_c")
    parser.add_argument("--t-checkpoint", required=True)
    parser.add_argument("--t-readout", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=0.0)
    parser.add_argument("--lambda-readout", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    train_rows = read_jsonl(Path(args.data_dir) / "train_C_only.jsonl")
    val_rows = read_jsonl(Path(args.data_dir) / "val_C_only.jsonl")
    test_rows = read_jsonl(Path(args.data_dir) / "test_C_only.jsonl")
    propagation_rows = read_jsonl(Path(args.data_dir) / "test_C_T.jsonl")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model(args.device)
    prefix_ids = get_answer_prefix_ids(tokenizer)

    train_c(
        model,
        tokenizer,
        train_rows,
        val_rows,
        test_rows,
        propagation_rows,
        prefix_ids,
        args.t_checkpoint,
        args.t_readout,
        parse_layers(args.layers),
        args.dim,
        args.epochs,
        args.lr,
        args.lambda_med,
        args.lambda_readout,
        args.train_batch_size,
        args.eval_batch_size,
        args.seed,
        str(result_dir / "das_c.pt"),
    )


if __name__ == "__main__":
    main()
