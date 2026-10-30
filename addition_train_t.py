import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from addition_das import LearnedSubspace, autoregressive_numeric_accuracy, autoregressive_token_iia, collect_layer_states, run_das_autoregressive, run_das_intervention, sequence_cross_entropy, sequence_iia, set_seed, slice_teacher_bank
from addition_data_load import build_all_banks, load_addition_rows
from addition_neural_network import load_gemma_model


@torch.no_grad()
def evaluate_t(model, tokenizer, bank, source_states, Q, layer, batch_size):
    teacher_logits = run_das_intervention(model, bank, source_states, Q, layer, batch_size, False)
    teacher_iia = sequence_iia(teacher_logits, bank["target_answer_ids"], bank["target_answer_mask"])

    generated_ids = run_das_autoregressive(model, bank, source_states, Q, layer, batch_size)
    token_iia = autoregressive_token_iia(generated_ids, bank["target_answer_ids"], bank["target_answer_mask"])
    numeric_iia = autoregressive_numeric_accuracy(generated_ids, bank["target_y"], tokenizer, bank["target_answer_mask"])

    return {
        "numeric_iia": numeric_iia,
        "token_iia": token_iia,
        "teacher_forced_iia": teacher_iia,
    }


def train_t(model, tokenizer, train_bank, validation_bank, test_bank, layer, subspace_dim=128, epochs=20, lr=1e-2, train_batch_size=32, eval_batch_size=32, seed=0, save_path=None):
    set_seed(seed)
    device = next(model.parameters()).device

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_positions"], layer, eval_batch_size)
    validation_source = collect_layer_states(model, validation_bank["source_input_ids"], validation_bank["source_attention_mask"], validation_bank["source_positions"], layer, eval_batch_size)
    test_source = None
    if test_bank is not None:
        test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_positions"], layer, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_validation_iia = -1.0
    best_basis = None
    best_epoch = None
    best_validation_metrics = None

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

        validation_metrics = evaluate_t(model, tokenizer, validation_bank, validation_source, alignment.basis(), layer, eval_batch_size)
        validation_iia = validation_metrics["numeric_iia"]

        print(
            f"[T layer {layer:02d} epoch {epoch:02d}] "
            f"loss={total_loss / len(permutation):.4f} "
            f"validation_numeric_iia={validation_metrics['numeric_iia']:.4f} "
            f"validation_token_iia={validation_metrics['token_iia']:.4f} "
            f"validation_teacher_iia={validation_metrics['teacher_forced_iia']:.4f}"
        )

        if validation_iia > best_validation_iia:
            best_validation_iia = validation_iia
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_validation_metrics = validation_metrics

    test_metrics = None
    if test_bank is not None:
        test_metrics = evaluate_t(model, tokenizer, test_bank, test_source, best_basis.to(device), layer, eval_batch_size)

    result = {
        "variable": "T",
        "metric_version": "autoregressive_numeric_v1",
        "layer": int(layer),
        "basis": best_basis,
        "token_position": "answer_start",
        "subspace_dim": int(subspace_dim),
        "best_epoch": int(best_epoch),
        "validation_iia": float(best_validation_metrics["numeric_iia"]),
        "validation_token_iia": float(best_validation_metrics["token_iia"]),
        "validation_teacher_forced_iia": float(best_validation_metrics["teacher_forced_iia"]),
        "seed": int(seed),
    }

    if test_metrics is not None:
        result["test_iia"] = float(test_metrics["numeric_iia"])
        result["test_token_iia"] = float(test_metrics["token_iia"])
        result["test_teacher_forced_iia"] = float(test_metrics["teacher_forced_iia"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)

    if test_metrics is None:
        print(f"[T BEST layer {layer:02d}] epoch={best_epoch} validation_numeric_iia={best_validation_metrics['numeric_iia']:.4f}")
    else:
        print(
            f"[T BEST layer {layer:02d}] epoch={best_epoch} "
            f"validation_numeric_iia={best_validation_metrics['numeric_iia']:.4f} "
            f"test_numeric_iia={test_metrics['numeric_iia']:.4f}"
        )
    return result


def fit_t_readout(model, train_bank, validation_bank, Q, layer, batch_size=32, seed=0, save_path=None):
    set_seed(seed)
    device = next(model.parameters()).device
    Q = Q.float().to(device)

    train_states = collect_layer_states(model, train_bank["input_ids"], train_bank["attention_mask"], train_bank["positions"], layer, batch_size).to(device)
    validation_states = collect_layer_states(model, validation_bank["input_ids"], validation_bank["attention_mask"], validation_bank["positions"], layer, batch_size).to(device)

    x_train = (train_states @ Q).detach()
    x_validation = (validation_states @ Q).detach()
    y_train = train_bank["target_t"].to(device)
    y_validation = validation_bank["target_t"].to(device)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train_normalized = (x_train - mean) / std
    x_validation_normalized = (x_validation - mean) / std

    best_accuracy = -1.0
    best = None

    for penalty in [1e-4, 1e-3, 1e-2]:
        readout = torch.nn.Linear(Q.shape[1], 10).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)

        for _ in range(200):
            optimizer.zero_grad()
            logits = readout(x_train_normalized)
            loss = F.cross_entropy(logits, y_train)
            loss = loss + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prediction = readout(x_validation_normalized).argmax(dim=-1)
                accuracy = float((prediction == y_validation).float().mean())

            if accuracy > best_accuracy:
                weights = readout.weight.detach() / std.unsqueeze(0)
                bias = readout.bias.detach() - weights @ mean
                best_accuracy = accuracy
                best = {
                    "W_T": weights.cpu(),
                    "b_T": bias.cpu(),
                    "layer": int(layer),
                    "basis": Q.detach().cpu(),
                    "validation_accuracy": float(accuracy),
                    "penalty": float(penalty),
                }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best, save_path)

    print(f"[T readout layer {layer:02d}] validation_accuracy={best_accuracy:.4f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    parser.add_argument("--result-dir", default="results_addition_t")
    parser.add_argument("--layer", type=int, required=True)
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

    model, tokenizer = load_gemma_model(args.device)
    rows = load_addition_rows(model, tokenizer, args.data_dir, args.cache_dir, args.eval_batch_size, args.max_new_tokens, args.force_rebuild_cache)
    banks = build_all_banks(tokenizer, rows)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "das_t.pt"
    readout_path = result_dir / "t_readout_LogisticRegression.pt"

    result = train_t(model, tokenizer, banks["T"]["train"], banks["T"]["validation"], banks["T"]["test"], args.layer, args.dim, args.epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.seed, checkpoint_path)
    fit_t_readout(model, banks["T_factual"]["train"], banks["T_factual"]["validation"], result["basis"], args.layer, args.eval_batch_size, args.seed, readout_path)


if __name__ == "__main__":
    main()
