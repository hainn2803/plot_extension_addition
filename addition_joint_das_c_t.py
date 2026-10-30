"""Train T first, fit T readout, then train C."""

import argparse
from pathlib import Path

from addition_gradual_das_c import train_c
from addition_gradual_das_t import fit_t_readout, get_answer_prefix_ids, load_gemma_model, parse_layers, read_jsonl, set_seed, train_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_filtered_pairs")
    parser.add_argument("--result-dir", default="results_addition_joint_gradual_all2.0")

    parser.add_argument("--t-layers", default="all")
    parser.add_argument("--c-layers", default="all")

    parser.add_argument("--t-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)

    parser.add_argument("--t-epochs", type=int, default=20)
    parser.add_argument("--c-epochs", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--c-lambda-med", type=float, default=2.0)
    parser.add_argument("--c-lambda-readout", type=float, default=2.0)

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    t_train_rows = read_jsonl(data_dir / "train_T_only.jsonl")
    t_val_rows = read_jsonl(data_dir / "val_T_only.jsonl")
    t_test_rows = read_jsonl(data_dir / "test_T_only.jsonl")

    c_train_rows = read_jsonl(data_dir / "train_C_only.jsonl")
    c_val_rows = read_jsonl(data_dir / "val_C_only.jsonl")
    c_test_rows = read_jsonl(data_dir / "test_C_only.jsonl")
    c_t_test_rows = read_jsonl(data_dir / "test_C_T.jsonl")

    model, tokenizer = load_gemma_model(args.device)
    prefix_ids = get_answer_prefix_ids(tokenizer)

    t_checkpoint = result_dir / "das_t.pt"
    t_readout = result_dir / "t_readout_LogisticRegression.pt"
    c_checkpoint = result_dir / "das_c.pt"

    print("\n=== TRAIN T ===")

    train_t(
        model,
        tokenizer,
        t_train_rows,
        t_val_rows,
        t_test_rows,
        prefix_ids,
        parse_layers(args.t_layers),
        args.t_dim,
        args.t_epochs,
        args.lr,
        args.train_batch_size,
        args.eval_batch_size,
        args.seed,
        str(t_checkpoint),
    )

    print("\n=== FIT T READOUT ===")

    fit_t_readout(
        model,
        str(t_checkpoint),
        t_train_rows,
        t_val_rows,
        tokenizer,
        prefix_ids,
        args.eval_batch_size,
        str(t_readout),
        args.seed,
    )

    print("\n=== TRAIN C ===")

    train_c(
        model,
        tokenizer,
        c_train_rows,
        c_val_rows,
        c_test_rows,
        c_t_test_rows,
        prefix_ids,
        str(t_checkpoint),
        str(t_readout),
        parse_layers(args.c_layers),
        args.c_dim,
        args.c_epochs,
        args.lr,
        args.c_lambda_med,
        args.c_lambda_readout,
        args.train_batch_size,
        args.eval_batch_size,
        args.seed,
        str(c_checkpoint),
    )


if __name__ == "__main__":
    main()
