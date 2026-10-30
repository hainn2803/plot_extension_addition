"""Sanity checks for the controlled two-digit addition DAS experiment.

This script checks:
1. Dataset labels and controlled-pair invariants.
2. Stored model-filter predictions.
3. Answer tokenization / shared prefix.
4. T-only positive controls for DAS, readout, and mediator.
5. Recomputed C direct/readout/mediator metrics.
6. Whether C readout is accidentally scored against source/base T.
7. Whether generated T states depend on teacher-forced future answer tokens.

Run:
    python3 addition_sanity_check.py --result-dir results_addition_joint

Use the result directory for the run you want to audit.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

from addition_gradual_das_c import run_c_intervention_and_capture, run_frozen_t_mediator
from addition_gradual_das_t import build_pair_bank, build_state_inputs, build_teacher_forcing_batch, collect_layer_states, get_answer_prefix_ids, load_gemma_model, read_jsonl, run_das_intervention, sequence_iia, set_seed


def parse_problem(prompt):
    numbers = re.findall(r"\d+", prompt)

    if len(numbers) < 2:
        raise ValueError(f"Could not parse two operands from: {prompt}")

    return int(numbers[0]), int(numbers[1])


def compute_variables(a, b):
    a_tens = a // 10
    a_ones = a % 10
    b_tens = b // 10
    b_ones = b % 10

    C = 1 if a_ones + b_ones >= 10 else 0
    ones_out = (a_ones + b_ones) % 10
    T = (a_tens + b_tens + C) % 10
    hundreds_out = 1 if a_tens + b_tens + C >= 10 else 0
    Y = a + b

    return {
        "C": C,
        "T": T,
        "ones_out": ones_out,
        "hundreds_out": hundreds_out,
        "Y": Y,
    }


def expected_y_from_digits(T, ones_out, hundreds_out):
    return 100 * hundreds_out + 10 * T + ones_out


def add_error(errors, split, pair_type, index, field, observed, expected):
    errors.append({
        "split": split,
        "pair_type": pair_type,
        "index": index,
        "field": field,
        "observed": observed,
        "expected": expected,
    })


def audit_common_row(row, split, pair_type, index, errors):
    base_a, base_b = parse_problem(row["base_input"])
    source_a, source_b = parse_problem(row["source_input"])

    base = compute_variables(base_a, base_b)
    source = compute_variables(source_a, source_b)

    for field in ["C", "T", "Y", "ones_out", "hundreds_out"]:
        row_field = f"base_{field}"

        if row[row_field] != base[field]:
            add_error(errors, split, pair_type, index, row_field, row[row_field], base[field])

    for field in ["C", "T", "Y", "ones_out", "hundreds_out"]:
        row_field = f"source_{field}"

        if row[row_field] != source[field]:
            add_error(errors, split, pair_type, index, row_field, row[row_field], source[field])

    if row.get("base_model_prediction") != row["base_Y"]:
        add_error(errors, split, pair_type, index, "base_model_prediction", row.get("base_model_prediction"), row["base_Y"])

    if row.get("source_model_prediction") != row["source_Y"]:
        add_error(errors, split, pair_type, index, "source_model_prediction", row.get("source_model_prediction"), row["source_Y"])

    return base, source


def audit_c_only(rows, split):
    errors = []

    for index, row in enumerate(rows):
        base, source = audit_common_row(row, split, "C_only", index, errors)

        if row["source_C"] == row["base_C"]:
            add_error(errors, split, "C_only", index, "source_C != base_C", row["source_C"], f"not {row['base_C']}")

        if row["source_T"] != row["base_T"]:
            add_error(errors, split, "C_only", index, "source_T == base_T", row["source_T"], row["base_T"])

        if row["source_Y"] != row["base_Y"]:
            add_error(errors, split, "C_only", index, "source_Y == base_Y", row["source_Y"], row["base_Y"])

        if row["source_ones_out"] != row["base_ones_out"]:
            add_error(errors, split, "C_only", index, "source_ones_out == base_ones_out", row["source_ones_out"], row["base_ones_out"])

        if row["source_hundreds_out"] != row["base_hundreds_out"]:
            add_error(errors, split, "C_only", index, "source_hundreds_out == base_hundreds_out", row["source_hundreds_out"], row["base_hundreds_out"])

        target_C = row["source_C"]
        target_T = ((parse_problem(row["base_input"])[0] // 10) + (parse_problem(row["base_input"])[1] // 10) + target_C) % 10
        target_Y = expected_y_from_digits(target_T, base["ones_out"], base["hundreds_out"])

        if row["C_after_C_intervene"] != target_C:
            add_error(errors, split, "C_only", index, "C_after_C_intervene", row["C_after_C_intervene"], target_C)

        if row["T_after_C_intervene"] != target_T:
            add_error(errors, split, "C_only", index, "T_after_C_intervene", row["T_after_C_intervene"], target_T)

        if row["Y_after_C_intervene"] != target_Y:
            add_error(errors, split, "C_only", index, "Y_after_C_intervene", row["Y_after_C_intervene"], target_Y)

        if row["ones_out_after_C_intervene"] != base["ones_out"]:
            add_error(errors, split, "C_only", index, "ones_out_after_C_intervene", row["ones_out_after_C_intervene"], base["ones_out"])

        if row["hundreds_out_after_C_intervene"] != base["hundreds_out"]:
            add_error(errors, split, "C_only", index, "hundreds_out_after_C_intervene", row["hundreds_out_after_C_intervene"], base["hundreds_out"])

        if row["T_after_C_intervene"] == row["source_T"]:
            add_error(errors, split, "C_only", index, "target T must differ from source T", row["T_after_C_intervene"], f"not {row['source_T']}")

        if row["Y_after_C_intervene"] == row["source_Y"]:
            add_error(errors, split, "C_only", index, "target Y must differ from source Y", row["Y_after_C_intervene"], f"not {row['source_Y']}")

    return errors


def audit_t_only(rows, split):
    errors = []

    for index, row in enumerate(rows):
        base, source = audit_common_row(row, split, "T_only", index, errors)

        if row["source_C"] != row["base_C"]:
            add_error(errors, split, "T_only", index, "source_C == base_C", row["source_C"], row["base_C"])

        if row["source_T"] == row["base_T"]:
            add_error(errors, split, "T_only", index, "source_T != base_T", row["source_T"], f"not {row['base_T']}")

        if row["source_ones_out"] != row["base_ones_out"]:
            add_error(errors, split, "T_only", index, "source_ones_out == base_ones_out", row["source_ones_out"], row["base_ones_out"])

        if row["source_hundreds_out"] != row["base_hundreds_out"]:
            add_error(errors, split, "T_only", index, "source_hundreds_out == base_hundreds_out", row["source_hundreds_out"], row["base_hundreds_out"])

        target_C = row["base_C"]
        target_T = row["source_T"]
        target_Y = expected_y_from_digits(target_T, base["ones_out"], base["hundreds_out"])

        if row["C_after_T_intervene"] != target_C:
            add_error(errors, split, "T_only", index, "C_after_T_intervene", row["C_after_T_intervene"], target_C)

        if row["T_after_T_intervene"] != target_T:
            add_error(errors, split, "T_only", index, "T_after_T_intervene", row["T_after_T_intervene"], target_T)

        if row["Y_after_T_intervene"] != target_Y:
            add_error(errors, split, "T_only", index, "Y_after_T_intervene", row["Y_after_T_intervene"], target_Y)

        if row["Y_after_T_intervene"] != row["source_Y"]:
            add_error(errors, split, "T_only", index, "Y_after_T_intervene == source_Y", row["Y_after_T_intervene"], row["source_Y"])

    return errors


def audit_c_t(rows, split):
    errors = []

    for index, row in enumerate(rows):
        base, source = audit_common_row(row, split, "C_T", index, errors)

        if row["source_C"] == row["base_C"]:
            add_error(errors, split, "C_T", index, "source_C != base_C", row["source_C"], f"not {row['base_C']}")

        if row["source_T"] == row["base_T"]:
            add_error(errors, split, "C_T", index, "source_T != base_T", row["source_T"], f"not {row['base_T']}")

        if row["source_ones_out"] != row["base_ones_out"]:
            add_error(errors, split, "C_T", index, "source_ones_out == base_ones_out", row["source_ones_out"], row["base_ones_out"])

        target_C = row["source_C"]
        target_T = ((parse_problem(row["base_input"])[0] // 10) + (parse_problem(row["base_input"])[1] // 10) + target_C) % 10
        target_Y = expected_y_from_digits(target_T, base["ones_out"], base["hundreds_out"])

        if row["C_after_C_intervene"] != target_C:
            add_error(errors, split, "C_T", index, "C_after_C_intervene", row["C_after_C_intervene"], target_C)

        if row["T_after_C_intervene"] != target_T:
            add_error(errors, split, "C_T", index, "T_after_C_intervene", row["T_after_C_intervene"], target_T)

        if row["Y_after_C_intervene"] != target_Y:
            add_error(errors, split, "C_T", index, "Y_after_C_intervene", row["Y_after_C_intervene"], target_Y)

        if row["T_after_C_intervene"] != row["source_T"]:
            add_error(errors, split, "C_T", index, "T_after_C_intervene == source_T", row["T_after_C_intervene"], row["source_T"])

        if row["Y_after_C_intervene"] != row["source_Y"]:
            add_error(errors, split, "C_T", index, "Y_after_C_intervene == source_Y", row["Y_after_C_intervene"], row["source_Y"])

    return errors


def print_errors(errors, limit=10):
    if len(errors) == 0:
        return

    print(f"Showing first {min(limit, len(errors))} errors:")

    for error in errors[:limit]:
        print(json.dumps(error))


def answer_content_length_distribution(tokenizer, prefix_ids):
    lengths = Counter()

    for value in range(10, 100):
        ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]
        content_ids = ids[len(prefix_ids):]
        lengths[len(content_ids)] += 1

    return lengths


def capture_c_to_t_prefix_only(model, tokenizer, rows, prefix_ids, source_c, Q_c, c_layer, t_layer, batch_size):
    device = next(model.parameters()).device
    base_prompts = []

    for row in rows:
        base_prompts.append(row["base_input"])

    input_ids, attention_mask, positions = build_state_inputs(tokenizer, base_prompts, prefix_ids)
    all_t = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        position = positions[start:end].to(device)
        donor = source_c[start:end].to(device)
        rows_index = torch.arange(len(ids), device=device)
        captured = {}

        def c_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_state = hidden[rows_index, position, :].float()
            delta = ((donor.float() - base_state) @ Q_c) @ Q_c.T
            new_hidden[rows_index, position, :] = (base_state + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def t_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            captured["t"] = hidden[rows_index, position, :].detach().float().cpu()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            c_handle.remove()
            t_handle.remove()

        all_t.append(captured["t"])

    return torch.cat(all_t, dim=0)


def first_token_accuracy(logits, target_ids):
    prediction = logits[:, 0, :].argmax(-1).detach().cpu()
    target = target_ids[:, 0].detach().cpu()
    return float((prediction == target).float().mean())


def build_first_content_token_targets(tokenizer, values, prefix_ids):
    targets = []

    for value in values:
        ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]
        content_ids = ids[len(prefix_ids):]

        if len(content_ids) == 0:
            raise ValueError(f"No answer content token for {value}")

        targets.append(content_ids[0])

    return torch.tensor(targets, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_filtered_pairs")
    parser.add_argument("--result-dir", default="results_addition_joint")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    print("============================================================")
    print("1. DATASET SANITY CHECK")
    print("============================================================")

    all_errors = []

    for split in ["train", "val", "test"]:
        c_rows = read_jsonl(data_dir / f"{split}_C_only.jsonl")
        t_rows = read_jsonl(data_dir / f"{split}_T_only.jsonl")
        ct_rows = read_jsonl(data_dir / f"{split}_C_T.jsonl")

        c_errors = audit_c_only(c_rows, split)
        t_errors = audit_t_only(t_rows, split)
        ct_errors = audit_c_t(ct_rows, split)

        print(f"{split.upper():5s} C_only: rows={len(c_rows):4d}, errors={len(c_errors)}")
        print(f"{split.upper():5s} T_only: rows={len(t_rows):4d}, errors={len(t_errors)}")
        print(f"{split.upper():5s} C_T:    rows={len(ct_rows):4d}, errors={len(ct_errors)}")

        all_errors.extend(c_errors)
        all_errors.extend(t_errors)
        all_errors.extend(ct_errors)

    if len(all_errors) > 0:
        print_errors(all_errors)
        raise RuntimeError(f"Dataset sanity check failed with {len(all_errors)} errors.")

    print("PASS: all dataset labels and controlled-pair invariants are correct.")

    print("\n============================================================")
    print("2. TOKENIZATION SANITY CHECK")
    print("============================================================")

    model, tokenizer = load_gemma_model(args.device)
    prefix_ids = get_answer_prefix_ids(tokenizer)
    device = next(model.parameters()).device

    length_distribution = answer_content_length_distribution(tokenizer, prefix_ids)

    print("Answer content-token length distribution for 10..99:")
    print(dict(sorted(length_distribution.items())))
    print("Shared prefix ids:", prefix_ids)
    print("Shared prefix text:", repr(tokenizer.decode(prefix_ids)))

    print("\n============================================================")
    print("3. LOAD CHECKPOINTS")
    print("============================================================")

    t_checkpoint = torch.load(result_dir / "das_t.pt", map_location="cpu", weights_only=False)
    t_readout = torch.load(result_dir / "t_readout_LogisticRegression.pt", map_location="cpu", weights_only=False)
    c_checkpoint = torch.load(result_dir / "das_c.pt", map_location="cpu", weights_only=False)

    Q_t = t_checkpoint["basis"].float().to(device)
    t_layer = int(t_checkpoint["layer"])

    W_t = t_readout["W_T"].float().to(device)
    b_t = t_readout["b_T"].float().to(device)

    Q_c = c_checkpoint["basis"].float().to(device)
    c_layer = int(c_checkpoint["layer"])

    print(f"T layer = {t_layer}")
    print(f"C layer = {c_layer}")
    print(f"Saved T test_iia = {float(t_checkpoint['test_iia']):.4f}")
    print(f"Saved C test_direct_iia = {float(c_checkpoint['test_direct_iia']):.4f}")
    print(f"Saved C test_T_mediator_iia = {float(c_checkpoint['test_T_mediator_iia']):.4f}")
    print(f"Saved C test_T_readout_acc = {float(c_checkpoint['test_T_readout_acc']):.4f}")

    print("\n============================================================")
    print("4. T-ONLY POSITIVE CONTROL")
    print("============================================================")

    t_rows = read_jsonl(data_dir / "test_T_only.jsonl")
    t_bank = build_pair_bank(tokenizer, t_rows, "T", prefix_ids)

    t_source = collect_layer_states(model, t_bank["source_input_ids"], t_bank["source_attention_mask"], t_bank["source_positions"], t_layer, args.eval_batch_size)

    t_das_logits = run_das_intervention(model, t_bank, t_source, Q_t, t_layer, args.eval_batch_size, False)
    t_mediator_logits = run_frozen_t_mediator(model, t_bank, t_source, Q_t, t_layer, args.eval_batch_size, False)

    t_das_iia = sequence_iia(t_das_logits, t_bank["target_answer_ids"], t_bank["target_answer_mask"])
    t_mediator_iia = sequence_iia(t_mediator_logits, t_bank["target_answer_ids"], t_bank["target_answer_mask"])
    max_t_logit_difference = float((t_das_logits - t_mediator_logits).abs().max())

    natural_t_logits = (t_source.to(device) @ Q_t) @ W_t.T + b_t
    natural_t_prediction = natural_t_logits.argmax(-1).cpu()
    natural_t_readout_acc = float((natural_t_prediction == t_bank["target_t"]).float().mean())

    print(f"Saved T test_iia:                    {float(t_checkpoint['test_iia']):.4f}")
    print(f"Recomputed T DAS IIA:                {t_das_iia:.4f}")
    print(f"Same natural T states via mediator:  {t_mediator_iia:.4f}")
    print(f"Natural source T readout accuracy:   {natural_t_readout_acc:.4f}")
    print(f"Max |T DAS logits - mediator logits|:{max_t_logit_difference:.8f}")

    if abs(t_das_iia - float(t_checkpoint["test_iia"])) > 1e-6:
        raise RuntimeError("Recomputed T DAS IIA does not match saved T test_iia.")

    if abs(t_mediator_iia - t_das_iia) > 1e-6:
        raise RuntimeError("Mediator does not reproduce T DAS on the same natural T states.")

    if max_t_logit_difference > 1e-5:
        raise RuntimeError("T DAS and mediator logits differ on the same natural T states.")

    print("PASS: T DAS, T mediator, and T readout machinery are internally consistent.")

    print("\n============================================================")
    print("5. RECOMPUTE C METRICS FROM SCRATCH")
    print("============================================================")

    c_rows = read_jsonl(data_dir / "test_C_only.jsonl")
    c_bank = build_pair_bank(tokenizer, c_rows, "C", prefix_ids)

    c_source = collect_layer_states(model, c_bank["source_input_ids"], c_bank["source_attention_mask"], c_bank["source_positions"], c_layer, args.eval_batch_size)

    direct_logits, generated_t = run_c_intervention_and_capture(model, c_bank, c_source, Q_c, c_layer, t_layer, args.eval_batch_size, False)
    mediator_logits = run_frozen_t_mediator(model, c_bank, generated_t, Q_t, t_layer, args.eval_batch_size, False)

    direct_iia = sequence_iia(direct_logits, c_bank["target_answer_ids"], c_bank["target_answer_mask"])
    mediator_iia = sequence_iia(mediator_logits, c_bank["target_answer_ids"], c_bank["target_answer_mask"])

    generated_t_logits = (generated_t.to(device) @ Q_t) @ W_t.T + b_t
    generated_t_prediction = generated_t_logits.argmax(-1).cpu()

    target_t = c_bank["target_t"].cpu()
    source_t = torch.tensor([row["source_T"] for row in c_rows], dtype=torch.long)
    base_t = torch.tensor([row["base_T"] for row in c_rows], dtype=torch.long)

    readout_target_acc = float((generated_t_prediction == target_t).float().mean())
    readout_source_acc = float((generated_t_prediction == source_t).float().mean())
    readout_base_acc = float((generated_t_prediction == base_t).float().mean())
    target_equals_source = float((target_t == source_t).float().mean())

    print(f"Recomputed C direct IIA:             {direct_iia:.4f}")
    print(f"Recomputed C -> T mediator IIA:      {mediator_iia:.4f}")
    print(f"Recomputed C -> T readout TARGET:    {readout_target_acc:.4f}")
    print(f"C -> T readout against SOURCE T:    {readout_source_acc:.4f}")
    print(f"C -> T readout against BASE T:      {readout_base_acc:.4f}")
    print(f"Fraction target_T == source_T:       {target_equals_source:.4f}")

    print("\nSaved checkpoint:")
    print(f"Saved C direct IIA:                  {float(c_checkpoint['test_direct_iia']):.4f}")
    print(f"Saved C -> T mediator IIA:           {float(c_checkpoint['test_T_mediator_iia']):.4f}")
    print(f"Saved C -> T readout accuracy:       {float(c_checkpoint['test_T_readout_acc']):.4f}")

    if abs(direct_iia - float(c_checkpoint["test_direct_iia"])) > 1e-6:
        raise RuntimeError("Recomputed C direct IIA does not match checkpoint.")

    if abs(mediator_iia - float(c_checkpoint["test_T_mediator_iia"])) > 1e-6:
        raise RuntimeError("Recomputed C mediator IIA does not match checkpoint.")

    if abs(readout_target_acc - float(c_checkpoint["test_T_readout_acc"])) > 1e-6:
        raise RuntimeError("Recomputed C readout accuracy does not match checkpoint.")

    if target_equals_source != 0.0:
        raise RuntimeError("C_only target_T unexpectedly equals source_T for some rows.")

    print("PASS: saved C metrics reproduce exactly from the raw test pairs and checkpoints.")

    print("\n============================================================")
    print("6. WRONG-LABEL CHECK")
    print("============================================================")

    print("For C_only, target T and source/base T must differ.")
    print(f"Accuracy vs correct T_after_C: {readout_target_acc:.4f}")
    print(f"Accuracy vs wrong source_T:    {readout_source_acc:.4f}")
    print(f"Accuracy vs wrong base_T:      {readout_base_acc:.4f}")

    if readout_target_acc <= readout_source_acc:
        raise RuntimeError("C readout is not more aligned with T_after_C than with source_T.")

    print("PASS: T readout is being scored against the counterfactual target, not the donor/base T.")

    print("\n============================================================")
    print("7. TEACHER-FORCING LEAK CHECK FOR generated_t")
    print("============================================================")

    generated_t_prefix_only = capture_c_to_t_prefix_only(model, tokenizer, c_rows, prefix_ids, c_source, Q_c, c_layer, t_layer, args.eval_batch_size)

    max_state_difference = float((generated_t - generated_t_prefix_only).abs().max())
    mean_state_difference = float((generated_t - generated_t_prefix_only).abs().mean())

    print(f"Max state difference:  {max_state_difference:.8f}")
    print(f"Mean state difference: {mean_state_difference:.8f}")

    if max_state_difference > 1e-5:
        raise RuntimeError("generated_t changes when future teacher-forced answer tokens are removed.")

    print("PASS: generated_t does not depend on future teacher-forced answer tokens.")

    print("\n============================================================")
    print("8. FIRST ANSWER-TOKEN CHECK")
    print("============================================================")

    target_y_values = [row["Y_after_C_intervene"] for row in c_rows]
    source_y_values = [row["source_Y"] for row in c_rows]

    target_first_ids = build_first_content_token_targets(tokenizer, target_y_values, prefix_ids)
    source_first_ids = build_first_content_token_targets(tokenizer, source_y_values, prefix_ids)

    direct_first_target_acc = first_token_accuracy(direct_logits, target_first_ids)
    direct_first_source_acc = first_token_accuracy(direct_logits, source_first_ids)
    mediator_first_target_acc = first_token_accuracy(mediator_logits, target_first_ids)
    mediator_first_source_acc = first_token_accuracy(mediator_logits, source_first_ids)

    print(f"Direct first-token acc vs TARGET Y_after_C: {direct_first_target_acc:.4f}")
    print(f"Direct first-token acc vs SOURCE/Base Y:    {direct_first_source_acc:.4f}")
    print(f"Mediator first-token acc vs TARGET Y_after_C:{mediator_first_target_acc:.4f}")
    print(f"Mediator first-token acc vs SOURCE/Base Y:   {mediator_first_source_acc:.4f}")

    print("\n============================================================")
    print("ALL SANITY CHECKS PASSED")
    print("============================================================")
    print("If this line is reached, then:")
    print("- dataset labels are recomputed correctly from the arithmetic problems")
    print("- C_only/T_only/C_T invariants hold")
    print("- stored model-filter predictions match natural answers")
    print("- T mediator reproduces T DAS on the same donor states")
    print("- C direct/readout/mediator metrics reproduce the saved checkpoint exactly")
    print("- C readout is scored against T_after_C, not source/base T")
    print("- generated_t is not leaking future teacher-forced answer tokens")
    print("- first-answer-token target/source accuracies are printed for an extra output sanity check")


if __name__ == "__main__":
    main()
