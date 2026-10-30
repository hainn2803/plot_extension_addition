import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def make_prompt(a, b):
    return f"Q: How much is {a} plus {b}? A:"


def get_variables(a, b):
    a_tens = a // 10
    a_ones = a % 10
    b_tens = b // 10
    b_ones = b % 10
    carry = 1 if a_ones + b_ones >= 10 else 0
    ones_out = (a_ones + b_ones) % 10
    tens_sum = a_tens + b_tens
    tens_out = (tens_sum + carry) % 10
    hundreds_out = 1 if tens_sum + carry >= 10 else 0
    answer = a + b

    return {
        "a": a,
        "b": b,
        "a_tens": a_tens,
        "a_ones": a_ones,
        "b_tens": b_tens,
        "b_ones": b_ones,
        "C": carry,
        "T": tens_out,
        "ones_out": ones_out,
        "hundreds_out": hundreds_out,
        "Y": answer,
        "prompt": make_prompt(a, b),
    }


def build_problem_pool():
    problems = []

    for a in range(10, 100):
        for b in range(10, 100):
            problem = get_variables(a, b)

            if problem["hundreds_out"] != 0:
                continue

            tens_sum = problem["a_tens"] + problem["b_tens"]
            if tens_sum == 9:
                continue

            problems.append(problem)

    return problems


def split_problem_pool(problems, seed, train_fraction=0.60, validation_fraction=0.20):
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Expected positive train/validation fractions with sum < 1")

    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def build_indices(problems):
    c_only_index = defaultdict(list)
    t_only_index = defaultdict(list)
    c_t_index = defaultdict(list)

    for problem in problems:
        c_only_key = (problem["ones_out"], problem["T"], problem["C"])
        t_only_key = (problem["ones_out"], problem["C"])
        c_t_key = (problem["a_tens"], problem["b_tens"], problem["ones_out"], problem["C"])

        c_only_index[c_only_key].append(problem)
        t_only_index[t_only_key].append(problem)
        c_t_index[c_t_key].append(problem)

    return c_only_index, t_only_index, c_t_index


def get_candidates(base, c_only_index, t_only_index, c_t_index):
    source_c = 1 - base["C"]

    c_only_key = (base["ones_out"], base["T"], source_c)
    c_only_candidates = list(c_only_index[c_only_key])

    t_only_key = (base["ones_out"], base["C"])
    t_only_candidates = []
    for candidate in t_only_index[t_only_key]:
        if candidate["T"] != base["T"]:
            t_only_candidates.append(candidate)

    c_t_key = (base["a_tens"], base["b_tens"], base["ones_out"], source_c)
    c_t_candidates = list(c_t_index[c_t_key])

    return c_only_candidates, t_only_candidates, c_t_candidates


def compact_problem(problem):
    return {
        "prompt": problem["prompt"],
        "a": problem["a"],
        "b": problem["b"],
        "C": problem["C"],
        "T": problem["T"],
        "ones_out": problem["ones_out"],
        "hundreds_out": problem["hundreds_out"],
        "Y": problem["Y"],
    }


def compute_after_c_intervention(base, source_c):
    tens_sum = base["a_tens"] + base["b_tens"]
    tens_out = (tens_sum + source_c) % 10
    hundreds_out = 1 if tens_sum + source_c >= 10 else 0
    answer = 100 * hundreds_out + 10 * tens_out + base["ones_out"]

    return {
        "C": source_c,
        "T": tens_out,
        "ones_out": base["ones_out"],
        "hundreds_out": hundreds_out,
        "Y": answer,
    }


def compute_after_t_intervention(base, source_t):
    answer = 100 * base["hundreds_out"] + 10 * source_t + base["ones_out"]

    return {
        "C": base["C"],
        "T": source_t,
        "ones_out": base["ones_out"],
        "hundreds_out": base["hundreds_out"],
        "Y": answer,
    }


def build_row(base, c_source, t_source, c_t_source, split_name, repeat_idx):
    c_target = compute_after_c_intervention(base, c_source["C"])
    t_target = compute_after_t_intervention(base, t_source["T"])
    c_t_target = compute_after_c_intervention(base, c_t_source["C"])

    return {
        "split": split_name,
        "repeat_idx": repeat_idx,
        "base": compact_problem(base),
        "C_only_counterfactual": {
            "source": compact_problem(c_source),
            "target_on_base": c_target,
        },
        "T_only_counterfactual": {
            "source": compact_problem(t_source),
            "target_on_base": t_target,
        },
        "C_T_counterfactual": {
            "source": compact_problem(c_t_source),
            "target_on_base": c_t_target,
        },
    }


def assert_row(row):
    base = row["base"]

    c_source = row["C_only_counterfactual"]["source"]
    c_target = row["C_only_counterfactual"]["target_on_base"]
    assert c_source["C"] != base["C"]
    assert c_source["T"] == base["T"]
    assert c_source["ones_out"] == base["ones_out"]
    assert c_source["Y"] == base["Y"]
    assert c_target["C"] == c_source["C"]
    assert c_target["T"] != base["T"]
    assert c_target["Y"] != base["Y"]
    assert c_target["ones_out"] == base["ones_out"]
    assert c_target["hundreds_out"] == base["hundreds_out"]

    t_source = row["T_only_counterfactual"]["source"]
    t_target = row["T_only_counterfactual"]["target_on_base"]
    assert t_source["C"] == base["C"]
    assert t_source["T"] != base["T"]
    assert t_source["ones_out"] == base["ones_out"]
    assert t_target["C"] == base["C"]
    assert t_target["T"] == t_source["T"]
    assert t_target["Y"] == t_source["Y"]
    assert t_target["ones_out"] == base["ones_out"]
    assert t_target["hundreds_out"] == base["hundreds_out"]

    c_t_source = row["C_T_counterfactual"]["source"]
    c_t_target = row["C_T_counterfactual"]["target_on_base"]
    assert c_t_source["C"] != base["C"]
    assert c_t_source["T"] != base["T"]
    assert c_t_source["ones_out"] == base["ones_out"]
    assert c_t_target["C"] == c_t_source["C"]
    assert c_t_target["T"] == c_t_source["T"]
    assert c_t_target["Y"] == c_t_source["Y"]


def choose_balanced_bases(eligible, num_unique_bases, rng, split_name):
    carry_zero = []
    carry_one = []

    for problem in eligible:
        if problem["C"] == 0:
            carry_zero.append(problem)
        else:
            carry_one.append(problem)

    rng.shuffle(carry_zero)
    rng.shuffle(carry_one)

    num_zero = num_unique_bases // 2
    num_one = num_unique_bases - num_zero

    if len(carry_zero) < num_zero or len(carry_one) < num_one:
        raise ValueError(
            f"{split_name}: not enough eligible unique bases. "
            f"Need C=0:{num_zero}, C=1:{num_one}; available C=0:{len(carry_zero)}, C=1:{len(carry_one)}. "
            "Increase --repeats-per-base or change the split seed."
        )

    selected = carry_zero[:num_zero] + carry_one[:num_one]
    rng.shuffle(selected)
    return selected


def build_rows(split_problems, num_rows, seed, split_name, repeats_per_base=3):
    if repeats_per_base < 1:
        raise ValueError("repeats_per_base must be positive")

    rng = random.Random(seed)
    c_only_index, t_only_index, c_t_index = build_indices(split_problems)
    eligible = []

    for base in split_problems:
        c_candidates, t_candidates, c_t_candidates = get_candidates(base, c_only_index, t_only_index, c_t_index)
        if len(c_candidates) > 0 and len(t_candidates) > 0 and len(c_t_candidates) > 0:
            eligible.append(base)

    num_unique_bases = math.ceil(num_rows / repeats_per_base)
    selected_bases = choose_balanced_bases(eligible, num_unique_bases, rng, split_name)
    rows = []

    for base in selected_bases:
        c_candidates, t_candidates, c_t_candidates = get_candidates(base, c_only_index, t_only_index, c_t_index)

        for repeat_idx in range(repeats_per_base):
            if len(rows) >= num_rows:
                break

            c_source = rng.choice(c_candidates)
            t_source = rng.choice(t_candidates)
            c_t_source = rng.choice(c_t_candidates)
            row = build_row(base, c_source, t_source, c_t_source, split_name, repeat_idx)
            assert_row(row)
            rows.append(row)

    rng.shuffle(rows)

    if len(rows) != num_rows:
        raise RuntimeError(f"{split_name}: expected {num_rows} rows, built {len(rows)}")

    return rows


def all_prompts(rows):
    prompts = set()

    for row in rows:
        prompts.add(row["base"]["prompt"])
        prompts.add(row["C_only_counterfactual"]["source"]["prompt"])
        prompts.add(row["T_only_counterfactual"]["source"]["prompt"])
        prompts.add(row["C_T_counterfactual"]["source"]["prompt"])

    return prompts


def check_prompt_disjointness(dataset_rows):
    prompt_sets = {}

    for split_name, rows in dataset_rows.items():
        prompt_sets[split_name] = all_prompts(rows)
        unique_bases = {row["base"]["prompt"] for row in rows}
        print(f"[{split_name}] rows={len(rows)} unique_bases={len(unique_bases)} all_prompts={len(prompt_sets[split_name])}")

    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = prompt_sets[left] & prompt_sets[right]
        if overlap:
            raise RuntimeError(f"Prompt leakage between {left} and {right}: {sorted(overlap)[:5]}")

    print("[split] all base/source prompts are disjoint across train/validation/test")


def write_jsonl(rows, path):
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=1200)
    parser.add_argument("--num-validation", type=int, default=300)
    parser.add_argument("--num-test", type=int, default=300)
    parser.add_argument("--repeats-per-base", type=int, default=3)
    parser.add_argument("--train-pool-fraction", type=float, default=0.60)
    parser.add_argument("--validation-pool-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="addition_controlled_data")
    args = parser.parse_args()

    problems = build_problem_pool()
    split_pools = split_problem_pool(problems, args.seed, args.train_pool_fraction, args.validation_pool_fraction)

    dataset_rows = {}
    dataset_rows["train"] = build_rows(split_pools["train"], args.num_train, args.seed + 11, "train", args.repeats_per_base)
    dataset_rows["validation"] = build_rows(split_pools["validation"], args.num_validation, args.seed + 23, "validation", args.repeats_per_base)
    dataset_rows["test"] = build_rows(split_pools["test"], args.num_test, args.seed + 37, "test", args.repeats_per_base)

    check_prompt_disjointness(dataset_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        write_jsonl(dataset_rows[split_name], output_dir / f"{split_name}.jsonl")

    print(f"Problem pool: {len(problems)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
