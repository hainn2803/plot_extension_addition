import argparse
import json
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

    C = 1 if a_ones + b_ones >= 10 else 0
    ones_out = (a_ones + b_ones) % 10
    tens_sum = a_tens + b_tens
    T = (tens_sum + C) % 10
    hundreds_out = 1 if tens_sum + C >= 10 else 0
    Y = a + b

    return {
        "a": a,
        "b": b,
        "a_tens": a_tens,
        "a_ones": a_ones,
        "b_tens": b_tens,
        "b_ones": b_ones,
        "C": C,
        "T": T,
        "ones_out": ones_out,
        "hundreds_out": hundreds_out,
        "Y": Y,
        "prompt": make_prompt(a, b),
    }


def build_problem_pool():
    problems = []

    for a in range(10, 100):
        for b in range(10, 100):
            problem = get_variables(a, b)

            # Keep the experiment on two-digit outputs only.
            if problem["hundreds_out"] != 0:
                continue

            # If a_tens + b_tens == 9, changing C can change hundreds_out.
            # Exclude those examples so the focused path is C -> T -> Y.
            tens_sum = problem["a_tens"] + problem["b_tens"]

            if tens_sum == 9:
                continue

            problems.append(problem)

    return problems


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
    source_C = 1 - base["C"]

    c_only_key = (base["ones_out"], base["T"], source_C)
    c_only_candidates = c_only_index[c_only_key]

    t_only_key = (base["ones_out"], base["C"])
    t_only_candidates = []

    for candidate in t_only_index[t_only_key]:
        if candidate["T"] != base["T"]:
            t_only_candidates.append(candidate)

    c_t_key = (base["a_tens"], base["b_tens"], base["ones_out"], source_C)
    c_t_candidates = c_t_index[c_t_key]

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


def compute_after_C_intervention(base, source_C):
    tens_sum = base["a_tens"] + base["b_tens"]
    T_after_C = (tens_sum + source_C) % 10
    hundreds_after_C = 1 if tens_sum + source_C >= 10 else 0

    if hundreds_after_C == 0:
        Y_after_C = 10 * T_after_C + base["ones_out"]
    else:
        Y_after_C = 100 + 10 * T_after_C + base["ones_out"]

    return {
        "C": source_C,
        "T": T_after_C,
        "ones_out": base["ones_out"],
        "hundreds_out": hundreds_after_C,
        "Y": Y_after_C,
    }


def compute_after_T_intervention(base, source_T):
    if base["hundreds_out"] == 0:
        Y_after_T = 10 * source_T + base["ones_out"]
    else:
        Y_after_T = 100 + 10 * source_T + base["ones_out"]

    return {
        "C": base["C"],
        "T": source_T,
        "ones_out": base["ones_out"],
        "hundreds_out": base["hundreds_out"],
        "Y": Y_after_T,
    }


def build_row(base, c_only_source, t_only_source, c_t_source):
    after_C = compute_after_C_intervention(base, c_only_source["C"])
    after_T = compute_after_T_intervention(base, t_only_source["T"])
    after_C_T = compute_after_C_intervention(base, c_t_source["C"])

    row = {
        "base": compact_problem(base),
        "C_only_counterfactual": {
            "source": compact_problem(c_only_source),
            "target_on_base": after_C,
        },
        "T_only_counterfactual": {
            "source": compact_problem(t_only_source),
            "target_on_base": after_T,
        },
        "C_T_counterfactual": {
            "source": compact_problem(c_t_source),
            "target_on_base": after_C_T,
        },
    }

    return row


def assert_row(row):
    base = row["base"]

    c_source = row["C_only_counterfactual"]["source"]
    c_target = row["C_only_counterfactual"]["target_on_base"]

    assert c_source["C"] != base["C"]
    assert c_source["T"] == base["T"]
    assert c_source["ones_out"] == base["ones_out"]
    assert c_source["hundreds_out"] == base["hundreds_out"]
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
    assert t_source["hundreds_out"] == base["hundreds_out"]

    assert t_target["C"] == base["C"]
    assert t_target["T"] == t_source["T"]
    assert t_target["Y"] == t_source["Y"]
    assert t_target["ones_out"] == base["ones_out"]
    assert t_target["hundreds_out"] == base["hundreds_out"]

    ct_source = row["C_T_counterfactual"]["source"]
    ct_target = row["C_T_counterfactual"]["target_on_base"]

    assert ct_source["C"] != base["C"]
    assert ct_source["T"] != base["T"]
    assert ct_source["ones_out"] == base["ones_out"]
    assert ct_source["hundreds_out"] == base["hundreds_out"]

    assert ct_target["C"] == ct_source["C"]
    assert ct_target["T"] == ct_source["T"]
    assert ct_target["Y"] == ct_source["Y"]


def build_rows(problems, num_rows, seed):
    random.seed(seed)

    c_only_index, t_only_index, c_t_index = build_indices(problems)

    carry_zero = []
    carry_one = []

    for base in problems:
        c_only_candidates, t_only_candidates, c_t_candidates = get_candidates(base, c_only_index, t_only_index, c_t_index)

        if len(c_only_candidates) == 0:
            continue

        if len(t_only_candidates) == 0:
            continue

        if len(c_t_candidates) == 0:
            continue

        if base["C"] == 0:
            carry_zero.append(base)
        else:
            carry_one.append(base)

    random.shuffle(carry_zero)
    random.shuffle(carry_one)

    num_zero = num_rows // 2
    num_one = num_rows - num_zero

    if len(carry_zero) < num_zero or len(carry_one) < num_one:
        raise ValueError(f"Not enough eligible bases. Available C=0: {len(carry_zero)}, C=1: {len(carry_one)}")

    selected_bases = carry_zero[:num_zero] + carry_one[:num_one]
    random.shuffle(selected_bases)

    rows = []

    for base in selected_bases:
        c_only_candidates, t_only_candidates, c_t_candidates = get_candidates(base, c_only_index, t_only_index, c_t_index)

        c_only_source = random.choice(c_only_candidates)
        t_only_source = random.choice(t_only_candidates)
        c_t_source = random.choice(c_t_candidates)

        row = build_row(base, c_only_source, t_only_source, c_t_source)
        assert_row(row)
        rows.append(row)

    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def print_example(row):
    base = row["base"]

    print("\nBASE")
    print(base["prompt"], base["Y"])
    print("C =", base["C"], "T =", base["T"], "ones_out =", base["ones_out"])

    for name in ["C_only_counterfactual", "T_only_counterfactual", "C_T_counterfactual"]:
        source = row[name]["source"]
        target = row[name]["target_on_base"]

        print("\n" + name)
        print("source:", source["prompt"], source["Y"])
        print("source C =", source["C"], "source T =", source["T"])
        print("target on base:", target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=1200)
    parser.add_argument("--num-val", type=int, default=300)
    parser.add_argument("--num-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="addition_controlled_data")
    args = parser.parse_args()

    total_rows = args.num_train + args.num_val + args.num_test

    problems = build_problem_pool()
    rows = build_rows(problems, total_rows, args.seed)

    random.seed(args.seed)
    random.shuffle(rows)

    train_rows = rows[:args.num_train]
    val_rows = rows[args.num_train:args.num_train + args.num_val]
    test_rows = rows[args.num_train + args.num_val:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_rows, output_dir / "train.jsonl")
    write_jsonl(val_rows, output_dir / "val.jsonl")
    write_jsonl(test_rows, output_dir / "test.jsonl")

    print("Problem pool:", len(problems))
    print("Train:", len(train_rows))
    print("Val:", len(val_rows))
    print("Test:", len(test_rows))
    print("Saved to:", output_dir)

    print_example(train_rows[0])


if __name__ == "__main__":
    main()
