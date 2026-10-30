import argparse
from pathlib import Path

from addition_data_load import PAIR_TYPES, SPLITS, read_jsonl


def check_raw_row(row):
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

    t_source = row["T_only_counterfactual"]["source"]
    t_target = row["T_only_counterfactual"]["target_on_base"]
    assert t_source["C"] == base["C"]
    assert t_source["T"] != base["T"]
    assert t_source["ones_out"] == base["ones_out"]
    assert t_target["T"] == t_source["T"]
    assert t_target["Y"] == t_source["Y"]

    c_t_source = row["C_T_counterfactual"]["source"]
    c_t_target = row["C_T_counterfactual"]["target_on_base"]
    assert c_t_source["C"] != base["C"]
    assert c_t_source["T"] != base["T"]
    assert c_t_source["ones_out"] == base["ones_out"]
    assert c_t_target["C"] == c_t_source["C"]
    assert c_t_target["T"] == c_t_source["T"]
    assert c_t_target["Y"] == c_t_source["Y"]


def check_raw_data(data_dir):
    data_dir = Path(data_dir)
    prompt_sets = {}

    for split in SPLITS:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        errors = 0
        prompts = set()

        for row in rows:
            try:
                check_raw_row(row)
            except Exception:
                errors += 1

            prompts.add(row["base"]["prompt"])
            prompts.add(row["C_only_counterfactual"]["source"]["prompt"])
            prompts.add(row["T_only_counterfactual"]["source"]["prompt"])
            prompts.add(row["C_T_counterfactual"]["source"]["prompt"])

        prompt_sets[split] = prompts
        print(f"RAW {split}: rows={len(rows)} all_prompts={len(prompts)} errors={errors}")

        if errors != 0:
            raise RuntimeError(f"Raw data invariant failures in {split}: {errors}")

    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = prompt_sets[left] & prompt_sets[right]
        if overlap:
            raise RuntimeError(f"Prompt overlap between {left} and {right}: {sorted(overlap)[:5]}")

    print("PASS: raw dataset invariants and full base/source split disjointness")


def check_filtered_pair(row, pair_type):
    if pair_type == "C_only":
        assert row["base_C"] != row["source_C"]
        assert row["base_T"] == row["source_T"]
        assert row["C_after_C_intervene"] == row["source_C"]
        assert row["T_after_C_intervene"] != row["base_T"]
        assert row["Y_after_C_intervene"] != row["base_Y"]
    elif pair_type == "T_only":
        assert row["base_C"] == row["source_C"]
        assert row["base_T"] != row["source_T"]
        assert row["T_after_T_intervene"] == row["source_T"]
        assert row["Y_after_T_intervene"] == row["source_Y"]
    elif pair_type == "C_T":
        assert row["base_C"] != row["source_C"]
        assert row["base_T"] != row["source_T"]
        assert row["C_after_C_intervene"] == row["source_C"]
        assert row["T_after_C_intervene"] == row["source_T"]
        assert row["Y_after_C_intervene"] == row["source_Y"]
    else:
        raise ValueError(pair_type)


def check_filtered_cache(cache_dir):
    cache_dir = Path(cache_dir)

    if not (cache_dir / "manifest.json").exists():
        print(f"SKIP: filtered cache does not exist at {cache_dir}")
        return

    for split in SPLITS:
        for pair_type in PAIR_TYPES:
            path = cache_dir / f"{split}_{pair_type}.jsonl"
            rows = read_jsonl(path)
            errors = 0

            for row in rows:
                try:
                    check_filtered_pair(row, pair_type)
                except Exception:
                    errors += 1

            print(f"FILTERED {split} {pair_type}: rows={len(rows)} errors={errors}")

            if errors != 0:
                raise RuntimeError(f"Filtered pair invariant failures in {path}: {errors}")

    print("PASS: filtered pair invariants")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--cache-dir", default="addition_filtered_cache")
    args = parser.parse_args()

    check_raw_data(args.data_dir)
    check_filtered_cache(args.cache_dir)


if __name__ == "__main__":
    main()
