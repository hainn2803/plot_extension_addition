import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from addition_constants import HF_TOKEN, MODEL_NAME


print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN


def load_gemma_model(device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR)
    model.eval()
    return model, tokenizer


def read_jsonl(path):
    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def extract_first_number(text):
    numbers = re.findall(r"\d+", text)

    if len(numbers) == 0:
        return None

    return int(numbers[0])


def collect_prompt_targets(rows):
    prompt_to_target = {}

    for row in rows:
        base = row["base"]
        c_source = row["C_only_counterfactual"]["source"]
        t_source = row["T_only_counterfactual"]["source"]
        ct_source = row["C_T_counterfactual"]["source"]

        prompt_to_target[base["prompt"]] = base["Y"]
        prompt_to_target[c_source["prompt"]] = c_source["Y"]
        prompt_to_target[t_source["prompt"]] = t_source["Y"]
        prompt_to_target[ct_source["prompt"]] = ct_source["Y"]

    return prompt_to_target


def run_model(model, tokenizer, prompt_to_target, batch_size, max_new_tokens):
    prompts = list(prompt_to_target.keys())
    predictions = {}

    model.eval()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = generated[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for prompt, generated_text in zip(batch_prompts, generated_texts):
            prediction = extract_first_number(generated_text)
            expected = prompt_to_target[prompt]

            predictions[prompt] = {
                "prediction": prediction,
                "expected": expected,
                "correct": prediction == expected,
                "generated_text": generated_text,
            }

        done = min(start + batch_size, len(prompts))
        print(f"Model checked {done}/{len(prompts)} prompts")

    return predictions


def build_C_only_pair(row, predictions):
    base = row["base"]
    source = row["C_only_counterfactual"]["source"]
    after = row["C_only_counterfactual"]["target_on_base"]

    pair = {
        "pair_type": "C_only",

        "base_input": base["prompt"],
        "source_input": source["prompt"],

        "base_C": base["C"],
        "source_C": source["C"],
        "base_T": base["T"],
        "source_T": source["T"],
        "base_Y": base["Y"],
        "source_Y": source["Y"],

        "C_after_C_intervene": after["C"],
        "T_after_C_intervene": after["T"],
        "Y_after_C_intervene": after["Y"],

        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_C_intervene": after["ones_out"],

        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_C_intervene": after["hundreds_out"],

        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] != pair["source_C"]
    assert pair["base_T"] == pair["source_T"]

    assert pair["C_after_C_intervene"] == pair["source_C"]
    assert pair["T_after_C_intervene"] != pair["base_T"]
    assert pair["Y_after_C_intervene"] != pair["base_Y"]

    return pair


def build_T_only_pair(row, predictions):
    base = row["base"]
    source = row["T_only_counterfactual"]["source"]
    after = row["T_only_counterfactual"]["target_on_base"]

    pair = {
        "pair_type": "T_only",

        "base_input": base["prompt"],
        "source_input": source["prompt"],

        "base_C": base["C"],
        "source_C": source["C"],
        "base_T": base["T"],
        "source_T": source["T"],
        "base_Y": base["Y"],
        "source_Y": source["Y"],

        "C_after_T_intervene": after["C"],
        "T_after_T_intervene": after["T"],
        "Y_after_T_intervene": after["Y"],

        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_T_intervene": after["ones_out"],

        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_T_intervene": after["hundreds_out"],

        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] == pair["source_C"]
    assert pair["base_T"] != pair["source_T"]

    assert pair["C_after_T_intervene"] == pair["base_C"]
    assert pair["T_after_T_intervene"] == pair["source_T"]
    assert pair["Y_after_T_intervene"] == pair["source_Y"]

    return pair


def build_C_T_pair(row, predictions):
    base = row["base"]
    source = row["C_T_counterfactual"]["source"]
    after = row["C_T_counterfactual"]["target_on_base"]

    pair = {
        "pair_type": "C_T",

        "base_input": base["prompt"],
        "source_input": source["prompt"],

        "base_C": base["C"],
        "source_C": source["C"],
        "base_T": base["T"],
        "source_T": source["T"],
        "base_Y": base["Y"],
        "source_Y": source["Y"],

        # This pair is used to evaluate propagation after intervening on C.
        "C_after_C_intervene": after["C"],
        "T_after_C_intervene": after["T"],
        "Y_after_C_intervene": after["Y"],

        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_C_intervene": after["ones_out"],

        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_C_intervene": after["hundreds_out"],

        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] != pair["source_C"]
    assert pair["base_T"] != pair["source_T"]

    assert pair["C_after_C_intervene"] == pair["source_C"]
    assert pair["T_after_C_intervene"] == pair["source_T"]
    assert pair["Y_after_C_intervene"] == pair["source_Y"]

    return pair


def filter_and_build_pairs(rows, predictions):
    c_pairs = []
    t_pairs = []
    ct_pairs = []

    for row in rows:
        base = row["base"]
        c_source = row["C_only_counterfactual"]["source"]
        t_source = row["T_only_counterfactual"]["source"]
        ct_source = row["C_T_counterfactual"]["source"]

        base_ok = predictions[base["prompt"]]["correct"]
        c_source_ok = predictions[c_source["prompt"]]["correct"]
        t_source_ok = predictions[t_source["prompt"]]["correct"]
        ct_source_ok = predictions[ct_source["prompt"]]["correct"]

        if base_ok and c_source_ok:
            c_pairs.append(build_C_only_pair(row, predictions))

        if base_ok and t_source_ok:
            t_pairs.append(build_T_only_pair(row, predictions))

        if base_ok and ct_source_ok:
            ct_pairs.append(build_C_T_pair(row, predictions))

    return c_pairs, t_pairs, ct_pairs


def save_predictions(predictions, path):
    rows = []

    for prompt, result in predictions.items():
        row = {
            "prompt": prompt,
            "prediction": result["prediction"],
            "expected": result["expected"],
            "correct": result["correct"],
            "generated_text": result["generated_text"],
        }
        rows.append(row)

    write_jsonl(rows, path)


def print_example(c_pairs, t_pairs, ct_pairs):
    if len(c_pairs) > 0:
        print("\n=== C_only example ===")
        print(json.dumps(c_pairs[0], indent=2))

    if len(t_pairs) > 0:
        print("\n=== T_only example ===")
        print(json.dumps(t_pairs[0], indent=2))

    if len(ct_pairs) > 0:
        print("\n=== C_T example ===")
        print(json.dumps(ct_pairs[0], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="addition_controlled_data")
    parser.add_argument("--output-dir", default="addition_filtered_pairs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model(args.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\n===== {split.upper()} =====")

        rows = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        print("Raw rows:", len(rows))

        prompt_to_target = collect_prompt_targets(rows)
        print("Unique prompts:", len(prompt_to_target))

        predictions = run_model(
            model,
            tokenizer,
            prompt_to_target,
            args.batch_size,
            args.max_new_tokens,
        )

        c_pairs, t_pairs, ct_pairs = filter_and_build_pairs(rows, predictions)

        write_jsonl(c_pairs, output_dir / f"{split}_C_only.jsonl")
        write_jsonl(t_pairs, output_dir / f"{split}_T_only.jsonl")
        write_jsonl(ct_pairs, output_dir / f"{split}_C_T.jsonl")
        save_predictions(predictions, output_dir / f"{split}_model_predictions.jsonl")

        print("C_only pairs:", len(c_pairs))
        print("T_only pairs:", len(t_pairs))
        print("C_T pairs:", len(ct_pairs))

        if split == "train":
            print_example(c_pairs, t_pairs, ct_pairs)

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
