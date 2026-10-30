import hashlib
import json
import re
from pathlib import Path

import torch


SPLITS = ["train", "validation", "test"]
PAIR_TYPES = ["C_only", "T_only", "C_T"]
CACHE_VERSION = 2


def read_jsonl(path):
    rows = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def extract_first_number(text):
    numbers = re.findall(r"\d+", text)

    if len(numbers) == 0:
        return None

    return int(numbers[0])


def collect_prompt_targets(rows):
    prompt_to_target = {}

    for row in rows:
        examples = [
            row["base"],
            row["C_only_counterfactual"]["source"],
            row["T_only_counterfactual"]["source"],
            row["C_T_counterfactual"]["source"],
        ]

        for example in examples:
            prompt = example["prompt"]
            target = int(example["Y"])

            if prompt in prompt_to_target and prompt_to_target[prompt] != target:
                raise ValueError(f"Conflicting targets for prompt: {prompt}")

            prompt_to_target[prompt] = target

    return prompt_to_target


@torch.no_grad()
def run_model_correctness(model, tokenizer, prompt_to_target, batch_size=32, max_new_tokens=4):
    max_answer_tokens = 0
    for value in range(10, 100):
        token_count = len(tokenizer(" " + str(value), add_special_tokens=False)["input_ids"])
        max_answer_tokens = max(max_answer_tokens, token_count)

    if max_new_tokens < max_answer_tokens:
        raise ValueError(f"max_new_tokens={max_new_tokens} is smaller than max two-digit answer token length={max_answer_tokens}")

    prompts = list(prompt_to_target.keys())
    predictions = {}

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
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

        print(f"[model filter] {end}/{len(prompts)} prompts")

    return predictions


def build_c_only_pair(row, predictions):
    base = row["base"]
    source = row["C_only_counterfactual"]["source"]
    target = row["C_only_counterfactual"]["target_on_base"]

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
        "C_after_C_intervene": target["C"],
        "T_after_C_intervene": target["T"],
        "Y_after_C_intervene": target["Y"],
        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_C_intervene": target["ones_out"],
        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_C_intervene": target["hundreds_out"],
        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] != pair["source_C"]
    assert pair["base_T"] == pair["source_T"]
    assert pair["C_after_C_intervene"] == pair["source_C"]
    assert pair["T_after_C_intervene"] != pair["base_T"]
    assert pair["Y_after_C_intervene"] != pair["base_Y"]
    return pair


def build_t_only_pair(row, predictions):
    base = row["base"]
    source = row["T_only_counterfactual"]["source"]
    target = row["T_only_counterfactual"]["target_on_base"]

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
        "C_after_T_intervene": target["C"],
        "T_after_T_intervene": target["T"],
        "Y_after_T_intervene": target["Y"],
        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_T_intervene": target["ones_out"],
        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_T_intervene": target["hundreds_out"],
        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] == pair["source_C"]
    assert pair["base_T"] != pair["source_T"]
    assert pair["T_after_T_intervene"] == pair["source_T"]
    assert pair["Y_after_T_intervene"] == pair["source_Y"]
    return pair


def build_c_t_pair(row, predictions):
    base = row["base"]
    source = row["C_T_counterfactual"]["source"]
    target = row["C_T_counterfactual"]["target_on_base"]

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
        "C_after_C_intervene": target["C"],
        "T_after_C_intervene": target["T"],
        "Y_after_C_intervene": target["Y"],
        "base_ones_out": base["ones_out"],
        "source_ones_out": source["ones_out"],
        "ones_out_after_C_intervene": target["ones_out"],
        "base_hundreds_out": base["hundreds_out"],
        "source_hundreds_out": source["hundreds_out"],
        "hundreds_out_after_C_intervene": target["hundreds_out"],
        "base_model_prediction": predictions[base["prompt"]]["prediction"],
        "source_model_prediction": predictions[source["prompt"]]["prediction"],
    }

    assert pair["base_C"] != pair["source_C"]
    assert pair["base_T"] != pair["source_T"]
    assert pair["C_after_C_intervene"] == pair["source_C"]
    assert pair["T_after_C_intervene"] == pair["source_T"]
    assert pair["Y_after_C_intervene"] == pair["source_Y"]
    return pair


def filter_rows(rows, predictions):
    result = {"C_only": [], "T_only": [], "C_T": []}

    for row in rows:
        base = row["base"]
        c_source = row["C_only_counterfactual"]["source"]
        t_source = row["T_only_counterfactual"]["source"]
        c_t_source = row["C_T_counterfactual"]["source"]

        base_ok = predictions[base["prompt"]]["correct"]
        c_ok = predictions[c_source["prompt"]]["correct"]
        t_ok = predictions[t_source["prompt"]]["correct"]
        c_t_ok = predictions[c_t_source["prompt"]]["correct"]

        if base_ok and c_ok:
            result["C_only"].append(build_c_only_pair(row, predictions))

        if base_ok and t_ok:
            result["T_only"].append(build_t_only_pair(row, predictions))

        if base_ok and c_t_ok:
            result["C_T"].append(build_c_t_pair(row, predictions))

    return result


def build_cache_manifest(model, data_dir, max_new_tokens):
    data_dir = Path(data_dir)
    model_name = getattr(model.config, "_name_or_path", model.__class__.__name__)
    manifest = {
        "cache_version": CACHE_VERSION,
        "model": str(model_name),
        "max_new_tokens": int(max_new_tokens),
        "raw_files": {},
    }

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        manifest["raw_files"][split] = file_sha256(path)

    return manifest


def cache_matches(cache_dir, expected_manifest):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        return False

    try:
        saved = json.loads(manifest_path.read_text())
    except Exception:
        return False

    if saved != expected_manifest:
        return False

    for split in SPLITS:
        for pair_type in PAIR_TYPES:
            if not (cache_dir / f"{split}_{pair_type}.jsonl").exists():
                return False

    return True


def load_addition_rows(model, tokenizer, data_dir="addition_controlled_data", cache_dir="addition_filtered_cache", batch_size=32, max_new_tokens=4, force_rebuild=False):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_manifest = build_cache_manifest(model, data_dir, max_new_tokens)
    use_cache = not force_rebuild and cache_matches(cache_dir, expected_manifest)

    if use_cache:
        print(f"[data] reusing filtered cache: {cache_dir}")
    else:
        print(f"[data] rebuilding filtered cache: {cache_dir}")

        for split in SPLITS:
            raw_rows = read_jsonl(data_dir / f"{split}.jsonl")
            prompt_to_target = collect_prompt_targets(raw_rows)
            predictions = run_model_correctness(model, tokenizer, prompt_to_target, batch_size, max_new_tokens)
            filtered = filter_rows(raw_rows, predictions)

            for pair_type in PAIR_TYPES:
                write_jsonl(filtered[pair_type], cache_dir / f"{split}_{pair_type}.jsonl")
                print(f"[{split}] {pair_type}: {len(filtered[pair_type])}")

        (cache_dir / "manifest.json").write_text(json.dumps(expected_manifest, indent=2) + "\n")

    rows = {}

    for split in SPLITS:
        rows[split] = {}
        for pair_type in PAIR_TYPES:
            rows[split][pair_type] = read_jsonl(cache_dir / f"{split}_{pair_type}.jsonl")

    return rows


def longest_common_prefix(sequences):
    if len(sequences) == 0:
        return []

    prefix = list(sequences[0])

    for sequence in sequences[1:]:
        length = min(len(prefix), len(sequence))
        new_length = 0

        for index in range(length):
            if prefix[index] != sequence[index]:
                break
            new_length += 1

        prefix = prefix[:new_length]
        if len(prefix) == 0:
            break

    return prefix


def get_answer_prefix_ids(tokenizer):
    sequences = []

    for value in range(10, 100):
        ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]
        sequences.append(ids)

    prefix_ids = longest_common_prefix(sequences)

    if len(prefix_ids) == 0:
        raise ValueError("No shared answer prefix token found for two-digit answers")

    print("[answer prefix] ids =", prefix_ids)
    print("[answer prefix] text =", repr(tokenizer.decode(prefix_ids)))
    return prefix_ids


def answer_content_ids(tokenizer, value, prefix_ids):
    ids = tokenizer(" " + str(value), add_special_tokens=False)["input_ids"]

    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f"Answer {value} does not begin with the shared prefix")

    content_ids = ids[len(prefix_ids):]
    if len(content_ids) == 0:
        raise ValueError(f"Answer {value} has no numeric token after the shared prefix")

    return content_ids


def pad_sequences(sequences, tokenizer):
    max_length = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    last_positions = torch.zeros(len(sequences), dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        pad = max_length - len(sequence)
        input_ids[row_index, pad:] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, pad:] = 1
        last_positions[row_index] = max_length - 1

    return input_ids, attention_mask, last_positions


def build_state_inputs(tokenizer, prompts, prefix_ids):
    sequences = []

    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        sequences.append(prompt_ids + prefix_ids)

    return pad_sequences(sequences, tokenizer)


def build_teacher_forcing_batch(tokenizer, prompts, target_values, prefix_ids):
    sequences = []
    intervention_positions = []
    label_positions = []
    target_ids = []

    for prompt, target_value in zip(prompts, target_values):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = answer_content_ids(tokenizer, target_value, prefix_ids)
        sequence = prompt_ids + prefix_ids + answer_ids[:-1]
        intervention_position = len(prompt_ids) + len(prefix_ids) - 1
        positions = []

        for answer_index in range(len(answer_ids)):
            positions.append(intervention_position + answer_index)

        sequences.append(sequence)
        intervention_positions.append(intervention_position)
        label_positions.append(positions)
        target_ids.append(answer_ids)

    max_seq_len = max(len(sequence) for sequence in sequences)
    max_answer_len = max(len(ids) for ids in target_ids)
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(sequences), max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_seq_len), dtype=torch.long)
    padded_intervention_positions = torch.zeros(len(sequences), dtype=torch.long)
    padded_label_positions = torch.zeros((len(sequences), max_answer_len), dtype=torch.long)
    padded_target_ids = torch.zeros((len(sequences), max_answer_len), dtype=torch.long)
    target_mask = torch.zeros((len(sequences), max_answer_len), dtype=torch.bool)

    for row_index, sequence in enumerate(sequences):
        pad = max_seq_len - len(sequence)
        input_ids[row_index, pad:] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, pad:] = 1
        padded_intervention_positions[row_index] = pad + intervention_positions[row_index]

        for answer_index in range(len(target_ids[row_index])):
            padded_label_positions[row_index, answer_index] = pad + label_positions[row_index][answer_index]
            padded_target_ids[row_index, answer_index] = target_ids[row_index][answer_index]
            target_mask[row_index, answer_index] = True

    return {
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": attention_mask,
        "intervention_positions": padded_intervention_positions,
        "label_positions": padded_label_positions,
        "target_answer_ids": padded_target_ids,
        "target_answer_mask": target_mask,
    }


def build_pair_bank(tokenizer, rows, intervention_name, prefix_ids):
    base_prompts = []
    source_prompts = []
    target_y = []
    target_t = []

    for row in rows:
        base_prompts.append(row["base_input"])
        source_prompts.append(row["source_input"])

        if intervention_name == "T":
            target_y.append(row["Y_after_T_intervene"])
            target_t.append(row["T_after_T_intervene"])
        elif intervention_name == "C":
            target_y.append(row["Y_after_C_intervene"])
            target_t.append(row["T_after_C_intervene"])
        else:
            raise ValueError(f"Unknown intervention: {intervention_name}")

    base_input_ids, base_attention_mask, base_positions = build_state_inputs(tokenizer, base_prompts, prefix_ids)
    source_input_ids, source_attention_mask, source_positions = build_state_inputs(tokenizer, source_prompts, prefix_ids)
    teacher = build_teacher_forcing_batch(tokenizer, base_prompts, target_y, prefix_ids)

    bank = {
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "base_positions": base_positions,
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "source_positions": source_positions,
        "target_y": torch.tensor(target_y, dtype=torch.long),
        "target_t": torch.tensor(target_t, dtype=torch.long),
    }
    bank.update(teacher)
    return bank


def build_factual_t_bank(tokenizer, rows, prefix_ids):
    prompts = []
    labels = []

    for row in rows:
        prompts.append(row["base_input"])
        labels.append(row["base_T"])
        prompts.append(row["source_input"])
        labels.append(row["source_T"])

    input_ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "target_t": torch.tensor(labels, dtype=torch.long),
    }


def build_all_banks(tokenizer, rows, prefix_ids=None):
    if prefix_ids is None:
        prefix_ids = get_answer_prefix_ids(tokenizer)

    banks = {
        "T": {},
        "C": {},
        "C_T": {},
        "T_factual": {},
        "prefix_ids": prefix_ids,
    }

    for split in SPLITS:
        banks["T"][split] = build_pair_bank(tokenizer, rows[split]["T_only"], "T", prefix_ids)
        banks["C"][split] = build_pair_bank(tokenizer, rows[split]["C_only"], "C", prefix_ids)
        banks["C_T"][split] = build_pair_bank(tokenizer, rows[split]["C_T"], "C", prefix_ids)
        banks["T_factual"][split] = build_factual_t_bank(tokenizer, rows[split]["T_only"], prefix_ids)

    return banks
