"""
Diagnose induced-mediator over-alignment / geometry shift in the addition task.

Question
--------
After a C intervention, does the induced T state become unusually aligned with
the learned T readout, compared with natural T states carrying the same T label?

This is the addition analogue of the MCQA observation where an upstream
intervention produced an AT state that was much more readout-aligned than the
natural/direct AT state.

For each result directory and each test bank (C_only, C_T), this script:

1. Creates the induced T state:
       base B + projected C from source S -> capture T at the learned T layer.

2. Chooses a NATURAL CLEAN T state with the same target T label from an
   independent test-prompt pool.

3. Compares induced vs natural-clean T in learned Q_T coordinates:
   - T readout accuracy
   - target-class margin
   - target probability
   - distance to natural same-class centroid
   - normalized centroid distance
   - nearest-neighbor distance to natural reference states
   - fraction outside the natural class 95% radius

4. For C_T only, source_T == target_T, so it additionally compares the induced
   T state directly against the natural SOURCE T state. This is the cleanest
   analogue of the MCQA direct/natural-vs-induced comparison.

Natural geometry reference:
    train + validation natural prompts, deduplicated across C_only/T_only/C_T.

Natural clean controls:
    test natural prompts, deduplicated across C_only/T_only/C_T, same T label,
    excluding the current base/source prompt when possible.

The script does NOT modify training and does NOT use teacher forcing to create
the induced T state. It runs only prompt + shared answer prefix and captures the
T state at answer_start.

Expected repository:
    hainn2803/plot_extension_addition
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from addition_das import collect_layer_states, set_seed
from addition_data_load import build_state_inputs, get_answer_prefix_ids, load_addition_rows
from addition_neural_network import load_gemma_model


PAIR_TYPES = ["C_only", "T_only", "C_T"]


def add_prompt_label(mapping, prompt, label):
    label = int(label)

    if prompt in mapping and mapping[prompt] != label:
        raise ValueError(f"Inconsistent T label for prompt: {prompt}")

    mapping[prompt] = label


def collect_natural_prompt_labels(rows, splits):
    mapping = {}

    for split in splits:
        for pair_type in PAIR_TYPES:
            for row in rows[split][pair_type]:
                add_prompt_label(mapping, row["base_input"], row["base_T"])
                add_prompt_label(mapping, row["source_input"], row["source_T"])

    return mapping


def group_prompts_by_label(prompt_to_t):
    grouped = {}

    for prompt, label in prompt_to_t.items():
        grouped.setdefault(int(label), []).append(prompt)

    return grouped


def build_prompt_bank(tokenizer, prompts, prefix_ids):
    input_ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
    }


@torch.no_grad()
def collect_prompt_states(model, tokenizer, prompts, prefix_ids, layer, batch_size):
    bank = build_prompt_bank(tokenizer, prompts, prefix_ids)

    return collect_layer_states(
        model,
        bank["input_ids"],
        bank["attention_mask"],
        bank["positions"],
        layer,
        batch_size,
    )


@torch.no_grad()
def capture_induced_t_after_c(
    model,
    tokenizer,
    pair_rows,
    prefix_ids,
    source_c_states,
    Q_c,
    c_layer,
    t_layer,
    batch_size,
):
    device = next(model.parameters()).device
    base_prompts = [row["base_input"] for row in pair_rows]
    base_bank = build_prompt_bank(tokenizer, base_prompts, prefix_ids)
    all_t = []

    for start in range(0, len(pair_rows), batch_size):
        end = min(start + batch_size, len(pair_rows))

        ids = base_bank["input_ids"][start:end].to(device)
        mask = base_bank["attention_mask"][start:end].to(device)
        positions = base_bank["positions"][start:end].to(device)
        source_c = source_c_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        captured = {}

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()

            current = hidden[rows, positions, :].float()
            delta = ((source_c.float() - current) @ Q_c) @ Q_c.T
            new_hidden[rows, positions, :] = (current + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def t_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["t"] = hidden[rows, positions, :].detach().float().cpu()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        t_handle = model.model.layers[t_layer].register_forward_hook(t_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        finally:
            c_handle.remove()
            t_handle.remove()

        if "t" not in captured:
            raise RuntimeError("T hook did not capture a state")

        all_t.append(captured["t"])

    return torch.cat(all_t, dim=0)


def choose_matched_clean_prompts(pair_rows, grouped_test_prompts, seed):
    rng = random.Random(seed)
    chosen = []

    for row_index, row in enumerate(pair_rows):
        target_t = int(row["T_after_C_intervene"])
        candidates = list(grouped_test_prompts.get(target_t, []))

        filtered = []

        for prompt in candidates:
            if prompt == row["base_input"]:
                continue

            if prompt == row["source_input"]:
                continue

            filtered.append(prompt)

        if len(filtered) == 0:
            filtered = candidates

        if len(filtered) == 0:
            raise RuntimeError(
                f"No natural clean prompt with T={target_t} for row {row_index}"
            )

        chosen.append(rng.choice(filtered))

    return chosen


def project(states, Q):
    return states.float() @ Q.float()


def readout_statistics(states, targets, Q, W, b):
    z = project(states, Q)
    logits = z @ W.float().T + b.float()
    targets = targets.long()

    predictions = logits.argmax(dim=-1)
    correct = predictions == targets

    row_index = torch.arange(len(targets))
    target_logits = logits[row_index, targets]

    competing = logits.clone()
    competing[row_index, targets] = -torch.inf
    best_other = competing.max(dim=-1).values
    margins = target_logits - best_other

    probabilities = torch.softmax(logits, dim=-1)
    target_probabilities = probabilities[row_index, targets]

    return {
        "z": z,
        "logits": logits,
        "prediction": predictions,
        "correct": correct,
        "margin": margins,
        "target_probability": target_probabilities,
    }


def build_reference_geometry(reference_z, reference_labels):
    geometry = {}

    for label in sorted(set(reference_labels.tolist())):
        mask = reference_labels == label
        z_class = reference_z[mask]

        if len(z_class) < 2:
            raise RuntimeError(
                f"Need at least 2 natural reference states for T={label}, got {len(z_class)}"
            )

        centroid = z_class.mean(dim=0)
        distances = torch.linalg.vector_norm(z_class - centroid, dim=1)

        geometry[int(label)] = {
            "z": z_class,
            "centroid": centroid,
            "distance_mean": float(distances.mean()),
            "distance_median": float(distances.median()),
            "distance_p95": float(torch.quantile(distances, 0.95)),
            "count": int(len(z_class)),
        }

    return geometry


def geometry_statistics(z, targets, geometry):
    centroid_distance = torch.empty(len(z), dtype=torch.float32)
    normalized_centroid_distance = torch.empty(len(z), dtype=torch.float32)
    nearest_neighbor_distance = torch.empty(len(z), dtype=torch.float32)
    outside_p95 = torch.empty(len(z), dtype=torch.bool)

    for label in sorted(set(targets.tolist())):
        indices = torch.where(targets == label)[0]

        if int(label) not in geometry:
            raise RuntimeError(f"No natural reference geometry for T={int(label)}")

        info = geometry[int(label)]
        z_class = z[indices]
        centroid = info["centroid"]
        reference = info["z"]

        distance = torch.linalg.vector_norm(z_class - centroid, dim=1)
        denominator = max(info["distance_median"], 1e-8)

        centroid_distance[indices] = distance
        normalized_centroid_distance[indices] = distance / denominator

        pairwise = torch.cdist(z_class, reference)
        nearest_neighbor_distance[indices] = pairwise.min(dim=1).values

        outside_p95[indices] = distance > info["distance_p95"]

    return {
        "centroid_distance": centroid_distance,
        "normalized_centroid_distance": normalized_centroid_distance,
        "nearest_neighbor_distance": nearest_neighbor_distance,
        "outside_p95": outside_p95,
    }


def bootstrap_mean_difference(a, b, seed, num_bootstrap):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError("Paired bootstrap requires equal-length arrays")

    difference = a - b
    mean_difference = float(difference.mean())

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        indices = rng.integers(0, len(difference), size=len(difference))
        bootstrap_means[i] = difference[indices].mean()

    low = float(np.quantile(bootstrap_means, 0.025))
    high = float(np.quantile(bootstrap_means, 0.975))

    return mean_difference, low, high


def summarize_condition(stats):
    return {
        "readout_accuracy": float(stats["correct"].float().mean()),
        "margin_mean": float(stats["margin"].mean()),
        "margin_median": float(stats["margin"].median()),
        "positive_margin_rate": float((stats["margin"] > 0).float().mean()),
        "target_probability_mean": float(stats["target_probability"].mean()),
        "centroid_distance_mean": float(stats["centroid_distance"].mean()),
        "centroid_distance_median": float(stats["centroid_distance"].median()),
        "normalized_centroid_distance_mean": float(
            stats["normalized_centroid_distance"].mean()
        ),
        "nearest_neighbor_distance_mean": float(
            stats["nearest_neighbor_distance"].mean()
        ),
        "outside_natural_p95_rate": float(stats["outside_p95"].float().mean()),
    }


def attach_geometry(readout_stats, geometry_stats):
    result = dict(readout_stats)
    result.update(geometry_stats)
    return result


def paired_comparison(induced, clean, seed, num_bootstrap):
    margin = bootstrap_mean_difference(
        induced["margin"].numpy(),
        clean["margin"].numpy(),
        seed,
        num_bootstrap,
    )

    probability = bootstrap_mean_difference(
        induced["target_probability"].numpy(),
        clean["target_probability"].numpy(),
        seed + 1,
        num_bootstrap,
    )

    centroid_distance = bootstrap_mean_difference(
        induced["centroid_distance"].numpy(),
        clean["centroid_distance"].numpy(),
        seed + 2,
        num_bootstrap,
    )

    normalized_distance = bootstrap_mean_difference(
        induced["normalized_centroid_distance"].numpy(),
        clean["normalized_centroid_distance"].numpy(),
        seed + 3,
        num_bootstrap,
    )

    nearest_neighbor = bootstrap_mean_difference(
        induced["nearest_neighbor_distance"].numpy(),
        clean["nearest_neighbor_distance"].numpy(),
        seed + 4,
        num_bootstrap,
    )

    return {
        "induced_minus_clean_margin_mean": margin[0],
        "margin_difference_95ci": [margin[1], margin[2]],
        "induced_minus_clean_target_probability_mean": probability[0],
        "target_probability_difference_95ci": [probability[1], probability[2]],
        "induced_minus_clean_centroid_distance_mean": centroid_distance[0],
        "centroid_distance_difference_95ci": [
            centroid_distance[1],
            centroid_distance[2],
        ],
        "induced_minus_clean_normalized_centroid_distance_mean": normalized_distance[0],
        "normalized_centroid_distance_difference_95ci": [
            normalized_distance[1],
            normalized_distance[2],
        ],
        "induced_minus_clean_nearest_neighbor_distance_mean": nearest_neighbor[0],
        "nearest_neighbor_distance_difference_95ci": [
            nearest_neighbor[1],
            nearest_neighbor[2],
        ],
        "fraction_induced_margin_gt_clean": float(
            (induced["margin"] > clean["margin"]).float().mean()
        ),
        "fraction_induced_centroid_distance_gt_clean": float(
            (
                induced["centroid_distance"]
                > clean["centroid_distance"]
            )
            .float()
            .mean()
        ),
    }


def print_condition(name, summary):
    print(
        f"{name:>14s} | "
        f"acc={summary['readout_accuracy']:.4f} | "
        f"margin={summary['margin_mean']:.4f} | "
        f"p(target)={summary['target_probability_mean']:.4f} | "
        f"centroid_dist={summary['centroid_distance_mean']:.4f} | "
        f"norm_dist={summary['normalized_centroid_distance_mean']:.4f} | "
        f"NN_dist={summary['nearest_neighbor_distance_mean']:.4f} | "
        f"outside_p95={summary['outside_natural_p95_rate']:.4f}"
    )


def print_paired(name, comparison):
    margin_ci = comparison["margin_difference_95ci"]
    distance_ci = comparison["centroid_distance_difference_95ci"]

    print(
        f"{name}: "
        f"Δmargin={comparison['induced_minus_clean_margin_mean']:+.4f} "
        f"[{margin_ci[0]:+.4f}, {margin_ci[1]:+.4f}] | "
        f"Δcentroid_dist={comparison['induced_minus_clean_centroid_distance_mean']:+.4f} "
        f"[{distance_ci[0]:+.4f}, {distance_ci[1]:+.4f}] | "
        f"P(margin_induced > margin_clean)="
        f"{comparison['fraction_induced_margin_gt_clean']:.4f}"
    )


def classwise_summary(targets, induced, clean):
    rows = []

    for label in sorted(set(targets.tolist())):
        mask = targets == label

        rows.append(
            {
                "T": int(label),
                "n": int(mask.sum()),
                "induced_accuracy": float(induced["correct"][mask].float().mean()),
                "clean_accuracy": float(clean["correct"][mask].float().mean()),
                "induced_margin_mean": float(induced["margin"][mask].mean()),
                "clean_margin_mean": float(clean["margin"][mask].mean()),
                "induced_centroid_distance_mean": float(
                    induced["centroid_distance"][mask].mean()
                ),
                "clean_centroid_distance_mean": float(
                    clean["centroid_distance"][mask].mean()
                ),
                "induced_outside_p95_rate": float(
                    induced["outside_p95"][mask].float().mean()
                ),
                "clean_outside_p95_rate": float(
                    clean["outside_p95"][mask].float().mean()
                ),
            }
        )

    return rows


def save_csv(rows, path):
    if len(rows) == 0:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def per_example_rows(
    pair_rows,
    clean_prompts,
    targets,
    induced,
    clean,
    source=None,
):
    rows = []

    for i, row in enumerate(pair_rows):
        item = {
            "index": i,
            "pair_type": row["pair_type"],
            "base_input": row["base_input"],
            "source_input": row["source_input"],
            "clean_input": clean_prompts[i],
            "base_T": int(row["base_T"]),
            "source_T": int(row["source_T"]),
            "target_T": int(targets[i]),
            "induced_prediction": int(induced["prediction"][i]),
            "clean_prediction": int(clean["prediction"][i]),
            "induced_correct": int(induced["correct"][i]),
            "clean_correct": int(clean["correct"][i]),
            "induced_margin": float(induced["margin"][i]),
            "clean_margin": float(clean["margin"][i]),
            "induced_target_probability": float(induced["target_probability"][i]),
            "clean_target_probability": float(clean["target_probability"][i]),
            "induced_centroid_distance": float(induced["centroid_distance"][i]),
            "clean_centroid_distance": float(clean["centroid_distance"][i]),
            "induced_normalized_centroid_distance": float(
                induced["normalized_centroid_distance"][i]
            ),
            "clean_normalized_centroid_distance": float(
                clean["normalized_centroid_distance"][i]
            ),
            "induced_nearest_neighbor_distance": float(
                induced["nearest_neighbor_distance"][i]
            ),
            "clean_nearest_neighbor_distance": float(
                clean["nearest_neighbor_distance"][i]
            ),
            "induced_outside_p95": int(induced["outside_p95"][i]),
            "clean_outside_p95": int(clean["outside_p95"][i]),
        }

        if source is not None:
            item.update(
                {
                    "source_prediction": int(source["prediction"][i]),
                    "source_correct": int(source["correct"][i]),
                    "source_margin": float(source["margin"][i]),
                    "source_target_probability": float(
                        source["target_probability"][i]
                    ),
                    "source_centroid_distance": float(
                        source["centroid_distance"][i]
                    ),
                    "source_normalized_centroid_distance": float(
                        source["normalized_centroid_distance"][i]
                    ),
                    "source_nearest_neighbor_distance": float(
                        source["nearest_neighbor_distance"][i]
                    ),
                    "source_outside_p95": int(source["outside_p95"][i]),
                }
            )

        rows.append(item)

    return rows


def validate_result_dir(result_dir, model):
    t_checkpoint_path = result_dir / "das_t.pt"
    c_checkpoint_path = result_dir / "das_c.pt"
    t_readout_path = result_dir / "t_readout_LogisticRegression.pt"

    for path in [t_checkpoint_path, c_checkpoint_path, t_readout_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    t_checkpoint = torch.load(
        t_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    c_checkpoint = torch.load(
        c_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    t_readout = torch.load(
        t_readout_path,
        map_location="cpu",
        weights_only=False,
    )

    Q_t = t_checkpoint["basis"].float()
    Q_c = c_checkpoint["basis"].float()
    W_t = t_readout["W_T"].float()
    b_t = t_readout["b_T"].float()
    t_layer = int(t_checkpoint["layer"])
    c_layer = int(c_checkpoint["layer"])

    hidden_size = int(model.config.hidden_size)

    if Q_t.shape[0] != hidden_size:
        raise ValueError(
            f"{result_dir}: Q_T width {Q_t.shape[0]} != hidden size {hidden_size}"
        )

    if Q_c.shape[0] != hidden_size:
        raise ValueError(
            f"{result_dir}: Q_C width {Q_c.shape[0]} != hidden size {hidden_size}"
        )

    if W_t.shape[1] != Q_t.shape[1]:
        raise ValueError(
            f"{result_dir}: T readout dim {W_t.shape[1]} "
            f"!= Q_T dim {Q_t.shape[1]}"
        )

    if c_layer >= t_layer:
        raise ValueError(
            f"{result_dir}: expected C layer < T layer, got {c_layer}, {t_layer}"
        )

    if "layer" in t_readout and int(t_readout["layer"]) != t_layer:
        raise ValueError(
            f"{result_dir}: T readout layer {t_readout['layer']} "
            f"!= T checkpoint layer {t_layer}"
        )

    if "basis" in t_readout:
        difference = (
            t_readout["basis"].float() - Q_t
        ).abs().max().item()

        if difference > 1e-5:
            raise ValueError(
                f"{result_dir}: T readout basis differs from T checkpoint "
                f"(max difference={difference:.6g})"
            )

    return {
        "Q_t": Q_t,
        "Q_c": Q_c,
        "W_t": W_t,
        "b_t": b_t,
        "t_layer": t_layer,
        "c_layer": c_layer,
    }


def evaluate_result_dir(
    model,
    tokenizer,
    rows,
    prefix_ids,
    result_dir,
    reference_prompts,
    reference_labels,
    grouped_test_prompts,
    batch_size,
    seed,
    num_bootstrap,
):
    loaded = validate_result_dir(result_dir, model)

    Q_t = loaded["Q_t"]
    Q_c_cpu = loaded["Q_c"]
    W_t = loaded["W_t"]
    b_t = loaded["b_t"]
    t_layer = loaded["t_layer"]
    c_layer = loaded["c_layer"]

    device = next(model.parameters()).device
    Q_c = Q_c_cpu.to(device)

    print("\n" + "=" * 88)
    print(f"RESULT DIR: {result_dir}")
    print(f"C layer={c_layer} | T layer={t_layer} | rank_T={Q_t.shape[1]}")
    print("=" * 88)

    print("[reference] collecting natural T states from train + validation ...")

    reference_states = collect_prompt_states(
        model,
        tokenizer,
        reference_prompts,
        prefix_ids,
        t_layer,
        batch_size,
    )

    reference_z = project(reference_states, Q_t)
    geometry = build_reference_geometry(reference_z, reference_labels)

    result_summary = {
        "result_dir": str(result_dir),
        "C_layer": c_layer,
        "T_layer": t_layer,
        "reference_size": len(reference_prompts),
        "reference_classes": {
            str(label): {
                "count": geometry[label]["count"],
                "distance_mean": geometry[label]["distance_mean"],
                "distance_median": geometry[label]["distance_median"],
                "distance_p95": geometry[label]["distance_p95"],
            }
            for label in sorted(geometry)
        },
        "banks": {},
    }

    for bank_index, pair_type in enumerate(["C_only", "C_T"]):
        pair_rows = rows["test"][pair_type]
        targets = torch.tensor(
            [row["T_after_C_intervene"] for row in pair_rows],
            dtype=torch.long,
        )

        base_prompts = [row["base_input"] for row in pair_rows]
        source_prompts = [row["source_input"] for row in pair_rows]

        print(f"\n--- {pair_type}: n={len(pair_rows)} ---")

        source_c_states = collect_prompt_states(
            model,
            tokenizer,
            source_prompts,
            prefix_ids,
            c_layer,
            batch_size,
        )

        induced_t_states = capture_induced_t_after_c(
            model,
            tokenizer,
            pair_rows,
            prefix_ids,
            source_c_states,
            Q_c,
            c_layer,
            t_layer,
            batch_size,
        )

        clean_prompts = choose_matched_clean_prompts(
            pair_rows,
            grouped_test_prompts,
            seed + 1000 * bank_index,
        )

        clean_t_states = collect_prompt_states(
            model,
            tokenizer,
            clean_prompts,
            prefix_ids,
            t_layer,
            batch_size,
        )

        induced_readout = readout_statistics(
            induced_t_states,
            targets,
            Q_t,
            W_t,
            b_t,
        )

        clean_readout = readout_statistics(
            clean_t_states,
            targets,
            Q_t,
            W_t,
            b_t,
        )

        induced_geometry = geometry_statistics(
            induced_readout["z"],
            targets,
            geometry,
        )

        clean_geometry = geometry_statistics(
            clean_readout["z"],
            targets,
            geometry,
        )

        induced = attach_geometry(induced_readout, induced_geometry)
        clean = attach_geometry(clean_readout, clean_geometry)

        induced_summary = summarize_condition(induced)
        clean_summary = summarize_condition(clean)

        comparison = paired_comparison(
            induced,
            clean,
            seed + 100 * bank_index,
            num_bootstrap,
        )

        print_condition("INDUCED", induced_summary)
        print_condition("NATURAL CLEAN", clean_summary)
        print_paired("INDUCED - CLEAN", comparison)

        source = None
        source_summary = None
        source_comparison = None

        if pair_type == "C_T":
            for row in pair_rows:
                if int(row["source_T"]) != int(row["T_after_C_intervene"]):
                    raise RuntimeError(
                        "C_T invariant failed: source_T != target T"
                    )

            source_t_states = collect_prompt_states(
                model,
                tokenizer,
                source_prompts,
                prefix_ids,
                t_layer,
                batch_size,
            )

            source_readout = readout_statistics(
                source_t_states,
                targets,
                Q_t,
                W_t,
                b_t,
            )

            source_geometry = geometry_statistics(
                source_readout["z"],
                targets,
                geometry,
            )

            source = attach_geometry(source_readout, source_geometry)
            source_summary = summarize_condition(source)

            source_comparison = paired_comparison(
                induced,
                source,
                seed + 777,
                num_bootstrap,
            )

            print_condition("NATURAL SOURCE", source_summary)
            print_paired("INDUCED - SOURCE", source_comparison)

            coordinate_distance = torch.linalg.vector_norm(
                induced["z"] - source["z"],
                dim=1,
            )

            source_comparison[
                "induced_source_Q_coordinate_distance_mean"
            ] = float(coordinate_distance.mean())

            source_comparison[
                "induced_source_Q_coordinate_distance_median"
            ] = float(coordinate_distance.median())

            print(
                "INDUCED vs SOURCE Q_T distance: "
                f"mean={coordinate_distance.mean().item():.4f} | "
                f"median={coordinate_distance.median().item():.4f}"
            )

        class_rows = classwise_summary(targets, induced, clean)

        print("\nPer target T:")
        print(
            "T | n | induced_acc | clean_acc | "
            "induced_margin | clean_margin | "
            "induced_dist | clean_dist | "
            "induced_out95 | clean_out95"
        )

        for row in class_rows:
            print(
                f"{row['T']:1d} | "
                f"{row['n']:3d} | "
                f"{row['induced_accuracy']:.4f} | "
                f"{row['clean_accuracy']:.4f} | "
                f"{row['induced_margin_mean']:+.4f} | "
                f"{row['clean_margin_mean']:+.4f} | "
                f"{row['induced_centroid_distance_mean']:.4f} | "
                f"{row['clean_centroid_distance_mean']:.4f} | "
                f"{row['induced_outside_p95_rate']:.4f} | "
                f"{row['clean_outside_p95_rate']:.4f}"
            )

        example_rows = per_example_rows(
            pair_rows,
            clean_prompts,
            targets,
            induced,
            clean,
            source,
        )

        csv_path = (
            result_dir
            / f"failure_mode_T_induced_vs_natural_{pair_type}.csv"
        )

        save_csv(example_rows, csv_path)

        class_csv_path = (
            result_dir
            / f"failure_mode_T_induced_vs_natural_{pair_type}_by_class.csv"
        )

        save_csv(class_rows, class_csv_path)

        result_summary["banks"][pair_type] = {
            "n": len(pair_rows),
            "induced": induced_summary,
            "natural_clean": clean_summary,
            "induced_minus_clean": comparison,
            "natural_source": source_summary,
            "induced_minus_source": source_comparison,
            "per_class": class_rows,
            "per_example_csv": str(csv_path),
        }

    summary_path = result_dir / "failure_mode_T_induced_vs_natural.json"
    summary_path.write_text(json.dumps(result_summary, indent=2) + "\n")

    print(f"\n[saved] {summary_path}")

    return result_summary


def print_cross_result_comparison(summaries):
    if len(summaries) < 2:
        return

    print("\n" + "=" * 88)
    print("CROSS-RESULT COMPARISON")
    print("=" * 88)

    for pair_type in ["C_only", "C_T"]:
        print(f"\n[{pair_type}]")

        for summary in summaries:
            bank = summary["banks"][pair_type]
            induced = bank["induced"]
            clean = bank["natural_clean"]
            comparison = bank["induced_minus_clean"]

            print(
                f"{summary['result_dir']}: "
                f"T_acc {induced['readout_accuracy']:.4f} vs "
                f"{clean['readout_accuracy']:.4f} | "
                f"margin {induced['margin_mean']:.4f} vs "
                f"{clean['margin_mean']:.4f} | "
                f"Δmargin={comparison['induced_minus_clean_margin_mean']:+.4f} | "
                f"outside95 {induced['outside_natural_p95_rate']:.4f} vs "
                f"{clean['outside_natural_p95_rate']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dirs",
        nargs="+",
        default=[
            "results_addition_standard",
            "results_addition_gradual_2.0",
        ],
    )

    parser.add_argument(
        "--data-dir",
        default="addition_controlled_data",
    )

    parser.add_argument(
        "--cache-dir",
        default="addition_filtered_cache",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    # Only used to match/reuse the existing filtered-data cache manifest.
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model(args.device)

    rows = load_addition_rows(
        model,
        tokenizer,
        args.data_dir,
        args.cache_dir,
        args.batch_size,
        args.max_new_tokens,
        False,
    )

    prefix_ids = get_answer_prefix_ids(tokenizer)

    reference_mapping = collect_natural_prompt_labels(
        rows,
        ["train", "validation"],
    )

    test_mapping = collect_natural_prompt_labels(
        rows,
        ["test"],
    )

    reference_prompts = list(reference_mapping.keys())
    reference_labels = torch.tensor(
        [reference_mapping[prompt] for prompt in reference_prompts],
        dtype=torch.long,
    )

    grouped_test_prompts = group_prompts_by_label(test_mapping)

    print("\n=== NATURAL T POOLS ===")
    print(
        f"reference train+validation prompts: {len(reference_prompts)}"
    )
    print(
        f"test clean-control prompts: {len(test_mapping)}"
    )

    print("test prompts per T:")

    for label in sorted(grouped_test_prompts):
        print(
            f"  T={label}: {len(grouped_test_prompts[label])}"
        )

    summaries = []

    for result_dir_string in args.result_dirs:
        result_dir = Path(result_dir_string)

        summary = evaluate_result_dir(
            model,
            tokenizer,
            rows,
            prefix_ids,
            result_dir,
            reference_prompts,
            reference_labels,
            grouped_test_prompts,
            args.batch_size,
            args.seed,
            args.num_bootstrap,
        )

        summaries.append(summary)

    print_cross_result_comparison(summaries)

    combined = {
        "diagnostic": "induced_T_vs_natural_T_readout_alignment_and_geometry",
        "result_dirs": summaries,
    }

    combined_path = Path(
        "failure_mode_T_induced_vs_natural_combined.json"
    )

    combined_path.write_text(
        json.dumps(combined, indent=2) + "\n"
    )

    print(f"\n[saved] {combined_path}")


if __name__ == "__main__":
    main()
