"""Different-question recipient transfer for Addition Standard/Gradual DAS.

Two tests, both on model-correct TEST recipients distinct from donor/base:
1. T replay: transplant original-source T (T_only) or T induced by
   C intervention (C_only/C_T) to a recipient with changed ones digit.
2. C direct transfer: transplant donor's C subspace into a new recipient
   and measure induced T readout and free-generated Y.

For each transfer, same/different refers to the recipient's natural
T or C. Counterfactual numeric Y uses the NEW recipient's ones digit.
"""

import torch

from addition_das import collect_layer_states
from addition_data_load import build_state_inputs
from addition_train_standard_grid import (
    capture_t_after_c,
    readout_logits,
    run_free_generation,
    score_sequences,
)


@torch.no_grad()
def make_bank(tokenizer, schedule, prefix_ids):
    prompts = [item["recipient"]["prompt"] for item in schedule]
    ids, attention_mask, positions = build_state_inputs(tokenizer, prompts, prefix_ids)
    return {
        "base_input_ids": ids,
        "base_attention_mask": attention_mask,
        "base_positions": positions,
        "target_y": torch.tensor([item["target_Y"] for item in schedule], dtype=torch.long),
        "target_t": torch.tensor([item["target_T"] for item in schedule], dtype=torch.long),
    }


@torch.no_grad()
def transfer_y(model, tokenizer, bank, donor, Q, layer, batch_size, generation_cap):
    sequences = run_free_generation(
        model, bank, donor, Q, layer, tokenizer, batch_size, generation_cap
    )
    correct, _ = score_sequences(sequences, bank["target_y"], tokenizer)
    return float(correct.float().mean())


@torch.no_grad()
def eval_recipient(
    model, tokenizer, banks, schedules, prefix_ids,
    Q_c, c_layer, Q_t, t_layer, W_t, b_t,
    batch_size, generation_cap,
):
    """Return per-family recipient metrics, using the same recipients across configs."""
    metrics = {"T_only": {}, "C_only": {}, "C_T": {}}
    device = next(model.parameters()).device

    # T -> Y: source T from a T_only donor, inserted in another addition question.
    t_bank = banks["T"]["test"]
    source_t = collect_layer_states(
        model, t_bank["source_input_ids"], t_bank["source_attention_mask"],
        t_bank["source_positions"], t_layer, batch_size,
    )
    for condition in ("same", "different"):
        schedule = schedules["T_only"]["T"][condition]
        recipient_bank = make_bank(tokenizer, schedule, prefix_ids)
        accuracy = transfer_y(
            model, tokenizer, recipient_bank, source_t, Q_t, t_layer,
            batch_size, generation_cap,
        )
        metrics["T_only"][f"Y_via_T_replay_recipient_{condition}"] = accuracy

    # C -> T -> Y: first induce T in the original donor base, THEN replay it
    # into an independent question with different ones digit.
    for family, donor_bank in (("C_only", banks["C"]["test"]), ("C_T", banks["C_T"]["test"])):
        source_c = collect_layer_states(
            model, donor_bank["source_input_ids"], donor_bank["source_attention_mask"],
            donor_bank["source_positions"], c_layer, batch_size,
        )
        induced_t = capture_t_after_c(
            model, donor_bank, source_c, Q_c, c_layer, t_layer, batch_size
        )
        for condition in ("same", "different"):
            t_schedule = schedules[family]["T"][condition]
            t_recipient = make_bank(tokenizer, t_schedule, prefix_ids)
            replay_acc = transfer_y(
                model, tokenizer, t_recipient, induced_t, Q_t, t_layer,
                batch_size, generation_cap,
            )
            metrics[family][f"Y_via_T_replay_recipient_{condition}"] = replay_acc

            # Also test directly applying the C handle to an independent
            # question, then checking that the new recipient's T and Y
            # change according to its own (T - C) tens sum and ones digit.
            c_schedule = schedules[family]["C"][condition]
            c_recipient = make_bank(tokenizer, c_schedule, prefix_ids)
            generated_t = capture_t_after_c(
                model, c_recipient, source_c, Q_c, c_layer, t_layer, batch_size
            )
            readout = readout_logits(generated_t, Q_t, W_t, b_t)
            t_correct = (
                readout.argmax(dim=-1).detach().cpu()
                == c_recipient["target_t"]
            )
            sequences = run_free_generation(
                model, c_recipient, source_c, Q_c, c_layer, tokenizer,
                batch_size, generation_cap,
            )
            y_correct, _ = score_sequences(
                sequences, c_recipient["target_y"], tokenizer
            )
            metrics[family][f"T_after_C_recipient_{condition}"] = float(
                t_correct.float().mean()
            )
            metrics[family][f"Y_after_C_recipient_{condition}"] = float(
                y_correct.float().mean()
            )
            metrics[family][f"Joint_T_Y_after_C_recipient_{condition}"] = float(
                (t_correct & y_correct).float().mean()
            )

    return metrics
