"""Deterministic, factual, different-question recipients for Addition transfer tests.

All pool prompts come from model-correct filtered TEST pairs. A recipient must
differ from BOTH donor prompts and have a different ones digit than the donor
base, making the changed arithmetic context explicit.

same/different refers to a recipient's natural value of the patched variable.
"""

import random


RECIPIENT_VERSION = 1


def recipient_pool(test_rows):
    by_prompt = {}
    for family in ("T_only", "C_only", "C_T"):
        for pair in test_rows[family]:
            for side in ("base", "source"):
                candidate = {
                    "prompt": str(pair[f"{side}_input"]),
                    "C": int(pair[f"{side}_C"]),
                    "T": int(pair[f"{side}_T"]),
                    "ones": int(pair[f"{side}_ones_out"]),
                    "Y": int(pair[f"{side}_Y"]),
                    "hundreds": int(pair[f"{side}_hundreds_out"]),
                }
                if candidate["hundreds"] != 0:
                    raise ValueError("Expected two-digit Addition pool with no hundreds carry")
                if candidate["Y"] != 10 * candidate["T"] + candidate["ones"]:
                    raise ValueError("Inconsistent factual recipient Y/T/ones")
                old = by_prompt.get(candidate["prompt"])
                if old is not None and old != candidate:
                    raise ValueError(f"Conflicting facts for {candidate['prompt']}")
                by_prompt[candidate["prompt"]] = candidate
    return list(by_prompt.values())


def target_for_recipient(recipient, kind, patched_value):
    """Return the correct target Y and T for a mediator- or C-swap."""
    if kind == "T":
        target_t = int(patched_value)
    elif kind == "C":
        # All filtered raw problems exclude hundreds carry and tens_sum=9.
        tens_sum = int(recipient["T"]) - int(recipient["C"])
        if not 0 <= tens_sum <= 8:
            raise ValueError("Recipient is outside controlled two-digit Addition task")
        target_t = tens_sum + int(patched_value)
    else:
        raise ValueError(f"Unknown intervention variable {kind}")
    if not 0 <= target_t <= 9:
        raise ValueError("Recipient intervention produced an invalid tens digit")
    return {"target_T": target_t, "target_Y": 10 * target_t + recipient["ones"]}


def make_schedule(donor_pairs, pool, kind, condition, seed):
    """A stable recipient per donor, independent of DAS layer/configuration.

    kind T: target is donor's counterfactual T, for T replay onto recipient.
    kind C: target is donor's source C, for direct C intervention on recipient.
    """
    if kind not in ("C", "T") or condition not in ("same", "different"):
        raise ValueError("kind must be C/T and condition same/different")
    rng = random.Random(seed)
    candidates_by_value = {value: [] for value in range(10 if kind == "T" else 2)}
    for recipient in pool:
        candidates_by_value[recipient[kind]].append(recipient)

    picked = []
    for donor in donor_pairs:
        patched_value = int(
            donor["T_after_C_intervene"]
            if kind == "T" and donor["pair_type"] != "T_only"
            else donor["T_after_T_intervene"]
            if kind == "T"
            else donor["source_C"]
        )
        if kind == "T":
            target_pool = (
                candidates_by_value[patched_value] if condition == "same"
                else [rec for rec in pool if rec["T"] != patched_value]
            )
        else:
            target_pool = (
                candidates_by_value[patched_value] if condition == "same"
                else candidates_by_value[1 - patched_value]
            )
        eligible = []
        for recipient in target_pool:
            if recipient["prompt"] == donor["base_input"]:
                continue
            if recipient["prompt"] == donor["source_input"]:
                continue
            if recipient["ones"] == int(donor["base_ones_out"]):
                continue
            eligible.append(recipient)

        if not eligible:
            raise RuntimeError(
                f"No {condition} recipient with different question/ones "
                f"for {donor['pair_type']} {kind}={patched_value}"
            )
        recipient = rng.choice(eligible)
        expected = target_for_recipient(recipient, kind, patched_value)
        picked.append({
            "recipient": recipient,
            "patched_value": patched_value,
            **expected,
        })
    return picked


def prepare_schedules(test_rows, seed):
    pool = recipient_pool(test_rows)
    print(f"[recipient] {len(pool)} unique model-correct TEST prompts")
    schedules = {}
    family_seeds = {"T_only": 1301, "C_only": 1302, "C_T": 1303}
    for family, offset in family_seeds.items():
        schedules[family] = {}
        for kind in (("T",) if family == "T_only" else ("T", "C")):
            schedules[family][kind] = {}
            for condition, delta in (("same", 0), ("different", 10000)):
                schedules[family][kind][condition] = make_schedule(
                    test_rows[family], pool, kind, condition,
                    seed + 5000 + offset + (0 if kind == "T" else 20000) + delta,
                )
    return schedules
