"""Evaluation profiles for Addition, matching MCQA eval_configs_<hash> layout.

Each unique dataset/evaluation setting gets its own output folder for both
Standard and Gradual DAS. Checkpoints live outside this folder.
"""

import hashlib
import json
from pathlib import Path

from addition_pipeline_utils import read_json, write_json


PROFILE_VERSION = 1
EVAL_PAIR_TYPES = ("T_only", "C_only", "C_T")


def profile_settings(data_signature, *, seed, eval_batch_size,
                     max_new_tokens, generation_cap, recipient_version):
    return {
        "profile_version": PROFILE_VERSION,
        "dataset": data_signature,
        "test_split": "test",
        "pair_types": list(EVAL_PAIR_TYPES),
        "seed": int(seed),
        "eval_batch_size": int(eval_batch_size),
        "max_new_tokens": int(max_new_tokens),
        "generation_cap": int(generation_cap),
        "recipient_version": int(recipient_version),
        "metric_version": "free_autoregressive_numeric_v2",
    }


def profile_name(settings):
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"eval_configs_{digest}"


def profile_directory(root, settings):
    folder = Path(root) / profile_name(settings)
    folder.mkdir(parents=True, exist_ok=True)
    settings_path = folder / "eval_settings.json"
    if settings_path.exists():
        if read_json(settings_path) != settings:
            raise RuntimeError(f"Existing evaluation profile mismatch: {settings_path}")
    else:
        write_json(settings_path, settings)
    return folder
