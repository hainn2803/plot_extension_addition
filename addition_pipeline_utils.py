"""Shared, resumable bookkeeping for the Addition C -> T -> Y layer pipeline."""

import csv
import hashlib
import json
from pathlib import Path

import torch

from addition_data_load import SPLITS, build_cache_manifest, file_sha256


def parse_layers(spec, num_layers):
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    layers = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a, b = map(int, item.split(":"))
            layers.extend(range(a, b))
        else:
            layers.append(int(item))
    layers = sorted(set(layers))
    if not layers or min(layers) < 0 or max(layers) >= num_layers:
        raise ValueError(f"Layer specification {spec!r} outside [0,{num_layers-1}]")
    return layers


def count_layers(model):
    return len(model.model.layers)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        return
    fields = fields or list(rows[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return file_sha256(path)


def dataset_signature(model, data_dir, cache_dir, max_new_tokens):
    # Check BOTH the model-correct filtered pairs and their raw-data manifest.
    manifest = build_cache_manifest(model, data_dir, max_new_tokens)
    filtered = {}
    for split in SPLITS:
        for pair in ("C_only", "T_only", "C_T"):
            name = f"{split}_{pair}.jsonl"
            filtered[name] = file_hash(Path(cache_dir) / name)
    return {"raw": manifest, "filtered": filtered}


def assert_run_manifest(root, expected):
    root = Path(root)
    path = root / "run_manifest.json"
    if path.exists():
        current = read_json(path)
        if current != expected:
            raise RuntimeError(
                f"Experiment settings/data differ from {path}. "
                "Use a NEW --result-dir (old checkpoints cannot be safely resumed)."
            )
    else:
        # Existing legacy checkpoints with no manifest are unsafe to reuse.
        checkpoints = list(root.glob("**/das_*.pt"))
        if checkpoints:
            raise RuntimeError(f"{root} contains legacy checkpoints without a run manifest")
        write_json(path, expected)


def load_checkpoint(path, variable, layer, method=None, t_layer=None):
    result = torch.load(path, map_location="cpu", weights_only=False)
    if result.get("variable") != variable or int(result.get("layer", -1)) != layer:
        raise RuntimeError(f"Wrong variable/layer in {path}")
    if result.get("metric_version") != "free_autoregressive_numeric_v2":
        raise RuntimeError(f"Incompatible IIA metric version in {path}")
    if method == "gradual":
        if result.get("method") != "gradual" or int(result.get("T_layer", -1)) != t_layer:
            raise RuntimeError(f"Wrong GDAS dependency in {path}")
    elif method == "standard" and variable == "C":
        if result.get("method") != "standard":
            raise RuntimeError(f"Wrong Standard DAS method in {path}")
    return result


def handle_paths(root, variable, layer):
    p = Path(root) / f"{variable}_handles" / f"{variable}_layer_{layer:02d}"
    lower = variable.lower()
    return p / f"das_{lower}.pt", p / f"{lower}_readout_LogisticRegression.pt"


def gradual_paths(root, c_layer, t_layer):
    p = Path(root) / "pairs" / f"C_{c_layer:02d}_T_{t_layer:02d}"
    return p / "das_c.pt", p / "c_readout_LogisticRegression.pt"


def check_artifacts(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing trained checkpoints/readouts: " + ", ".join(missing))
    return {str(path): file_hash(path) for path in paths}


def evaluation_signature(method, c_layer, t_layer, artifacts, data_sig, generation_cap):
    return {
        "method": method,
        "C_layer": c_layer,
        "T_layer": t_layer,
        "artifacts": check_artifacts(artifacts),
        "data": data_sig,
        "generation_cap": generation_cap,
        "metric_version": "free_autoregressive_numeric_v2",
    }
