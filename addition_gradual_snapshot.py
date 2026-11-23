"""Freeze SDAS T artifacts INTO the GDAS run for standalone evaluation.

The copied checkpoints are byte-for-byte identical to those used to train
conditional GDAS C handles. Existing runs can be backfilled on CPU, without
loading the model, by executing this file.
"""

import argparse
import os
import re
import shutil
import uuid
from pathlib import Path

from addition_pipeline_utils import (
    file_hash,
    gradual_paths,
    handle_paths,
    read_json,
    write_json,
)


TRAINING_SELECTION_NAME = "training_pairs.json"


def artifact_hashes_by_name(items):
    """Match artifact hashes across directories, never by absolute source path."""
    result = {}
    for path, digest in items.items():
        name = Path(path).name
        if name in result:
            raise ValueError(f"Duplicate artifact basename in signature: {name}")
        result[name] = digest
    return result


def compare_artifact_hashes(expected, actual, context):
    if artifact_hashes_by_name(expected) != artifact_hashes_by_name(actual):
        raise RuntimeError(f"Frozen T checkpoint/readout mismatch for {context}")


def verified_pair(c, t):
    c, t = int(c), int(t)
    if c < 0 or c >= t:
        raise ValueError(f"Invalid GDAS pair: C{c}->T{t}")
    return c, t


def normalize_pairs(pairs):
    return sorted({verified_pair(p["C_layer"], p["T_layer"]) for p in pairs},
                  key=lambda pair: (pair[1], pair[0]))


def copy_immutable(source, destination):
    """Never silently overwrite a checkpoint frozen by a previous run."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != source_hash:
            raise RuntimeError(
                f"Existing GDAS T snapshot differs from SDAS source: {destination}"
            )
        return
    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source, temp)
        if file_hash(temp) != source_hash:
            raise RuntimeError(f"Failed to copy frozen T artifact: {source}")
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def check_pair_upstream(root, c, t, t_hashes):
    """Validate any existing C checkpoint was trained with this exact T."""
    c_path, c_readout = gradual_paths(root, c, t)
    signature_path = c_path.parent / "train_signature.json"
    if not signature_path.exists():
        if c_path.exists() or c_readout.exists():
            raise RuntimeError(f"Existing GDAS pair lacks dependency signature: {c_path.parent}")
        return
    signature = read_json(signature_path)
    if int(signature.get("C_layer", -1)) != c or int(signature.get("T_layer", -1)) != t:
        raise RuntimeError(f"Incorrect GDAS pair identity at {signature_path}")
    compare_artifact_hashes(
        signature["upstream_artifacts"], t_hashes, signature_path
    )


def freeze_t_snapshots(gradual_root, standard_root, pairs):
    gradual_root = Path(gradual_root)
    standard_root = Path(standard_root)
    pairs = normalize_pairs(
        [{"C_layer": c, "T_layer": t} for c, t in pairs]
    )
    if not pairs:
        raise ValueError("No GDAS (C,T) pairs supplied")

    manifest = read_json(gradual_root / "run_manifest.json")
    if manifest.get("method") != "gradual":
        raise RuntimeError(f"Not a GDAS run: {gradual_root}")
    if manifest.get("standard_root") != str(standard_root.resolve()):
        raise RuntimeError(
            "The SDAS source is not the one recorded in the GDAS run manifest"
        )

    pair_file = gradual_root / TRAINING_SELECTION_NAME

    # Preserve the entire prior selection when extending an existing run.
    # Running only a subset of layers must never drop recorded pairs.
    recorded = set()
    if pair_file.exists():
        previous = read_json(pair_file)
        if (previous.get("method") != "gradual" or
                previous.get("selection_source") != "training_pairs_snapshot"):
            raise RuntimeError(f"Unexpected GDAS selection format: {pair_file}")
        recorded = set(normalize_pairs(previous["pairs"]))

    # Also preserve any already-created checkpoint directories, including a
    # pair whose training began before its selection was fully recorded.
    on_disk = set()
    handles = gradual_root / "trained_handles"
    for folder in handles.glob("T_L*__C_L*"):
        match = re.fullmatch(r"T_L(\d+)__C_L(\d+)", folder.name)
        if not match or not folder.is_dir():
            continue
        if any((folder / name).exists() for name in (
                "train_signature.json", "das_c.pt",
                "c_readout_LogisticRegression.pt")):
            t, c = map(int, match.groups())
            on_disk.add(verified_pair(c, t))

    requested = set(pairs)
    preserved = recorded | on_disk
    pairs = sorted(preserved | requested, key=lambda pair: (pair[1], pair[0]))
    payload = {
        "method": "gradual",
        "selection_source": "training_pairs_snapshot",
        "pairs": [{"C_layer": c, "T_layer": t} for c, t in pairs],
    }

    # Check first; do not snapshot a source T handle that differs from the
    # dependency recorded by an already-trained conditional GDAS C handle.
    by_t = {}
    for c, t in pairs:
        source = handle_paths(standard_root, "T", t)
        for p in source:
            if not p.is_file():
                raise FileNotFoundError(p)
        hashes = {str(p): file_hash(p) for p in source}
        check_pair_upstream(gradual_root, c, t, hashes)
        by_t[t] = source

    for t, source_paths in sorted(by_t.items()):
        dest_paths = handle_paths(gradual_root, "T", t)
        for src, dest in zip(source_paths, dest_paths):
            copy_immutable(src, dest)
        print(f"[frozen T snapshot] L{t}: {dest_paths[0].parent}", flush=True)

    # Write only after every old/new pair and frozen T artifact is verified.
    if not pair_file.exists() or read_json(pair_file) != payload:
        write_json(pair_file, payload)
    print(
        f"[GDAS self-contained] {pair_file}: "
        f"{len(preserved)} preserved, {len(requested - preserved)} new, "
        f"{len(pairs)} total pairs",
        flush=True,
    )


def validate_gradual_pair_snapshot(root, c, t):
    """Verify eval uses exactly the frozen T that trained GDAS C."""
    c_path, _ = gradual_paths(root, c, t)
    signature_path = c_path.parent / "train_signature.json"
    if not signature_path.is_file():
        raise FileNotFoundError(signature_path)
    signature = read_json(signature_path)
    if int(signature.get("C_layer", -1)) != c or int(signature.get("T_layer", -1)) != t:
        raise RuntimeError(f"GDAS pair mismatch: {signature_path}")
    t_paths = handle_paths(root, "T", t)
    local_hashes = {}
    for path in t_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen GDAS T snapshot: {path}")
        local_hashes[str(path)] = file_hash(path)
    compare_artifact_hashes(
        signature["upstream_artifacts"], local_hashes, signature_path
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", default="results_addition_standard_all_layers_seed0")
    parser.add_argument("--result-dir", default="results_addition_gradual_selected_seed0")
    parser.add_argument("--selection", help="Source selection JSON (default SDAS selected layers)")
    args = parser.parse_args()

    standard_root = Path(args.standard_root)
    gradual_root = Path(args.result_dir)
    source_selection = Path(args.selection) if args.selection else (
        standard_root / "selected_layers_standard.json"
    )
    selection = read_json(source_selection)
    if selection.get("method") not in ("standard", "gradual"):
        raise ValueError(f"Invalid selection at {source_selection}")
    pairs = normalize_pairs(selection["pairs"])
    # freeze_t_snapshots safely merges the requested selection with existing
    # recorded/on-disk pairs, verifying every frozen T dependency.
    freeze_t_snapshots(gradual_root, standard_root, pairs)


if __name__ == "__main__":
    main()
