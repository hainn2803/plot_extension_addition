"""One-time, CPU-only, verified migration of Addition results to MCQA layout.

SDAS:
  C_handles/C_layer_16 -> trained_handles/C_L16
  T_handles/T_layer_18 -> trained_handles/T_L18

GDAS:
  pairs/C_16_T_18     -> trained_handles/T_L18__C_L16
  T_handles/T_layer_18 -> trained_handles/T_L18

Both:
  evaluation_<method>/C_16_T_18/metrics.json
      -> eval_configs/T_L18__C_L16/{metrics.json, eval_faithfulness.*}
  eval_configs/{config_summary.csv,all_config_metrics.csv}

Copy+SHA256-verify each artifact before retiring its old location.
Does not load the model or invoke CUDA, and never retrains/evaluates.
"""

import argparse
import re
from pathlib import Path

from addition_eval_reports import import_legacy, rebuild_summary
from addition_gradual_snapshot import (
    artifact_hashes_by_name,
    copy_immutable,
    freeze_t_snapshots,
)
from addition_pipeline_utils import (
    file_hash, gradual_paths, handle_paths, read_json, write_json,
)


def verified_relocate(old, new):
    old = Path(old)
    new = Path(new)
    if not old.is_file():
        return False
    if old.resolve() == new.resolve():
        return False
    copy_immutable(old, new)
    if file_hash(old) != file_hash(new):
        raise RuntimeError(f"Artifact relocation verification failed: {old} -> {new}")
    old.unlink()
    return True


def remove_if_empty(folder):
    folder = Path(folder)
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()


def migrate_handles(root, method):
    root = Path(root)
    moved = 0
    for variable in ("T", "C"):
        top = root / f"{variable}_handles"
        if not top.exists():
            continue
        for folder in sorted(top.glob(f"{variable}_layer_*")):
            match = re.fullmatch(rf"{variable}_layer_(\d+)", folder.name)
            if match is None:
                continue
            layer = int(match.group(1))
            new_paths = handle_paths(root, variable, layer)
            old_paths = (
                folder / f"das_{variable.lower()}.pt",
                folder / f"{variable.lower()}_readout_LogisticRegression.pt",
            )
            for old, new in zip(old_paths, new_paths):
                moved += int(verified_relocate(old, new))
            remove_if_empty(folder)
        remove_if_empty(top)
    print(f"[{method}] migrated {moved} Standard/frozen T files")


def rebase_calibration_signature(signature, root):
    """Relocate CAL signature paths with checkpoint content verification."""
    if "artifacts" not in signature:
        raise RuntimeError("Missing artifact hashes in CAL signature")
    variable, layer = signature["variable"], int(signature["layer"])
    expected = artifact_hashes_by_name(signature["artifacts"])
    targets = handle_paths(root, variable, layer)
    actual = {path.name: file_hash(path) for path in targets}
    if expected != actual:
        raise RuntimeError(f"Changed CAL artifact for {variable} L{layer}")
    result = dict(signature)
    result["artifacts"] = {str(path): file_hash(path) for path in targets}
    return result


def migrate_calibration_metadata(root):
    root = Path(root)
    count = 0
    for path in sorted((root / "calibration_standard").glob("*_layer_*.json")):
        saved = read_json(path)
        if "signature" not in saved:
            continue
        converted = rebase_calibration_signature(saved["signature"], root)
        if converted != saved["signature"]:
            saved["signature"] = converted
            write_json(path, saved)
            count += 1
    selection = root / "selected_layers_standard.json"
    if selection.exists():
        chosen = read_json(selection)
        signatures = chosen.get("calibration_signatures", {})
        for key, sig in signatures.items():
            if key != f"{sig['variable']}_L{sig['layer']}":
                raise RuntimeError(f"Incorrect selected-layer CAL signature: {key}")
            signatures[key] = rebase_calibration_signature(sig, root)
        write_json(selection, chosen)
    print(f"[standard] rebased {count} cached CAL records and selected layers")


def migrate_conditional_c(root):
    root = Path(root)
    top = root / "pairs"
    if not top.exists():
        return
    moved = 0
    for folder in sorted(top.glob("C_*_T_*")):
        match = re.fullmatch(r"C_(\d+)_T_(\d+)", folder.name)
        if match is None:
            continue
        c, t = map(int, match.groups())
        c_path, c_readout = gradual_paths(root, c, t)
        for old, new in (
            (folder / "das_c.pt", c_path),
            (folder / "c_readout_LogisticRegression.pt", c_readout),
            (folder / "train_signature.json", c_path.parent / "train_signature.json"),
        ):
            moved += int(verified_relocate(old, new))
        remove_if_empty(folder)
    remove_if_empty(top)
    print(f"[gradual] migrated {moved} conditional C checkpoint/metadata files")


def retire_legacy_evaluations(root, method, pairs):
    """Move old metrics only after new metrics and reports exist."""
    old_root = Path(root) / f"evaluation_{method}"
    from addition_eval_reports import config_dir, legacy_dir

    for c, t in pairs:
        old = legacy_dir(root, method, c, t)
        new = config_dir(root, c, t)
        original = old / "metrics.json"
        translated = new / "metrics.json"
        if original.exists() and translated.exists():
            # Verify numeric content identity before dropping obsolete copy.
            if read_json(original) != read_json(translated):
                raise RuntimeError(
                    f"Canonical/legacy evaluation data mismatch: {original}"
                )
            if not (new / "eval_faithfulness.csv").exists():
                raise RuntimeError(f"Missing MCQA CSV report at {new}")
            if not (new / "eval_faithfulness.txt").exists():
                raise RuntimeError(f"Missing MCQA text report at {new}")
            original.unlink()
        remove_if_empty(old)

    # Old summary is regenerated from the preserved per-config JSON,
    # with MCQA filenames, but do not delete other unknown legacy files.
    old_summary = old_root / "summary.csv"
    new_summary = Path(root) / "eval_configs" / "config_summary.csv"
    if old_summary.is_file() and new_summary.is_file():
        old_summary.unlink()
    remove_if_empty(old_root)


def restore_eval(root, method, pairs):
    root = Path(root)
    manifest = read_json(root / "run_manifest.json")
    data_sig = manifest["data"]
    import_legacy(root, method, pairs, data_sig)
    results = rebuild_summary(root, data_sig)
    retire_legacy_evaluations(root, method, pairs)
    print(f"[{method}] {len(results)} MCQA-format eval configs at {root / 'eval_configs'}")


def migrate_standard(root):
    root = Path(root)
    if read_json(root / "run_manifest.json").get("method") != "standard":
        raise RuntimeError(f"Wrong Standard root: {root}")
    migrate_handles(root, "standard")
    migrate_calibration_metadata(root)
    selection = read_json(root / "selected_layers_standard.json")
    pairs = [
        (int(row["C_layer"]), int(row["T_layer"]))
        for row in selection["pairs"]
    ]
    restore_eval(root, "standard", pairs)


def migrate_gradual(root, standard_root):
    root = Path(root)
    standard_root = Path(standard_root)
    if read_json(root / "run_manifest.json").get("method") != "gradual":
        raise RuntimeError(f"Wrong Gradual root: {root}")

    # Move conditional C checkpoints and existing local T copies before
    # verifying T against the dependency recorded during GDAS training.
    migrate_conditional_c(root)
    migrate_handles(root, "gradual")
    pair_file = root / "training_pairs.json"
    if pair_file.exists():
        selection = read_json(pair_file)
    else:
        selection = read_json(standard_root / "selected_layers_standard.json")
    pairs = [
        (int(row["C_layer"]), int(row["T_layer"]))
        for row in selection["pairs"]
    ]
    freeze_t_snapshots(root, standard_root, pairs)
    restore_eval(root, "gradual", pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-root", default="results_addition_standard_all_layers_seed0"
    )
    parser.add_argument(
        "--gradual-root", default="results_addition_gradual_selected_seed0"
    )
    parser.add_argument(
        "--method", choices=("standard", "gradual", "both"), default="both"
    )
    args = parser.parse_args()

    if args.method in ("both", "standard"):
        migrate_standard(args.standard_root)
    if args.method in ("both", "gradual"):
        migrate_gradual(args.gradual_root, args.standard_root)
    print("[done] MCQA layout migrated; no model, no GPU, no retraining")


if __name__ == "__main__":
    main()
