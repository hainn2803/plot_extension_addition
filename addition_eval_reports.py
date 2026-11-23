"""Common MCQA-style evaluation layout for both Addition SDAS and GDAS.

Both methods use <result-dir>/eval_configs_<hash>/T_L{t}__C_L{c}/ with
metrics.json, eval_faithfulness.txt and eval_faithfulness.csv.
The shared evaluation root also holds config_summary.csv and
all_config_metrics.csv. Old evaluation_<method>/ results are imported
without rerunning inference.
"""

import shutil
from pathlib import Path

from addition_pipeline_utils import read_json, write_csv


FAMILIES = ("T_only", "C_only", "C_T")


def config_dir(eval_dir, c_layer, t_layer):
    return Path(eval_dir) / f"T_L{int(t_layer)}__C_L{int(c_layer)}"


def legacy_dir(root, method, c_layer, t_layer):
    return Path(root) / f"evaluation_{method}" / f"C_{int(c_layer):02d}_T_{int(t_layer):02d}"


def import_legacy(root, method, pairs, data_sig, eval_dir, seed):
    """Import earlier result layouts to a hashed MCQA evaluation profile.

    Legacy directories did not identify the evaluation seed in their path.
    Only seed=0 may import legacy data without an explicit recipient_seed.
    Importing an already-run evaluation never runs GPU inference.
    """
    root = Path(root)
    eval_dir = Path(eval_dir)
    if seed != 0:
        return 0
    count = 0
    for c, t in pairs:
        destination = config_dir(eval_dir, c, t) / "metrics.json"
        if destination.exists():
            continue
        sources = [
            legacy_dir(root, method, c, t) / "metrics.json",
            config_dir(root / "eval_configs", c, t) / "metrics.json",
        ]
        for old in sources:
            if not old.is_file():
                continue
            payload = read_json(old)
            if (payload.get("method") != method or
                    payload.get("C_layer") != c or payload.get("T_layer") != t):
                raise RuntimeError(f"Unexpected legacy evaluation identity: {old}")
            if payload.get("signature", {}).get("data") != data_sig:
                continue
            if payload.get("recipient") and payload.get("recipient_seed", 0) != seed:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, destination)
            count += 1
            print(f"[import old evaluation] {old} -> {destination}", flush=True)
            break
    return count

def classify_metric(metric):
    if "recipient_" in metric:
        return "recipient"
    if metric.startswith("Y_restored_"):
        return "restoration"
    if metric.startswith("Y_via_"):
        return "mediator"
    return "correctness"


def detail_rows(payload):
    records = []
    for dataset in FAMILIES:
        intervention = "T" if dataset == "T_only" else "C"
        groups = (payload.get("metrics", {}).get(dataset, {}),
                  payload.get("recipient", {}).get(dataset, {}))
        for metrics in groups:
            for metric, value in metrics.items():
                records.append({
                    "C_layer": int(payload["C_layer"]),
                    "T_layer": int(payload["T_layer"]),
                    "family": classify_metric(metric),
                    "intervention": intervention,
                    "dataset": dataset,
                    "metric": metric,
                    "value": float(value),
                })
    return records


def flatten(payload):
    row = {
        "method": payload["method"],
        "C_layer": payload["C_layer"],
        "T_layer": payload["T_layer"],
        "C_cal_iia": payload["calibration"]["C_cal_iia"],
        "T_cal_iia": payload["calibration"]["T_cal_iia"],
    }
    for name, value in payload.get("clean", {}).items():
        row[f"clean_{name}"] = value
    for dataset in FAMILIES:
        for metric, value in payload.get("metrics", {}).get(dataset, {}).items():
            row[f"{dataset}_{metric}"] = value
        for metric, value in payload.get("recipient", {}).get(dataset, {}).items():
            row[f"{dataset}_{metric}"] = value
    return row


def report_text(payload):
    c = int(payload["C_layer"])
    t = int(payload["T_layer"])
    lines = [
        f"CONFIG: C=L{c} T=L{t}",
        f"METHOD: {payload['method']}",
        "",
        "=== CLEAN READOUT SANITY ===",
    ]
    for name in ("C_base", "C_source", "T_base", "T_source"):
        if name in payload.get("clean", {}):
            lines.append(f"{name} = {float(payload['clean'][name]):.4f}")
    lines += ["", "=== 1. CORRECTNESS / READOUT ===", ""]
    for dataset in FAMILIES:
        lines.append(f"NODE CORRECTNESS — {dataset}")
        metrics = payload.get("metrics", {}).get(dataset, {})
        for key, value in metrics.items():
            if classify_metric(key) == "correctness":
                lines.append(f"{key} = {float(value):.4f}")
        lines.append("")
    lines += ["=== 2. MEDIATOR REPLAY ===", ""]
    for dataset in ("C_only", "C_T"):
        lines.append(f"MEDIATOR — {dataset}")
        for key, value in payload.get("metrics", {}).get(dataset, {}).items():
            if classify_metric(key) == "mediator":
                lines.append(f"{key} = {float(value):.4f}")
        lines.append("")
    lines += ["=== 3. RECIPIENT TRANSFER (different question) ===", ""]
    for dataset in FAMILIES:
        recipient_metrics = payload.get("recipient", {}).get(dataset, {})
        if not recipient_metrics:
            continue
        lines.append(f"RECIPIENT — {dataset}")
        for key, value in recipient_metrics.items():
            lines.append(f"{key} = {float(value):.4f}")
        lines.append("")
    lines += ["=== 4. RESTORATION ===", ""]
    for dataset in ("C_only", "C_T"):
        lines.append(f"RESTORATION — {dataset}")
        for key, value in payload.get("metrics", {}).get(dataset, {}).items():
            if classify_metric(key) == "restoration":
                lines.append(f"{key} = {float(value):.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


DETAIL_FIELDS = [
    "C_layer", "T_layer", "family", "intervention", "dataset", "metric", "value"
]


def write_config_reports(config_folder, payload):
    config_folder = Path(config_folder)
    config_folder.mkdir(parents=True, exist_ok=True)
    (config_folder / "eval_faithfulness.txt").write_text(report_text(payload))
    write_csv(
        config_folder / "eval_faithfulness.csv",
        detail_rows(payload),
        DETAIL_FIELDS,
    )


def rebuild_summary(eval_dir, data_sig):
    """Recreate MCQA-style reports for one eval_configs_<hash> profile."""
    evaluation_root = Path(eval_dir)
    summary_rows = []
    details = []
    for path in sorted(evaluation_root.glob("T_L*__C_L*/metrics.json")):
        payload = read_json(path)
        if payload.get("signature", {}).get("data") != data_sig:
            continue
        if path.parent != config_dir(evaluation_root, payload["C_layer"], payload["T_layer"]):
            raise RuntimeError(f"Result stored in wrong configuration directory: {path}")
        write_config_reports(path.parent, payload)
        summary_rows.append(flatten(payload))
        details.extend(detail_rows(payload))
    summary_rows.sort(key=lambda row: (int(row["T_layer"]), int(row["C_layer"])))
    if summary_rows:
        fields = list(summary_rows[0])
        for row in summary_rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        write_csv(evaluation_root / "config_summary.csv", summary_rows, fields)
        write_csv(evaluation_root / "all_config_metrics.csv", details, DETAIL_FIELDS)
    return summary_rows
