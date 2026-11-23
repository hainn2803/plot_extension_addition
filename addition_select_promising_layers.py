"""Select promising Addition C/T layers using calibration (validation) IIA only.

Standard: choose each variable's layers within 5 percentage POINTS of its maximum.
GDAS: T is chosen globally, then C is chosen conditionally for each T
because GDAS C handles depend on a frozen T handle/readout.
No test metrics are ever used for layer selection.
"""

import argparse
import csv
from pathlib import Path

from addition_pipeline_utils import write_json, write_csv


def rows_from_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def within_threshold(rows, key, threshold_pct):
    if not rows:
        raise ValueError("No completed calibration rows to select from")
    best = max(float(row[key]) for row in rows)
    cutoff = best - threshold_pct / 100.0
    return [row for row in rows if float(row[key]) >= cutoff - 1e-12]


def select_standard(rows, threshold_pct):
    picked = {}
    for variable in ("C", "T"):
        subset = [row for row in rows if row["variable"] == variable]
        winners = within_threshold(subset, "cal_iia", threshold_pct)
        picked[variable] = sorted({int(row["layer"]) for row in winners})
    pairs = [
        {"C_layer": c, "T_layer": t}
        for t in picked["T"]
        for c in picked["C"]
        if c < t
    ]
    return {"layers": picked, "pairs": pairs}


def select_gradual(rows, threshold_pct):
    # The T handle is shared with Standard DAS. C is conditional on that T.
    by_t = {}
    for row in rows:
        t = int(row["T_layer"])
        c = int(row["C_layer"])
        if c >= t:
            continue
        by_t.setdefault(t, []).append(row)
    if not by_t:
        raise ValueError("No complete GDAS C<T pair rows in summary")

    t_rows = [{"T_layer": t, "T_cal_iia": max(float(row["T_cal_iia"]) for row in group)}
              for t, group in by_t.items()]
    chosen_t = within_threshold(t_rows, "T_cal_iia", threshold_pct)
    pairs = []
    for item in sorted(chosen_t, key=lambda x: x["T_layer"]):
        t = item["T_layer"]
        chosen_c = within_threshold(by_t[t], "C_cal_iia", threshold_pct)
        for row in sorted(chosen_c, key=lambda x: int(x["C_layer"])):
            pairs.append({"C_layer": int(row["C_layer"]), "T_layer": t})
    return {
        "layers": {
            "T": sorted({pair["T_layer"] for pair in pairs}),
            "C": sorted({pair["C_layer"] for pair in pairs}),
        },
        "pairs": pairs,
        "selection_note": "C selected conditionally on each frozen T handle",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--method", choices=("standard", "gradual"), default="standard")
    parser.add_argument("--threshold-pct", type=float, default=5.0,
                        help="Max CAL numeric IIA minus this many percentage points")
    parser.add_argument("--summary", help="Override default <result-dir>/summary.csv")
    parser.add_argument("--output", help="Override default selected_layers_<method>.json")
    args = parser.parse_args()
    if not 0 <= args.threshold_pct <= 100:
        parser.error("--threshold-pct must be in [0,100]")

    root = Path(args.result_dir)
    summary_path = Path(args.summary) if args.summary else root / "summary.csv"
    rows = rows_from_csv(summary_path)
    if args.method == "standard":
        chosen = select_standard(rows, args.threshold_pct)
    else:
        chosen = select_gradual(rows, args.threshold_pct)
    if not chosen["pairs"]:
        raise ValueError("No C<T candidate pair survived selection; inspect per-variable layers")

    output = Path(args.output) if args.output else root / f"selected_layers_{args.method}.json"
    result = {
        "method": args.method,
        "threshold_pct": args.threshold_pct,
        "calibration_summary": str(summary_path),
        **chosen,
    }
    write_json(output, result)
    write_csv(output.with_suffix(".csv"), chosen["pairs"], ["C_layer", "T_layer"])
    print(f"Selected C layers: {chosen['layers']['C']}")
    print(f"Selected T layers: {chosen['layers']['T']}")
    print(f"Valid C<T configurations: {len(chosen['pairs'])}")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
