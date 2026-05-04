#!/usr/bin/env python3
"""Build synthetic_results.csv from per-step run-{1,2,3}.json files.

The synthetic benchmark (openvm-benchmark-proving) emits a flat metrics
JSON without the "app_proof" group prefix that spec.py expects, so we
read the raw timing fields directly here.

For each step, we read the three measurement runs and report the median
of each metric.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYNTHETIC_DIR = HERE / "synthetic"
OUT = HERE / "synthetic_results.csv"

STEPS = [
    ("step-00-baseline", "Baseline (VPMM_PAGE_SIZE=16777216)"),
    ("step-01-multistream-round0", "Multi-stream Round 0 (8 streams)"),
    ("step-02-multistream-gkr-input", "Multi-stream GKR input eval (8 streams)"),
    ("step-03-stacked-mle-sync", "Batch stacked-reduction MLE sync"),
    ("step-04-stacking-scatter", "Batch stacking scatter kernel"),
    ("step-05-batch-mle-round-kernels", "Batch stacked-reduction MLE round kernels"),
    ("step-06-prealloc-gkr-buffers", "Pre-allocate GKR input buffers"),
    ("step-07-prealloc-round0-buffers", "Pre-allocate Round 0 buffers + round-robin balance"),
    ("step-08-gpu-round0-overlap-logup", "GPU Round 0 poly extract + overlap logup precompute"),
]

# Metric name in the synthetic JSON -> column alias used in our CSV.
METRICS = [
    ("stark_prove_excluding_trace_time_ms",       "stark_excl_trace_ms"),
    ("prover.rap_constraints_time_ms",            "constraints_ms"),
    ("prover.rap_constraints.logup_gkr_time_ms",  "logup_gkr_ms"),
    ("prover.rap_constraints.round0_time_ms",     "round0_ms"),
    ("prover.rap_constraints.mle_rounds_time_ms", "mle_rounds_ms"),
    ("prover.openings_time_ms",                   "openings_ms"),
    ("prover.openings.whir_time_ms",              "whir_ms"),
    ("prover.openings.stacked_reduction_time_ms", "stacked_reduction_ms"),
    ("prover.main_trace_commit_time_ms",          "trace_commit_ms"),
]


def load_run(path: Path) -> dict[str, float]:
    """Load a single run-N.json and return {metric_name: value}."""
    with path.open() as f:
        d = json.load(f)
    out: dict[str, float] = {}
    for entry in d.get("counter", []) + d.get("gauge", []):
        name = entry["metric"]
        val = float(entry["value"])
        # Some metrics appear multiple times (e.g. round-tagged); we take the sum
        # to mirror what extract_metrics() does. For the prover.* timings we use,
        # there is exactly one entry per run, so this is a no-op.
        out[name] = out.get(name, 0.0) + val
    return out


def main() -> None:
    rows = []
    for step_dir, label in STEPS:
        runs = []
        for n in (1, 2, 3):
            p = SYNTHETIC_DIR / step_dir / f"run-{n}.json"
            if not p.exists():
                print(f"Missing: {p}", file=sys.stderr)
                continue
            runs.append(load_run(p))
        if len(runs) != 3:
            print(f"Skipping {step_dir} (got {len(runs)} runs)", file=sys.stderr)
            continue

        row = {"step": step_dir, "label": label}
        for metric_name, alias in METRICS:
            vals = [r.get(metric_name, 0.0) for r in runs]
            row[f"{alias}_run1"] = round(vals[0])
            row[f"{alias}_run2"] = round(vals[1])
            row[f"{alias}_run3"] = round(vals[2])
            row[alias] = round(statistics.median(vals))
        rows.append(row)

    if not rows:
        print("No rows to write.", file=sys.stderr)
        return

    # CSV header: step, label, then for each metric: median + 3 individual runs
    header = ["step", "label"]
    for _, alias in METRICS:
        header += [alias, f"{alias}_run1", f"{alias}_run2", f"{alias}_run3"]

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT}")

    # Print a quick summary table to stdout for the report.
    print()
    print(f"{'Step':<48} {'Stark excl.':>12} {'(runs)':>30}")
    base = rows[0]["stark_excl_trace_ms"]
    for r in rows:
        med = r["stark_excl_trace_ms"]
        ratio = base / med if med > 0 else 0
        runs = f"({r['stark_excl_trace_ms_run1']}/{r['stark_excl_trace_ms_run2']}/{r['stark_excl_trace_ms_run3']})"
        print(f"{r['label']:<48} {med:>10}ms {runs:>30}  speedup={ratio:.2f}x")


if __name__ == "__main__":
    main()
