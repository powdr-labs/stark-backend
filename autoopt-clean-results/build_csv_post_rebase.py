#!/usr/bin/env python3
"""Build the post-rebase pairing CSV from autoopt-clean-results/pairing-post-rebase/<step>/apc{000,100,300}.json.

Identical schema to autoopt-clean-results/results.csv (which holds the pre-rebase
pairing numbers); the only thing that changed is the underlying stark-backend
state these were measured against.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "pairing-post-rebase"
OUT = HERE / "results_post_rebase.csv"

sys.path.insert(0, "/home/georg/autoopt")
from spec import extract_metrics  # type: ignore

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

TIME_FIELDS = [
    ("app_proof_time_excluding_trace_ms", "stark_excl_trace_ms"),
    ("app_rap_constraints_time_ms", "constraints_ms"),
    ("app_rap_logup_gkr_time_ms", "logup_gkr_ms"),
    ("app_rap_round0_time_ms", "round0_ms"),
    ("app_rap_mle_rounds_time_ms", "mle_rounds_ms"),
    ("app_openings_time_ms", "openings_ms"),
    ("app_openings_whir_time_ms", "whir_ms"),
    ("app_openings_stacked_reduction_time_ms", "stacked_reduction_ms"),
    ("app_trace_commit_time_ms", "trace_commit_ms"),
    ("app_proof_time_ms", "app_proof_time_ms"),
    ("total_proof_time_ms", "total_ms"),
]


def main() -> None:
    field_headers = [f[1] for f in TIME_FIELDS]
    rows = []

    for step_dir, label in STEPS:
        for apc in ("000", "100", "300"):
            path = SOURCE_DIR / step_dir / f"apc{apc}.json"
            if not path.exists():
                print(f"Missing: {path}", file=sys.stderr)
                continue
            with path.open() as f:
                data = json.load(f)
            m = extract_metrics(step_dir, data)
            row = {"step": step_dir, "label": label, "apc": int(apc)}
            for key, alias in TIME_FIELDS:
                val = m.get(key)
                row[alias] = round(val) if val is not None else ""
            rows.append(row)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "label", "apc"] + field_headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT}")

    # Quick summary: stark_excl_trace_ms per step per APC
    print(f"\n{'Step':<48} {'APC 0':>8} {'APC 100':>10} {'APC 300':>10}")
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], {})[r["apc"]] = r["stark_excl_trace_ms"]
    for step_dir, label in STEPS:
        if step_dir not in by_step:
            continue
        s = by_step[step_dir]
        a0 = s.get(0, "—")
        a1 = s.get(100, "—")
        a3 = s.get(300, "—")
        print(f"{label:<48} {a0:>8} {a1:>10} {a3:>10}")


if __name__ == "__main__":
    main()
