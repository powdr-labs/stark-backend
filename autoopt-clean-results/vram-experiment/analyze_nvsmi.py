#!/usr/bin/env python3
"""Compare nvidia-smi device-wide VRAM peaks against the VPMM-reported peaks.

Reads each `nvsmi-<label>.csv` (timestamp, memory.used [MiB]) and the matching
`metrics-<label>.json` (gpu_mem.* gauges from MemTracker), reports:
- max(memory.used) from nvidia-smi (device-wide, includes CUDA context, kernels,
  framework state, NOT just the VPMM allocator)
- max(gpu_mem.current_bytes) and max(gpu_mem.reserved_bytes) from VPMM gauges
- delta = nvsmi_peak - vpmm_current_peak (expected to be positive: at least the
  CUDA context overhead, ~300-500 MiB on a 4090)
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

LABELS = sys.argv[1:] or ["step-00-apc100", "step-08-apc100"]

MIB = 1024 * 1024


def nvsmi_peak_mib(csv_path: Path) -> tuple[int, str]:
    peak = 0
    peak_ts = ""
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            ts = row[0].strip()
            used_str = row[1].strip().replace(" MiB", "")
            try:
                used = int(used_str)
            except ValueError:
                continue
            if used > peak:
                peak = used
                peak_ts = ts
    return peak, peak_ts


def vpmm_peaks(json_path: Path) -> dict:
    with open(json_path) as f:
        metrics = json.load(f)
    current = 0
    reserved = 0
    for entry in metrics.get("gauge", []):
        m = entry.get("metric")
        v = float(entry.get("value", 0))
        labels = dict(entry.get("labels", []))
        if labels.get("group") != "app_proof":
            continue
        if m == "gpu_mem.current_bytes" and v > current:
            current = v
        elif m == "gpu_mem.reserved_bytes" and v > reserved:
            reserved = v
    return {
        "current_mib": current / MIB,
        "reserved_mib": reserved / MIB,
        "current_bytes": current,
        "reserved_bytes": reserved,
    }


print(
    f"{'label':<20} {'nvsmi peak':>12} {'VPMM cur':>12} {'VPMM rsv':>12} {'nvsmi-VPMM':>12}"
)
print("-" * 74)
for label in LABELS:
    nvsmi_csv = HERE / f"nvsmi-{label}.csv"
    metrics_json = HERE / f"metrics-{label}.json"
    if not nvsmi_csv.exists():
        print(f"{label:<20} (missing nvsmi csv)")
        continue
    if not metrics_json.exists():
        print(f"{label:<20} (missing metrics json)")
        continue
    nvsmi_peak, nvsmi_ts = nvsmi_peak_mib(nvsmi_csv)
    vp = vpmm_peaks(metrics_json)
    delta = nvsmi_peak - vp["current_mib"]
    print(
        f"{label:<20} {nvsmi_peak:>10} MiB {vp['current_mib']:>10.1f} MiB "
        f"{vp['reserved_mib']:>10.1f} MiB {delta:>+10.1f} MiB"
    )
    print(f"  nvsmi peak at {nvsmi_ts}")
