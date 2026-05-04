#!/usr/bin/env python3
"""Plot GPU VRAM over time during pairing prove, baseline vs optimized.

Reads per-step `apc{000,100,300}.json` files from
`autoopt-clean-results/step-XX-*/` (committed on the autoopt-2026-04-12-clean
branch), extracts the `gpu_mem.*` gauge time-series, groups module emissions
into proving phases, and produces:

  - vram-apc{000,100,300}.svg : per-segment VRAM-over-time, baseline (step 0)
    vs optimized (step 8), with phase-coloured background
  - peak-by-step-apc{100,300}.svg : per-step peak VRAM bar chart with the
    baseline GKR peak as a reference line
  - peaks.csv : machine-readable peaks for every (step, apc, segment)

Run from the repo root:
    python3 autoopt-clean-results/vram-experiment/plot_vram.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent

STEPS = [
    ("step-00-baseline", "Baseline"),
    ("step-01-multistream-round0", "1: Multi-stream Round 0"),
    ("step-02-multistream-gkr-input", "2: Multi-stream GKR input eval"),
    ("step-03-stacked-mle-sync", "3: Batch stacked MLE sync"),
    ("step-04-stacking-scatter", "4: Batch stacking scatter"),
    ("step-05-batch-mle-round-kernels", "5: Batch MLE round kernels"),
    ("step-06-prealloc-gkr-buffers", "6: Pre-alloc GKR buffers"),
    ("step-07-prealloc-round0-buffers", "7: Pre-alloc Round 0 buffers"),
    ("step-08-gpu-round0-overlap-logup", "8: GPU Round 0 + overlap logup"),
]

# Map module label -> phase. Order in PHASES defines plotting order along x.
PHASES = [
    ("setup", "#cbd5e1"),
    ("tracegen", "#bae6fd"),
    ("trace_commit", "#bbf7d0"),
    ("gkr", "#fecaca"),
    ("round0", "#fed7aa"),
    ("openings", "#ddd6fe"),
]


def module_phase(module: str) -> str:
    """Group emission module labels into phases."""
    if module in ("set initial memory",):
        return "setup"
    if module == "generate mem proving ctxs" or module.startswith("tracegen."):
        return "tracegen"
    if module in ("prover.rs_code_matrix", "prover.stacked_commit"):
        return "trace_commit"
    if module in ("prover.before_gkr_input_evals", "prover.gkr_input_evals") or module.startswith("frac_sumcheck."):
        return "gkr"
    if module.startswith("prover.batch_constraints.") or module == "prover.rap_constraints":
        return "round0"
    if module in ("prover.merkle_tree", "prover.openings", "prover.prove_whir_opening"):
        # `prover.merkle_tree` here is the WHIR opening's merkle tree, not the
        # trace-commit merkle tree (those emissions don't appear separately;
        # they're inside `stacked_commit`).
        return "openings"
    return "other"


@dataclass
class Sample:
    ts_ms: float
    current_mib: float
    local_peak_mib: float
    reserved_mib: float
    module: str
    phase: str


def load_app_proof_samples(path: Path) -> dict[str, list[Sample]]:
    """Return {segment_label: sorted list of Samples} for the app_proof group."""
    with path.open() as f:
        d = json.load(f)
    records: dict[tuple, dict[str, float]] = defaultdict(dict)
    for c in d["gauge"]:
        if not c["metric"].startswith("gpu_mem."):
            continue
        labs = dict(c["labels"])
        if labs.get("group") != "app_proof":
            continue
        seg = labs.get("segment", "<none>")
        mod = labs.get("module", "<none>")
        records[(seg, mod)][c["metric"]] = float(c["value"])

    by_seg: dict[str, list[Sample]] = defaultdict(list)
    for (seg, mod), m in records.items():
        if "gpu_mem.timestamp_ms" not in m:
            continue
        by_seg[seg].append(
            Sample(
                ts_ms=m["gpu_mem.timestamp_ms"],
                current_mib=m.get("gpu_mem.current_bytes", 0) / 2**20,
                local_peak_mib=m.get("gpu_mem.local_peak_bytes", 0) / 2**20,
                reserved_mib=m.get("gpu_mem.reserved_bytes", 0) / 2**20,
                module=mod,
                phase=module_phase(mod),
            )
        )
    for seg in by_seg:
        by_seg[seg].sort(key=lambda s: s.ts_ms)
    return dict(sorted(by_seg.items()))


def gkr_peak(samples: list[Sample]) -> float:
    gkr = [s.current_mib for s in samples if s.phase == "gkr"]
    return max(gkr) if gkr else 0.0


def overall_peak(samples: list[Sample]) -> tuple[float, str, str]:
    """Return (peak_mib, peak_phase, peak_module)."""
    if not samples:
        return 0.0, "", ""
    s = max(samples, key=lambda x: x.current_mib)
    return s.current_mib, s.phase, s.module


def plot_segment(ax, samples: list[Sample], color: str, label: str):
    if not samples:
        return
    t0 = samples[0].ts_ms
    xs = [(s.ts_ms - t0) / 1000.0 for s in samples]
    ys_cur = [s.current_mib / 1024 for s in samples]
    ys_res = [s.reserved_mib / 1024 for s in samples]
    ax.plot(xs, ys_cur, color=color, marker="o", markersize=3, linewidth=1.6, label=f"{label} (current)")
    ax.plot(xs, ys_res, color=color, linestyle="--", linewidth=1.0, alpha=0.55, label=f"{label} (reserved)")


def add_phase_bands(ax, samples: list[Sample]):
    if not samples:
        return
    t0 = samples[0].ts_ms
    # Build phase intervals based on consecutive samples that share a phase.
    phase_colors = dict(PHASES)
    intervals: list[tuple[float, float, str]] = []
    cur_phase = samples[0].phase
    cur_start = (samples[0].ts_ms - t0) / 1000.0
    for i in range(1, len(samples)):
        x = (samples[i].ts_ms - t0) / 1000.0
        if samples[i].phase != cur_phase:
            intervals.append((cur_start, x, cur_phase))
            cur_phase = samples[i].phase
            cur_start = x
    intervals.append(
        (cur_start, (samples[-1].ts_ms - t0) / 1000.0 + 0.01, cur_phase)
    )
    seen_phases = set()
    for x0, x1, phase in intervals:
        c = phase_colors.get(phase, "#f3f4f6")
        ax.axvspan(x0, x1, color=c, alpha=0.45, ec="none", zorder=0)
        if phase not in seen_phases:
            seen_phases.add(phase)
            ax.text(
                (x0 + x1) / 2,
                ax.get_ylim()[1] * 0.97,
                phase,
                ha="center",
                va="top",
                fontsize=8,
                color="#334155",
            )


def plot_apc(apc: int, base_samples: dict[str, list[Sample]],
             opt_samples: dict[str, list[Sample]],
             out_path: Path) -> None:
    segs = sorted(set(base_samples) | set(opt_samples), key=lambda s: int(s) if s.isdigit() else -1)
    n = len(segs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    base_color = "#1d4ed8"
    opt_color = "#dc2626"

    for ax, seg in zip(axes, segs):
        bs = base_samples.get(seg, [])
        os_ = opt_samples.get(seg, [])
        # Determine y-range ahead of time for placing phase labels.
        all_y_mib = [s.current_mib for s in bs + os_] + [s.reserved_mib for s in bs + os_]
        y_max = (max(all_y_mib) if all_y_mib else 1) / 1024 * 1.07
        ax.set_ylim(0, y_max)

        plot_segment(ax, bs, base_color, "baseline")
        plot_segment(ax, os_, opt_color, "step 8 (optimized)")
        # Phase bands derived from the baseline (longer / canonical timeline).
        # If baseline missing, fall back to optimized.
        ref = bs if bs else os_
        add_phase_bands(ax, ref)

        # GKR peak reference line: max GKR current_mib across either branch.
        gkr_ref = max(gkr_peak(bs), gkr_peak(os_)) / 1024
        if gkr_ref > 0:
            ax.axhline(gkr_ref, color="#0f172a", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.text(
                ax.get_xlim()[1] * 0.99 if ax.get_xlim()[1] > 0 else 1,
                gkr_ref,
                f"  GKR peak ≈ {gkr_ref:.2f} GiB",
                va="bottom",
                ha="right",
                fontsize=8,
                color="#0f172a",
            )

        ax.set_title(f"APC={apc}, app_proof segment {seg}", fontsize=11)
        ax.set_xlabel("time within segment (s)")
        ax.set_ylabel("GPU memory (GiB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"GPU VRAM during pairing APC={apc} prove "
        f"(baseline: step-00 / optimized: step-08-gpu-round0-overlap-logup)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_peak_by_step(apc: int, peaks_by_step: dict[str, float],
                      gkr_peaks_by_step: dict[str, float],
                      out_path: Path) -> None:
    labels = [label for _, label in STEPS]
    overall = [peaks_by_step.get(d, 0) / 1024 for d, _ in STEPS]
    gkr = [gkr_peaks_by_step.get(d, 0) / 1024 for d, _ in STEPS]
    baseline_gkr = gkr[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(STEPS)))
    bar_overall = ax.bar(xs, overall, width=0.6, color="#dc2626", label="overall peak (current_bytes)")
    ax.bar(xs, gkr, width=0.4, color="#0ea5e9", alpha=0.7, label="GKR phase peak")
    if baseline_gkr > 0:
        ax.axhline(baseline_gkr, color="#0f172a", linestyle=":", linewidth=1.2,
                   label=f"baseline GKR peak ({baseline_gkr:.2f} GiB)")

    for i, v in enumerate(overall):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Peak GPU memory (GiB)")
    ax.set_title(f"Peak VRAM by autoopt step — pairing APC={apc} (max across app_proof segments)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    out_dir = HERE
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "peaks.csv"

    # 1) Per-(step, apc) load samples
    samples_by_step: dict[tuple[str, int], dict[str, list[Sample]]] = {}
    for step_dir, _label in STEPS:
        for apc in (0, 100, 300):
            jp = RESULTS / step_dir / f"apc{apc:03d}.json"
            if not jp.exists():
                print(f"  WARN missing {jp}", flush=True)
                continue
            samples_by_step[(step_dir, apc)] = load_app_proof_samples(jp)

    # 2) Time-series plots for baseline vs step-08 across APCs
    for apc in (0, 100, 300):
        base = samples_by_step.get(("step-00-baseline", apc), {})
        opt = samples_by_step.get(("step-08-gpu-round0-overlap-logup", apc), {})
        plot_apc(apc, base, opt, out_dir / f"vram-apc{apc:03d}.svg")

    # 3) Per-(step, apc, segment) peaks → CSV
    rows: list[dict] = []
    for (step_dir, apc), per_seg in samples_by_step.items():
        for seg, samples in per_seg.items():
            ov, ph, mod = overall_peak(samples)
            rows.append({
                "step": step_dir,
                "apc": apc,
                "segment": seg,
                "gkr_peak_mib": round(gkr_peak(samples), 1),
                "overall_peak_mib": round(ov, 1),
                "peak_phase": ph,
                "peak_module": mod,
            })
    rows.sort(key=lambda r: (r["apc"], r["step"], r["segment"]))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {csv_path}")

    # 4) Console summary: per-APC max-across-segments overall and GKR peak per branch
    print("\n=== Peak VRAM summary (max across segments) ===")
    print(f"{'APC':>4} {'branch':<14} {'overall (GiB)':>14} {'gkr (GiB)':>12} {'overall - gkr':>14}")
    for apc in (0, 100, 300):
        for label, step in (("baseline", "step-00-baseline"),
                            ("step-08", "step-08-gpu-round0-overlap-logup")):
            ss = samples_by_step.get((step, apc), {})
            if not ss:
                continue
            ov = max((overall_peak(s)[0] for s in ss.values()), default=0)
            gp = max((gkr_peak(s) for s in ss.values()), default=0)
            print(f"{apc:>4} {label:<14} {ov/1024:>14.2f} {gp/1024:>12.2f} {(ov-gp)/1024:>+14.3f}")

    # 5) Per-step bar chart for APC=100 and APC=300
    for apc in (100, 300):
        peaks_by_step: dict[str, float] = {}
        gkr_by_step: dict[str, float] = {}
        for step_dir, _label in STEPS:
            ss = samples_by_step.get((step_dir, apc), {})
            peaks_by_step[step_dir] = max((overall_peak(s)[0] for s in ss.values()), default=0)
            gkr_by_step[step_dir] = max((gkr_peak(s) for s in ss.values()), default=0)
        plot_peak_by_step(apc, peaks_by_step, gkr_by_step,
                          out_dir / f"peak-by-step-apc{apc:03d}.svg")


if __name__ == "__main__":
    main()
