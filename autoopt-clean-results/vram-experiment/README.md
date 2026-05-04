# Peak GPU VRAM check for the autoopt optimizations

Axiom asked whether the autoopt optimizations push peak GPU VRAM above the
GKR fractional sumcheck peak. **No: peak `current_bytes` is byte-for-byte
identical across all 9 steps for every (APC, segment), and is always at
`frac_sumcheck.gkr_rounds`.**

| APC | overall peak (GiB) | GKR peak (GiB) | overall − GKR |
|----:|-------------------:|---------------:|--------------:|
|   0 | 14.23              | 14.23          | 0.000         |
| 100 | 14.85              | 14.85          | 0.000         |
| 300 | 14.64              | 14.64          | 0.000         |

Spot-check, APC=100 segment 0: `current_bytes = 15942253696` for step-00,
step-01, step-08 (identical to the byte). Full breakdown in
[`peaks.csv`](./peaks.csv).

## Plots

- [`vram-apc{000,100,300}.svg`](./vram-apc100.svg) — VRAM over time per
  `app_proof` segment, baseline (step-00) vs optimized (step-08), with
  proving phases shaded (setup → tracegen → trace_commit → gkr → round0 →
  openings) and the GKR peak as a dotted reference line.
- [`peak-by-step-apc{100,300}.svg`](./peak-by-step-apc100.svg) — peak VRAM
  by autoopt step 00…08; all bars equal.

## Reproducing the analysis

```bash
python3 autoopt-clean-results/vram-experiment/plot_vram.py
```

This is pure offline analysis on the metrics JSONs already committed under
`autoopt-clean-results/step-XX-*/apc{000,100,300}.json` — no build, no
prove run. Reproducing those JSONs themselves is described in the
appendix.

## Why this makes physical sense

The fractional-sumcheck leaves buffer is sized by Σᵢ 2 × interactionsᵢ ×
2^heightᵢ in `Frac<EF>` words — a function of the AIR set and trace
heights, not of how each phase is scheduled. The 8 autoopt steps re-stream
Round 0 / GKR input eval, batch MLE kernels, pre-allocate per-thread chip
buffers, and overlap logup precompute, all without enlarging the GKR
leaves buffer or any other allocation. The peak is stable to the byte.

---

## Appendix: how the metrics JSONs were generated

The `gpu_mem.*` time-series come from `MemTracker::emit_metrics_with_label`
in
[`crates/cuda-common/src/memory_manager/mod.rs`](../../crates/cuda-common/src/memory_manager/mod.rs)
(lines ~201–220). Each emission writes four gauges (`gpu_mem.timestamp_ms`,
`current_bytes`, `local_peak_bytes`, `reserved_bytes`) labelled with proof
`group`, `segment`, and `module`. The committed metrics files are the
post-rebase pairing run that produced the timing numbers in
[`../report.md`](../report.md).

### Stark-backend commits used (one per `step-XX-*/` directory)

All on `autoopt-2026-04-12-clean` (this branch).

| dir                                  | commit     | description |
|--------------------------------------|------------|-------------|
| `step-00-baseline`                   | `1d719aa9` | tip of `v2-powdr-beta.2-remove-columns-air` (`Remove ColumnsAir`) — the autoopt baseline |
| `step-01-multistream-round0`         | `0e5835aa` | Multi-stream Round 0 |
| `step-02-multistream-gkr-input`      | `2ba73c0e` | Multi-stream GKR input eval |
| `step-03-stacked-mle-sync`           | `ac0bce3b` | Batch stacked-reduction MLE sync |
| `step-04-stacking-scatter`           | `b04f7627` | Batch stacking scatter kernel |
| `step-05-batch-mle-round-kernels`    | `5a4a2b5a` | Batch MLE round kernels |
| `step-06-prealloc-gkr-buffers`       | `bc5deaf9` | Pre-allocate GKR input buffers |
| `step-07-prealloc-round0-buffers`    | `a2a4a5b1` | Pre-allocate Round 0 buffers + round-robin balance |
| `step-08-gpu-round0-overlap-logup`   | `32074f9e` | GPU Round 0 poly extract + overlap logup precompute |

### Powdr / openvm wiring used to drive the prove

Same as documented in [`../run_benchmark.sh`](../run_benchmark.sh):

- `/home/georg/powdr` on `georgwiese/openvm-deps-update` with commit
  `ad3b6557e Add artifact compile/load support to CLI` cherry-picked back
  in (gives the `--artifact` flag used to load the pre-compiled pairing
  guests).
- `/home/georg/openvm` on `georgwiese/columns-air-trait`.
- The local-development `[patch."https://github.com/powdr-labs/stark-backend.git"]`
  and `[patch."https://github.com/powdr-labs/openvm.git"]` blocks in
  powdr's `Cargo.toml` uncommented so the workspace resolves both to the
  local checkouts.
- Pre-compiled pairing guest artifacts at
  `/home/georg/powdr/results/pairing/apc{000,100,300}.cbor`.

### Run command

For each step, with `stark-backend` checked out at the corresponding
commit and powdr/openvm wired up as above:

```bash
VPMM_PAGE_SIZE=16777216 autoopt-clean-results/run_benchmark.sh <step-tag>
```

`run_benchmark.sh` builds `powdr_openvm_riscv -r --features metrics,cuda`,
then for each APC ∈ {000, 100, 300} runs

```
$PROVE_BIN prove --artifact <apc${APC}.cbor> --input "0" \
                 --metrics <step-tag>/apc${APC}.json --recursion
```

writing the metrics JSON consumed by `plot_vram.py`.

### Hardware

NVIDIA GeForce RTX 4090 (24 GiB VRAM) — same machine as the timing numbers
in `../report.md`.
