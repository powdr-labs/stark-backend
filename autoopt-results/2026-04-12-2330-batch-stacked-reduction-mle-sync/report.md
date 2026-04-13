# Report: Batch Stacked Reduction MLE Sync

## Description

The stacked reduction MLE rounds in `batch_sumcheck_poly_eval()` performed a synchronous D2H copy (`to_host()` with full GPU pipeline drain) per trace-matrix window per MLE round. With ~400+ windows and 17 MLE rounds at APC 300, this created ~7,000+ pipeline-draining sync points per segment. Both CUDA kernels already use atomic accumulation into a shared `d_accum` buffer, and the final result is a sum across all windows, so per-window syncs are unnecessary. This optimization replaces them with a single `fill_zero` + `to_host` per round (~17 syncs per segment total), eliminating the dominant source of CPU-GPU synchronization overhead.

## Implementation

Four changes across two files:

**1. Degenerate kernel wrapper** (`crates/cuda-backend/src/cuda/stacked_reduction.rs`):
Changed `eq_ub_ptr` parameter from `&DeviceBuffer<EF>` to `*const EF` to accept raw pointer offsets into a pre-uploaded buffer.

**2. Struct field** (`crates/cuda-backend/src/stacked_reduction.rs`):
Replaced `d_eq_ub: DeviceBuffer<EF>` (sized to max window) with `d_eq_ub_all: DeviceBuffer<EF>` (sized to full `unstacked_cols.len()`). Removed now-unused `max_window_len` computation.

**3. Restructured `batch_sumcheck_poly_eval` inner loop** (`crates/cuda-backend/src/stacked_reduction.rs`):
- Upload `eq_ub_per_trace` to device once per round (was per-window H2D copy)
- Single `fill_zero` per round (was per-window)
- All windows accumulate atomically into same `d_accum` buffer
- Single `to_host` per round (was per-window with pipeline drain)
- Removed per-window `s_evals_batch` collection and final `sum()` — the atomic accumulator already holds the total

**4. Removed per-window `d_eq_ub` reallocation** (was lines 858-859):
No longer needed since `d_eq_ub_all` has fixed capacity for all columns.

No deviations from the plan. All changes compile cleanly with zero warnings.

## Results

### STARK excl trace (ms) — from spec.py

| APC | Baseline | Before Task | After Task | vs Baseline | vs Before |
|-----|----------|-------------|------------|-------------|-----------|
| 0   | 2153     | 2150        | 2152       | -1ms, 1.00x | +2ms, 1.00x |
| 100 | 2155     | 2166        | 2055       | -100ms, 1.05x lower | -111ms, 1.05x lower |
| 300 | 2455     | 2460        | 2268       | -187ms, 1.08x lower | -192ms, 1.08x lower |

### Stacked Reduction total (ms) — from spec.py (app proof)

| APC | Baseline | Before Task | After Task | vs Baseline | vs Before |
|-----|----------|-------------|------------|-------------|-----------|
| 0   | 113      | 113         | 83         | -30ms, 1.36x lower | -30ms, 1.36x lower |
| 100 | 202      | 201         | 95         | -107ms, 2.13x lower | -106ms, 2.12x lower |
| 300 | 311      | 312         | 126        | -185ms, 2.47x lower | -186ms, 2.48x lower |

### Stacked Reduction MLE Rounds (ms) — from JSON metrics (app proof segments only)

| APC | Baseline | Before Task | After Task | vs Baseline | vs Before |
|-----|----------|-------------|------------|-------------|-----------|
| 0   | 28       | 28          | 16         | -12ms, 1.75x lower | -12ms, 1.75x lower |
| 100 | 113      | 113         | 43         | -70ms, 2.63x lower | -70ms, 2.63x lower |
| 300 | 282      | 280         | 95         | -187ms, 2.97x lower | -185ms, 2.95x lower |

### Other metrics (APC 300, ms) — from spec.py (unchanged)

| Metric | Baseline | Before | After |
|--------|----------|--------|-------|
| Constraints | 1634 | 1639 | 1638 |
| LogUp GKR | 790 | 790 | 793 |
| Round 0 | 662 | 665 | 662 |
| WHIR | 100 | 100 | 99 |
| Trace Commit | 406 | 406 | 402 |

## Assessment

**Clear success.** The optimization achieved its primary goal and exceeded expectations:

- **Stacked Reduction MLE Rounds at APC 300**: 280ms -> 95ms (2.95x lower). The plan estimated 50-70% reduction; achieved 66% reduction.
- **Stacked Reduction total at APC 300**: 312ms -> 126ms (2.48x lower). The plan estimated 45-65% reduction; achieved 60% reduction.
- **STARK excl trace at APC 300**: 2460ms -> 2268ms (7.8% lower, -192ms). The plan estimated 4-8% reduction; achieved 7.8%.
- **No APC 0 regression**: STARK excl trace at APC 0 is flat (2150 -> 2152ms, noise). Stacked Reduction actually improved (113ms -> 83ms, 1.36x lower).
- **Correctness verified**: All three APC configs completed prove+verify successfully.

The complexity is minimal — the change is a straightforward restructuring of the synchronization pattern with no protocol changes. Well worth the small amount of added code.

## Future Work

- **What worked well**: The hypothesis was exactly right — the per-window `to_host()` calls with their `record_and_wait()` pipeline drains were the dominant overhead. Eliminating them gave a near-3x speedup on MLE rounds at APC 300.

- **Further improvements to Stacked Reduction**: The remaining 126ms at APC 300 is split between ~95ms MLE rounds and ~29ms round 0. The 95ms MLE rounds is now closer to actual GPU kernel time, leaving less sync overhead to optimize. Multi-stream parallelism for independent height-group kernels could help further.

- **Round 0 optimization**: Round 0 contributes 662ms at APC 300 and is the largest single component of STARK excl trace. Previous optimization attempts (pipelining, CPU parallelization) yielded only ~1-3% improvement. The bottleneck appears to be in D2H copy latency and per-AIR overhead, not CPU compute or kernel launch serialization.

- **eq_ub incorporation into stable values**: The existing `PERF[jpw]` comment (removed by this change) noted that most of `eq_ub` could be folded into `eq_stable` and `k_rot_stable`, which would eliminate the per-round H2D upload of `eq_ub_per_trace`. With the current optimization, this upload is once per round (not per window), so it's a smaller win, but could still save a few ms.

- **Scaling observation**: The Stacked Reduction MLE rounds now scale much better with APCs: from 16ms (APC 0) to 95ms (APC 300), a 5.9x increase for a workload that has 6.3x more AIR instances. Previously it was 28ms to 280ms (10x increase), indicating severe sync overhead that masked the true scaling.
