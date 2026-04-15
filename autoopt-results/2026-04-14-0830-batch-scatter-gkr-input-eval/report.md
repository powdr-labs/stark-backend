# Report: Batch SCATTER-mode GKR Input Evaluation

## Description

At APC 300, ~305 of the 791 GKR input eval kernel instances use SCATTER mode (buffer_size <= 10), each launching a tiny kernel (4-16 blocks) that underutilizes the GPU's 128 SMs. The optimization batches all SCATTER-mode AIR evaluations into a single descriptor-array CUDA kernel launch per segment, running on a dedicated background thread concurrent with the existing 8 multi-stream GLOBAL-mode workers. This was expected to eliminate ~305 kernel launches and their CPU-side overhead while freeing the worker threads to focus on compute-heavy GLOBAL-mode AIRs.

The plan predicted 10-20ms improvement in LogUp GKR at APC 300, acknowledging the improvement would be close to the rollback threshold. The prior `gkr-input-round-robin` task (reverted) had shown that scheduling improvements at this phase have limited impact because aggregate kernel execution time dominates.

## Implementation

### Files Changed

1. **`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`**:
   - Added `#include "eval_ctx.cuh"` for `BlockCtx` struct
   - Defined `GkrInputScatterCtx` descriptor struct matching Rust `#[repr(C)]` layout
   - Implemented `batched_evaluate_interactions_scatter_kernel` — identical DAG evaluation logic as the existing `evaluate_interactions_gkr_kernel<false>`, but reading per-AIR pointers from descriptor array
   - Added `_batched_gkr_input_eval_scatter` extern "C" launcher

2. **`crates/cuda-backend/src/cuda/logup_zerocheck.rs`**:
   - Added Rust `GkrInputScatterCtx` struct with `Send + Sync` impls
   - Added FFI extern declaration and safe wrapper `batched_gkr_input_eval_scatter`

3. **`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`**:
   - Restructured `log_gkr_input_evals` to partition work items into SCATTER and GLOBAL groups
   - Single-threaded path (< 100 AIRs, i.e., APC 0) unchanged
   - Multi-threaded path: background thread handles all SCATTER AIRs (concatenated H2D uploads, descriptor construction, single batched kernel launch, sequential lifting); 8 worker threads handle only GLOBAL AIRs
   - Each lifted SCATTER AIR gets its own section of the tmp buffer (non-overlapping offsets)

### Key Decisions

- **Per-AIR tmp buffer offsets**: The initial implementation (following the plan) used a shared single-offset tmp buffer for all lifted SCATTER AIRs. This caused a correctness bug (NonzeroRootSum) because the batched kernel writes to all lifted AIRs' tmp sections concurrently. Fixed by allocating the total sum of all lifted AIR fracs sizes and giving each AIR its own offset.

- **APC 0 fallback preserved**: The `work_items.len() >= 100` guard ensures APC 0 (99 AIRs) uses the original single-threaded path with no batching.

### Deviations from Plan

- The plan specified `max_scatter_tmp` as the maximum across lifted SCATTER AIRs (reusing a shared buffer sequentially). Changed to sum of all lifted AIRs' sizes to avoid concurrent write conflicts during the batched kernel.

## Results

### APC 300 (target metric)

| Metric | Baseline | Before Task | After Task (median of 3) | vs Baseline | vs Before |
|--------|----------|-------------|--------------------------|-------------|-----------|
| STARK excl trace | 2455 ms | 1302 ms | 1297 ms | -1158 ms (1.89x lower) | -5 ms (1.00x lower) |
| Constraints | 1634 ms | 874 ms | 863 ms | -771 ms (1.89x lower) | -11 ms (1.01x lower) |
| LogUp GKR | 790 ms | 518 ms | 510 ms | -280 ms (1.55x lower) | -8 ms (1.02x lower) |
| Round 0 | 662 ms | 183 ms | 178 ms | -484 ms (3.72x lower) | -5 ms (1.03x lower) |
| MLE Rounds | 180 ms | 171 ms | 173 ms | -7 ms (1.04x lower) | +2 ms (1.01x higher) |
| Openings | 413 ms | 177 ms | 176 ms | -237 ms (2.35x lower) | -1 ms (1.01x lower) |
| Trace Commit | 406 ms | 250 ms | 253 ms | -153 ms (1.60x lower) | +3 ms (1.01x higher) |

Individual after runs: 510ms, 528ms, 510ms (LogUp GKR). Median: 510ms. Range: 18ms.

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155 ms | 1716 ms | 1687 ms | -468 ms (1.28x lower) | -29 ms (1.02x lower) |
| LogUp GKR | 775 ms | 855 ms | 837 ms | +62 ms (0.93x higher) | -18 ms (1.02x lower) |

### APC 000 (regression check)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153 ms | 2157 ms | 2146 ms | -7 ms (1.00x) | -11 ms (1.01x lower) |
| LogUp GKR | 993 ms | 1030 ms | 1022 ms | +29 ms (0.97x higher) | -8 ms (1.01x lower) |

No regressions at APC 0 — single-threaded fallback works correctly.

## Assessment

**Result: Marginal improvement, below rollback threshold.**

The optimization produces a median LogUp GKR improvement of ~8ms at APC 300 (518ms → 510ms), which is below the 10ms rollback threshold specified in the plan. At APC 100, the single run showed a larger 18ms improvement, but this may be within noise given the high run-to-run variance observed at APC 300 (18ms range across 3 runs).

The plan correctly predicted this outcome as a risk: "the per-thread savings (~1.6ms) are modest" and "the improvement is expected to be 10-20ms, close to the rollback threshold." The fundamental issue is that the SCATTER AIRs' aggregate kernel time (~63ms GPU time from nsight profiling) is already hidden behind the GLOBAL AIRs' processing (~464ms) in the multi-stream setup. The batching reduces CPU-side overhead per AIR but doesn't significantly change the critical-path thread's completion time.

**Decision: Revert.** The improvement is not reliably measurable above noise and fails the rollback criterion.

## Future Work

- **What worked**: The descriptor-array batched kernel pattern works correctly and produces valid proofs. The BlockCtx + per-AIR context approach can be reused for other batching needs.

- **Why limited impact**: SCATTER AIRs are already fast relative to GLOBAL AIRs. The bottleneck in GKR input eval is the GLOBAL AIRs, not the SCATTER ones. The background thread for SCATTER runs concurrently, so its absolute runtime doesn't matter — only whether removing SCATTER from the main workers' queues helps. The savings per worker thread (~1.6ms) are modest compared to the 60ms+ critical-path duration.

- **More promising targets**: To significantly improve LogUp GKR at APC 300, focus on:
  1. Reducing GLOBAL AIR kernel launch overhead (kernel fusion across small GLOBAL AIRs)
  2. Better load balancing of GLOBAL AIRs across worker threads
  3. Overlapping GKR input eval with the preceding logup_combination_precompute phase

- **Potential combination**: This SCATTER batching could be valuable if combined with other optimizations that reduce GLOBAL processing time enough to make the SCATTER thread visible on the critical path. As a standalone optimization, the impact is too small.
