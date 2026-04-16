# Report: Batch Stacked Reduction Round 0 Descriptors

## Description

Batch the per-trace stacked reduction Round 0 kernel launches (sumcheck block sums + PLE fold) using descriptor arrays. At APC 300, the stacked reduction Round 0 loop calls `stacked_reduction_sumcheck_round0` and `stacked_reduction_fold_ple` once per backing trace matrix (~400 per segment), many of which are tiny (1-2 columns, height 2^5-2^8). Grouping same-height matrices into single batched kernel launches was expected to reduce ~800 kernel launches per segment to ~100, saving kernel launch overhead and improving SM utilization for the numerous tiny launches.

## Implementation

### CUDA kernels (`crates/cuda-backend/cuda/src/stacked_reduction.cu`)
- Added `StackedR0Desc` struct: `{trace_ptr, lambda_pows, trace_width}` for batched Round 0.
- Added `FoldPleDesc` struct: `{src, dst, trace_width}` for batched PLE fold.
- Added `upper_bound_search()` device function for binary search on column prefix sums.
- Added `batched_stacked_reduction_round0_block_sum_kernel`: 2D grid (blocks_per_row, total_columns). Each block uses binary search on `col_prefix_sums` to find its descriptor and local column.
- Added `batched_stacked_reduction_fold_ple_kernel`: Same column-based binary search approach.
- Added `_batched_stacked_reduction_sumcheck_round0` launcher: launches batched block_sum kernel, then per-descriptor `final_reduce_block_sums<true>`.
- Added `_batched_stacked_reduction_fold_ple` launcher.

### FFI bindings (`crates/cuda-backend/src/cuda/stacked_reduction.rs`)
- Added `StackedR0Desc` and `FoldPleDesc` repr(C) structs with Send impls.
- Added extern "C" declarations for both batched launchers.
- Added safe Rust wrappers: `batched_stacked_reduction_sumcheck_round0`, `batched_stacked_reduction_fold_ple`.

### Rust orchestration (`crates/cuda-backend/src/stacked_reduction.rs`)
- **Round 0** (`batch_sumcheck_uni_round0_poly`): Replaced per-trace loop with height-grouped batching. Traces are grouped by height, merged, and groups with >=10 traces use the batched kernel path. Small groups use the existing per-trace path.
- **PLE fold** (`fold_ple_evals`): Within each per-commit outer loop, traces are grouped by height and large groups (>=10) use the batched kernel.
- Added `BATCH_THRESHOLD = 10` module-level constant.
- Pre-computes max `block_sums` size upfront to avoid per-trace resize.

No deviations from the plan.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 300** | | | | | |
| STARK excl trace | 2455ms | 1127ms | 1122ms | -1333ms, 2.19x lower | -5ms, 1.00x |
| Stacked Reduction | 311ms | 74ms | 74ms | -237ms, 4.20x lower | 0ms, 1.00x |
| Round 0 | 662ms | 182ms | 188ms | -474ms, 3.52x lower | +6ms, 1.03x higher |
| LogUp GKR | 790ms | 379ms | 368ms | -422ms, 2.15x lower | -11ms, 1.03x lower |
| MLE Rounds | 180ms | 161ms | 162ms | -18ms, 1.11x lower | +1ms, 1.01x higher |
| Trace Commit | 406ms | 197ms | 197ms | -209ms, 2.06x lower | 0ms, 1.00x |
| WHIR | 100ms | 127ms | 128ms | +28ms, 1.28x higher | +1ms, 1.01x |
| **APC 100** | | | | | |
| STARK excl trace | — | 1302ms | 1301ms | — | -1ms, 1.00x |
| Round 0 | — | 165ms | 162ms | — | -3ms, 1.02x lower |
| Stacked Reduction | — | 70ms | 70ms | — | 0ms, 1.00x |
| **APC 0** | | | | | |
| STARK excl trace | — | 1793ms | 1796ms | — | +3ms, 1.00x |
| Round 0 | — | 179ms | 176ms | — | -3ms, 1.02x lower |
| Stacked Reduction | — | 77ms | 77ms | — | 0ms, 1.00x |

All changes are within measurement noise (±10ms).

## Assessment

The optimization did **not** improve performance. All metrics are within measurement noise at all APC configurations. No regression at APC 0.

The expected 7ms improvement from eliminating ~1400 kernel launches was not realized. The per-launch CUDA overhead is ~2-3µs (confirmed by the previous `batch-fold-ple-descriptor-array` task), giving a theoretical savings of ~3-4ms — which is well below the measurement noise floor of ±10ms.

Additionally, the batched path introduces its own overhead:
- CPU-side descriptor array construction and sorting (~0.1ms)
- H2D upload of descriptor and prefix sum arrays (~0.1ms)  
- Binary search in the kernel for each block (~negligible GPU time)

The net savings after subtracting batching overhead is estimated at ~2-3ms, which is undetectable.

The stacked reduction Round 0 has already been optimized to 74ms at APC 300 (down from 311ms baseline), and the remaining time is dominated by GPU kernel execution (not launch overhead). The Stacked Reduction is now only 6.6% of STARK excl trace — too small a target for kernel launch batching to have a measurable impact.

## Future Work

- **Stacked Reduction is at diminishing returns**: At 74ms (6.6% of STARK excl trace at APC 300), further optimizations to stacked reduction have negligible impact on the overall metric.
- **Kernel launch overhead is not a bottleneck**: The CUDA driver pipeline already hides most inter-kernel launch gaps for kernels that execute on the default stream sequentially. Batching only removes gaps that were already hidden.
- **Focus should shift to remaining large components**: LogUp GKR (368ms, 33%), Round 0 (188ms, 17%), MLE Rounds (162ms, 14%), and Trace Commit (197ms, 18%) are the dominant costs at APC 300.
- **Kernel fusion** (combining multiple small sequential kernels into one) could potentially help more than kernel batching for the stacked reduction, since it would reduce actual GPU scheduling overhead rather than just CUDA API overhead.
