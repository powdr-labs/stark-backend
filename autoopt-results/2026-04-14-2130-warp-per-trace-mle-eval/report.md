# Report: 2026-04-14-2130-warp-per-trace-mle-eval

## Description

MLE Rounds at APC 300 (159ms) scales negatively vs APC 0 (112ms, ratio 0.70x) despite 2.3x fewer cells. Small APC-generated traces (num_y ≤ 512) are routed to the `zerocheck_monomial_kernel` which parallelizes over monomials with a fixed block size of 256 threads. Since most AIRs have only 5–20 monomials, thread utilization is 2–8%. The optimization introduced a warp-per-trace kernel that flips the parallelism axis from monomials to y-values: each thread handles one y-value and loops over all monomials, producing per-trace sums via warp reduction.

The expected improvement was 15–25ms on MLE Rounds at APC 300 from eliminating wasted threads, removing the `tmp_sums` intermediary buffer, and eliminating the secondary `batched_final_reduce_block_sums` kernel launch.

## Implementation

### CUDA kernels (`crates/cuda-backend/cuda/src/logup_zerocheck/batch_mle_monomial.cu`)

1. **`warp_zerocheck_monomial_kernel`**: Warp-per-trace zerocheck evaluation. 4 warps per block (128 threads). Each warp handles one trace; thread lane = y_int. Active lanes loop over ALL monomials serially, apply eq_xi, then warp_reduce_sum. Lane 0 writes directly to scattered output position.

2. **`warp_logup_monomial_kernel`**: Same pattern for logup, evaluating both numerator and denominator monomials per trace. Two warp_reduce_sum calls (numer + denom). Adds bus_term_sum contribution per y_int.

3. **`scatter_fpext_blocks_kernel`**: Utility kernel to scatter-copy contiguous blocks of FpExt elements to non-contiguous positions in a destination buffer.

4. **C launchers**: `_warp_zerocheck_monomial_batched`, `_warp_logup_monomial_batched`, `_scatter_fpext_blocks`.

### Rust FFI (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)

Added extern "C" declarations and safe wrappers for all three new CUDA functions plus a `scatter_frac_blocks` helper that reinterprets `Frac<EF>` as `2×EF`.

### Rust logic (`crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs`)

Modified both `ZerocheckMonomialBatch` and `LogupMonomialBatch` to use a two-path strategy:

- **Partition in `new()`**: Traces with `num_y ≤ 32` AND `num_monomials ≤ 32` (zerocheck) or `max(num_numer, num_denom) ≤ 32` (logup) go to the warp path. Remaining traces use the existing block path.
- **Two separate air_ctxs**: Warp path and block path each have their own device context arrays with local indexing.
- **Evaluate**: Warp kernel writes directly to scattered positions in the output buffer via `output_offsets`. Block kernel writes to its own contiguous buffer, then results are scatter-copied to the correct positions.
- **Empty partition handling**: Added `to_device_or_empty()` helper to avoid panics when one path has zero traces.

### Deviations from plan

- Used separate `warp_air_ctxs` and `block_air_ctxs` arrays instead of a single shared `air_ctxs` array. The shared approach required the block kernel's `BlockCtx.air_idx` to use original indices, which conflicted with the reduction kernel's `seg_idx = blockIdx.x` expectation.
- Added scatter-copy kernel instead of modifying the existing reduction kernel for scattered output.

## Results

| Metric | Baseline | Before Task | After Task (avg) | vs Baseline | vs Before |
|--------|----------|-------------|------------------|-------------|-----------|
| STARK excl trace APC 300 | 2455ms | 1301ms | 1304ms | -1151ms, 1.88x lower | +3ms, 1.00x (noise) |
| MLE Rounds APC 300 | 180ms | 158ms | 163ms | -17ms, 1.10x lower | +5ms, 1.03x higher (noise) |
| LogUp GKR APC 300 | 790ms | 536ms | 531ms | -259ms, 1.49x lower | -5ms, 1.01x lower (noise) |
| Round 0 APC 300 | 662ms | 182ms | 180ms | -482ms, 3.68x lower | -2ms, 1.01x lower (noise) |
| STARK excl trace APC 0 | 2153ms | 2150ms | 2160ms | -7ms, 1.00x (noise) | +10ms, 1.00x (noise) |
| MLE Rounds APC 0 | 118ms | 113ms | 115ms | -3ms, 1.03x lower | +2ms, 1.02x (noise) |

After-task values are the average of 2 runs at APC 300.

## Assessment

The optimization **did not improve performance**. MLE Rounds at APC 300 showed no measurable change (158ms → 163ms avg, +5ms within noise). STARK excl trace was unchanged (1301ms → 1304ms avg). No regression at APC 0.

**Root cause**: The `zerocheck_monomial_kernel` itself accounts for only ~17ms of the 158ms MLE Rounds total (from nsight profiling). Even a hypothetical 100% elimination of kernel time would save at most 17ms. The warp kernel doesn't eliminate computation — it reorganizes it. The actual savings come from:

1. Eliminating `tmp_sums` allocation (~0.3ms per call × few calls)
2. Eliminating secondary `batched_final_reduce_block_sums` kernel (~0.5ms per call × few calls)
3. Fewer blocks in flight

These savings total ~2–5ms, well within measurement noise.

**Why the parallelism flip didn't help as expected:**

- For **late_eval traces** (num_y=1, the most common case at APC 300): the warp kernel has only 1 active lane per 32-thread warp (3.1% utilization), which is comparable to the block kernel's 5–15 active threads per 256-thread block (2–6% utilization). The warp approach provides no utilization advantage here.
- For **early-round traces** (num_y > 1): many have num_y > 32 and are not warp-eligible. The traces that ARE warp-eligible (num_y ≤ 32) are a small subset.
- The dominant MLE Rounds costs are other kernels (interpolation, fold, sumcheck polynomial construction), not the monomial evaluation kernels.

## Future Work

- The fundamental bottleneck in MLE Rounds is not monomial kernel utilization but rather the number of kernel launches and the kernel execution time of other components (interpolation, fold).
- A more impactful approach would be to fuse multiple MLE round operations into fewer kernel launches, reducing launch overhead.
- For the monomial evaluation specifically: a persistent kernel that processes all traces from a queue (rather than one kernel launch per batch) could eliminate launch overhead entirely.
- The warp kernel infrastructure (scatter output, partitioned batches) could be useful if combined with other optimizations that increase the warp path's share of total time.
