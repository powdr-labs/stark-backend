# Report: Column-Batched NTT for RS Code Matrix

## Description

The optimization aimed to improve NTT memory access efficiency in `rs_code_matrix` by processing columns in L2-cache-sized batches instead of all columns simultaneously. At APC 300, the `batch_ntt` processes ~53K columns (each 64KB) in a single call, creating a 3.4GB working set that far exceeds the RTX 4090's 72MB L2 cache. By splitting into batches of ~1000 columns (~64MB per batch), the hypothesis was that the second NTT step would read data still L2-resident from the first step, improving effective bandwidth by 2x and saving 5-15ms on the NTT portion of Trace Commit.

## Implementation

Two files were modified:

1. **`crates/cuda-backend/src/ntt.rs`**: Added `batch_ntt_column_batched()` function that:
   - Falls back to `batch_ntt` when total working set fits in ~60MB (L2 budget)
   - Computes `batch_cols = floor(60MB / col_bytes)` to determine columns per batch
   - Performs bit-reversal on entire buffer once (not batched — scatter/gather has no L2 reuse benefit)
   - Iterates column batches, creating `non_owning` DeviceBuffer views and running the same NTT step schedule as `batch_ntt`

2. **`crates/cuda-backend/src/stacked_pcs.rs`** (lines 254-264): Replaced separate `bit_rev()` + `batch_ntt()` calls with single `batch_ntt_column_batched()` call with `bit_reverse=true`.

No deviations from the plan.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **STARK excl trace APC 300** | 2455ms | 1121ms | 1115ms avg | -1340ms, 2.20x lower | -6ms, 1.01x lower |
| **STARK excl trace APC 100** | 1669ms | 1308ms | 1305ms | -364ms, 1.28x lower | -3ms, 1.00x lower |
| **STARK excl trace APC 0** | 2153ms | 1834ms | 1782ms avg | -371ms, 1.21x lower | -52ms, 1.03x lower |
| **Trace Commit APC 300** | 406ms | 197ms | 191ms avg | -215ms, 2.13x lower | -6ms, 1.03x lower |
| **Trace Commit APC 100** | 348ms | 286ms | 277ms | -71ms, 1.26x lower | -9ms, 1.03x lower |
| **Trace Commit APC 0** | 518ms | 476ms | 465ms avg | -53ms, 1.11x lower | -11ms, 1.02x lower |

APC 300 after-task values are averages of 2 runs (run 1: 1107/190ms, run 2: 1123/192ms).
APC 0 after-task values are averages of 2 runs (run 1: 1779/467ms, run 2: 1784/463ms).

No regression at APC 0 — STARK excl trace 1834ms → 1782ms avg (within noise + slight TC improvement).

## Assessment

**This optimization did not meaningfully improve performance.** The Trace Commit improvement at APC 300 was ~6ms (3%), right at the measurement noise floor. The STARK excl trace change of -6ms at APC 300 is within run-to-run variance.

The primary hypothesis — that L2-sized batching would allow the second NTT step to read data still L2-resident from the first step — did not materialize at scale. The likely explanation:

1. **All threads access the batch concurrently**: The NTT CUDA kernel launches all threads at once across the entire batch. Even with a 64MB batch, the effective working set during any given kernel execution is the full batch, not a sub-portion. L2 eviction happens within the first kernel step, not between steps.

2. **DRAM page locality benefit is real but small**: The ~6ms Trace Commit improvement across all APC configs (~6-11ms) is consistent with modest DRAM page locality improvement from reducing the concurrent address range. But this benefit is much smaller than the 5-15ms expected from L2 reuse.

3. **Kernel launch overhead partially offsets gains**: At APC 300, 53 batches × 2 steps = 106 kernel launches (vs 2). At ~10us each, this adds ~1ms of overhead. At APC 0, 49 batches × 3 steps = 147 launches adding ~1.5ms.

## Future Work

- The NTT is fundamentally memory-bandwidth-bound, not cache-bound. The kernel's thread mapping processes all elements concurrently within each batch, defeating inter-step L2 reuse.
- A kernel-level change that processes NTT stages 1-7 and 8-14 for each column within a single kernel launch (fused multi-stage) could achieve actual L2 reuse, but requires modifying the CUDA kernel code.
- The Trace Commit at APC 300 (197ms) has been reduced from 406ms baseline primarily by the batched stacking scatter kernel. The NTT portion (~50ms estimated from the plan) represents ~25% of Trace Commit — further improvements would need kernel-level optimization or algorithmic changes.
- Profiling the NTT kernel with nsight compute to measure actual L2 hit rate would confirm whether the batching changes the cache behavior at all.
