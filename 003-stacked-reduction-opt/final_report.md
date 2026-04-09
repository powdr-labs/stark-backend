# Final Report: Combined GPU Prover Optimizations

## Summary

Five optimizations were applied to improve GPU prover STARK excl. trace time for APC configurations:

| # | Optimization | Commit | APC300 Impact |
|---|-------------|--------|---------------|
| 1 | Round 0 Parallel CUDA Streams (8 threads) | 2ef4093, 6fc1602 | Round 0: 670ms -> 233ms (-65%) |
| 2 | Stacked Reduction Deferred Sync | 1d61faf (cherry-pick c868e4b) | Stacked Red: 311ms -> 90ms (-71%) |
| 3 | GKR Input Eval Batching | 8e437d1 (cherry-pick 514d9d0) | GKR: 800ms -> 597ms (-25%) |
| 4 | Batched Degenerate Stacked Reduction | e728b55 | Stacked Red: 128ms -> 90ms (-30%) |
| 5 | GKR Threshold 10->32 (REVERTED) | - | REVERTED: Hurts RTX 4090 occupancy |

## Combined Results

### STARK (excl. trace) Phase Times

| Config | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| APC000 | 2176ms | 2147ms | **-1.3%** |
| APC100 | 2179ms | 1647ms | **-24.4%** |
| APC300 | 2491ms | 1621ms | **-34.9%** |

### APC300 Detailed Phase Breakdown

| Phase | Baseline | Optimized | Delta |
|-------|----------|-----------|-------|
| **STARK excl. trace** | **2491ms** | **1621ms** | **-870ms (-34.9%)** |
| LogUp GKR | 800ms | 597ms | -203ms (-25.4%) |
| Round 0 | 670ms | 233ms | -437ms (-65.2%) |
| MLE Rounds | 181ms | 183ms | +2ms (noise) |
| Stacked Reduction | 311ms | 90ms | -221ms (-71.1%) |
| Trace Commit | 408ms | 405ms | -3ms (noise) |
| WHIR | 100ms | 100ms | 0ms |

### APC Scaling

| Metric | APC000 | APC300 | Ratio |
|--------|--------|--------|-------|
| Cells | 1.90B | 811M | 0.43x |
| **Baseline STARK** | **2176ms** | **2491ms** | **1.14x (WORSE)** |
| **Optimized STARK** | **2147ms** | **1621ms** | **0.76x (BETTER!)** |
| Target (proportional) | 2147ms | 913ms | 0.43x |

Before optimization, APC300 STARK was 14% WORSE than APC000 despite 2.35x fewer cells.
After optimization, APC300 is 24% BETTER, following the correct direction.

### Total Proof Time

| Config | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| APC000 | 5.13s | 5.00s | -0.13s (-2.5%) |
| APC100 | 6.11s | 5.50s | -0.61s (-10.0%) |
| APC300 | 6.98s | 6.05s | **-0.93s (-13.3%)** |

## Remaining Gap Analysis

Target APC300 STARK: ~913ms. Achieved: 1621ms. Gap: 708ms.

| Phase | Time | Proportion | Scalable? |
|-------|------|------------|-----------|
| GKR (fractional sumcheck) | 597ms | 37% | Partially (tree height) |
| Trace Commit | 405ms | 25% | Partially (Merkle tree) |
| Round 0 | 233ms | 14% | Could batch kernels further |
| MLE Rounds | 183ms | 11% | CPU overhead dominated |
| WHIR | 100ms | 6% | Scales well |
| Stacked Reduction | 90ms | 6% | Well optimized |

The gap is dominated by GKR fractional sumcheck (inherently sequential per-round due to Fiat-Shamir) and Trace Commit (cryptographic hashing). These are algorithmic/cryptographic bottlenecks that don't scale proportionally with cells.

## Nsight Profiling Insights

From the optimized binary profile:
- 47K CUDA kernel launches (down from 57K after degenerate batching)
- 142K cudaMemcpyAsync calls (1739ms total) - dominated by tiny H2D transfers
- 1223 cudaStreamSynchronize (1164ms) - inherent to protocol rounds
- Top kernel: apc_apply_bus_kernel (604ms, trace gen, out of scope)

## What Was Tested But Rejected

- **GKR threshold 10->32**: Increases register pressure, reduces occupancy on RTX 4090. Net negative.
- **Round 0 kernel batching (cross-AIR)**: Cherry-picked from pacheco branch but had NttEvalContext API incompatibility and conflicts with parallel streams. Deferred.
- **l_skip tuning**: Previous report found l_skip=4 is already optimal.

## Technical Notes

- All optimizations compile and pass verification (proofs verify correctly).
- Parallel streams use `std::thread::scope` + `--default-stream=per-thread` CUDA compilation.
- Deferred D2H avoids the global COPY_EVENT mutex serialization.
- Batched kernels use BlockCtx/context-struct pattern for per-AIR dispatch.
- Branch: `003-stacked-reduction-opt`, 6 commits on top of `v2-powdr-07-04`.
