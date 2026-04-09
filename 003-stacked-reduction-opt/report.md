# Report: Combined GPU Prover Optimizations (003-stacked-reduction-opt)

## Overall Idea

Multiple GPU prover optimizations targeting per-AIR kernel launch overhead, GPU-CPU synchronization barriers, and memory allocation inefficiency. The optimizations compound: parallel streams overlap kernel launches, batching reduces launch count, deferred sync eliminates barriers, and VPMM pre-commitment avoids page faults.

## Implementation

All changes on branch `003-stacked-reduction-opt` (12 commits on top of `001-round0-parallel-streams`):

1. **Stacked Reduction Deferred Sync** (cherry-pick c868e4b): Batch all window kernel launches, single D2H after.
2. **GKR Input Eval Batching** (cherry-pick 514d9d0): BlockCtx/GkrInputCtx pattern, 2-12 launches instead of ~791.
3. **8 Round 0 Threads + Interleaved Assignment**: Better GPU overlap + load balancing.
4. **Batched Degenerate Stacked Reduction**: DegenerateWindowCtx, one kernel for all degenerate windows.
5. **Batched Round 0 Zerocheck**: Round0ZerocheckCtx/Round0BlockCtx, batches 91% of traces.
6. **Logup Round 0 Rules Caching**: Cache DAG at keygen.
7. **Parallel Logup Precompute**: 8 threads for precompute_logup_combinations.
8. **Selector Caching**: Share DeviceBuffers across same-height traces.
9. **Parallel eq_3b Computation**: 8 threads for CPU-bound eval_eq_mle.
10. **VPMM 8192-page Pre-commitment**: Pre-commit ~16GB at startup.

## Results

### Comparison: Baseline vs All Optimizations

| Phase | APC000 Base | APC000 Opt | Factor | APC100 Base | APC100 Opt | Factor | APC300 Base | APC300 Opt | Factor |
|-------|-------------|------------|--------|-------------|------------|--------|-------------|------------|--------|
| **STARK excl. trace** | **2176ms** | **2161ms** | **0.99x** | **2179ms** | **1614ms** | **0.74x** | **2491ms** | **1558ms** | **0.63x** |
| LogUp GKR | 1010ms | 1034ms | 1.02x | 783ms | 646ms | 0.83x | 800ms | 605ms | 0.76x |
| Round 0 | 179ms | 171ms | 0.96x | 470ms | 162ms | 0.34x | 670ms | 170ms | 0.25x |
| MLE Rounds | 118ms | 118ms | 1.00x | 149ms | 150ms | 1.01x | 181ms | 182ms | 1.01x |
| Stacked Reduction | 113ms | 81ms | 0.72x | 205ms | 80ms | 0.39x | 311ms | 89ms | 0.29x |
| Trace Commit | 519ms | 524ms | 1.01x | 418ms | 420ms | 1.00x | 408ms | 407ms | 1.00x |
| WHIR | 218ms | 229ms | 1.05x | 137ms | 138ms | 1.01x | 100ms | 100ms | 1.00x |
| **Total** | **5127ms** | **5060ms** | **0.99x** | **6106ms** | **5440ms** | **0.89x** | **6981ms** | **5919ms** | **0.85x** |

### Relative to APC000 (optimized)

| Phase | APC000 Opt | APC300 Opt | APC300/APC000 |
|-------|------------|------------|---------------|
| STARK excl. trace | 2161ms | 1558ms | 0.72x |
| Cells | 1.90B | 811M | 0.43x |
| Target (proportional) | 2161ms | 919ms | 0.43x |

APC300 STARK is now 28% lower than APC000 (was 14% higher before optimization).

## Future Work

1. **GKR leaves VPMM overhead**: Segment 0 GKR input evals takes 316ms vs 2ms for segment 1. The 314ms is from VPMM pool contiguous block search. A direct cudaMallocAsync path for this specific buffer (`with_capacity_direct`) is implemented but needs testing and validation that it doesn't regress other phases.

2. **GKR fractional sumcheck**: ~400ms sequential Fiat-Shamir. Cannot be optimized without protocol changes.

3. **Trace Commit**: ~407ms Poseidon2 hashing. Cannot be optimized without faster hash implementation.

4. **Logup Round 0 kernel batching**: CUDA-level batching of barycentric/NTT evaluation could reduce the remaining 735 logup kernel launches.

5. **SP1-inspired constraint batching**: Fold all AIR constraints into a single sumcheck instance, as SP1 does with the Hypercube approach.
