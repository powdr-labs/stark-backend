# Report: Multi-stream Stacked Reduction Round 0 and PLE Fold

## Description

Multi-stream the per-trace sequential kernel loops in Stacked Reduction's `batch_sumcheck_uni_round0_poly` and `fold_ple_evals`. These functions launch ~797 sequential small CUDA kernels on a single stream at APC 300 (~39ms GPU time). The optimization distributes traces across 8 OS threads with per-thread `cudaStreamPerThread` — the same pattern that improved Round 0 constraint eval by 1.89x and GKR input eval by 1.40x — to enable concurrent kernel execution and reduce wall time.

## Implementation

### Files modified:
- `crates/cuda-backend/src/stacked_reduction.rs` — Main implementation
- `crates/cuda-backend/src/cuda/stacked_reduction.rs` — Made `_stacked_reduction_sumcheck_round0` and `_stacked_reduction_fold_ple` FFI functions public for direct use from worker threads
- `crates/cuda-backend/src/error.rs` — Added `CurrentStreamSync` variant to `StackedReductionError`

### Key changes:

**`batch_sumcheck_uni_round0_poly`**: Pre-computed per-trace work items (trace_ptr, height, width, lambda_offset, n_value), sorted by descending height. For >= 100 traces, pre-allocated per-thread `d_g_pos`, `d_g_neg[0..l_skip]`, and `d_block_sums` buffers (8 threads × 14 buffers = 112 DeviceBuffers). Dispatched work items round-robin across threads. After threads join, reduced per-thread accumulators on CPU (D2H copy + element-wise addition) and uploaded the sums back to device for `reconstruct_s0_from_g`.

**`fold_ple_evals`**: Pre-computed per-trace `dst_offset` values (the running sum of folded heights × widths). For >= 100 traces per stacked_per_commit group, dispatched traces across 8 threads. Each thread writes to non-overlapping regions of the shared `folded_evals` buffer (no per-thread accumulation needed). Called `current_stream_sync()` after `fill_zero` to ensure visibility to worker streams.

**Deviation from plan**: Used `usize` pointer casts for thread safety instead of `SendPtr` wrapper, as the wrapper approach still triggered Rust's `Send` checker on raw pointers inside closures.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1382ms | 1558ms | -897ms, 1.58x lower | +176ms, 1.13x higher |
| Constraints | - | 947ms | 1134ms | - | +187ms, 1.20x higher |
| LogUp GKR | 790ms | 531ms | 724ms | -66ms, 1.09x lower | +193ms, 1.36x higher |
| Round 0 | 662ms | 242ms | 238ms | -424ms, 2.78x lower | -4ms, unchanged |
| MLE Rounds | 180ms | 172ms | 170ms | -10ms, 1.06x lower | -2ms, unchanged |
| Openings | 413ms | 176ms | 175ms | -238ms, 2.36x lower | -1ms, unchanged |
| Stacked Reduction | 311ms | 75ms | 74ms | -237ms, 4.20x lower | -1ms, unchanged |
| Trace Commit | 406ms | 256ms | 247ms | -159ms, 1.64x lower | -9ms, noise |

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155ms | 1745ms | 1979ms | -176ms, 1.09x lower | +234ms, 1.13x higher |
| LogUp GKR | 775ms | 858ms | 1091ms | +316ms, 1.41x higher | +233ms, 1.27x higher |
| Stacked Reduction | 202ms | 68ms | 70ms | -132ms, 2.89x lower | +2ms, unchanged |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2145ms | 2172ms | +19ms, unchanged | +27ms, noise |
| LogUp GKR | 993ms | 1016ms | 1040ms | +47ms, noise | +24ms, noise |
| Stacked Reduction | 113ms | 76ms | 76ms | -37ms, 1.49x lower | 0ms, unchanged |

## Assessment

**Failed.** The optimization did not improve Stacked Reduction performance and caused severe cross-phase regression in LogUp GKR.

**Stacked Reduction itself barely changed**: 75ms → 74ms at APC 300 (-1ms), far below the 15ms rollback threshold. The expected ~31ms savings from concurrent kernel execution did not materialize.

**LogUp GKR regressed significantly**: +193ms at APC 300, +233ms at APC 100. No regression at APC 0 (sequential fallback, < 100 traces). This confirms the regression is from the multi-stream runtime behavior, not binary codegen changes.

**Root cause analysis**:

1. **Buffer allocation overhead dominates savings**: The multi-stream path pre-allocates 112 DeviceBuffers (8 threads × 14 buffers: 1 d_g_pos + 12 d_g_neg + 1 d_block_sums per thread). Each allocation goes through the CUDA memory pool via `cudaMallocAsync`. While individual allocations are fast (~5μs each), 112 allocations add ~1ms overhead, and the subsequent deallocation through `cudaFreeAsync` (112 calls on thread scope exit) disrupts the CUDA memory pool state.

2. **Cross-phase memory pool degradation**: The 112 buffer allocations+frees in Segment 0's Stacked Reduction phase alter the CUDA memory pool's internal state (fragmentation, free-list structure). When Segment 1's LogUp GKR subsequently allocates its working buffers, the pool serves them from different physical locations or requires more coalescing, degrading memory access patterns. This explains why GKR (memory-bandwidth-bound) regresses while compute-bound phases (Round 0, MLE) are unaffected.

3. **CPU reconstruction dominates Stacked Reduction**: The 75ms Stacked Reduction total includes ~39ms GPU kernel time and ~36ms CPU NTT reconstruction. Even eliminating all GPU time would only reduce total to ~36ms. The multi-stream GPU savings (~31ms) are partially offset by D2H reduction overhead (~11ms: 10MB D2H at 12GB/s + 10M field additions on CPU).

4. **fold_ple multi-stream ineffective**: The fold_ple phase contributes only ~9ms GPU time. Multi-streaming would save at most 7ms, but the per-group trace count threshold (>= 100) means only the common_main group activates multi-threading.

## Future Work

- **Avoid per-thread DeviceBuffer allocations**: The multi-stream pattern works well for Round 0 and GKR input eval because those phases don't need per-thread accumulation buffers. Stacked Reduction's `final_reduce_block_sums<true>` uses non-atomic `+=`, requiring per-thread output buffers. Switching to `atomicAdd` in the CUDA kernel would eliminate all per-thread buffers, making multi-streaming feasible without memory pool disruption.

- **Reduce CPU reconstruction time**: The NTT-based polynomial multiplication in `reconstruct_s0_from_g` takes ~36ms on CPU. Moving this to GPU (batched NTT on device) would reduce the CPU bottleneck that limits multi-stream GPU savings.

- **Memory pool warming**: Allocating and immediately freeing scratch buffers before the critical GKR phase could stabilize the pool state. This is fragile and environment-dependent.

- **Kernel fusion**: Instead of multi-streaming many small kernels, fuse the per-trace loop into a single batched kernel (similar to the MLE round batching). This avoids threading overhead entirely and could reduce the ~39ms GPU time to <5ms with a single kernel launch.
