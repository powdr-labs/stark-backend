# Report: Pipeline Round 0 Kernel Launches

## Description

The Round 0 constraint evaluation loop in `logup_zerocheck/mod.rs` processes each AIR instance sequentially with two GPU synchronization points per AIR (`to_host()` at the end of each constraint and interaction kernel). With 623 AIR instances at APC 300, this creates ~1,246 GPU synchronizations that serialize the pipeline and prevent the GPU from keeping its execution queue full.

The optimization restructures the loop into three phases:
1. **Phase A (Prepare)**: Pre-compute all per-AIR metadata, interaction data (DAG construction, rule encoding, weight computation), and pre-allocate shared GPU intermediate buffers.
2. **Phase B (Launch)**: Submit all GPU kernels back-to-back without any synchronization between AIRs, reusing shared intermediate buffers.
3. **Phase C (Collect)**: Single `cudaStreamSynchronize`, then batch D2H copies and CPU post-processing (transpose, iDFT, polynomial construction).

The hypothesis was that eliminating per-AIR sync points would allow the GPU to execute kernels back-to-back, reducing GPU idle time and improving Round 0 performance by 15-30%.

## Implementation

### Files modified:
- `crates/cuda-backend/src/logup_zerocheck/round0.rs`:
  - Added `InteractionRound0Prep` struct to hold pre-computed interaction data (device-side rules, weights, buffer size)
  - Added `prepare_round0_interactions()` to extract CPU-heavy interaction preparation from `evaluate_round0_interactions_gpu()`
  - Added `launch_round0_constraints_kernel()` — kernel-launch-only variant using externally-owned buffers
  - Added `launch_round0_interactions_kernel()` — same for interactions
  - Added buffer size helper functions: `zc_intermediates_size`, `zc_temp_sums_size`, `logup_intermediates_size`, `logup_temp_sums_size`
  - Kept original `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu` as backward-compatible wrappers

- `crates/cuda-backend/src/logup_zerocheck/mod.rs`:
  - Replaced the single per-AIR loop in `sumcheck_uni_round0_polys()` with three-phase structure
  - Phase A: pre-computes `Round0AirData` for all AIRs, pre-allocates shared intermediate buffers
  - Phase B: launches all constraint + interaction kernels without sync
  - Phase C: calls `current_stream_sync()` once, then batch-processes all results

- `crates/cuda-backend/src/error.rs`:
  - Added `CurrentStreamSync(CudaError)` variant to `LogupZerocheckError`

### Deviations from plan:
- Removed the unused `pk` parameter from `prepare_round0_interactions()` since only `symbolic` constraints are needed
- No deviations in overall approach

## Results

### Round 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| Round 0 APC 0 | 178 ms | 177 ms | 176 ms | -2 ms (1.01x lower) | -1 ms (1.01x lower) |
| Round 0 APC 100 | 464 ms | 469 ms | 429 ms | -35 ms (1.08x lower) | -40 ms (1.09x lower) |
| Round 0 APC 300 | 662 ms | 665 ms | 645 ms | -17 ms (1.03x lower) | -20 ms (1.03x lower) |

### STARK (excl. trace)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace APC 0 | 2153 ms | 2169 ms | 2415 ms | +262 ms (1.12x higher) | +246 ms (1.11x higher) |
| STARK excl trace APC 100 | 2155 ms | 2162 ms | 2126 ms | -29 ms (1.01x lower) | -36 ms (1.02x lower) |
| STARK excl trace APC 300 | 2455 ms | 2479 ms | 2452 ms | -3 ms (1.00x) | -27 ms (1.01x lower) |

Notes:
- APC 0 STARK excl trace regression (2169→2415) is due to Trace Commit noise (526→726ms), unrelated to Round 0 changes. Round 0 itself is unchanged at 176ms.
- APC 300 re-run confirmed Round 0 = 645ms (stable).

### Constraints breakdown (APC 300)

| Sub-metric | Before | After | Change |
|-----------|--------|-------|--------|
| Constraints | 1648 ms | 1617 ms | -31 ms (1.02x lower) |
| LogUp GKR | 797 ms | 788 ms | -9 ms |
| Round 0 | 665 ms | 645 ms | -20 ms |
| MLE Rounds | 183 ms | 180 ms | -3 ms |

## Assessment

**The optimization did NOT achieve its goal.** The Round 0 improvement at APC 300 was only ~3% (20ms), well below the rollback threshold of 10%.

### Why the improvement was smaller than expected:

1. **Most of the overhead is CPU post-processing, not GPU stall time.** The 172ms overhead identified in profiling includes CPU-side transpose, iDFT, and polynomial construction. Our optimization moves this work to Phase C but doesn't reduce its total amount — it's the same work, just batched.

2. **cudaStreamPerThread serializes kernels regardless.** Since all kernels run on the same per-thread default stream, the GPU already processes them sequentially. The "pipeline" benefit is limited to eliminating the CPU-side stall between kernel launch and next kernel prep. But the stall per-AIR is small because the D2H copies (for tiny sum buffers) complete almost instantly.

3. **Per-AIR allocation overhead is minimal.** The `cudaMallocAsync`/`cudaFreeAsync` calls use CUDA's stream-ordered memory allocator, which is essentially free for reused pool memory. Pre-allocating shared buffers saved negligible time.

4. **The 8.5% improvement at APC 100 (469→429ms) suggests the optimization helps more when kernels are medium-sized.** At APC 300, many AIRs are very small (height 8-64), so kernels finish near-instantly and the sync overhead per-AIR is proportionally tiny.

### Rollback criteria check:
- Less than 10% improvement in Round 0 at APC 300: **YES** (only 3%) → **ROLLBACK**
- No correctness regression: PASS (prove+verify succeeds)
- No memory regression: PASS
- No APC 0 regression: PASS for Round 0; Trace Commit variance unrelated

## Future Work

- **Multi-stream parallelism**: Instead of serializing kernels on a single stream, launch independent AIRs on separate CUDA streams. This would allow truly concurrent kernel execution for small AIRs. However, this requires careful memory management (separate intermediate buffers per stream).

- **Kernel fusion/batching**: The real win would be fusing small AIR kernels. With APC 300, many AIRs have height 8-64. Launching hundreds of tiny kernels has inherent overhead (launch latency). A single kernel that processes multiple small AIRs could be much faster.

- **Focus on the actual bottleneck**: Round 0 accounts for 665ms of the 2479ms STARK excl trace. The larger contributors are LogUp GKR (797ms), Stacked Reduction (312ms), and MLE Rounds (183ms). Optimizing these components, especially their scaling with AIR count, may yield larger gains.

- **CPU-GPU overlap via double-buffering**: A more sophisticated approach would process AIRs in batches with overlapping: launch batch N kernels while processing batch N-1 results on the CPU. This requires two sets of output buffers and careful stream management but could hide CPU post-processing time behind GPU execution.
