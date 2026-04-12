# Task: Sub-batch Round 0 Coset-Parallel Kernel Launches

Task name: 2026-04-11-1145-subbatch-round0-coset-parallel

## Problem Statement

Round 0 of the constraint evaluation phase at APC 300 launches ~1,470 small coset-parallel GPU kernels (539 zerocheck + 396 logup with shared memory, plus 196 + 339 without), each utilizing only 3-11% of the GPU's 128 SMs. These kernels execute serially on a single CUDA stream, leaving ~90% of GPU compute idle during each kernel.

Profiling data (nsys, APC 300 current):
- `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel<1,0>`: 221ms total, 539 instances, avg 410us
- `logup_r0_ntt_eval_interactions_coset_parallel_kernel<1,0>`: 196ms total, 396 instances, avg 494us
- `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel<0,0>`: 20ms total, 196 instances, avg 101us
- `logup_r0_ntt_eval_interactions_coset_parallel_kernel<0,0>`: 29ms total, 339 instances, avg 85us
- Total coset-parallel GPU time: 466ms out of 569ms Round 0 wall time

By comparison, at APC 0 (137+78+38+177 = 430 instances), the same kernel types total only 46ms. The 10x GPU time increase for ~3.4x more instances reflects severe SM underutilization: each small-AIR kernel launches only 4-12 thread blocks on a 128-SM GPU.

Round 0 is the largest scaling bottleneck: it increases by +396ms from APC 0 (173ms) to APC 300 (569ms), accounting for the majority of the +67ms net STARK excl trace gap (2134ms vs 2201ms).

## Proposed Approach

Batch multiple small coset-parallel AIR instances into single kernel launches using sub-batching with a memory budget. This directly improves SM utilization by launching many more thread blocks per kernel call.

**Key design: Sub-batching (not all-or-nothing)**

A previous attempt (commit 74134b4e, reverted at 61dc6616) implemented all-or-nothing batching with a 128MB memory budget check that disabled batching entirely when any AIR had a large skip_domain. This optimization refines that approach:

1. **Group AIRs into sub-batches** sized to fit within a memory budget (e.g., 64-128MB per batch). Sort coset-parallel AIRs by `skip_domain` and `num_cosets`, then greedily fill batches up to the budget. This means:
   - Small-skip AIRs (the majority at APC 300): batches of 16-64 AIRs
   - Large-skip AIRs: batches of 2-4 AIRs, or individual launches if too large

2. **Reuse existing batched CUDA kernels** from commit 74134b4e. The kernels were verified correct (94/94 tests pass). They use a 2D grid `(total_x_blocks, max_num_cosets)` with per-AIR metadata structs (`ZerocheckBatchMeta`, `LogupBatchMeta`) and `air_for_block` mapping. The `batched_final_reduce_block_sums` reduction kernel handles multi-segment output.

3. **Sub-batch by compatible parameters**: Group AIRs with the same `num_cosets` together to avoid coset-dimension padding waste. Within each group, limit batch size by total output buffer memory.

4. **Mechanism for improvement**: A batch of 16 AIRs, each launching 8 thread blocks, produces a combined grid of 128 blocks -- enough to saturate the 128-SM RTX 4090. Instead of 16 serial kernels at 5-10% SM utilization, we get 1 kernel at ~100% utilization. The GPU completes the same work in ~1/16th the wall time.

**What changes:**
- CUDA: Re-introduce the batched coset-parallel kernels from 74134b4e (zerocheck + logup)
- Rust: Replace the all-or-nothing batching logic with sub-batch grouping in Phase 1 of `sumcheck_uni_round0_polys`. AIRs are classified, grouped into sub-batches, and each sub-batch gets a single kernel launch. Individual launches remain for non-eligible AIRs (large buffer_size, GLOBAL=true).

## Expected Outcome

- **Round 0 APC 300**: 569ms -> 200-300ms (3-5x reduction in coset-parallel GPU time through SM saturation, plus reduced launch overhead from ~1470 -> ~50-100 launches)
- **STARK excl trace APC 300**: 2201ms -> ~1850-1950ms (270-370ms improvement)
- **Scaling ratio (APC 300/APC 0)**: 1.031 -> ~0.88-0.92 (APC 300 becomes faster than APC 0 for STARK excl trace)
- **APC 0 impact**: Minimal change (Round 0 at APC 0 is dominated by non-coset-parallel large-AIR kernels)
- **Cumulative vs baseline APC 300**: 2478ms -> ~1850-1950ms (-21% to -25%)

## Success Criteria

- **Primary**: STARK excl trace APC 300 decreases by >= 150ms vs current (2201ms -> <= 2051ms)
- **Secondary**: Round 0 APC 300 decreases by >= 30% (569ms -> <= 398ms)
- **Scaling**: APC 300/APC 0 STARK excl trace ratio < 1.00 (currently 1.031)
- **Correctness**: All 94 `openvm-cuda-backend` tests pass; proof verification succeeds for all APC configs
- **Rollback threshold**: If Round 0 APC 300 >= 540ms (within 5% of current), revert

## Considered Alternatives

1. **Multi-stream Round 0 execution** (concurrent kernels via multiple CUDA streams): Would achieve similar SM utilization improvement without new CUDA kernels. However, the current CUDA infrastructure hardcodes `cudaStreamPerThread` in DeviceBuffer allocation, D2H transfers, and memset operations. Enabling multi-stream requires deep cross-cutting changes to `cuda-common` (stream pool, parameterized stream in all CUDA calls, VPMM stream-aware allocation). Estimated 3x more complex than sub-batching, with similar expected improvement. **Rejected: too complex for this iteration.**

2. **Batch LogUp GKR input evaluation kernels**: GKR input eval has 791 kernel launches at APC 300 totaling 428ms GPU time -- another per-AIR scaling bottleneck. However, Round 0 has the larger absolute scaling penalty (+396ms APC 0->300 vs -199ms for LogUp GKR) and has proven CUDA kernels already in git history. **Deferred: good candidate for next iteration.**

3. **Pre-allocate reusable device buffers**: Eliminate per-AIR `DeviceBuffer::with_capacity` overhead in Round 0. Expected savings: 20-30ms (the overhead gap is only ~53ms, much of which is CUDA runtime dispatch, not malloc). **Rejected: too low impact to justify as standalone task.**

4. **Batch calculate_zero_hash in Trace Commit**: 584 instances at 187us each = 109ms total. Batching would save 50-80ms. **Deferred: smaller impact, good future candidate.**

5. **All-or-nothing Round 0 batching (repeat previous attempt)**: The memory budget check that disabled batching entirely for large skip_domains would still apply. The pairing benchmark likely triggers the check, making the optimization a no-op. **Rejected: sub-batching directly addresses this flaw.**
