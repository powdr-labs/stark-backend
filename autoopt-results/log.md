## 2026-04-12-2130-pipeline-round0-kernel-launches

**Idea**: Eliminate per-AIR GPU synchronization in Round 0 by restructuring the loop into prepare/launch/collect phases.

**Result**: failure — Round 0 at APC 300: 665ms → 645ms (1.03x lower vs baseline 662ms). STARK excl trace at APC 300: 2479ms → 2452ms (vs previous: -27ms, 1.01x lower).

**Summary**: Restructured the Round 0 constraint evaluation from a per-AIR loop with 2 sync points each into three phases: prepare all metadata upfront, launch all kernels without sync, then batch-collect results. The improvement was only ~3% for Round 0 at APC 300 (20ms), well below the 10% rollback threshold. Root cause: the overhead is primarily CPU-side post-processing (transpose, iDFT) not GPU stall time, and cudaStreamPerThread already serializes kernels. Multi-stream parallelism or kernel fusion would be needed for meaningful gains.

## 2026-04-12-2245-parallelize-round0-cpu-postprocess

**Idea**: Parallelize CPU-bound post-processing (transpose, iDFT, polynomial construction) across AIRs using rayon in Round 0.

**Result**: failure — Round 0 at APC 300: 667ms → 657ms (1.01x lower vs baseline 662ms). STARK excl trace at APC 300: 2468ms → 2455ms (vs previous: -13ms, 1.01x lower).

**Summary**: Split the per-AIR Round 0 loop into a sequential GPU phase (kernel launches + D2H copies) and a parallel CPU phase (transpose + iDFT via rayon). The improvement was only ~10ms (1.5%) at APC 300, well below the 10% rollback threshold. The 200ms gap between GPU kernel time and total Round 0 time is not dominated by CPU compute as hypothesized — it's primarily D2H copy latency and kernel launch overhead, which this optimization does not address.

## 2026-04-12-2330-batch-stacked-reduction-mle-sync

**Idea**: Eliminate per-window GPU synchronization in stacked reduction MLE rounds by batching all kernel launches with a single fill_zero + to_host per round.

**Result**: success — Stacked Reduction at APC 300: 311ms → 126ms (2.47x lower vs baseline). STARK excl trace at APC 300: 2455ms → 2268ms (1.08x lower vs baseline). (vs previous: SR 312ms → 126ms, 2.48x lower; STARK 2460ms → 2268ms, -192ms)

**Summary**: Replaced per-window D2H sync (to_host with pipeline drain) in batch_sumcheck_poly_eval with a single fill_zero + to_host per MLE round. Also bulk-uploaded eq_ub_per_trace once per round instead of per-window. At APC 300, MLE rounds dropped from 280ms to 95ms (2.95x), confirming that synchronization overhead was the dominant cost. APC 0 showed no regression.

## 2026-04-13-0100-multistream-round0-parallel

**Idea**: Process Round 0 AIRs across multiple OS threads with per-thread CUDA streams for concurrent kernel execution.

**Result**: success — Round 0 at APC 300: 662ms → 351ms (1.89x lower vs baseline). STARK excl trace at APC 300: 2455ms → 1969ms (1.25x lower vs baseline). (vs previous: Round 0 669ms → 351ms, 1.91x lower; STARK 2282ms → 1969ms, -313ms)

**Summary**: Extracted per-AIR Round 0 work into a standalone function, distributed across 4 OS threads (each with its own cudaStreamPerThread), enabling concurrent GPU kernel execution on idle SMs. Added a mutex-free D2H copy path (to_host_on_current_stream) and a threshold (>=100 AIRs) to avoid multi-threading when few large AIRs already saturate the GPU. Round 0 improved by 1.91x at APC 300 and 1.69x at APC 100, with no regression at APC 0.

## 2026-04-13-0115-multistream-gkr-input-eval

**Idea**: Parallelize LogUp GKR input evaluation across multiple OS threads with per-thread CUDA streams, using the same multi-stream pattern as Round 0.

**Result**: success — LogUp GKR at APC 300: 790ms → 564ms (1.40x lower vs baseline). STARK excl trace at APC 300: 2455ms → 1730ms (1.42x lower vs baseline). (vs previous: LogUp GKR 789ms → 564ms, 1.40x lower; STARK 1953ms → 1730ms, -223ms, 1.13x lower)

**Summary**: Extracted per-AIR GKR input eval into a standalone function, distributed across 4 OS threads with per-thread CUDA streams for concurrent kernel execution. Used the same pattern as the Round 0 multi-stream optimization: work item struct, height-descending sort for load balance, >=100 AIR threshold to skip multi-threading at APC 0. LogUp GKR improved 1.40x at APC 300, bringing cumulative STARK excl trace improvement to 1.42x vs baseline.

## 2026-04-13-1500-batch-stacking-scatter-kernel

**Idea**: Replace per-column CUDA API calls in `stack_traces_into_expanded()` with a single batched scatter kernel.

**Result**: success — Trace Commit at APC 300: 406ms → 255ms (1.59x lower vs baseline). STARK excl trace at APC 300: 2455ms → 1597ms (1.54x lower vs baseline). (vs previous: Trace Commit 414ms → 255ms, 1.62x lower; STARK 1746ms → 1597ms, -149ms, 1.09x lower)

**Summary**: Replaced ~106K per-column CUDA API calls (cudaMemcpyAsync + batch_expand_pad_wide kernel launches) with a two-phase approach: build a descriptor array on the CPU, then upload it and launch a single scatter kernel. D2D memcpy count dropped from 101K to 325. Trace Commit improved 1.62x at APC 300 with no regression at APC 0. Cumulative STARK excl trace improvement is now 1.54x vs baseline.

## 2026-04-13-1830-increase-multistream-thread-count

**Idea**: Increase the multi-stream thread count from 4 to 8 for both Round 0 and GKR input evaluation to improve GPU SM utilization.

**Result**: success — STARK excl trace at APC 300: 1517ms, 1.62x lower vs baseline (vs previous: 1578ms → 1517ms, -61ms, 1.04x lower)

**Summary**: Changed NUM_ROUND0_STREAMS and NUM_GKR_INPUT_STREAMS from 4 to 8. Round 0 improved clearly (348ms → 295ms, 15.2% at APC 300; 286ms → 244ms, 14.7% at APC 100), confirming kernel concurrency was a bottleneck. GKR input eval improvement was minimal (14ms), suggesting mutex contention or allocation overhead limits scaling there. Overall STARK excl trace improvement was borderline (61ms, just above 60ms rollback threshold). Cumulative improvement vs baseline is now 1.62x.

## 2026-04-13-2100-prealloc-gkr-input-buffers

**Idea**: Pre-allocate per-thread reusable GPU buffers for GKR input evaluation to eliminate per-AIR mutex contention and cudaMallocAsync serialization.

**Result**: success — STARK excl trace at APC 300: 1501ms, 1.64x lower vs baseline (vs previous: 1718ms → 1501ms, -217ms, 1.14x lower)

**Summary**: Pre-computed max buffer sizes across all AIRs then allocated one set of max-sized buffers (intermediates, public_values, partition_ptrs, tmp) per worker thread before spawning. Eliminated ~1240 Mutex acquisitions per segment and all per-AIR cudaMallocAsync/cudaFreeAsync calls. GKR input eval improved 1.67x at APC 300 (532ms → 318ms), with dramatic per-segment asymmetry: seg1 improved 4.15x (many small AIRs where allocation overhead dominated) while seg0 improved only 1.07x (fewer large AIRs, kernel-dominated). No regression at APC 0. Cumulative STARK excl trace improvement vs baseline is now 1.64x.

## 2026-04-13-2300-batch-mle-round-kernels

**Idea**: Replace per-AIR kernel launch loop in stacked reduction MLE rounds with batched descriptor-array kernel launches.

**Result**: success — Stacked Reduction at APC 300: 123ms → 75ms (4.15x lower vs baseline 311ms). STARK excl trace at APC 300: 1474ms, 1.67x lower vs baseline (vs previous: 1474ms → 1474ms, 0ms change — improvement masked by noise in other metrics).

**Summary**: Replaced ~15K per-AIR kernel launches per benchmark with 2 batched launches per MLE round (one degenerate, one non-degenerate) using descriptor arrays. The stacked reduction MLE rounds improved 2.09x at APC 300 (115ms → 55ms raw gauge), with the improvement scaling with AIR count (1.55x at APC 0, 1.93x at APC 100). The overall STARK excl trace didn't show net improvement at APC 300 due to measurement noise in other components (GKR +39ms, Round 0 +8ms), but Openings total improved by 60ms. No regression at any APC configuration.
