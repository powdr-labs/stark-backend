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

## 2026-04-13-prealloc-round0-buffers

**Idea**: Pre-allocate per-thread reusable GPU buffers for Round 0 constraint and interaction evaluation to eliminate per-AIR cudaMallocAsync/cudaFreeAsync serialization.

**Result**: failure — Round 0 at APC 300: 296ms → 443ms (1.50x higher vs before). STARK excl trace at APC 300: 1779ms, 1.38x lower vs baseline (vs previous: 1442ms → 1779ms, +337ms, 1.23x higher).

**Summary**: Applied the GKR pre-allocation pattern to Round 0, but the approach failed due to fundamental differences: (1) Round 0 processes AIRs in small batches of 20-42, never reaching the >=100 multi-threading threshold; (2) max buffer sizes are ~1GB per thread (driven by max_temp_bytes), vs ~100MB for GKR (driven by TASK_SIZE); (3) pre-allocating 1GB for the largest AIR caused systemic GPU memory pressure that degraded even unrelated phases (GKR +188ms, Leaf Recursion +100ms). Key learning: the pre-allocation pattern only works when buffer sizes are small relative to GPU memory AND many AIRs are processed per call.

## 2026-04-13-1100-batch-mle-interpolation

**Idea**: Replace per-AIR `interpolate_columns_gpu` kernel launches in MLE sumcheck rounds with a single batched descriptor-array kernel launch per round.

**Result**: success — MLE Rounds at APC 300: 183ms → 168ms (1.09x lower vs before, 1.07x lower vs baseline 180ms). STARK excl trace at APC 300: 1418ms, 1.73x lower vs baseline (vs previous: 1441ms → 1418ms, -23ms, 1.02x lower).

**Summary**: Replaced ~5000 per-AIR interpolation kernel launches with a single batched kernel per MLE round using descriptor arrays. One big buffer allocation replaces ~5000 individual allocations, and one kernel launch replaces ~5000 launches. Used a multi-block-per-descriptor design with binary search (initial one-block-per-trace design caused APC 0 regression due to GPU underutilization for large AIRs). The improvement was modest (15ms MLE Rounds, 23ms STARK excl trace at APC 300) because residual per-trace `main_ptrs.to_device()` overhead (~5000 calls) was not addressed, and per-call CUDA API overhead was lower than estimated.

## 2026-04-13-1430-batch-mle-main-ptrs-upload

**Idea**: Replace per-trace `main_ptrs.to_device()` calls in MLE sumcheck rounds with a single batched device upload per round.

**Result**: failure — MLE Rounds at APC 300: 170ms → 165ms (-5ms, below 10ms rollback threshold). STARK excl trace at APC 300: 1421ms, 1.73x lower vs baseline (vs previous: 1417ms → 1421ms, +4ms noise).

**Summary**: Replaced ~5600 per-trace `main_ptrs.to_device()` calls (confirmed by nsys H2D count drop from 30035 to 24436) with 2 batched uploads per round (one for late_eval, one for early_eval). The implementation is mechanically correct but the improvement was only 5ms — per-call CUDA API overhead is ~1μs (not ~15-20μs as estimated) because cudaMallocAsync uses pool caching, the mutex is uncontended in single-threaded MLE rounds, and 16-48 byte H2D copies are negligible on PCIe. The dominant MLE Rounds cost is kernel execution, not API overhead.

## 2026-04-13-1600-batch-round0-descriptor-arrays

**Idea**: Replace per-AIR Round 0 kernel launches with batched descriptor-array kernels for both zerocheck and logup evaluation.

**Result**: failure — Infrastructure implemented (CUDA kernels, FFI bindings, Rust batch helpers) but orchestration integration blocked by Rust optimizer interference causing cudaErrorIllegalAddress.

**Summary**: Implemented batched CUDA kernels (Round0ZcCtx/Round0LogupCtx descriptors, 1D grid with BlockCtx mapping, batched reduction) and Rust FFI bindings. However, adding the Rust orchestration code that routes small AIRs to the batched path consistently caused CUDA illegal memory access errors at APC 300, even when the batched code path was never executed. The root cause appears to be that adding significant code to the large `sumcheck_uni_round0_polys` function changes Rust optimizer behavior for the existing multi-threaded CUDA kernel launch path. The CUDA+FFI infrastructure is committed as a foundation for future integration.

## 2026-04-13-1700-batch-round0-caller-routing

**Idea**: Re-attempt batched Round 0 descriptor-array kernels using caller-level routing to avoid modifying the sensitive `sumcheck_uni_round0_polys` function.

**Result**: failure — Implementation complete but blocked by pre-existing GPU OOM in gkr_input (8GB allocation on 24GB GPU). APC 0 confirms no regression (batched path inactive, Round 0 = 178ms unchanged).

**Summary**: Moved all batched orchestration to a separate `round0_batched.rs` module with `#[inline(never)]`, adding only a `.filter()` to the existing function. This avoided the Rust optimizer interference from the previous attempt (APC 0 proves+verifies correctly). Fixed a `cudaErrorIllegalAddress` bug where GLOBAL-mode batched kernels crashed when intermediates buffers were null (identified via compute-sanitizer). Added memory budget filtering to prevent intermediates OOM. However, APC 300/100 benchmarks failed with OOM in the GKR input phase (unrelated to this optimization) after a clean CUDA rebuild increased GPU memory overhead.

## 2026-04-13-1930-round0-interleaved-work-balance

**Idea**: Replace contiguous work chunking in Round 0 with interleaved (round-robin) assignment to balance GPU kernel time and CPU post-processing across threads.

**Result**: success — Round 0 at APC 300: 313ms → 277ms (2.39x lower vs baseline 662ms). STARK excl trace at APC 300: 1433ms → 1411ms, 1.74x lower vs baseline (vs previous: -22ms, 1.02x lower).

**Summary**: Replaced contiguous `chunks(chunk_size)` dispatch with round-robin index assignment on the height-sorted work items list. Diagnostic instrumentation showed 26ms thread spread (83-109ms) with contiguous chunking; round-robin reduced this to 7ms (93-100ms). The plan's LPT approach using `height` as cost proxy was abandoned because it assigned only 1 AIR to the 3 highest-height threads (finishing in 13ms) while packing 66+ small AIRs onto other threads (finishing in 118ms). The per-AIR fixed overhead (~1.8ms) dominates for small AIRs, making height alone a poor cost proxy. Round-robin balances both count and height naturally.

## 2026-04-13-2035-gkr-input-round-robin

**Idea**: Replace contiguous work chunking in GKR input evaluation with interleaved (round-robin) assignment across multi-stream threads.

**Result**: failure — GKR input evals at APC 300: 304ms → 296ms (-8ms median, 1.03x lower). STARK excl trace at APC 300: 1403ms → 1406ms (+3ms, within noise). Below 8ms rollback threshold.

**Summary**: Applied the same round-robin pattern that worked for Round 0 (saving 36ms) to GKR input eval. Thread balance improved (seg0 spread 29ms, seg1 spread 6ms), but the net impact was only ~8ms on GKR input and undetectable at STARK excl trace level. The pre-allocated buffer optimization (previous task) had already eliminated the main source of thread imbalance (per-AIR mutex contention), leaving little room for scheduling improvements. Also discovered that `RUST_LOG=warn` blocks INFO-level tracing spans, preventing `TimingMetricsLayer` from recording timing metrics.

## 2026-04-13-2200-prealloc-round0-threshold-buffers

**Idea**: Pre-allocate per-thread reusable GPU buffers for Round 0 constraint and interaction evaluation using an adaptive 95th-percentile size threshold.

**Result**: success — Round 0 at APC 300: 276ms → 242ms (2.74x lower vs baseline 662ms). STARK excl trace at APC 300: 1372ms, 1.79x lower vs baseline (vs previous: 1407ms → 1372ms, -35ms, 1.03x lower).

**Summary**: Pre-computed per-AIR buffer sizes during Phase 1 work item preparation (required adding `logup_round0_buffer_size` to `AirDataGpu` to avoid per-AIR DAG reconstruction). Used 95th-percentile sizes for pre-allocation instead of maximum — the max is 193 MB/thread (driven by a few large AIRs), while p95 is 3.1 MB/thread. The failed previous attempt (`prealloc-round0-buffers`) used max sizes, causing 1.5 GB total pre-allocation and systemic GPU memory pressure. The percentile approach covers 95% of AIRs with negligible memory impact. No regression at APC 0.

## 2026-04-14-0030-rightsize-gkr-input-kernel-grid

**Idea**: Right-size CUDA kernel grid for small GLOBAL-mode AIRs in GKR input eval by using `min(TASK_SIZE, permutation_height)` instead of always `TASK_SIZE`.

**Result**: failure — STARK excl trace at APC 300: 1364ms → 1358/1383ms (within noise, 0ms net change). GKR input evals seg0: 242ms → 238-254ms (noise). No improvement vs baseline or previous.

**Summary**: Changed the CUDA launcher `count` from `TASK_SIZE` to `min(TASK_SIZE, permutation_height)` for GLOBAL-mode kernels, reducing grid size for small AIRs (e.g., 4 blocks instead of 256 for height=1024). The hypothesis was that oversized grids saturated all 128 SMs, preventing inter-stream concurrency. The change had no measurable impact, suggesting SM saturation is not the concurrency bottleneck — likely either memory bandwidth limits throughput regardless of SM count, or the few large AIRs (height >= TASK_SIZE) dominate per-thread wall time and are unaffected by this change.

## 2026-04-14-0140-multistream-stacked-reduction-round0

**Idea**: Multi-stream the per-trace sequential kernel loops in Stacked Reduction's Round 0 sumcheck and PLE fold across 8 OS threads with per-thread CUDA streams.

**Result**: failure — Stacked Reduction at APC 300: 75ms → 74ms (-1ms, unchanged). STARK excl trace at APC 300: 1382ms → 1558ms (+176ms, 1.13x higher). LogUp GKR regressed +193ms at APC 300, +233ms at APC 100. No regression at APC 0.

**Summary**: Applied the multi-stream pattern (used successfully for Round 0 and GKR input eval) to Stacked Reduction. The optimization required per-thread accumulation buffers (112 DeviceBuffers total) because the final_reduce kernel uses non-atomic `+=`. The buffer allocations/frees through the CUDA memory pool disrupted pool state, causing a ~200ms regression in the memory-bandwidth-bound LogUp GKR phase in subsequent segments. The Stacked Reduction itself barely improved because (1) CPU NTT reconstruction dominates at ~36ms of the 75ms total, limiting max GPU savings, and (2) the D2H reduction overhead (~11ms for 10MB) partially offsets the GPU kernel concurrency gains. Key learning: the multi-stream pattern only works when it doesn't require per-thread accumulation buffers that significantly increase allocation count.

## 2026-04-14-0300-precompute-logup-round0-interaction-rules

**Idea**: Pre-compute the logup Round 0 interaction evaluation DAG, rules, and weight-index mapping at keygen time to eliminate per-AIR CPU overhead in `evaluate_round0_interactions_gpu`.

**Result**: failure — Round 0 at APC 300: 243ms → 243ms (0ms change). STARK excl trace at APC 300: 1386ms → 1373ms (-13ms, within noise). (vs baseline: 1.79x lower, unchanged from previous)

**Summary**: Moved DAG construction, rule compilation, encoding, and H2D upload from the per-AIR hot path to keygen time. The implementation is correct and architecturally clean, but the per-AIR overhead was ~0.05-0.1ms (not ~0.8ms as estimated), making the total savings ~5ms across 8 threads — well within measurement noise. The plan's estimate was based on overstated per-expression costs; in practice, the DAGs are small (2-10 expressions per AIR), pointer deduplication is O(1), and CUDA pool allocation handles tiny H2D uploads with near-zero latency.

## 2026-04-14-0330-gpu-round0-poly-extraction

**Idea**: Move per-AIR polynomial extraction (D2H sync + transpose + iDFT + Lagrange interpolation + coefficient adjustment) from CPU to a GPU kernel using pre-computed transformation matrices.

**Result**: success — Round 0 at APC 300: 245ms → 202ms (3.28x lower vs baseline 662ms). STARK excl trace at APC 300: 1383ms → 1336ms, 1.84x lower vs baseline (vs previous: -47ms, 1.04x lower).

**Summary**: Replaced per-AIR D2H sync + CPU post-processing with GPU-side matrix-vector multiply using pre-computed transformation matrices that capture the entire pipeline (transpose + iDFT + unshift + Lagrange interpolation + coefficient adjustment). Polynomial coefficients are written directly into a shared device batch array, with a single D2H copy replacing ~1246 per-AIR pipeline drains. Round 0 improved 21% at APC 300 with no regression at APC 0. The pre-computed matrix approach is simpler and more robust than implementing the Bowers iDFT on GPU.

## 2026-04-14-0730-overlap-logup-precompute-round0

**Idea**: Overlap logup combination precomputation (d_eq_3b upload + precompute kernels) with Round 0 multi-stream evaluation by running it on a background thread.

**Result**: success — STARK excl trace at APC 300: 1370ms → 1306ms, 1.88x lower vs baseline (vs previous: -64ms, 1.05x lower). Round 0: 203ms → 182ms (-21ms, 1.12x lower).

**Summary**: Spawned d_eq_3b upload + logup_combinations precompute on a background thread inside the existing thread::scope block, concurrent with Round 0 worker threads. The precompute output is only consumed during MLE rounds, making it fully independent of Round 0. Per-segment Round 0 improved by ~10ms (93ms → 82ms). LogUp GKR also improved by 44ms, likely due to improved CUDA memory pool state from overlapped allocations. No regression at APC 0 (single-threaded path unchanged).
