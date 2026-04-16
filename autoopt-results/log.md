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

## 2026-04-14-0830-batch-scatter-gkr-input-eval

**Idea**: Batch SCATTER-mode GKR input evaluation AIRs into a single descriptor-array CUDA kernel launch per segment on a dedicated background thread.

**Result**: failure — LogUp GKR at APC 300: 518ms → 510ms (-8ms median, 1.02x lower vs before, below 10ms rollback threshold). STARK excl trace at APC 300: 1302ms → 1297ms (-5ms, within noise). (vs baseline: 1.89x lower, unchanged from previous)

**Summary**: Implemented a batched SCATTER kernel using the descriptor-array pattern (BlockCtx + GkrInputScatterCtx), running on a dedicated background thread concurrent with 8 GLOBAL worker threads. Fixed a correctness bug where multiple lifted SCATTER AIRs shared the same tmp buffer during concurrent kernel writes (needed per-AIR offsets). The optimization produces correct proofs but the improvement (~8ms median) is below the 10ms threshold because SCATTER AIRs' aggregate GPU time (~63ms) is already hidden behind the longer GLOBAL processing (~464ms). The batching reduces CPU-side overhead per SCATTER AIR but doesn't change the GLOBAL-dominated critical path.

## 2026-04-14-1200-bulk-alloc-mle-fold-buffers

**Idea**: Replace per-matrix GPU buffer allocation in MLE fold rounds with bulk arena allocation (one cudaMallocAsync per fold group instead of ~1869).

**Result**: failure — MLE Rounds at APC 300: 175ms → 162ms (-13ms, 1.08x lower, below 20ms rollback threshold). STARK excl trace at APC 300: 1331ms → 1306ms (vs baseline: 1.88x lower, vs previous: -25ms, 1.02x lower).

**Summary**: Introduced ArenaMatrix/MatrixRef/FoldArena types to replace ~37K per-round cudaMallocAsync/cudaFreeAsync calls with ~40 bulk allocations. The implementation is correct (valid proofs at all APC configs, no APC 0 regression), but the improvement was only ~13ms on MLE Rounds — well below the expected 40-55ms. The CUDA pool allocator's per-call overhead is ~0.3-0.5μs (not ~0.8μs as estimated from nsight), making the true allocation overhead ~15-20ms, of which we recovered ~13ms. The dominant MLE Rounds cost is kernel execution time, not allocation overhead.

## 2026-04-14-1430-warp-tiled-gkr-intermediates

**Idea**: Reorganize GKR input evaluation GLOBAL-mode intermediates buffer from thread-interleaved layout (stride=65536) to warp-tiled layout (stride=32) for better GPU cache locality.

**Result**: failure — LogUp GKR at APC 300: 529ms → 534ms (no change, within noise). STARK excl trace at APC 300: 1314ms → 1317ms (vs baseline: 1.86x lower, vs previous: +3ms, within noise).

**Summary**: Changed the intermediates pointer computation from `base + thread_id` with stride `65536` to `base + warp_id * buffer_size * 32 + lane_id` with stride `32`, reducing per-thread inter-node stride from 1MB to 512B. nsight confirmed kernel total GPU time unchanged (458ms vs ~464ms). The bottleneck is memory bandwidth from main/preprocessed trace reads, not intermediates access locality — intermediates are only ~10-20% of total memory traffic, and the 72MB L2 cache already absorbs the old layout's coalesced warp-level accesses adequately.

## 2026-04-14-1630-pingpong-mle-fold-buffers

**Idea**: Replace per-round DeviceMatrix allocations in MLE fold with pre-allocated ping-pong buffers using non-owning DeviceBuffer views.

**Result**: success — MLE Rounds at APC 300: 171ms → 161ms (1.12x lower vs baseline 180ms). STARK excl trace at APC 300: 1288ms, 1.91x lower vs baseline (vs previous: 1308ms → 1288ms, -20ms, 1.02x lower).

**Summary**: Added `owns_memory` flag to DeviceBuffer enabling non-owning views, then pre-allocated two ping-pong buffers per fold set (mat_evals and sels). Each fold round writes foldable output into the alternate buffer and creates new non-owning views, eliminating ~37K cudaMallocAsync + ~37K cudaFreeAsync calls. Non-foldable matrices retain their existing views without D2D copies (an initial attempt with per-round D2D copies for non-foldable matrices actually regressed performance). No regression at APC 0. Cumulative STARK excl trace improvement vs baseline is now 1.91x.

## 2026-04-14-1800-gpu-gkr-transcript-processing

**Idea**: Move per-round GKR fractional sumcheck post-processing (D2H + reconstruct_s_evals + Poseidon2 transcript observe/sample) to a GPU kernel to eliminate CPU-GPU roundtrips in the FoldEval inner loop.

**Result**: failure — LogUp GKR at APC 300: 527ms → 534ms (+7ms, no improvement). STARK excl trace at APC 300: 1294ms → 1305ms (vs baseline: 1.88x lower, vs previous: +11ms, within noise).

**Summary**: Implemented a <<<1,1>>> GPU postprocess kernel (reconstruct s_evals + sponge observe/sample + accumulator update), device-pointer compute kernel variants (so challenges stay on GPU between rounds), and batch D2H + CPU transcript replay after all inner rounds. The optimization produced correct results (debug_assert on GPU-vs-CPU sponge challenges passed) but had no measurable impact because the per-round CPU work (~15-25μs for reconstruct + sponge) was already fast, the D2H of 32 bytes is near-instant, and the kernel launch overhead of the postprocess kernel (~5-10μs each × 40 rounds) offset savings. The inter-kernel gaps seen in nsight profiling are dominated by CUDA driver/launch overhead, not CPU arithmetic.

## 2026-04-14-2000-matrix-base-ptr-interpolation

**Idea**: Replace the flat per-column pointer array (~106K entries) with per-matrix base pointers (~2K entries) in the batched interpolation kernel to reduce CPU collection and H2D transfer overhead in MLE rounds.

**Result**: success (marginal) — STARK excl trace at APC 300: 1293ms avg, 1.90x lower vs baseline (vs previous: 1299ms → 1293ms, -6ms, 1.00x lower). MLE Rounds: 163ms → 159ms (-4ms).

**Summary**: Replaced O(106K) per-column pointer collection with O(2K) per-matrix base pointer collection, and 850KB H2D with ~40KB H2D per MLE round. The CUDA kernel computes column addresses from matrix base + offset via a short linear scan (2-5 matrices per trace). The improvement was marginal (4ms on MLE Rounds, 6ms avg on STARK excl trace) because CPU iteration overhead was lower than estimated (~3-4ns/column not 6-10ns) and H2D transfer of 850KB is already fast at PCIe 4.0. No regression at APC 0.

## 2026-04-14-2130-warp-per-trace-mle-eval

**Idea**: Warp-per-trace monomial MLE evaluation kernel that flips the parallelism axis from monomials to y-values for traces with few monomials (≤32) and small num_y (≤32).

**Result**: failure — MLE Rounds at APC 300: 158ms → 163ms (no change, within noise). STARK excl trace at APC 300: 1301ms → 1304ms (vs baseline: 1.88x lower, unchanged from previous).

**Summary**: Implemented warp-per-trace CUDA kernels for both zerocheck and logup monomial evaluation, with a two-path partition (warp + block) and scatter output. All 94 tests pass, no regression at APC 0. The optimization had no measurable impact because the monomial kernel accounts for only ~17ms of 158ms MLE Rounds total — even eliminating it entirely would barely be detectable. The savings from removing tmp_sums allocation and secondary reduction kernels (~2-5ms) are within noise. For late_eval traces (num_y=1, the dominant case), the warp approach has similar utilization (1/32 vs 5-15/256) to the block approach.

## 2026-04-15-0030-drain-gpu-pipeline-before-stark

**Idea**: Add explicit `current_stream_sync()` before the `stark_prove_excluding_trace` timing span to drain async trace gen GPU kernels and eliminate pipeline stall from the STARK metric.

**Result**: failure — STARK excl trace at APC 300: 1300ms → 1312ms (no change, within noise). Drain pipeline time: 0ms for all segments. (vs baseline: 1.87x lower, unchanged from previous)

**Summary**: Added `drain_pending_device_ops()` to the `ProverDevice` trait and restructured `Coordinator::prove` to sync the CUDA stream before starting the STARK timing span. The drain completes in 0ms for all segments at all APC configurations, conclusively disproving the pipeline stall hypothesis. By the time `prove()` is called, all trace gen GPU kernels have already completed — the caller's code path includes implicit synchronization points that drain the pipeline before the prover starts. The ~160ms Trace Commit time in APC 300 seg0 is genuine commit work, not pipeline stall.

## 2026-04-15-0100-batch-global-gkr-input-eval

**Idea**: Replace per-AIR GLOBAL-mode GKR input evaluation kernel launches with a single batched descriptor-array kernel for small AIRs (height ≤ TASK_SIZE).

**Result**: failure — LogUp GKR at APC 300: 528ms → 525ms (no change, within noise). STARK excl trace at APC 300: 1292ms → 1291ms (vs baseline: 1.90x lower, unchanged from previous)

**Summary**: Implemented a batched CUDA kernel with per-AIR descriptors and BlockCtx mapping, plus Rust orchestration to partition work items, upload flat device buffers, and handle height normalization. Without a memory budget cap, the batched kernel caused severe regression (528ms → 694-944ms at APC 300) because the aggregate intermediates buffer (576 MB for ~211 AIRs) vastly exceeds the 72 MB L2 cache, destroying the cache reuse that makes the existing per-stream sequential approach fast. Adding a 64 MB L2 budget cap causes batching to be disabled for all significant workloads. Key learning: kernel launch overhead (~1.5ms per segment) is negligible; the per-AIR sequential approach's intermediates buffer reuse is the fundamental performance advantage that batching cannot replicate.

## 2026-04-15-0430-multistream-mle-round-eval

**Idea**: Overlap logup and zerocheck evaluation paths within each MLE sumcheck round using OS threads with per-thread CUDA streams.

**Result**: failure — MLE Rounds at APC 300: 162ms → 165ms (no change, within noise). STARK excl trace at APC 300: 1305ms → 1298ms (vs baseline: 1.89x lower, unchanged from previous)

**Summary**: Spawned logup and zerocheck evaluation onto separate threads with per-thread CUDA streams when ≥50 early traces (APC 300 has ~350). Also replaced all to_host() with to_host_on_current_stream() to avoid global COPY_EVENT mutex. The optimization had no measurable impact because the remaining MLE Rounds time (~162ms over 14 rounds) is dominated by actual GPU kernel compute, not idle SMs or launch overhead. Previous optimizations (batched kernels, batched interpolation, pingpong buffers) already eliminated the per-round scheduling overhead that multi-streaming could have helped with. Work imbalance between logup and zerocheck threads and MEMORY_MANAGER mutex contention further limit any potential overlap benefit.

## 2026-04-15-0800-cache-codeword-buffer-across-segments

**Idea**: Pre-warm the GPU memory pool (VPMM) by allocating and freeing a 256MB buffer in GpuDevice::new() to eliminate cold-start cuMemCreate+cuMemMap overhead in the first segment's rs_code_matrix.

**Result**: failure — APC 300: STARK excl trace 1292ms → 1250ms (-42ms, 1.96x lower vs baseline). APC 0: STARK excl trace 2134ms → 2271ms (+137ms regression, exceeds 20ms rollback threshold).

**Summary**: The warmup successfully eliminated the codeword allocation cold-start at APC 300 (rs_code_matrix seg 0: 58ms → 14ms), but caused a consistent +177ms LogUp GKR regression at APC 0 (verified across 4 runs). This is the same VPMM pool state sensitivity pattern seen in multistream-stacked-reduction-round0 and batch-global-gkr-input-eval — changing the pool's free region layout disrupts memory access patterns for bandwidth-bound GKR kernels. An alternative using VPMM initial_pages (pre-allocation at pool construction time) barely helped (58ms → 51ms) because pages get consumed by intermediate allocations before proving starts.

## 2026-04-15-1600-fix-batch-round0-revert-and-reapply

**Idea**: Pre-compute logup Round 0 interaction DAG rules at keygen time to eliminate per-AIR CPU-side DAG reconstruction in Phase 2 and enable a future batched kernel path.

**Result**: failure — Round 0 at APC 300: 181ms → 210ms (+29ms, 1.16x higher). STARK excl trace at APC 300: 1111ms → 1154ms (+43ms, 1.04x higher vs before). (vs baseline: 2.13x lower, slight regression from previous 2.21x)

**Summary**: Two approaches were tried. Approach 1 (Phase 1 pre-computation per the plan) caused a severe +96ms regression at APC 300 by serializing work that Phase 2 parallelized across 8 threads. Approach 2 (keygen-time pre-computation, maintaining Phase 2 parallelism) still regressed by ~28ms at APC 300. The likely cause is that the removed per-AIR CPU work (DAG construction, rule encoding) was providing beneficial pacing between GPU kernel launches across 8 threads — eliminating it causes tighter kernel launch bursts and increased GPU command queue contention. The batch kernel path (Step 5) was not attempted because the prerequisite pre-computation steps failed.

## 2026-04-15-1800-reenable-batch-round0-small-airs

**Idea**: Re-enable the batch Round 0 descriptor-array kernel path for small AIRs, which was reverted due to GPU hang. The OOM blocker was assumed resolved by VPMM page size increase.

**Result**: failure — Batch CUDA kernel hangs when >12 AIRs are batched. GPU shows 100% utilization but kernel never completes. With only 12 batchable AIRs (of ~300 at APC 300), optimization saves negligible time. STARK excl trace at APC 300: 1110ms (unchanged, 2.21x lower vs baseline 2455ms).

**Summary**: Fully implemented the batch eval-only + deferred extraction design (CUDA kernels, FFI bindings, Rust orchestration, Phase 2 integration). Systematic diagnosis via binary search narrowed the hang to >12 batched AIRs: 1-12 AIRs complete correctly, 15+ hang at current_stream_sync. The plan assumed the previous hang was OOM-related (resolved by VPMM 16 MiB pages), but the root cause is a bug in the batch CUDA kernel itself — likely an out-of-bounds intermediates buffer access that scales with AIR count. The kernel code was restored from reverted commit 0d1bb2f9 unchanged; the bug exists in that kernel. Key learning: the batch approach for Round 0 has both an unsolved kernel bug AND the same L2 cache thrashing issue that blocked batch-global-gkr-input-eval.

## 2026-04-15-2100-precompute-mle-fold-layout

**Idea**: Pre-compute fold descriptor arrays for all MLE rounds and bulk-upload once, fusing the two per-round fold_pingpong calls into a single kernel launch.

**Result**: failure — MLE Rounds at APC 300: 166ms → 166ms (0ms change). STARK excl trace at APC 300: 1144ms → 1119ms (vs baseline: 2.19x lower, vs previous: -25ms, noise).

**Summary**: Replaced 112 per-round to_device() calls and 28 kernel launches with 4 bulk uploads and 14 fused launches. All 94 tests pass, no regression at APC 0. The improvement was undetectable because the total CUDA API overhead eliminated (~0.3-0.5ms across 2 segments) is 100x below the measurement noise floor. MLE Rounds is now firmly kernel-execution-bound (~113ms of 166ms); further improvements require algorithmic or kernel-level changes, not CUDA API overhead reduction.

## 2026-04-15-2230-prealloc-whir-fold-buffers

**Idea**: Pre-allocate and reuse WHIR sumcheck fold buffers via a pingpong pattern to eliminate per-round VPMM allocation churn.

**Result**: failure — WHIR at APC 300: 130ms → 126ms (-4ms, below 5ms rollback threshold). STARK excl trace at APC 300: 1117ms → 1115ms (vs baseline: 2.20x lower, vs previous: -2ms, noise).

**Summary**: Pre-allocated one pair of scratch DeviceBuffers at maximum size and used std::mem::swap to alternate between scratch and current buffers for non-final inner rounds. All 14 WHIR tests pass, no regression at APC 0. The improvement was only ~4ms because only 6 VPMM allocations are eliminated (inner rounds 0-2 of WHIR round 0), and per-VPMM-operation overhead is ~0.5-1ms (not ~2ms as estimated). The pool state stabilization hypothesis did not materialize — WHIR's 26ms regression from baseline is driven by factors other than intra-WHIR allocation patterns.

## 2026-04-16-0030-hoist-mle-trace-ctx-construction

**Idea**: Pre-allocate per-trace main_ptrs DeviceBuffers before the MLE round loop and reuse via copy_to() instead of to_device() to eliminate ~8400 cudaMallocAsync/cudaFreeAsync cycles per proof.

**Result**: success (marginal) — MLE Rounds at APC 300: 166ms → 162ms avg (-4ms, 1.02x lower). STARK excl trace at APC 300: 1182ms → 1118ms avg (2.20x lower vs baseline 2455ms). (vs previous: MLE -4ms, STARK -64ms but dominated by WHIR noise)

**Summary**: Added a d_main_ptrs_pool field to LogupZerocheckGpu, pre-allocated one DeviceBuffer per trace alongside the existing ping-pong fold buffers, and replaced to_device() with copy_to() + non_owning views in both Case A (late_eval) and Case B (early_eval) TraceCtx construction. The 4ms MLE Rounds improvement matches the data-backed 3-5ms prediction from the plan, confirming the CUDA pool allocator's per-cycle alloc+free overhead is ~0.5us. No regression at APC 0. MLE Rounds is now firmly kernel-execution-bound.

## 2026-04-16-0200-batch-fold-ple-descriptor-array

**Idea**: Replace per-trace sequential `fold_ple_from_evals` kernel launches (~797 at APC 300) with batched descriptor-array kernel launches (2 total: one rotate=false, one rotate=true).

**Result**: failure — Round 0 at APC 300: 181ms → 181ms (0ms change). STARK excl trace at APC 300: 1115ms → 1112ms (vs baseline: 2.21x lower, vs previous: -3ms, noise).

**Summary**: Implemented a batched CUDA kernel with FoldPleDesc descriptors and binary search block mapping, replacing ~797 per-trace kernel launches with 2 batched launches. All 94 tests pass, no regression at APC 0. The improvement was undetectable because per-launch CUDA overhead (~2-3µs × 797 = ~2ms) is at the noise floor, and the default stream already executes kernels sequentially — batching only removes launch gaps that the CUDA driver pipeline already hides. fold_ple at 9.5ms total GPU time is only 0.8% of STARK excl trace, making it too small a target for meaningful gains.

## 2026-04-16-0600-tune-vpmm-page-size-for-whir

**Idea**: Tune VPMM page size from 16 MiB to an intermediate value (4-8 MiB) to recover the 25ms WHIR regression while preserving large-allocation benefits.

**Result**: failure — STARK excl trace at APC 300: 1110ms → 1120ms (+10ms, 1.01x higher vs before). APC 0: 1804ms → 1907ms (+103ms regression). (vs baseline: 2.19x lower, unchanged from previous 2.21x)

**Summary**: Swept page sizes 4, 8, and 16 MiB (12 MiB is not viable due to VA_SIZE alignment). WHIR recovered partially at 8 MiB (-10ms) and fully at 4 MiB (-22ms), but LogUp GKR regressed by 2-3x more at each step (8 MiB: +24ms GKR, 4 MiB: +75ms GKR). APC 0 regressed by +103ms at 8 MiB. The WHIR regression from baseline is a tolerable cost of the 16 MiB page size; no intermediate value improves the overall metric. The GKR sensitivity to VPMM pool state is the dominant constraint.

## 2026-04-16-0830-gpu-stacked-reduction-poly-extraction

**Idea**: Move Stacked Reduction Round 0 polynomial reconstruction (iDFT + polynomial multiplication + accumulation) from CPU to GPU using batch_ntt_small and AoS/SoA conversion primitives.

**Result**: failure — Stacked Reduction at APC 300: 75ms → 74ms (-1ms, below 5ms rollback threshold). STARK excl trace at APC 300: 1107ms → 1124ms (vs baseline: 2.18x lower, vs previous: +17ms, noise).

**Summary**: Replaced per-bucket D2H + CPU NTT pipeline with a 10-step GPU kernel pipeline (AoS→SoA → iDFT → zero-pad → DFT → SoA→AoS → pointwise mul → iDFT → accumulate). All 94 tests pass, no regression at APC 0. The improvement was only 1-2ms because (1) CPU NTT of 256-512 EF elements is already ~10-20µs per operation, making total CPU work ~2-3ms not ~10ms as estimated, (2) the 180 GPU kernel launches add ~1-1.5ms of CUDA overhead that nearly offsets the savings, and (3) D2H sync overhead was ~50µs/call not ~200µs. The CPU polynomial reconstruction accounts for only ~3% of Stacked Reduction time — too small a target.

## 2026-04-16-1100-tune-gkr-precompute-m-parameters

**Idea**: Tune GKR fractional sumcheck PrecomputeM parameters (MIN_N, TARGET_BLOCKS, TAIL_TILE) to reduce GKR inner round cost at APC 300.

**Result**: failure — LogUp GKR at APC 300: 367ms → 363ms (-4ms, within noise). STARK excl trace at APC 300: 1113ms → 1113ms (0ms change). (vs baseline: 2.21x lower, unchanged from previous)

**Summary**: Swept 10 parameter configurations via environment variables. Raising MIN_N (shifting layers from PrecomputeM to FoldEval) made GKR monotonically worse: MIN_N=24 +6ms, MIN_N=26 +24ms, MIN_N=28 +68ms, disabled +87ms. Lowering MIN_N (more PrecomputeM layers) had no measurable benefit: MIN_N=20 -2ms, MIN_N=18 -4ms, MIN_N=16 +19ms. TARGET_BLOCKS and TAIL_TILE sweeps were also within noise (±14ms). The current defaults are near-optimal. Key finding: PrecomputeM is faster than FoldEval for layers at rem_n≥22 because its windowed approach (w=3) amortizes per-round D2H sync + CPU transcript overhead, saving ~2-3ms per window vs 3 FoldEval rounds. Fixed a buffer sizing bug where the work buffer used the compile-time MIN_N constant instead of the runtime env var value, causing CUDA crashes when MIN_N is overridden to a higher value.

## 2026-04-16-1430-column-batched-ntt-for-rs-code-matrix

**Idea**: Split the forward NTT in rs_code_matrix into L2-cache-sized column batches (~1000 columns, ~64MB per batch) to improve memory locality for the 3.4GB working set at APC 300.

**Result**: failure — Trace Commit at APC 300: 197ms → 191ms (-6ms, at noise floor). STARK excl trace at APC 300: 1121ms → 1115ms (-6ms, within noise). (vs baseline: 2.20x lower, unchanged from previous)

**Summary**: Added `batch_ntt_column_batched` that splits NTT into L2-sized batches with non-owning DeviceBuffer views. The hypothesis that L2 reuse between NTT steps would improve bandwidth did not materialize because all CUDA threads access the full batch concurrently within each kernel launch, causing L2 eviction before the second step begins. The ~6ms Trace Commit improvement is consistent with modest DRAM page locality gains from reducing the concurrent address range, but is indistinguishable from measurement noise. The NTT is fundamentally bandwidth-bound, and kernel-level fusion (not Rust-level batching) would be needed to achieve inter-step L2 reuse.

## 2026-04-16-1600-batch-stacked-reduction-round0-descriptors

**Idea**: Batch per-trace stacked reduction Round 0 kernel launches (sumcheck block sums + PLE fold) using descriptor arrays to reduce ~800 kernel launches per segment to ~100.

**Result**: failure — Stacked Reduction at APC 300: 74ms → 74ms (0ms change). STARK excl trace at APC 300: 1127ms → 1122ms (-5ms, within noise). (vs baseline: 2.19x lower, unchanged from previous)

**Summary**: Implemented batched CUDA kernels with column-based binary search descriptor mapping for both Round 0 block_sum and PLE fold. Height-grouped orchestration in Rust batches same-height traces (>=10 threshold) into single kernel launches. All 94 tests pass, no regression at APC 0. The improvement was undetectable because per-launch CUDA overhead (~2-3µs × 797 = ~2ms) is at the noise floor, and the CUDA driver pipeline already hides inter-kernel launch gaps for sequential kernels on the default stream. Stacked Reduction at 74ms (6.6% of STARK excl trace) is now too small a target for kernel launch batching.
