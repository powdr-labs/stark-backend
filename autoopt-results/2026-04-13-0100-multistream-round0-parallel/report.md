# Report: Multi-stream Round 0 Parallel AIR Processing

## Description

Round 0 constraint evaluation processes 623 AIRs (at APC 300) sequentially on a single CUDA stream. Each AIR launches kernels that use only 1-4 of the RTX 4090's 128 SMs, leaving >95% of GPU compute idle. By distributing AIRs across multiple OS threads (each with its own `cudaStreamPerThread`), kernels from different threads execute concurrently on idle SMs, recovering GPU utilization. CPU-side work (symbolic constraint construction, interaction DAG building, iDFT post-processing) also parallelizes across threads.

## Implementation

### Files changed

1. **`crates/cuda-common/src/copy.rs`**: Added `MemCopyD2HStreamSync` trait with `to_host_on_current_stream()` method. Uses `cudaStreamSynchronize(cudaStreamPerThread)` instead of the global `COPY_EVENT` mutex for D2H transfers, enabling contention-free multi-threaded D2H copies.

2. **`crates/cuda-backend/src/logup_zerocheck/mod.rs`**:
   - Added `NUM_ROUND0_STREAMS = 4` constant (number of worker threads).
   - Added `Round0AirResult` struct and `Round0AirWorkItem` struct to encapsulate per-AIR inputs/outputs.
   - Extracted the per-AIR loop body (constraint eval + interaction eval + D2H + post-processing) into a standalone `process_air_round0()` function.
   - Restructured the Round 0 loop into 3 phases:
     - **Phase 1**: Build work items, sort by descending height for load balance, sync main thread's stream.
     - **Phase 2**: Process AIRs in parallel via `std::thread::scope` with N worker threads (each gets its own CUDA stream via `cudaStreamPerThread`).
     - **Phase 3**: Scatter results back into `batch_sp_poly`.
   - Added an AIR count threshold (`>= 100`) to avoid multi-threading when few large AIRs would cause memory pool contention without concurrent kernel benefits. Without this threshold, APC 0 (99 AIRs, ~20 per segment) showed a ~170ms regression in LogUp GKR due to multi-thread VPMM allocation patterns affecting subsequent memory pool behavior across segments.

### Key decisions

- **Threshold of 100 AIRs**: Initial implementation with no threshold caused a consistent 6-11% STARK excl trace regression at APC 0 due to multi-threaded VPMM usage polluting the memory pool for subsequent proving phases. Setting the threshold to 100 ensures APC 0 (~20 AIRs/segment) uses the sequential path while APC 100 (~120 AIRs/segment) and APC 300 (~300 AIRs/segment) use multi-threading.
- **Memory budget division**: Each worker thread gets `memory_limit_bytes / NUM_ROUND0_STREAMS` as its temp buffer budget, keeping total GPU memory within the original envelope.
- **Sort by descending height**: Prevents large AIRs from clustering in one chunk, creating a long tail while other threads are idle.

### Deviations from plan

- The plan did not include the AIR count threshold. This was added after observing the APC 0 regression during measurement.
- The plan set the threshold at `num_threads <= 1`; actual threshold is at AIR count `>= 100`.

## Results

### STARK excl trace (target metric)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 0** | 2153 ms | 2124 ms | 2139 ms | -14 ms, 1.01x lower | +15 ms, 1.01x higher |
| **APC 100** | 2155 ms | 2055 ms | 1870 ms | -285 ms, 1.15x lower | -185 ms, 1.10x lower |
| **APC 300** | 2455 ms | 2282 ms | 1969 ms | -486 ms, 1.25x lower | -313 ms, 1.16x lower |

### Round 0 (direct target)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 0** | 178 ms | 179 ms | 177 ms | -1 ms, 1.01x lower | -2 ms, 1.01x lower |
| **APC 100** | 464 ms | 464 ms | 274 ms | -190 ms, 1.69x lower | -190 ms, 1.69x lower |
| **APC 300** | 662 ms | 669 ms | 351 ms | -311 ms, 1.89x lower | -318 ms, 1.91x lower |

### Full STARK breakdown at APC 300

| Component | Baseline | Before Task | After Task | vs Baseline | vs Before |
|-----------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455 ms | 2282 ms | 1969 ms | 1.25x lower | 1.16x lower |
| Constraints | 1634 ms | 1644 ms | 1331 ms | 1.23x lower | 1.24x lower |
| LogUp GKR | 790 ms | 792 ms | 797 ms | ~same | ~same |
| Round 0 | 662 ms | 669 ms | 351 ms | 1.89x lower | 1.91x lower |
| MLE Rounds | 180 ms | 182 ms | 181 ms | ~same | ~same |
| Openings | 413 ms | 227 ms | 228 ms | 1.81x lower | ~same |
| WHIR | 100 ms | 101 ms | 103 ms | ~same | ~same |
| Stacked Reduction | 311 ms | 125 ms | 125 ms | 2.49x lower | ~same |
| Trace Commit | 406 ms | 410 ms | 407 ms | ~same | ~same |

### Full STARK breakdown at APC 100

| Component | Baseline | Before Task | After Task | vs Baseline | vs Before |
|-----------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155 ms | 2055 ms | 1870 ms | 1.15x lower | 1.10x lower |
| Constraints | 1393 ms | 1399 ms | 1211 ms | 1.15x lower | 1.16x lower |
| LogUp GKR | 775 ms | 784 ms | 783 ms | ~same | ~same |
| Round 0 | 464 ms | 464 ms | 274 ms | 1.69x lower | 1.69x lower |
| MLE Rounds | 150 ms | 148 ms | 149 ms | ~same | ~same |

## Assessment

This optimization is a clear success, far exceeding the plan's targets:

- **Round 0 at APC 300**: 669ms → 351ms (1.91x, target was 1.15-1.25x)
- **STARK excl trace at APC 300**: 2282ms → 1969ms (1.16x, target was 1.04-1.08x)
- **No regression at APC 0**: STARK excl trace within noise (2124ms → 2139ms, +0.7%)

The cumulative improvement vs baseline at APC 300 is now 1.25x for STARK excl trace (from 2455ms to 1969ms), combining the stacked reduction MLE sync optimization (previous task) with this multi-stream Round 0 parallelism.

The implementation is relatively simple (~120 lines of new code) and the approach is sound: it leverages CUDA's native per-thread default streams to achieve concurrent kernel execution without explicit stream management.

## Future Work

- **Memory pool contention**: The global `MemoryManager` mutex is a serialization point. With 4 threads doing ~10 allocations each per AIR, there's ~6,230 lock acquisitions per Round 0 at APC 300. Pre-allocating per-thread reusable buffers could eliminate most allocations and further improve concurrency.
- **Adaptive thread count**: The current threshold (100 AIRs) is static. A more sophisticated approach could estimate the benefit based on the distribution of AIR heights — e.g., if >50% of AIRs have height <= 64, use multi-threading regardless of count.
- **Extend to MLE rounds**: The same multi-stream approach could potentially benefit MLE constraint evaluation rounds, which also process AIRs sequentially.
- **Combined with kernel fusion**: Small AIRs (height 8-64) produce kernels that barely occupy 1 SM. Fusing multiple small AIR evaluations into a single kernel launch would reduce launch overhead and could be combined with multi-streaming for medium-sized AIRs.
- **LogUp GKR regression investigation**: Multi-threading with `NUM_ROUND0_STREAMS=4` at APC 0 caused a consistent ~170ms regression in LogUp GKR (which runs in a subsequent segment's proving phase). This suggests VPMM pool fragmentation from multi-threaded alloc/free patterns. Understanding this interaction could allow lowering or removing the threshold.
