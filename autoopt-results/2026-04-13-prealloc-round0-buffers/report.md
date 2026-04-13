# Report: Pre-allocate per-thread GPU buffers for Round 0

## Description

Pre-allocate reusable per-thread GPU buffers for Round 0 constraint and interaction evaluation to eliminate per-AIR `cudaMallocAsync`/`cudaFreeAsync` and `MemoryManager` mutex overhead. This followed the same pattern that achieved a 1.67x improvement for GKR input evaluation in the previous task (`2026-04-13-2100-prealloc-gkr-input-buffers`).

The plan was to compute the maximum buffer sizes across all AIRs, allocate one set of max-sized buffers per worker thread, and reuse them across AIRs instead of allocating/freeing per-AIR.

## Implementation

### Files changed
- `crates/cuda-backend/src/logup_zerocheck/round0.rs`: Modified `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu` to accept pre-allocated `&mut DeviceBuffer` parameters instead of allocating internally. Added `compute_logup_round0_buffer_size()` helper function.
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`: Added `Round0ThreadBuffers` struct, `logup_r0_buffer_size` field to `Round0AirWorkItem`, max-size pre-computation loop, memory budget safety valve, and updated thread spawning to pass pre-allocated buffers.
- `crates/cuda-backend/src/pkey.rs`: Added `logup_round0_buffer_size: u32` field to `AirDataGpu` to cache the logup round0 buffer_size during keygen, avoiding expensive per-segment recomputation.

### Key deviations from plan
1. **Keygen caching**: The plan called for computing `logup_r0_buffer_size` per work item via `compute_logup_round0_buffer_size()`. Initial implementation did this, adding ~200ms of CPU overhead per benchmark. Fixed by caching in the proving key during keygen (`AirDataGpu.logup_round0_buffer_size`).
2. **Module visibility**: Changed `round0` module from private to `pub(crate)` to allow `pkey.rs` to import `compute_logup_round0_buffer_size`.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **STARK excl trace APC 300** | 2455ms | 1442ms | 1779ms | 676ms lower (1.38x) | 337ms higher (1.23x) |
| **STARK excl trace APC 100** | — | 1604ms | 1953ms | — | 349ms higher (1.22x) |
| **STARK excl trace APC 0** | 2153ms | 2134ms | 2165ms | 12ms higher (1.01x) | 31ms higher (1.01x) |
| **Round 0 APC 300** | 662ms | 296ms | 443ms | 219ms lower (1.49x) | 147ms higher (1.50x) |
| **Round 0 APC 100** | — | 250ms | 385ms | — | 135ms higher (1.54x) |
| **Round 0 APC 0** | 178ms | 177ms | 177ms | 1ms lower (1.01x) | 0ms (1.00x) |
| **LogUp GKR APC 300** | 790ms | 539ms | 727ms | 63ms lower (1.09x) | 188ms higher (1.35x) |
| **LogUp GKR APC 100** | — | 644ms | 855ms | — | 211ms higher (1.33x) |
| **LogUp GKR APC 0** | 993ms | 1001ms | 1030ms | 37ms higher (1.04x) | 29ms higher (1.03x) |

## Assessment

**The optimization FAILED.** It caused significant regression across all metrics at APC 100 and APC 300, and must be reverted.

### Root causes of failure

**1. Batching invalidates the plan's core assumption.** The plan assumed Round 0 processes ~310 AIRs per segment in a single call. In reality, AIRs are processed in small batches of 20-42 per call (due to memory-limited batching in the prover). The `>= 100 AIR` multi-threading threshold is **never reached**, so `num_threads = 1` for all batches. The pre-allocation reduces to a single set of max-sized buffers per batch.

**2. Max-sized buffers are enormous.** The Round 0 buffer sizes scale with `max_temp_bytes` (= 5 GiB / 8 = 640 MiB per thread). The max buffer sizes across all AIRs are ~984 MB per thread (96M elements for zc_intermediates + 144M for logup_intermediates + temp_sums). This is fundamentally different from GKR pre-alloc where buffer sizes are bounded by `TASK_SIZE * buffer_size` (a small constant).

**3. Systemic GPU memory pressure.** Pre-allocating ~1 GB of GPU memory for the largest AIR in each batch (even though most AIRs in the batch only need <10 MB) creates memory pressure that degrades performance of ALL subsequent GPU operations. This explains the ~190ms LogUp GKR regression even though GKR code was untouched — the CUDA memory pool state after Round 0's large allocation/deallocation cycle adversely affects subsequent allocation patterns.

### Why GKR pre-alloc worked but Round 0 doesn't

| Factor | GKR Pre-alloc | Round 0 Pre-alloc |
|--------|--------------|-------------------|
| AIRs per call | ~310 (all at once) | 20-42 (small batches) |
| Multi-threading | 8 threads (threshold met) | 1 thread (threshold never met) |
| Max buffer size | ~100 MB per thread | ~984 MB per thread |
| Buffer size driver | `TASK_SIZE * buffer_size` (constant) | `max_temp_bytes` (proportional to memory limit) |
| Alloc/free pairs eliminated | ~1240 per segment | ~80 per batch |

## Future Work

- **Batch-level optimization**: Instead of pre-allocating max-sized buffers, consider restructuring the batching to process more AIRs per call (e.g., lowering the memory limit threshold for batching). This would allow the multi-threading threshold to be reached.
- **Tiered pre-allocation**: Pre-allocate at a "typical" size (e.g., median AIR buffer size) and fall back to per-AIR allocation for the few large AIRs that exceed it. This avoids the 1 GB max-size allocation.
- **Per-stream memory pools**: Instead of pre-allocating through the global `MemoryManager` mutex, use CUDA per-stream memory pools (`cudaMemPoolCreate`) to eliminate cross-stream contention without large pre-allocations.
- **Keygen-cached logup buffer_size**: The `logup_round0_buffer_size` field added to `AirDataGpu` is valuable regardless — it eliminates the expensive per-segment DAG reconstruction for buffer size computation. This should be kept even if the pre-allocation is reverted.
- **Profile memory pool behavior**: Use `CUDA_MEMPOOL_LOGGING` to understand how the memory pool handles the allocation/deallocation patterns and identify the source of cross-phase interference.
