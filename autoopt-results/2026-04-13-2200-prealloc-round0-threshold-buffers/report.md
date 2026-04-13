# Report: Pre-allocate per-thread Round 0 buffers with adaptive size threshold

## Description

Eliminate per-AIR GPU memory allocation overhead in Round 0 constraint and interaction evaluation by pre-allocating reusable per-thread buffer pools. Each of the 8 concurrent worker threads processes ~78 AIRs at APC 300, with each AIR requiring 2 intermediates buffers and 2 temp_sums buffers (across zerocheck and logup evaluation). Without pre-allocation, each buffer allocation/free pair acquires the global `MEMORY_MANAGER` mutex, causing ~8 serialized mutex acquisitions per AIR × 623 AIRs × 2 segments ≈ 9,968 contended mutex operations.

The key innovation vs the previous failed `prealloc-round0-buffers` attempt is using the **95th percentile** of buffer sizes rather than the maximum. At APC 300, the max intermediates buffer is ~48M elements (193 MB) driven by a few large AIRs, while the 95th percentile is ~786K elements (3.1 MB) — a 61x difference. Pre-allocating at the 95th percentile covers 95%+ of AIRs while keeping total allocation under 100 MB across all threads. The few large AIRs above the threshold fall back to dynamic allocation.

## Implementation

### Files changed

1. **`crates/cuda-backend/src/pkey.rs`**: Added `logup_round0_buffer_size: u32` field to `AirDataGpu`. Pre-computed at keygen time by constructing the interaction DAG (same logic as `evaluate_round0_interactions_gpu`) and extracting `rules.buffer_size`. This makes the logup buffer_size available during Phase 1 work item preparation without rebuilding the DAG per AIR.

2. **`crates/cuda-backend/src/logup_zerocheck/round0.rs`**:
   - Added `Round0ThreadBuffers` struct with 4 fields: `zc_intermediates`, `zc_temp_sums`, `logup_intermediates`, `logup_temp_sums`.
   - Modified `evaluate_round0_constraints_gpu` to accept `prealloc_intermediates: Option<&mut DeviceBuffer<F>>` and `prealloc_temp_sums: Option<&mut DeviceBuffer<EF>>`. When the pre-allocated buffer is large enough, the function uses it directly (no mutex, no allocation). Otherwise falls back to dynamic allocation.
   - Same pattern for `evaluate_round0_interactions_gpu`.

3. **`crates/cuda-backend/src/logup_zerocheck/mod.rs`**:
   - Added buffer size fields to `Round0AirWorkItem`: `zc_intermed_cap`, `zc_temp_sums_cap`, `logup_intermed_cap`, `logup_temp_sums_cap`.
   - Phase 1: Pre-compute buffer sizes per work item using FFI functions (`_zerocheck_r0_intermediates_buffer_size`, etc.) and the pre-computed `logup_round0_buffer_size`.
   - Between Phase 1 and Phase 2: Compute 95th-percentile buffer sizes, memory budget check (reduce threads or skip pre-allocation if > 2GB), pre-allocate per-thread buffer pools.
   - Modified `process_air_round0` to accept `&mut Round0ThreadBuffers`, check per-AIR sizes against pre-allocated capacity, and pass appropriate `Option` to evaluation functions.
   - Phase 2: Each thread receives its own buffer set; single-thread path also uses the buffer pool.

### Deviations from plan

1. **Percentile threshold instead of max**: The plan specified using the maximum buffer size across all work items. Empirical measurement showed the max intermediates is 48M elements (193 MB/thread × 8 threads = 1.5 GB), which triggered thread count reduction to 2 and caused GPU memory pressure (+350ms STARK excl trace regression). Switched to 95th percentile, which is 786K elements (3.1 MB/thread) — covers 95% of AIRs with negligible memory impact.

2. **No result buffer pre-allocation**: The plan included pre-allocating sp_evals and s_evals (result buffers). These were omitted to avoid the ManuallyDrop complexity for D2H copies. Result buffers are small (48-64 elements), so their allocation overhead is minimal. This simplification avoids 4 of the 12 pre-allocatable mutex acquisitions but keeps the implementation straightforward.

## Results

### APC 300 (median of 3 runs)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455 ms | 1407 ms | 1372 ms | -1083 ms, 1.79x lower | -35 ms, 1.03x lower |
| Round 0 | 662 ms | 276 ms | 242 ms | -420 ms, 2.74x lower | -34 ms, 1.14x lower |
| LogUp GKR | 790 ms | 533 ms | 530 ms | -260 ms, 1.49x lower | -3 ms (noise) |
| MLE Rounds | 180 ms | 169 ms | 170 ms | -10 ms, 1.06x lower | +1 ms (noise) |
| Openings | 413 ms | 176 ms | 176 ms | -237 ms, 2.35x lower | 0 ms |
| Stacked Reduction | 311 ms | 75 ms | 75 ms | -236 ms, 4.15x lower | 0 ms |
| Trace Commit | 406 ms | 247 ms | 248 ms | -158 ms, 1.64x lower | +1 ms (noise) |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153 ms | 2164 ms | 2159 ms | +6 ms (noise) | -5 ms (noise) |
| Round 0 | 178 ms | 178 ms | 177 ms | -1 ms (noise) | -1 ms (noise) |

### Stability

Before APC 300 runs: 1407, 1401, 1412 → median 1407 ms, spread 11 ms
After APC 300 runs: 1372, 1365, 1381 → median 1372 ms, spread 16 ms

The 35 ms improvement exceeds the combined spread (~27 ms), confirming it is above the noise floor.

## Assessment

**Success.** Round 0 improved by 34 ms (12.3%) at APC 300, bringing cumulative STARK excl trace improvement to 1.79x vs baseline. The improvement is consistent across all 3 runs and there are no regressions in any other phase at either APC configuration.

The improvement is at the lower end of the plan's expected range (36-76 ms) because:
1. Only intermediates and temp_sums are pre-allocated (4 of 6 buffers per AIR), leaving result buffer allocations dynamic.
2. The 95th percentile threshold means ~5% of AIRs still fall back to dynamic allocation, though these are the largest AIRs where allocation time is dominated by kernel execution.
3. At APC 300, segments have 306-317 work items. With 8 threads, each handles ~39 AIRs. Of these, ~2 exceed the threshold per thread, contributing 4 extra mutex acquisitions each — a modest overhead.

## Future Work

- **Pre-allocate result buffers (sp_evals, s_evals)**: Would eliminate 4 more mutex acquisitions per AIR. Requires ManuallyDrop view for correct D2H copy length. Small incremental benefit expected (result buffers are tiny).
- **Pre-allocate d_rules, d_numer_weights, d_denom_weights**: These are data-dependent per AIR (logup weights depend on runtime eq_3bs and beta_pows). Could pre-allocate max-sized buffers and use `copy_to` instead of `to_device`.
- **Pre-allocate d_main_parts**: Per-AIR `to_device()` for a small Vec of pointers. Tiny allocation but still a mutex acquisition.
- **Adaptive percentile**: Currently hardcoded at 95%. Could be tuned based on total buffer memory vs GPU capacity. Higher percentile = fewer fallbacks but more memory pressure.
- **Lock-free allocation**: The underlying bottleneck is the global `Mutex<MemoryManager>`. A lock-free pool or per-thread allocator would eliminate the serialization point entirely, benefiting all phases. This is a larger architectural change.
