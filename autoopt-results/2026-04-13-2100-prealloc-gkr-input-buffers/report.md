# Report: Pre-allocate per-thread GPU buffers for GKR input evaluation

## Description

Pre-allocate per-thread reusable GPU buffers (`intermediates`, `public_values`, `partition_ptrs`, `tmp`) for GKR input evaluation to eliminate per-AIR memory management overhead and unlock multi-stream concurrency.

Previously, each of the ~310 per-segment AIRs allocated and freed `intermediates` and `d_public_values` DeviceBuffers through the global `Mutex<MemoryManager>`, which also called `cudaMallocAsync`/`cudaFreeAsync` on the per-thread CUDA stream pool. This serialized the 8 worker threads via both the Rust mutex and implicit CUDA pool synchronization, producing ~1.0x effective parallelism despite 8 threads.

The optimization computes the maximum buffer sizes across all AIRs before spawning worker threads, then pre-allocates one set of max-sized buffers per thread. Workers reuse these buffers across AIRs via `copy_to` (H2D copy into pre-allocated memory) instead of allocating/freeing per AIR.

## Implementation

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

### Changes:

1. **Added `GkrThreadBuffers` struct** (line 36-41): Holds per-thread pre-allocated `intermediates: DeviceBuffer<EF>`, `public_values: DeviceBuffer<F>`, `partition_ptrs: DeviceBuffer<u64>`, `tmp: DeviceBuffer<Frac<EF>>`.

2. **Max buffer size pre-computation** (lines 254-279): After sorting work items, iterates all AIRs to determine maximum sizes for each buffer type. Pure CPU work, no GPU calls.

3. **Memory budget check** (lines 291-303): If total pre-allocation would exceed 2 GB, halves `num_threads` until it fits. Safety valve for edge cases with very large buffer sizes.

4. **Per-thread buffer pool allocation** (lines 305-321): Allocates `num_threads` sets of buffers on the main thread's stream before spawning workers. Zero-length buffers use `DeviceBuffer::new()` (no allocation).

5. **Modified `process_gkr_input_air`** (lines 112-210): Changed signature to accept `&mut GkrThreadBuffers`. Key changes:
   - `intermediates`: removed per-AIR allocation, uses `&buffers.intermediates` directly
   - `d_public_values`: uses `copy_to(&mut buffers.public_values)` for non-empty values, `DeviceBuffer::new()` for empty (preserving null-pointer semantics)
   - `partition_ptrs`: uses `buffers.partition_ptrs` directly, removed resize check
   - `tmp`: uses `buffers.tmp` directly, removed resize check
   - `frac_vector_scalar_multiply_ext_fp`: fixed to use `(height * num_interactions) as u32` instead of `tmp.len() as u32` to avoid unnecessary GPU work on trailing stale elements

6. **Updated thread spawning** (lines 323-349): Single-thread path uses first buffer set; multi-thread path zips `thread_buffers` with work item chunks.

### Deviations from plan:
None. Implementation followed the plan exactly.

## Results

### STARK (excl. trace) — Primary metric

| Config  | Baseline | Before Task | After Task | vs Baseline        | vs Before          |
|---------|----------|-------------|------------|--------------------|--------------------|
| APC 0   | 2153 ms  | 2155 ms     | 2152 ms    | -1ms (1.00x)       | -3ms (1.00x)       |
| APC 100 | 2155 ms  | 1837 ms     | 1632 ms    | -523ms (1.32x lower) | -205ms (1.13x lower) |
| APC 300 | 2455 ms  | 1718 ms     | 1501 ms    | -954ms (1.64x lower) | -217ms (1.14x lower) |

### LogUp GKR — Sub-component

| Config  | Baseline | Before Task | After Task | vs Baseline        | vs Before          |
|---------|----------|-------------|------------|--------------------|--------------------|
| APC 0   | 993 ms   | 1016 ms     | 1020 ms    | +27ms (0.97x)      | +4ms (1.00x)       |
| APC 100 | 775 ms   | 848 ms      | 652 ms     | -123ms (1.19x lower) | -196ms (1.30x lower) |
| APC 300 | 790 ms   | 755 ms      | 536 ms     | -254ms (1.47x lower) | -219ms (1.41x lower) |

### GKR Input Eval — Direct target

| Config  | Before Task (sum) | After Task (sum) | Improvement |
|---------|-------------------|------------------|-------------|
| APC 0   | 407 ms            | 410 ms           | -3ms (1.01x worse) |
| APC 100 | 499 ms            | 323 ms           | -176ms (1.55x lower) |
| APC 300 | 532 ms            | 318 ms           | -214ms (1.67x lower) |

### Per-segment GKR Input Eval — APC 300

| Segment | Before | After | Change |
|---------|--------|-------|--------|
| seg0    | 275 ms | 256 ms| -19ms (1.07x lower) |
| seg1    | 257 ms | 62 ms | -195ms (4.15x lower) |

The dramatic per-segment asymmetry (seg1 improved 4.15x while seg0 only 1.07x) confirms the hypothesis: seg1 has many small AIRs where per-AIR allocation overhead dominated, while seg0 has fewer, larger AIRs where kernel execution time dominates.

### Rollback criteria check:
1. STARK excl trace APC 300 improvement: 217ms > 60ms threshold -- PASS
2. STARK excl trace APC 0 regression: -3ms < 65ms threshold -- PASS
3. GPU OOM: none -- PASS
4. Prove+verify: all configs passed -- PASS
5. GKR input eval improvement at APC 300: 40.2% > 15% threshold -- PASS

## Assessment

This optimization is a clear success. At APC 300, GKR input eval improved by 1.67x (532ms to 318ms) and STARK excl trace improved by 1.14x (1718ms to 1501ms) with zero regression at APC 0. The cumulative STARK excl trace improvement vs baseline is now 1.64x.

The approach of pre-allocating max-sized buffers per thread is a well-understood pattern that eliminated two sources of serialization: (1) ~1240 Rust mutex acquisitions per segment, and (2) implicit CUDA memory pool cross-stream synchronization from cudaMallocAsync/cudaFreeAsync. The memory overhead is minimal (~100MB per thread for intermediates buffers, well within the 24GB RTX 4090 budget).

The dramatic improvement in seg1 (4.15x for GKR input eval) vs modest improvement in seg0 (1.07x) confirms that allocation overhead was the primary bottleneck for segments with many small AIRs. For segments with fewer large AIRs, kernel execution time dominates and this optimization has diminishing returns.

## Future Work

- **What worked well**: The pre-allocation pattern is simple, low-risk, and highly effective at eliminating mutex contention. The memory budget check provides a safety valve. The same pattern could be applied to other multi-threaded GPU phases.

- **Further improvements to GKR input eval**: seg0 at APC 300 still takes 256ms — this is dominated by kernel execution time for a few large AIRs, not allocation overhead. Improving seg0 would require kernel-level optimizations (e.g., fusing multiple small kernels, reducing per-kernel launch overhead).

- **Apply pre-allocation to Round 0**: Round 0 uses a similar multi-stream pattern with per-thread buffer creation. Pre-allocating those buffers could yield similar improvements, especially as the thread count was recently increased to 8.

- **Reduce intermediates buffer size**: Currently pre-allocates to the maximum across ALL AIRs in the segment. A smarter approach could partition work items into groups with similar buffer size requirements, reducing per-thread memory usage.

- **Pool-based allocation**: Instead of pre-allocating fixed max-sized buffers, a lock-free ring buffer or per-thread allocator could handle variable-sized allocations without global mutex contention. This would be more complex but more memory-efficient.
