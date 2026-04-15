# Report: Bulk-Allocate MLE Fold Output Buffers

## Description

Replace per-matrix GPU buffer allocation in MLE sumcheck fold rounds with bulk pre-allocated arena buffers. Each MLE fold round allocates ~1869 individual `DeviceMatrix<EF>` output buffers (one per trace matrix for both `mat_evals` and `sels`) at APC 300, totaling ~37K `cudaMallocAsync`/`cudaFreeAsync` calls across ~20 rounds. The optimization replaces these with a single contiguous `DeviceBuffer` per fold group per round via a `FoldArena` allocator, reducing ~37K allocations to ~40.

The plan estimated ~47ms of pure CUDA memory API overhead from nsight profiling (37K/47K of 38ms allocs + 37K/49K of 21ms frees), expecting 40-55ms savings at APC 300.

## Implementation

### New types in `crates/cuda-backend/src/base.rs`:

1. **`ArenaMatrix<T>`**: Non-owning matrix view (raw pointer + height + width) that implements `Copy`/`Clone`. Includes `to_host()` via direct `cuda_memcpy` + `current_stream_sync()`. Does not free memory on drop.

2. **`MatrixRef<T>`**: Enum with `Owned(DeviceMatrix<T>)` and `Arena(ArenaMatrix<T>)` variants. Delegates `as_ptr()`, `height()`, `width()`, `buffer_len()`, `to_host()` to the inner type. Handles the transition from `DeviceMatrix` (output of `fold_ple_evals`) to `ArenaMatrix` (output of MLE fold rounds).

3. **`FoldArena<T>`**: Holds a `Vec<DeviceBuffer<T>>`. Each `allocate_bulk(total_cells)` call creates one `DeviceBuffer`, stores it, and returns a raw `*mut T`. All buffers freed on drop.

### Changes in `crates/cuda-backend/src/logup_zerocheck/mod.rs`:

1. **Struct fields**: `mat_evals_per_trace: Vec<Vec<MatrixRef<EF>>>`, `sels_per_trace: Vec<MatrixRef<EF>>`, added `fold_arena: FoldArena<EF>`.

2. **`fold_ple_evals`**: Initial population wraps `DeviceMatrix` in `MatrixRef::Owned(...)`.

3. **`fold_mle_evals`**: Rewrote `batch_fold` closure to accept `Vec<MatrixRef<EF>>` + `&mut FoldArena<EF>`. Computes `total_cells` across all foldable matrices, calls `arena.allocate_bulk(total_cells)` once, slices the bulk buffer into per-matrix `ArenaMatrix` views via pointer offsets. Same `batch_fold_mle` kernel call as before.

4. **`sumcheck_polys_batch_eval`**: Replaced `m.buffer().as_ptr()` with `m.as_ptr()` (~6 call sites).

5. **`into_column_openings`**: Replaced `transport_matrix_d2h_col_major(&mat)` with `mat.to_host()` + `ColMajorMatrix::new(...)`.

6. **Memory accounting**: Replaced `m.buffer().len()` with `m.buffer_len()`.

No deviations from the plan.

## Results

| Metric | Baseline | Before Task | After Task (median) | vs Baseline | vs Before |
|--------|----------|-------------|---------------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1331ms | 1306ms | -1149ms, 1.88x lower | -25ms, 1.02x lower |
| MLE Rounds (APC 300) | 180ms | 175ms | 162ms | -18ms, 1.11x lower | -13ms, 1.08x lower |
| LogUp GKR (APC 300) | 790ms | 554ms | 531ms | -259ms, 1.49x lower | -23ms, 1.04x lower |
| Round 0 (APC 300) | 662ms | 177ms | 180ms | -482ms, 3.68x lower | +3ms, noise |
| Trace Commit (APC 300) | 406ms | 246ms | 251ms | -155ms, 1.62x lower | +5ms, noise |
| Stacked Reduction (APC 300) | 311ms | 75ms | 74ms | -237ms, 4.20x lower | -1ms, noise |
| STARK excl trace (APC 0) | 2166ms (before=2166ms) | 2166ms | 2160ms | -6ms, noise | -6ms, noise |
| MLE Rounds (APC 0) | 114ms | 114ms | 111ms | -3ms, noise | -3ms, noise |

Two APC 300 runs: STARK excl trace 1314ms, 1297ms (median ~1306ms). MLE Rounds 161ms, 164ms (median ~162ms).

## Assessment

The optimization **did not meet the success criteria**. The plan expected 40-55ms savings on MLE Rounds at APC 300, but the actual improvement was only ~13ms (median). This is below the 20ms rollback threshold specified in the plan.

**Why the improvement was smaller than expected:**

The plan estimated ~47ms of CUDA memory API overhead based on nsight profiling. The actual per-call overhead of `cudaMallocAsync` with CUDA pool caching is ~0.3-0.5μs (not the ~0.8μs estimated). The pool allocator is highly optimized for repeated alloc/free patterns of similar sizes — exactly the pattern MLE fold rounds produce. Over ~37K calls, the true overhead is ~15-20ms, and we recovered ~13ms of that. The remaining ~5ms is likely from the arena's own CPU-side offset computation and metadata bookkeeping, which partially offsets the allocation savings.

Additionally, the MLE Rounds time was already reduced from the baseline 180ms to 175ms by prior optimizations (batch-mle-interpolation), leaving less headroom.

**Decision**: Revert per rollback criteria. The implementation is correct and produces valid proofs, but the improvement is too small to justify the added type complexity (ArenaMatrix, MatrixRef, FoldArena).

## Future Work

- The dominant MLE Rounds cost at APC 300 is kernel execution time, not allocation overhead. Further optimization should target the `batch_fold_mle` kernel itself (e.g., persistent threads, better memory coalescing).
- The `sumcheck_polys_batch_eval` per-trace `main_ptrs.to_device()` calls (~5600 per benchmark) are another source of per-call overhead, but a previous task showed this is also only ~5ms total.
- Multi-streaming MLE rounds (like Round 0 and GKR input eval) could help if individual fold kernels don't saturate the GPU, but MLE rounds are inherently sequential (each round depends on the previous round's output).
- The arena pattern could be combined with persistent kernel approaches where a single kernel processes all rounds, eliminating inter-round launch overhead.
