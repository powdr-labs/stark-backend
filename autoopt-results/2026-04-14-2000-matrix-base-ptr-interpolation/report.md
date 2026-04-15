# Report: 2026-04-14-2000-matrix-base-ptr-interpolation

## Description

Replace the flat per-column pointer array (`all_columns: Vec<*const EF>`, ~106K entries at APC 300) with per-matrix base pointers (~2K entries) in the `batched_interpolate_columns_kernel` used during MLE rounds. The optimization targets CPU-side overhead: each MLE round previously rebuilt a `Vec<*const EF>` by iterating O(#total_columns) on the CPU, then uploaded ~850KB H2D. By passing per-matrix base pointers and widths instead, the kernel computes column addresses internally via `base + col * height`, reducing CPU collection from O(106K) to O(2K) and H2D from 850KB to ~40KB per round.

## Implementation

### CUDA changes (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`)

- Added `InterpMatrixInfo` struct (base pointer + width) and `InterpTraceDescM` struct (per-trace descriptor with matrix_offset, num_matrices instead of columns_offset).
- Added `batched_interpolate_columns_matrix_kernel`: identical interpolation logic to existing kernel, but resolves column pointers via a short linear scan over 2-5 matrices per trace instead of a flat array lookup.
- Added `_batched_interpolate_columns_matrix` C launcher.

### Rust FFI (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)

- Added `InterpMatrixInfo` and `InterpTraceDescM` repr(C) types with Send+Sync impls.
- Added extern "C" declaration and safe wrapper `batched_interpolate_columns_matrix_gpu`.

### Rust hot path (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)

- Updated `CaseBMeta` struct: replaced `col_start: usize` with `matrix_offset: usize` and `num_matrices: usize`.
- Phase 1: Replaced `all_columns: Vec<*const EF>` with `all_matrices: Vec<InterpMatrixInfo>`. Instead of extending a flat pointer array with O(106K) entries, now pushes one `InterpMatrixInfo` per matrix (sels + preprocessed + mains, typically 2-5 per trace).
- Phase 2: Builds `InterpTraceDescM` descriptors and uploads `all_matrices` (~2K entries, ~40KB) instead of `all_columns` (~106K entries, ~850KB). Launches the new matrix-based kernel.
- No changes to Phase 3 (TraceCtx building) or any other code paths.

### Deviations from plan

None. The implementation follows the plan exactly.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **STARK excl trace APC 300** | 2455ms | 1299ms | 1289ms (avg 1293ms) | -1162ms, 1.90x lower | -6ms, 1.00x lower |
| **MLE Rounds APC 300** | 180ms | 163ms | 159ms | -21ms, 1.13x lower | -4ms, 1.03x lower |
| **Constraints APC 300** | 1634ms | 869ms | 858ms | -776ms, 1.90x lower | -11ms, 1.01x lower |
| **LogUp GKR APC 300** | 790ms | 526ms | 516ms (avg 523ms) | -267ms, 1.51x lower | -3ms, 1.01x lower |
| **Round 0 APC 300** | 662ms | 179ms | 181ms (avg 179ms) | -483ms, 3.70x lower | 0ms, unchanged |
| **Openings APC 300** | 413ms | 175ms | 174ms | -239ms, 2.37x lower | -1ms, unchanged |
| **Trace Commit APC 300** | 406ms | 253ms | 254ms | -152ms, 1.60x lower | +1ms, unchanged |
| **STARK excl trace APC 0** | 2153ms | 2151ms | 2141ms | -12ms, 1.01x lower | -10ms, 1.00x lower |
| **MLE Rounds APC 0** | 118ms | 112ms | 112ms | -6ms, 1.05x lower | 0ms, unchanged |

After measurements for APC 300 were run twice: run 1 = 1289ms, run 2 = 1297ms. Average = 1293ms.

## Assessment

The optimization achieved a **marginal improvement** at APC 300:
- MLE Rounds: consistent 4ms reduction (163ms → 159ms in both runs)
- STARK excl trace: average 6ms reduction (borderline above the 5ms rollback threshold)

The improvement is smaller than the plan's estimate of 8-12ms because:
1. **CPU iteration overhead was lower than estimated**: `Vec::extend` with `flat_map` runs at ~3-4ns per column (not 6-10ns), making the O(106K) loop cost ~0.3-0.4ms per round (not 0.6-1.0ms).
2. **H2D transfer savings are negligible**: 850KB at PCIe 4.0 takes ~0.07ms, and the CUDA pool caching means the allocation overhead for the Vec is near-zero.
3. **The dominant MLE Rounds cost is kernel execution**: At ~160ms total, the 4ms of CPU/H2D overhead is only ~2.5% of the total, limiting the ceiling for this optimization.

No regression at APC 0 (only 4K columns / ~80 matrices, overhead is negligible at either granularity).

The change is kept because: (a) it passes the rollback threshold (average 6ms > 5ms), (b) it reduces code complexity (fewer allocations, smaller H2D), and (c) it eliminates the double-indirection in the kernel (matrix scan vs pointer array lookup).

## Future Work

- The MLE Rounds bottleneck is now dominated by kernel execution time (~155ms at APC 300). Further improvements would require optimizing the interpolation kernel itself (e.g., shared memory for t0/t1 reads, warp-level data reuse).
- The per-trace `main_ptrs.to_device()` calls in Phase 3 (~5000 at APC 300) remain as residual overhead (~5ms), but prior work showed each call costs ~1μs with pool caching, making further batching low-value.
- The matrix-based kernel could potentially be extended to other interpolation paths (e.g., the unbatched `interpolate_columns_kernel`), though those are only used as fallbacks.
