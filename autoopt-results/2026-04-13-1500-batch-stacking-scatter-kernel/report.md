# Report: Batch Stacking Scatter Kernel

## Description

The `stack_traces_into_expanded()` function iterates over every trace column (~106K at APC 300, ~53K per segment) issuing individual CUDA API calls (`cudaMemcpyAsync` or `batch_expand_pad_wide` kernel launch) per column. At ~1-5us CPU overhead per call, this loop costs ~100-150ms of CPU-side stalling. Replacing the per-column loop with a two-phase approach (build descriptor array on CPU, then single batched scatter kernel on GPU) eliminates this CPU overhead. The optimization is proportional to the number of columns, so it primarily benefits high-APC configs.

## Implementation

### Files changed

1. **`crates/cuda-backend/cuda/src/matrix.cu`**: Added `StackColDesc` struct and `stack_columns_kernel` — each CUDA block handles one column descriptor, with threads cooperatively copying elements from `src` to `dst` with configurable stride. Added `_stack_columns` extern "C" launcher.

2. **`crates/cuda-backend/src/cuda/matrix.rs`**: Added `StackColDesc` Rust struct (`#[repr(C)]`, matching CUDA layout), FFI extern declaration for `_stack_columns`, and safe wrapper `stack_columns()`.

3. **`crates/cuda-backend/src/stacked_pcs.rs`**: Rewrote `stack_traces_into_expanded()` into two phases:
   - Phase 1: CPU-only loop building a `Vec<StackColDesc>` with pointer arithmetic (no CUDA calls).
   - Phase 2: Single H2D upload of descriptors + single kernel launch.

4. **`crates/cuda-backend/src/error.rs`**: Replaced `StackTracesError::MemCopy` and `BatchExpandPadWide` variants with `DescriptorUpload` and `StackColumns`.

### Key decisions

- Used 256 threads per block (matching plan). Most columns have height 8-256, so this is sufficient.
- Descriptor array is ~24 bytes per column * 53K = ~1.3MB per segment — small overhead.
- Removed `cuda_memcpy` and `batch_expand_pad_wide` imports from `stacked_pcs.rs` as they're no longer used there.

### Deviations from plan

None. The implementation followed the plan exactly.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1746ms | 1597ms | -858ms, 1.54x lower | -149ms, 1.09x lower |
| Trace Commit | 406ms | 414ms | 255ms | -151ms, 1.59x lower | -159ms, 1.62x lower |
| Constraints | 1634ms | 1103ms | 1115ms | -519ms, 1.47x lower | +12ms (noise) |
| LogUp GKR | 790ms | 571ms | 588ms | -202ms, 1.34x lower | +17ms (noise) |
| Round 0 | 662ms | 350ms | 345ms | -317ms, 1.92x lower | -5ms (noise) |
| MLE Rounds | 180ms | 181ms | 180ms | 0ms | -1ms |
| Openings | 413ms | 227ms | 225ms | -188ms, 1.84x lower | -2ms |
| WHIR | 100ms | 100ms | 101ms | +1ms | +1ms |
| Stacked Reduction | 311ms | 125ms | 124ms | -187ms, 2.51x lower | -1ms |

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | — | 1967ms | 1683ms | — | -284ms, 1.17x lower |
| Trace Commit | — | 418ms | 340ms | — | -78ms, 1.23x lower |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2137ms | 2157ms | +4ms (noise) | +20ms (noise) |
| Trace Commit | 518ms | 521ms | 531ms | +13ms (noise) | +10ms (noise) |

### Nsys validation

D2D memcpy operations dropped from 101,838 to 325 across the APC 300 benchmark, confirming the per-column loop is eliminated. The remaining 325 D2D copies are from other code paths.

## Assessment

The optimization achieved its goal. Trace Commit at APC 300 improved by 1.62x (414ms -> 255ms), exceeding the 18% target from the plan. The STARK excl trace improvement at APC 300 is -149ms (1.09x), meeting the 4% target.

APC 0 shows no regression (within noise), as expected since it only has ~4K columns.

APC 100 also benefits significantly: Trace Commit improved 1.23x (418ms -> 340ms).

The optimization is clean and minimal — it replaces ~45 lines of per-column CUDA calls with ~20 lines of descriptor building plus a single kernel launch. The complexity is low and the code is easier to understand.

Cumulative STARK excl trace improvement vs baseline at APC 300: **1.54x** (2455ms -> 1597ms).

## Future Work

- The `fill_zero()` call zeros the entire stacked buffer before the scatter kernel writes non-zero data. At APC 300 the buffer is ~400MB. A "scatter-with-zero" variant that also handles zero-padding within the kernel could eliminate this separate memset, potentially saving another 10-20ms.
- The scatter kernel uses 256 threads per block, but most columns have very few elements (8-64). A warp-per-column approach (32 threads) with multiple columns per block could improve SM utilization.
- The descriptor H2D upload (~1.3MB per segment) could use pinned memory for slightly faster transfer, though the current pageable path is already fast enough at ~1ms.
- Combining this scatter kernel with the MLE interpolation step that follows could save one full GPU memory pass over the stacked buffer.
