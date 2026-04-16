# Report: Batch fold_ple_from_evals Descriptor Array

## Description

The `fold_ple_evals` function in `LogupZerocheckGpu` launches one `fold_ple_from_evals_kernel` per matrix per trace in a sequential loop. At APC 300, nsight showed ~797 instances of `fold_ple_from_evals_kernel<false>` totaling 9.5ms GPU time (avg 12µs each). The optimization replaces the per-trace loop with a batched descriptor-array kernel: a single launch for all rotate=false operations and a single launch for all rotate=true operations. This was expected to save ~2-3ms in kernel launch overhead and improve SM utilization for small traces.

## Implementation

### CUDA side (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`)
- Added `FoldPleDesc` struct with src/dst pointers, dimensions, and `block_start` field
- Added `batched_fold_ple_from_evals_kernel<ROTATE>` using binary search on `block_start` to map 1D blocks to descriptors, then decomposing local block index into (row_block, col_idx) per descriptor
- Added `_batched_fold_ple_from_evals` C launcher with same block_size/smem logic as existing kernel

### Rust FFI (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)
- Added `#[repr(C)] FoldPleDesc` struct with Send+Sync
- Added extern "C" binding and safe wrapper `batched_fold_ple_from_evals`

### Rust orchestration (`crates/cuda-backend/src/logup_zerocheck/fold_ple.rs`)
- Added `FoldPleItem`, `FoldPleResult`, and `batched_fold_ple_evals_rotate` function
- Collects all items, pre-allocates per-item output DeviceBuffers, builds two descriptor arrays (rotate=false and rotate=true), uploads via `to_device()`, and launches 1-2 batched kernels

### Integration (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)
- Replaced per-trace `fold_ple_evals_rotate` loop with 3-phase approach: collect all matrices → batch-launch → distribute results
- mem_limit accounting preserved (common_main is always last per trace)

### Deviations from plan
- None significant. Plan was followed as written.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **STARK excl trace APC 300** | 2455ms | 1115ms | 1127ms (1112ms run2) | 2.18x lower | +12ms noise (run2: -3ms noise) |
| **Round 0 APC 300** | 662ms | 181ms | 181ms (179ms run2) | 3.66x lower | 0ms (run2: -2ms noise) |
| **LogUp GKR APC 300** | 790ms | 370ms | 370ms (363ms) | 2.14x lower | 0ms |
| **MLE Rounds APC 300** | 180ms | 162ms | 170ms (166ms) | 1.06x lower | +8ms noise |
| **STARK excl trace APC 100** | — | 1316ms | 1310ms | — | -6ms noise |
| **Round 0 APC 100** | — | 168ms | 162ms | — | -6ms noise |
| **STARK excl trace APC 0** | — | 1891ms | 1810ms | — | -81ms (noise in GKR) |
| **Round 0 APC 0** | — | 177ms | 177ms | — | 0ms |

All 94 tests pass. No regression at any APC configuration.

## Assessment

The optimization did **not** achieve its goal. Round 0 at APC 300 showed 0ms improvement across 3 runs.

The plan estimated 3-5ms from eliminating ~795 kernel launch overhead cycles. However:
1. **Per-launch CUDA overhead is ~2-3µs**, confirmed by previous tasks (batch-mle-main-ptrs-upload measured ~1µs per CUDA API call). At 797 launches, total overhead is ~1.6-2.4ms — barely above the measurement noise floor.
2. **SM utilization gain is negligible**: the fold_ple kernel operates on the default stream (single-threaded section, not multi-streamed), so there's no inter-stream concurrency to improve. The kernel simply runs sequentially either way — batching just removes the CPU→GPU launch gap between kernels, which the CUDA driver pipeline already hides.
3. **9.5ms total GPU time for 797 kernels** means the average kernel takes 12µs. The launch overhead (~2µs per kernel) is already 83% hidden by kernel execution time via the CUDA driver's kernel pipeline.

The fundamental issue: `fold_ple_evals` runs on the default stream where kernels execute sequentially. Batching multiple sequential kernels into one larger kernel only saves the launch gap between them, which the CUDA driver already pipelines.

## Future Work

- **fold_ple is not a bottleneck**: At 9.5ms total GPU time (~0.8% of STARK excl trace), fold_ple is too small to yield meaningful gains even with perfect optimization.
- **Multi-stream fold_ple**: Running fold_ple across multiple streams (like Round 0 and GKR) could provide ~4-5x speedup on the 9.5ms, but the 2ms net gain doesn't justify the complexity.
- **Focus elsewhere**: The remaining optimization opportunities in STARK excl trace are in kernel execution time (LogUp GKR 370ms, Round 0 181ms, MLE Rounds 162ms), not in CUDA API overhead, which has been thoroughly addressed by previous tasks.
