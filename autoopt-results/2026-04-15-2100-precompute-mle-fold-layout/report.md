# Report: Pre-compute MLE Fold Layout

## Description

Pre-compute fold descriptor arrays (input_ptrs, output_ptrs, log_heights, widths) for all MLE rounds and bulk-upload them to the GPU once before the MLE loop. Additionally, fuse the two per-round `fold_pingpong` calls (mat_evals + selectors) into a single kernel launch by concatenating their descriptor entries. This replaces 112 per-round `to_device()` calls and 28 kernel launches per segment with 4 bulk uploads and 14 launches.

The optimization was expected to save 5-15ms at APC 300 based on measured per-call overhead of ~0.3-0.5μs (from `bulk-alloc-mle-fold-buffers` report) and ~0.3ms inter-kernel gap per launch.

## Implementation

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Added:
- `FoldPlan` struct: holds 4 pre-uploaded DeviceBuffer arrays (input_ptrs, output_ptrs, log_heights, widths) and per-round `FoldRoundMeta` (start index, counts, max_output_cells, view reconstruction offsets)
- `FoldRoundMeta` struct: per-round metadata including mat/sel entry counts and output offsets for view reconstruction
- `build_fold_plan()` method: iterates over all rounds, determines foldable matrices for both mat and sel groups, computes input/output pointers using physical buffer addresses (respecting the alternating odd/even write-buffer pattern), and uploads 4 flat arrays to device
- Modified `fold_mle_evals()`: when `fold_plan` is Some, creates non-owning DeviceBuffer slices into the pre-uploaded arrays and launches a single fused kernel per round. Falls back to the original two-call `fold_pingpong` path when fold_plan is None.

**No CUDA kernel changes** — the existing `batch_fold_mle_kernel` processes matrices uniformly by `blockIdx.y` index, so concatenating mat + sel entries into a single descriptor array requires no kernel modification.

**Key invariant**: output offsets at round R match input offsets at round R+1 because both are computed from the same sorted iteration order. The alternating write-buffer pattern (odd rounds → phys_B, even rounds → phys_A) is maintained by pre-computing physical buffer pointers before any Rust-side swaps.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1144ms | 1119ms | -1336ms, 2.19x lower | -25ms, 1.02x lower |
| MLE Rounds (APC 300) | 180ms | 166ms | 166ms | -14ms, 1.08x lower | 0ms, unchanged |
| Constraints (APC 300) | 1634ms | 742ms | 721ms | -913ms, 2.27x lower | -21ms, 1.03x lower |
| LogUp GKR (APC 300) | 790ms | 391ms | 367ms | -423ms, 2.15x lower | -24ms, noise |
| Round 0 (APC 300) | 662ms | 183ms | 184ms | -478ms, 3.60x lower | +1ms, noise |
| STARK excl trace (APC 100) | - | 1306ms | 1306ms | - | 0ms, unchanged |
| MLE Rounds (APC 100) | - | 143ms | 146ms | - | +3ms, noise |
| STARK excl trace (APC 0) | 2153ms | 1799ms | 1787ms | -366ms, 1.20x lower | -12ms, noise |
| MLE Rounds (APC 0) | 118ms | 119ms | 117ms | -1ms, noise | -2ms, noise |

Second APC 300 run confirmed: MLE Rounds = 164ms, STARK excl trace = 1109ms (consistent with first run).

## Assessment

The optimization did **not** achieve its goal. MLE Rounds at APC 300 was unchanged (166ms → 166ms), well below the 3ms rollback threshold. The STARK excl trace changes (-25ms at APC 300) are within measurement noise, as evidenced by similar-magnitude fluctuations in unrelated components (LogUp GKR -24ms).

**Why it failed**: The dominant cost in MLE Rounds is GPU kernel execution time (~113ms of 166ms), not CUDA API overhead. The savings from eliminating per-round descriptor uploads and halving kernel launches are:
- 112 × to_device() at ~0.3-0.5μs each: ~34-56μs
- 112 × cudaFreeAsync at ~0.3μs each: ~34μs
- 14 kernel launches saved at ~5-10μs each: ~70-140μs
- Total: ~0.14-0.23ms per segment, ~0.3-0.5ms total

This is 100x below the measurement noise floor (~10ms), making the optimization undetectable.

## Future Work

- The MLE Rounds bottleneck is now firmly in kernel execution time (~113ms at APC 300). Previous optimizations have already eliminated allocation overhead (pingpong buffers), per-round descriptor uploads (this task), and per-AIR kernel launch overhead (batched interpolation). Further MLE Rounds improvements require either:
  - **Algorithmic changes**: reducing the number of MLE rounds (currently 14)
  - **Kernel-level optimization**: improving the fold kernel's arithmetic throughput or memory bandwidth utilization
  - **Work reduction**: avoiding computation on matrices that have already reached height 1 (the kernel grid already handles this via height checks, but launch overhead for empty matrix entries could be eliminated)
- The pre-computed fold plan adds ~200 lines of code for zero measurable benefit. The complexity is not justified by the savings.
- The CPU-side view reconstruction (~per-round iteration over ~1000 matrices to create DeviceMatrix objects) could theoretically be optimized, but at ~10μs per iteration, the 14 rounds × 10μs = 140μs total is negligible.
