# Report: Parallel LogUp Precompute (008-parallel-logup-precompute)

## Idea

Parallelize `precompute_logup_combinations` calls across 8 threads. Each call launches 2 GPU kernels per AIR; overlapping them on separate CUDA streams allows concurrent execution.

## Implementation

Wrapped the `precompute_logup_combinations` loop in a parallel iterator (8 threads). Each thread operates on its own CUDA stream, enabling the GPU to overlap kernel execution across multiple AIRs simultaneously.

Files changed:
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`

## Results

| Phase      | APC 0 | APC 100 | APC 300 |
|------------|-------|---------|---------|
| **Before** STARK   | -     | -       | ~1621ms |
| **After** STARK    | -     | -       | ~1572ms |
| **Before** Round 0 | -     | -       | 216ms   |
| **After** Round 0  | -     | -       | 183ms   |

APC300 STARK improved by ~49ms; Round 0 improved by ~33ms (~15%).

## Future Work

- The thread count (8) could be tuned based on GPU stream concurrency limits and workload characteristics.
- Other per-AIR loops in the prover could benefit from similar parallelization.
