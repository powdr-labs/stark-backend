# Report: Parallel eq_3b (010-parallel-eq3b)

## Idea

Parallelize `eval_eq_mle` computation for eq_3b weights across 8 threads. This computation is CPU-bound and was a serial bottleneck.

## Implementation

Split the eq_3b weight computation across 8 parallel threads using rayon or equivalent threading. Each thread computes a portion of the eq_mle evaluations independently.

Files changed:
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`

## Results

| Phase  | APC 0 | APC 100 | APC 300 |
|--------|-------|---------|---------|
| **Before** STARK | -     | -       | ~1570ms |
| **After** STARK  | -     | -       | ~1568ms |

APC300 STARK improved by ~2ms. The small gain indicates this CPU computation was already a minor fraction of total time.

## Future Work

- Moving eq_3b computation to the GPU could eliminate the CPU overhead entirely.
- The parallelization framework established here can be reused for other CPU-bound preprocessing steps.
