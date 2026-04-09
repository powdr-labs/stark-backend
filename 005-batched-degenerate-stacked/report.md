# Report: Batched Degenerate Stacked (005-batched-degenerate-stacked)

## Idea

Batched the `stacked_reduction_sumcheck_mle_round_degenerate_kernel` calls. Collected all degenerate windows into a `DegenerateWindowCtx` array and launched a single kernel with one block per window, reducing 10,348 individual launches to 1.

## Implementation

Created a `DegenerateWindowCtx` structure to hold per-window parameters. The host collects all degenerate windows across all MLE rounds into a single context array, then launches one batched kernel where each CUDA block processes one degenerate window independently.

Files changed:
- `crates/cuda-backend/cuda/src/stacked_reduction.cu`
- `crates/cuda-backend/src/cuda/stacked_reduction.rs`
- `crates/cuda-backend/src/stacked_reduction.rs`

## Results

| Phase              | APC 0 | APC 100 | APC 300 |
|--------------------|-------|---------|---------|
| **Before** Stacked Reduction | -     | -       | 128ms   |
| **After** Stacked Reduction  | -     | -       | 90ms    |

APC300 Stacked Reduction improved by ~30%.

## Future Work

- The non-degenerate windows could benefit from a similar batching approach.
- Further gains may come from fusing degenerate and non-degenerate kernels into a single dispatch with a flag.
