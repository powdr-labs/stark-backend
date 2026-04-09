# Report: Stacked Deferred Sync (003-stacked-deferred-sync)

## Idea

Cherry-picked commit c868e4b. Eliminated ~32,500 per-window GPU-CPU sync barriers in stacked reduction MLE rounds. The original implementation synchronized between the host and device after every window kernel launch, creating massive overhead from repeated round-trips.

## Implementation

Restructured `batch_sumcheck_poly_eval` to:
1. Upload all `eq_ub` data once at the start
2. Zero `d_accum` once
3. Launch all window kernels without intermediate synchronization
4. Perform a single device-to-host transfer after all kernels complete

Files changed:
- `crates/cuda-backend/src/cuda/stacked_reduction.rs`
- `crates/cuda-backend/src/stacked_reduction.rs`

## Results

| Phase              | APC 0 | APC 100 | APC 300 |
|--------------------|-------|---------|---------|
| **Before** Stacked Reduction | -     | -       | 311ms   |
| **After** Stacked Reduction  | -     | -       | 126ms   |
| **Before** STARK   | -     | -       | 2491ms  |
| **After** STARK    | -     | -       | 2088ms  |

APC300 Stacked Reduction improved by ~60%. Combined with Round 0 parallel streams, STARK total improved by ~16%.

## Future Work

- The remaining ~126ms may contain irreducible compute time, but further kernel fusion across windows could reduce launch overhead.
- Similar deferred-sync patterns could be applied to other multi-kernel phases.
