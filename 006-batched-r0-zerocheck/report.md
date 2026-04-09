# Report: Batched Round 0 Zerocheck (006-batched-r0-zerocheck)

## Idea

Batched Round 0 zerocheck evaluation using a BlockCtx/Round0ZerocheckCtx pattern. Groups small AIRs by matching parameters: (height, num_x, num_cosets, constraint_degree). For APC300, this batches 279 out of 306 traces (91%) into 12 groups.

## Implementation

Introduced `Round0ZerocheckCtx` to describe per-AIR zerocheck work. AIRs with identical structural parameters are grouped and dispatched as a single kernel launch. The remaining non-matching AIRs are launched individually.

Files changed:
- `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`
- `crates/cuda-backend/src/cuda/logup_zerocheck.rs`
- `crates/cuda-backend/src/logup_zerocheck/round0.rs`
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`

## Results

| Phase      | APC 0 | APC 100 | APC 300 |
|------------|-------|---------|---------|
| **Before** Round 0 | -     | -       | ~233ms  |
| **After** Round 0  | -     | -       | ~222ms  |

APC300 Round 0 improved by ~5ms. The modest improvement is because parallel CUDA streams already hide most of the launch overhead for small AIRs.

## Future Work

- The 27 unbatched traces (9%) could be handled with a more flexible grouping strategy that pads or generalizes parameters.
- Benefit would be larger on systems without parallel stream support or with higher kernel launch overhead.
