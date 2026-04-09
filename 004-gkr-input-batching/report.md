# Report: GKR Input Batching (004-gkr-input-batching)

## Idea

Cherry-picked commit 514d9d0. Batched GKR input evaluation across AIRs using a BlockCtx/GkrInputCtx pattern. Each CUDA block handles one AIR. Grouped launches by the GLOBAL flag, reducing ~791 individual kernel launches to just 2.

## Implementation

Introduced `BlockCtx` and `GkrInputCtx` structures to describe per-AIR work items. The kernel dispatcher groups AIRs by their GLOBAL flag and issues one batched launch per group. Each CUDA block independently processes its assigned AIR using the context array.

Files changed:
- `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`
- `crates/cuda-backend/src/cuda/logup_zerocheck.rs`
- `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

## Results

| Phase          | APC 0 | APC 100 | APC 300 |
|----------------|-------|---------|---------|
| **Before** LogUp GKR | -     | -       | ~800ms  |
| **After** LogUp GKR  | -     | -       | ~600ms  |

APC300 LogUp GKR improved by ~25%.

## Future Work

- The remaining 2 launches (one per GLOBAL flag value) could potentially be unified with a flag field in the context struct.
- The BlockCtx pattern established here became a reusable template for subsequent batching optimizations (006, etc.).
