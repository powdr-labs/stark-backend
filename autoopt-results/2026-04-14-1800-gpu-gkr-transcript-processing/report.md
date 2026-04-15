# Report: GPU-Side GKR Transcript Processing

## Description

Eliminate per-round CPU-GPU roundtrips in the GKR fractional sumcheck by moving the post-round processing (reconstruct s_evals, Poseidon2 transcript observe/sample, accumulator updates) to a GPU kernel. Currently, each FoldEval inner round launches a compute kernel, does a blocking D2H copy of 2 extension field elements, runs CPU-side arithmetic and transcript operations, then launches the next round's kernel. The plan estimated 14ms of inter-kernel gap per segment from these roundtrips (28ms total across 2 segments at APC 300). By implementing a <<<1,1>>> GPU kernel that fuses the post-round processing and adding device-pointer variants of the compute kernels (so the challenge stays on GPU between rounds), we aimed to eliminate all mid-loop D2H copies.

## Implementation

### Files changed:

1. **`crates/cuda-backend/cuda/include/sponge.cuh`** (new): Extracted `DeviceSpongeState`, `sponge_observe()`, `sponge_sample()` from `sponge.cu` to a shared header for reuse by the postprocess kernel.

2. **`crates/cuda-backend/cuda/src/sponge.cu`**: Updated to `#include "sponge.cuh"`, removing inlined definitions.

3. **`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_postprocess.cu`** (new): Single-thread GPU kernel `gkr_round_postprocess_kernel` that:
   - Reconstructs s_evals from GPU-computed s'(1), s'(2) (matching `reconstruct_s_evals` in Rust)
   - Observes 3 extension field elements (12 BabyBear) in the GPU sponge
   - Samples 1 extension field element (4 BabyBear) as the new challenge
   - Updates eq_r_acc and prev_s_eval accumulators in-place on device

4. **`crates/cuda-backend/cuda/src/logup_zerocheck/gkr.cu`**: Templated `compute_round_and_fold_kernel` and `compute_round_and_fold_inplace_kernel` with `<bool r_prev_from_ptr>` to optionally read `r_prev` from a device pointer. Added `_frac_compute_round_and_fold_dptr` and `_frac_compute_round_and_fold_inplace_dptr` launcher functions.

5. **`crates/cuda-backend/src/cuda/logup_zerocheck.rs`**: Added FFI declarations and safe Rust wrappers for `gkr_round_postprocess`, `frac_compute_round_and_fold_dptr`, `frac_compute_round_and_fold_inplace_dptr`.

6. **`crates/cuda-backend/src/sponge.rs`**: Added `as_duplex_sponge_gpu()` method to `GpuFiatShamirTranscript` trait (returns `Option<&mut DuplexSpongeGpu>`).

7. **`crates/cuda-backend/src/logup_zerocheck/fractional.rs`**: 
   - Specialized `fractional_sumcheck_gpu` and all helper functions from generic `TS: FiatShamirTranscript<SC>` to concrete `DuplexSpongeGpu`.
   - Replaced the FoldEval inner loop: after round 0 (CPU), syncs sponge H2D, uploads accumulators, runs inner rounds with GPU postprocess + device-pointer compute kernels, then batch D2H + CPU transcript replay.

8. **`crates/cuda-backend/src/logup_zerocheck/mod.rs`**: Updated caller to use `transcript.as_duplex_sponge_gpu()`.

### Deviations from plan:
- The plan suggested approach (a) for Change 7 (specialize entire call chain). Instead used `Option<&mut DuplexSpongeGpu>` via trait method to avoid breaking the BN254 `MultiField32ChallengerGpu` implementation.
- Used `lambda` as dummy value for `r_prev_val` in dptr launchers (instead of constructing a zero FpExt, which is not possible from host code due to `__device__`-only constructors).

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1294ms | 1305ms | -1150ms, 1.88x lower | +11ms, 1.01x higher |
| LogUp GKR (APC 300) | 790ms | 527ms | 534ms | -256ms, 1.48x lower | +7ms, 1.01x higher |
| Round 0 (APC 300) | 662ms | 175ms | 180ms | -482ms, 3.68x lower | +5ms, 1.03x higher |
| MLE Rounds (APC 300) | 180ms | 159ms | 158ms | -22ms, 1.14x lower | -1ms, unchanged |
| STARK excl trace (APC 000) | 2231ms | 2153ms | 2165ms | -66ms, 1.03x lower | +12ms, 1.01x higher |
| LogUp GKR (APC 000) | 1085ms | 1034ms | 1041ms | -44ms, 1.04x lower | +7ms, 1.01x higher |

Second APC 300 run: STARK excl trace = 1339ms, LogUp GKR = 575ms (higher noise).

**No improvement.** The optimization is within measurement noise at best, with a slight regression tendency (+7-11ms on STARK excl trace, +7ms on LogUp GKR at APC 300).

## Assessment

The optimization did not achieve its goal. The LogUp GKR improvement was 0ms (actually -7ms regression), well below the plan's 10ms rollback threshold. The likely causes:

1. **D2H cost was overestimated**: The `d_sum_evals.to_host()` of 32 bytes (2 EF) is very fast (~1-2μs). The blocking sync to wait for the compute kernel to finish happens regardless — whether we copy 32 bytes or launch a postprocess kernel, we still must wait for the compute kernel.

2. **Kernel launch overhead offsets savings**: Each <<<1,1>>> postprocess kernel launch adds ~5-10μs of CPU-side overhead (driver call + kernel argument setup). Over ~20 inner rounds × 2 segments = ~0.2-0.4ms, partially offsetting any savings.

3. **CPU work was fast**: The CPU-side `reconstruct_s_evals` (12 field ops) and Poseidon2 sponge operations (1-2 permutations, ~10μs each) are very efficient. Total CPU work per round was ~15-25μs, not the ~350μs estimated from nsight inter-kernel gaps.

4. **The inter-kernel gap is not CPU transit time**: The 14ms nsight gap likely includes kernel launch overhead, CUDA driver overhead, and memory allocation (for d_sum_evals.to_host), not just the D2H + CPU arithmetic. The GPU postprocess approach only eliminates the CPU arithmetic portion.

**Key learning**: The per-round overhead in the fractional sumcheck FoldEval path is dominated by CUDA API overhead (kernel launches, tiny memory operations) and driver overhead, not by CPU-side arithmetic or transcript operations. Eliminating CPU arithmetic alone is insufficient when the kernel launch pattern creates the same number of synchronization points.

## Future Work

- **Fuse postprocess into compute kernel**: Instead of a separate <<<1,1>>> kernel, fuse the transcript processing into the final reduction of the compute kernel itself. The last warp in the block reduction could perform the reconstruct + sponge + accumulator update, eliminating one kernel launch per round entirely.
- **Multi-round kernel**: Launch a single kernel that performs multiple consecutive sumcheck rounds internally (compute → postprocess → compute → postprocess) without returning to the host. This eliminates all inter-round kernel launch overhead.
- **Overlap approaches**: The FoldEval inner rounds are inherently sequential (each depends on the previous challenge), so concurrency-based approaches won't help. Focus on reducing fixed per-round overhead instead.
- **Profile after all optimizations**: The nsight profiling that motivated this task was done on an earlier codebase. Re-profile to identify the true current bottleneck in LogUp GKR — it may have shifted from CPU transit to memory bandwidth or kernel execution time.
