# Report: GPU-side Round 0 Polynomial Extraction

## Description

The Round 0 multi-stream loop processes ~623 AIRs at APC 300, with each AIR performing two kernel launches (zerocheck and logup evaluation) followed by two `cudaStreamSynchronize` calls and CPU post-processing (transpose, inverse DFT, Lagrange interpolation, coefficient adjustment). These per-AIR D2H synchronizations drain the GPU pipeline ~1246 times per benchmark, blocking each thread's GPU stream while the CPU does ~4us of post-processing between kernel launches.

The optimization replaces the per-AIR D2H sync + CPU post-processing with a small GPU kernel that applies a pre-computed transformation matrix directly on device. The transformation matrix captures the entire pipeline (transpose + iDFT + unshift + Lagrange interpolation + coefficient adjustment) as a single matrix-vector multiply. Polynomial coefficients are written directly into a shared device-side batch array, and a single D2H copy at the end replaces the ~1246 per-AIR pipeline drains.

## Implementation

### CUDA kernel (`crates/cuda-backend/cuda/src/logup_zerocheck/round0_extract.cu`)
- New file with two kernels: `round0_extract_zerocheck_kernel` and `round0_extract_logup_kernel`
- Each kernel is launched as `<<<1, min(output_size, 64)>>>` (one warp or two warps)
- Each thread computes one output coefficient via matrix-vector multiply against the pre-computed transform
- The logup kernel additionally unpacks `FracExt` (numerator/denominator) and applies optional normalization
- Two `extern "C"` launcher functions: `_round0_extract_zerocheck_poly` and `_round0_extract_logup_polys`

### Rust FFI bindings (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)
- Added `extern "C"` declarations for the two new launcher functions

### Pre-computed transformation matrices (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)
- `Round0ExtractTables` struct holds per-unique-degree device buffers for zerocheck and logup transforms
- `compute_round0_extract_tables(d, l_skip)` computes transforms by running `UnivariatePoly::from_geometric_cosets_evals_idft` on unit vectors, capturing the exact CPU pipeline as a matrix
- Transforms are computed once during setup (2-3 unique degree values, ~200KB total GPU memory)

### Integration changes (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)
- Added batch offset fields (`zc_batch_offset`, `numer_batch_offset`, `denom_batch_offset`) to `Round0AirWorkItem`
- Pre-allocate device batch array (`3 * num_present_airs * max_poly_len` elements, ~1.9MB)
- Modified `process_air_round0` to launch GPU extraction kernels instead of D2H + CPU post-processing
- Each worker thread calls `current_stream_sync()` once at end (vs ~156 syncs before)
- Post-loop: single D2H copy of batch array, reconstruct `batch_sp_poly` using pre-computed per-AIR polynomial lengths

### Deviations from plan
- Instead of implementing the Bowers iDFT butterfly pattern on the GPU (error-prone), used a pre-computed combined transformation matrix that captures the entire pipeline. This is simpler, more robust, and avoids matching the exact DFT implementation.
- The `Round0AirResult` struct was removed entirely since `process_air_round0` now returns `Result<(), Error>`.
- Removed unused `MemCopyD2HStreamSync` import.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1383ms | 1336ms | -1119ms, 1.84x lower | -47ms, 1.04x lower |
| Round 0 (APC 300) | 662ms | 245ms | 202ms | -460ms, 3.28x lower | -43ms, 1.21x lower |
| LogUp GKR (APC 300) | 790ms | 535ms | 530ms | -260ms, 1.49x lower | -5ms (noise) |
| MLE Rounds (APC 300) | 180ms | 170ms | 170ms | -10ms, 1.06x lower | 0ms |
| Trace Commit (APC 300) | 406ms | 255ms | 254ms | -152ms, 1.60x lower | -1ms (noise) |
| Stacked Reduction (APC 300) | 311ms | 75ms | 75ms | -236ms, 4.15x lower | 0ms |
| WHIR (APC 300) | 100ms | 101ms | 99ms | -1ms (noise) | -2ms (noise) |
| STARK excl trace (APC 0) | 2153ms | 2162ms | 2147ms | -6ms (noise) | -15ms (noise) |
| Round 0 (APC 0) | 178ms | 177ms | 176ms | -2ms (noise) | -1ms (noise) |

Second APC 300 run confirmed: Round 0 = 202ms, STARK excl trace = 1350ms (consistent).

## Assessment

The optimization achieved its goal. Round 0 at APC 300 improved by 43ms (21% reduction), exceeding the 25ms rollback threshold. STARK excl trace improved by 47ms. No regression in any other metric at any APC configuration.

The improvement came from eliminating ~1246 per-AIR `cudaStreamSynchronize` calls and the CPU post-processing that blocked GPU kernel pipelining between evaluation and the next AIR's setup. The pre-computed transformation matrix approach is simpler than the plan's proposed Bowers iDFT implementation while achieving the same result.

At APC 0, the improvement is negligible (~1ms) because the single-threaded path processes only ~20 AIRs per segment and each kernel already saturates the GPU.

Cumulative STARK excl trace improvement vs baseline is now 1.84x at APC 300.

## Future Work

- The remaining per-AIR overhead in Round 0 includes H2D copies for `d_main_parts`, `d_numer_weights`, `d_denom_weights`, `d_rules` and the interaction DAG construction. These could potentially be batched or pre-computed.
- The transformation matrix approach generalizes to any linear post-processing of GPU kernel outputs. Similar patterns could be applied to other evaluation phases.
- The extraction kernel currently uses a simple single-warp launch. For larger constraint degrees (d > 4), a multi-block kernel with shared memory for the transform matrix could improve performance.
- The `evaluate_round0_interactions_gpu` function still constructs the interaction DAG per-AIR. Pre-computing this at keygen time (attempted in a previous task but with 0ms improvement due to small DAG sizes) could be revisited if DAG sizes grow.
