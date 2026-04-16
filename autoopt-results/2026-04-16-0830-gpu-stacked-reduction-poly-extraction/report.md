# Report: GPU Stacked Reduction Polynomial Extraction

## Description

Move the Stacked Reduction Round 0 polynomial reconstruction (`reconstruct_s0_from_g` + `ntt_multiply_and_add`) from CPU to GPU using a multi-kernel pipeline. The current CPU-side implementation performs per-bucket D2H transfers (each syncing the stream), followed by serial iDFT, DFT, pointwise multiplication, and inverse DFT on the host. The GPU pipeline eliminates per-bucket D2H syncs and replaces CPU NTT operations with batched GPU NTT calls via `batch_ntt_small`.

The expected improvement was ~10ms on the 75ms Stacked Reduction span (~3.6ms from eliminating D2H syncs + ~6-8ms from moving CPU NTT to GPU).

## Implementation

### Files Changed

- **`crates/cuda-backend/cuda/src/stacked_reduction.cu`**: Added two new CUDA kernels:
  - `ef_pointwise_mul3_sum_kernel`: Pointwise multiply 3 pairs of EF vectors and sum
  - `ef_accumulate_kernel`: Element-wise EF accumulation (no atomics, same-stream ordering)
  - Plus launcher functions for both

- **`crates/cuda-backend/src/cuda/stacked_reduction.rs`**: Added FFI declarations and safe Rust wrappers for the two new kernels (`ef_pointwise_mul3_sum`, `ef_accumulate`)

- **`crates/cuda-backend/src/error.rs`**: Added `ReconstructGpu(CudaError)` variant to `StackedReductionError`

- **`crates/cuda-backend/src/stacked_reduction.rs`**: 
  - Added imports for `batch_ntt_small`, `split_ext_to_base_col_major_matrix`, `batch_expand_pad`, `transpose_fp_to_fpext_vec`, and the new kernel wrappers
  - Added `reconstruct_s0_from_g_gpu` method: Pre-computes all E evaluations on CPU, uploads once, then runs GPU pipeline per bucket
  - Added `gpu_ntt_multiply_and_accumulate` helper: 10-step GPU pipeline per bucket (AoS→SoA → iDFT → zero-pad → DFT → SoA→AoS → pointwise mul → AoS→SoA → iDFT → SoA→AoS → accumulate)
  - Modified `reconstruct_s0_from_g` to dispatch to GPU path when `l_skip + 1 <= MAX_NTT_LEVEL` (always true for current l_skip=8)

### Key Decisions

- **E evaluations pre-computed on CPU**: E polynomials are small (256-512 coefficients), known ahead of time, and DFT'd on CPU in ~microseconds. Single H2D upload of all E data (~216KB).
- **No zero-check for inactive buckets**: The GPU work per zero bucket is ~microseconds, not worth a D2H to check.
- **CPU fallback guard**: If `l_skip + 1 > MAX_NTT_LEVEL (10)`, the existing CPU path is used. For current l_skip=8, the GPU path is always taken.

### Deviations from Plan

None — the implementation follows the plan's GPU pipeline design exactly.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 300** | | | | | |
| STARK excl trace | 2455ms | 1107ms | 1124ms | -1331ms, 2.18x lower | +17ms, 1.02x higher |
| Stacked Reduction | 311ms | 75ms | 74ms | -237ms, 4.20x lower | -1ms, 1.01x lower |
| Openings | 413ms | 201ms | 200ms | -213ms, 2.07x lower | -1ms, 1.01x lower |
| **APC 100** | | | | | |
| STARK excl trace | — | 1303ms | 1295ms | — | -8ms, 1.01x lower |
| Stacked Reduction | — | 71ms | 70ms | — | -1ms, 1.01x lower |
| **APC 0** | | | | | |
| STARK excl trace | 2153ms | 1794ms | 1802ms | -351ms, 1.19x lower | +8ms, 1.00x higher |
| Stacked Reduction | 113ms | 79ms | 77ms | -36ms, 1.47x lower | -2ms, 1.03x lower |

All changes are within measurement noise (~±20ms for STARK excl trace, ~±3ms for Stacked Reduction).

## Assessment

**The optimization did NOT achieve its goal.** Stacked Reduction improved by only 1-2ms at all APC configurations, well below the 5ms rollback threshold. STARK excl trace showed no meaningful change.

The root cause is that the CPU-side polynomial reconstruction was already fast:
- **D2H sync overhead was overestimated**: The CUDA memory pool's `to_host()` overhead is ~50µs per call (not ~200µs). For ~18 calls across 2 segments, total D2H overhead is ~0.9ms.
- **CPU NTT work was overestimated**: `Radix2BowersSerial` iDFT/DFT of 256-512 EF elements takes ~10-20µs per operation. The ~60-80 NTTs across all buckets complete in ~1-2ms total.
- **GPU kernel launch overhead nearly offsets savings**: The 10-kernel pipeline per bucket × 9 buckets × 2 segments = 180 kernel launches adds ~0.9-1.8ms of CUDA launch overhead.
- **E evaluation H2D upload adds overhead**: The ~216KB upload of pre-computed E evaluations adds ~0.1ms per segment.

Net effect: ~2ms saved from eliminated D2H syncs and CPU NTT, offset by ~1.5ms of GPU kernel launch overhead and E data upload. The theoretical maximum improvement was ~3-4ms (not 10ms as estimated), and the actual improvement (~1ms) is at the measurement noise floor.

## Future Work

- **The CPU polynomial reconstruction is not a bottleneck**: At 2-3ms of the 75ms Stacked Reduction total, it's only ~3-4% of the component and <0.3% of STARK excl trace. No further optimization of this path is worthwhile.
- **Stacked Reduction's dominant cost is the per-trace kernel loop** (`batch_sumcheck_uni_round0_poly` GPU kernels) which computes G0/G1/G2. Multi-streaming this was attempted previously and failed due to accumulation buffer issues.
- **The GPU NTT pipeline infrastructure** (AoS↔SoA + batch_ntt_small) works correctly and could be reused for other small polynomial operations, but the kernel launch overhead limits applicability for sub-1000-element operations.
- **Kernel fusion** (combining multiple steps of the pipeline into a single kernel) could reduce launch overhead but adds significant complexity for minimal gain at these scales.
