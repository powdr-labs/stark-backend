/// GPU-side polynomial extraction for Round 0.
///
/// Instead of D2H copying evaluation results and doing iDFT + unshift +
/// Lagrange interpolation + coefficient adjustment on the CPU, we apply a
/// pre-computed transformation matrix on the GPU. This eliminates per-AIR
/// cudaStreamSynchronize calls and CPU post-processing.
///
/// The transformation matrix is computed on the host during setup and
/// captures the entire pipeline (transpose + iDFT + unshift + Lagrange
/// interpolation + coefficient adjustment for zerocheck, or transpose +
/// iDFT + unshift + Lagrange interpolation for logup).

#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include <algorithm>
#include <cstdint>

/// Zerocheck polynomial extraction.
/// Each thread computes one output coefficient via mat-vec multiply:
///   output[i] = sum_j transform[i * input_size + j] * evals[j]
__global__ void round0_extract_zerocheck_kernel(
    FpExt* __restrict__ d_out,
    const FpExt* __restrict__ d_evals,
    const FpExt* __restrict__ d_transform,
    uint32_t input_size,
    uint32_t output_size
) {
    uint32_t i = threadIdx.x;
    if (i >= output_size) return;

    FpExt sum;
    const FpExt* row = d_transform + static_cast<uint64_t>(i) * input_size;
    for (uint32_t j = 0; j < input_size; j++) {
        sum += row[j] * d_evals[j];
    }
    d_out[i] = sum;
}

/// Logup polynomial extraction.
/// Unpacks FracExt into numer/denom, applies normalization to numer,
/// then does mat-vec multiply for each component.
__global__ void round0_extract_logup_kernel(
    FpExt* __restrict__ d_out_numer,
    FpExt* __restrict__ d_out_denom,
    const FracExt* __restrict__ d_evals,
    const FpExt* __restrict__ d_transform,
    uint32_t input_size,
    uint32_t output_size,
    FpExt norm_factor
) {
    uint32_t i = threadIdx.x;
    if (i >= output_size) return;

    FpExt numer_sum;
    FpExt denom_sum;
    const FpExt* row = d_transform + static_cast<uint64_t>(i) * input_size;
    for (uint32_t j = 0; j < input_size; j++) {
        FpExt coeff = row[j];
        numer_sum += coeff * (d_evals[j].p * norm_factor);
        denom_sum += coeff * d_evals[j].q;
    }
    d_out_numer[i] = numer_sum;
    d_out_denom[i] = denom_sum;
}

extern "C" int _round0_extract_zerocheck_poly(
    FpExt* d_out,
    const FpExt* d_evals,
    const FpExt* d_transform,
    uint32_t input_size,
    uint32_t output_size
) {
    if (output_size == 0 || input_size == 0) return 0;
    uint32_t block_size = std::min(output_size, 64u);
    round0_extract_zerocheck_kernel<<<1, block_size>>>(
        d_out, d_evals, d_transform, input_size, output_size
    );
    return CHECK_KERNEL();
}

extern "C" int _round0_extract_logup_polys(
    FpExt* d_out_numer,
    FpExt* d_out_denom,
    const FracExt* d_evals,
    const FpExt* d_transform,
    uint32_t input_size,
    uint32_t output_size,
    FpExt norm_factor
) {
    if (output_size == 0 || input_size == 0) return 0;
    uint32_t block_size = std::min(output_size, 64u);
    round0_extract_logup_kernel<<<1, block_size>>>(
        d_out_numer, d_out_denom, d_evals,
        d_transform, input_size, output_size, norm_factor
    );
    return CHECK_KERNEL();
}
