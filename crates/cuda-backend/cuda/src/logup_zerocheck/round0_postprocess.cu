/// GPU round-0 postprocess: coset evaluations → polynomial coefficients.
///
/// Replaces the per-trace CPU pipeline of:
///   transpose → IDFT → unshift → Lagrange interpolation → coefficient assembly
///
/// One block per trace, skip_domain threads per block.
/// Each thread handles one ntt_idx row across all cosets.

#include "device_ntt.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

using namespace device_ntt;

namespace round0_postprocess {

constexpr uint32_t MAX_COSETS = 4;

// Per-trace context for zerocheck GPU postprocess
struct ZcPostprocessCtx {
    uint32_t eval_offset;   // offset into d_evals for this trace (in FpExt units)
    uint32_t coeff_offset;  // offset into d_coeffs for this trace (in FpExt units)
};

// Per-trace context for logup GPU postprocess
struct LogupPostprocessCtx {
    uint32_t eval_offset;   // offset into d_evals (in FpExt units, interleaved numer/denom)
    uint32_t numer_offset;  // offset into d_numer_coeffs
    uint32_t denom_offset;  // offset into d_denom_coeffs
    Fp norm_factor;         // F::ONE for non-lifted, inverse(1 << |n|) for lifted
};

// ============================================================================
// Shared helper: column-wise IDFT + unshift + Lagrange interpolation
// ============================================================================

// Operates on a skip_domain × num_cosets matrix loaded into per-thread registers.
// Each thread holds one row (ntt_idx) of values across all cosets.
//
// After this function, vals[c] contains the polynomial coefficient for
// coset index c at this thread's ntt_idx row.
//
// The IDFT is across the skip_domain dimension (columns). Since threads
// represent different ntt_idx values, the IDFT uses warp shuffles.
//
// Parameters:
//   vals[MAX_COSETS]: in = evals in row-major, out = coefficients
//   lagrange_basis: [num_cosets * num_cosets] precomputed basis matrix
//   shift_inv: [num_cosets] precomputed unshift factors
//   ntt_idx: this thread's row index
//   l_skip: log2(skip_domain)
//   num_cosets: actual number of cosets (1-4)
template <bool needs_shmem>
__device__ void evals_to_coeffs(
    FpExt vals[MAX_COSETS],
    const FpExt *__restrict__ lagrange_basis,
    const FpExt *__restrict__ shift_inv,
    Fp *__restrict__ shmem_buf,
    uint32_t ntt_idx,
    uint32_t l_skip,
    uint32_t num_cosets
) {
    uint32_t skip_domain = 1u << l_skip;

    // Step 1: Column-wise IDFT.
    // Each column c is a length-skip_domain polynomial evaluation.
    // Threads hold row ntt_idx; the IDFT requires cross-thread communication.
    // We use the existing device NTT for each column sequentially.
    for (uint32_t c = 0; c < num_cosets; c++) {
        // The IDFT operates on Fp, but our values are FpExt.
        // Do the IDFT on each of the 4 BabyBear limbs independently.
        for (uint32_t limb = 0; limb < 4; limb++) {
            Fp limb_val = vals[c].elems[limb];
            ntt_natural_to_bitrev</*intt=*/true, needs_shmem>(
                limb_val, shmem_buf, ntt_idx, l_skip
            );
            vals[c].elems[limb] = limb_val;
        }
        // After iNTT, vals[c] is in bit-reversed order.
        // Unreverse: the output of from_geometric_cosets_evals_idft uses
        // Radix2BowersSerial which produces natural-order coefficients.
        // The device NTT produces bit-reversed order, so we need to shuffle.
        uint32_t rev_idx = rev_len(ntt_idx, l_skip);
        // Exchange vals[c] between thread ntt_idx and thread rev_idx
        for (uint32_t limb = 0; limb < 4; limb++) {
            Fp other = Fp::fromRaw(__shfl_sync(0xffffffff, vals[c].elems[limb].asRaw(), rev_idx));
            vals[c].elems[limb] = other;
        }
    }

    // Step 2: Unshift coefficients.
    // shift_inv[c] = (init * shift^c)^{-1}
    // Per row t (ntt_idx), accumulate: val[c] *= shift_inv[c]^t
    // Since shift_inv is precomputed per-coset, we compute the power inline.
    for (uint32_t c = 0; c < num_cosets; c++) {
        FpExt factor = FpExt(Fp::one());
        FpExt base = shift_inv[c];
        // Compute base^ntt_idx via repeated squaring
        uint32_t exp = ntt_idx;
        FpExt pow = FpExt(Fp::one());
        while (exp > 0) {
            if (exp & 1) pow = pow * base;
            base = base * base;
            exp >>= 1;
        }
        vals[c] = vals[c] * pow;
    }

    // Step 3: Lagrange interpolation across cosets.
    // For this row, compute: out[k] = sum_c(vals[c] * lagrange_basis[c * num_cosets + k])
    FpExt result[MAX_COSETS];
    for (uint32_t k = 0; k < num_cosets; k++) {
        result[k] = FpExt(Fp::zero());
        for (uint32_t c = 0; c < num_cosets; c++) {
            result[k] = result[k] + vals[c] * lagrange_basis[c * num_cosets + k];
        }
    }

    // Copy result back to vals
    for (uint32_t k = 0; k < num_cosets; k++) {
        vals[k] = result[k];
    }
}

// ============================================================================
// Zerocheck postprocess kernel
// ============================================================================

// One block per trace. skip_domain threads per block.
// Converts coset evals → final sp_0 coefficients.
template <bool needs_shmem>
__global__ void zc_evals_to_coeffs_kernel(
    const FpExt *__restrict__ d_evals,
    FpExt *__restrict__ d_coeffs,
    const ZcPostprocessCtx *__restrict__ ctxs,
    const FpExt *__restrict__ lagrange_basis,
    const FpExt *__restrict__ shift_inv,
    uint32_t skip_domain,
    uint32_t num_cosets
) {
    extern __shared__ char smem[];
    Fp *shmem_buf = reinterpret_cast<Fp *>(smem);

    uint32_t trace_idx = blockIdx.x;
    ZcPostprocessCtx ctx = ctxs[trace_idx];
    uint32_t ntt_idx = threadIdx.x;
    uint32_t l_skip = __ffs(skip_domain) - 1;

    if (ntt_idx >= skip_domain) return;

    // Load evals: transpose from [coset * skip_domain + ntt_idx] to per-thread vals[coset]
    FpExt vals[MAX_COSETS];
    for (uint32_t c = 0; c < num_cosets; c++) {
        vals[c] = d_evals[ctx.eval_offset + c * skip_domain + ntt_idx];
    }
    for (uint32_t c = num_cosets; c < MAX_COSETS; c++) {
        vals[c] = FpExt(Fp::zero());
    }

    // Convert to coefficients
    evals_to_coeffs<needs_shmem>(vals, lagrange_basis, shift_inv, shmem_buf, ntt_idx, l_skip, num_cosets);

    // vals[k] now holds coefficients: the polynomial coefficient for degree
    // (k * skip_domain + ntt_idx). The output layout is column-major:
    // coeffs[k * skip_domain + ntt_idx] = vals[k]

    // Apply vanishing polynomial: sp_0[i] = -q[i] + (i >= skip_domain ? q[i - skip_domain] : 0)
    // sp_0_deg = (num_cosets + 1) * skip_domain - 1
    // Output length: (num_cosets + 1) * skip_domain
    uint32_t sp_0_stride = (num_cosets + 1) * skip_domain;
    uint32_t coeff_base = ctx.coeff_offset;

    // For each output coefficient degree i = k * skip_domain + ntt_idx:
    // sp_0[i] = -q[i] + (i >= skip_domain ? q[i - skip_domain] : 0)
    // Since q has num_cosets * skip_domain coefficients, q[j] = 0 for j >= num_cosets * skip_domain

    // k=0: sp_0[ntt_idx] = -q[ntt_idx]
    d_coeffs[coeff_base + ntt_idx] = FpExt(Fp::zero()) - vals[0];

    // k=1..num_cosets-1: sp_0[k*skip+ntt] = -q[k*skip+ntt] + q[(k-1)*skip+ntt]
    for (uint32_t k = 1; k < num_cosets; k++) {
        d_coeffs[coeff_base + k * skip_domain + ntt_idx] =
            FpExt(Fp::zero()) - vals[k] + vals[k - 1];
    }

    // k=num_cosets: sp_0[num_cosets*skip+ntt] = q[(num_cosets-1)*skip+ntt]
    // (q[num_cosets*skip+ntt] = 0 since q has only num_cosets*skip_domain coeffs)
    d_coeffs[coeff_base + num_cosets * skip_domain + ntt_idx] = vals[num_cosets - 1];
}

// ============================================================================
// Logup postprocess kernel
// ============================================================================

// One block per trace. skip_domain threads per block.
// Converts interleaved FracExt coset evals → separate numer/denom coefficients.
template <bool needs_shmem>
__global__ void logup_evals_to_coeffs_kernel(
    const FpExt *__restrict__ d_evals,       // interleaved [p0, q0, p1, q1, ...]
    FpExt *__restrict__ d_numer_coeffs,
    FpExt *__restrict__ d_denom_coeffs,
    const LogupPostprocessCtx *__restrict__ ctxs,
    const FpExt *__restrict__ lagrange_basis,
    const FpExt *__restrict__ shift_inv,
    uint32_t skip_domain,
    uint32_t num_cosets
) {
    extern __shared__ char smem[];
    Fp *shmem_buf = reinterpret_cast<Fp *>(smem);

    uint32_t trace_idx = blockIdx.x;
    LogupPostprocessCtx ctx = ctxs[trace_idx];
    uint32_t ntt_idx = threadIdx.x;
    uint32_t l_skip = __ffs(skip_domain) - 1;

    if (ntt_idx >= skip_domain) return;

    // Load numer evals from interleaved layout
    // The reduced output is interleaved: [p0, q0, p1, q1, ...]
    // Total per trace: 2 * num_cosets * skip_domain FpExt values
    FpExt numer_vals[MAX_COSETS];
    FpExt denom_vals[MAX_COSETS];
    for (uint32_t c = 0; c < num_cosets; c++) {
        uint32_t base = ctx.eval_offset + 2 * (c * skip_domain + ntt_idx);
        numer_vals[c] = d_evals[base];
        denom_vals[c] = d_evals[base + 1];
    }
    for (uint32_t c = num_cosets; c < MAX_COSETS; c++) {
        numer_vals[c] = FpExt(Fp::zero());
        denom_vals[c] = FpExt(Fp::zero());
    }

    // Apply numerator normalization for lifted traces
    FpExt norm = FpExt(ctx.norm_factor);
    for (uint32_t c = 0; c < num_cosets; c++) {
        numer_vals[c] = numer_vals[c] * norm;
    }

    // Convert numer evals to coefficients
    evals_to_coeffs<needs_shmem>(numer_vals, lagrange_basis, shift_inv, shmem_buf, ntt_idx, l_skip, num_cosets);

    // Sync before reusing shmem for denom
    __syncthreads();

    // Convert denom evals to coefficients
    evals_to_coeffs<needs_shmem>(denom_vals, lagrange_basis, shift_inv, shmem_buf, ntt_idx, l_skip, num_cosets);

    // Write output: column-major layout [k * skip_domain + ntt_idx]
    uint32_t coeffs_per_trace = num_cosets * skip_domain;
    for (uint32_t k = 0; k < num_cosets; k++) {
        d_numer_coeffs[ctx.numer_offset + k * skip_domain + ntt_idx] = numer_vals[k];
        d_denom_coeffs[ctx.denom_offset + k * skip_domain + ntt_idx] = denom_vals[k];
    }
}

// ============================================================================
// Launchers
// ============================================================================

extern "C" int _round0_zc_postprocess(
    const FpExt *d_evals,
    FpExt *d_coeffs,
    const ZcPostprocessCtx *ctxs,
    const FpExt *lagrange_basis,
    const FpExt *shift_inv,
    uint32_t skip_domain,
    uint32_t num_cosets,
    uint32_t num_traces
) {
    if (num_traces == 0) return 0;

    bool needs_shmem = skip_domain > WARP_SIZE;
    dim3 grid(num_traces);
    dim3 block(skip_domain);
    size_t shmem = needs_shmem ? sizeof(Fp) * skip_domain : 0;

    if (needs_shmem) {
        zc_evals_to_coeffs_kernel<true><<<grid, block, shmem>>>(
            d_evals, d_coeffs, ctxs, lagrange_basis, shift_inv, skip_domain, num_cosets
        );
    } else {
        zc_evals_to_coeffs_kernel<false><<<grid, block, shmem>>>(
            d_evals, d_coeffs, ctxs, lagrange_basis, shift_inv, skip_domain, num_cosets
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _round0_logup_postprocess(
    const FpExt *d_evals,
    FpExt *d_numer_coeffs,
    FpExt *d_denom_coeffs,
    const LogupPostprocessCtx *ctxs,
    const FpExt *lagrange_basis,
    const FpExt *shift_inv,
    uint32_t skip_domain,
    uint32_t num_cosets,
    uint32_t num_traces
) {
    if (num_traces == 0) return 0;

    bool needs_shmem = skip_domain > WARP_SIZE;
    dim3 grid(num_traces);
    dim3 block(skip_domain);
    size_t shmem = needs_shmem ? sizeof(Fp) * skip_domain : 0;

    if (needs_shmem) {
        logup_evals_to_coeffs_kernel<true><<<grid, block, shmem>>>(
            d_evals, d_numer_coeffs, d_denom_coeffs, ctxs,
            lagrange_basis, shift_inv, skip_domain, num_cosets
        );
    } else {
        logup_evals_to_coeffs_kernel<false><<<grid, block, shmem>>>(
            d_evals, d_numer_coeffs, d_denom_coeffs, ctxs,
            lagrange_basis, shift_inv, skip_domain, num_cosets
        );
    }
    return CHECK_KERNEL();
}

} // namespace round0_postprocess
