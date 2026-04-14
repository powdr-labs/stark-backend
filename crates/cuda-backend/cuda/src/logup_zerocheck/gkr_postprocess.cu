// GPU-side GKR round postprocessing kernel.
// Replaces per-round D2H copy + CPU reconstruct_s_evals + CPU transcript observe/sample.

#include "fpext.h"
#include "sponge.cuh"
#include <cstdint>

namespace fractional_sumcheck_gkr {

// Lagrange interpolation of degree-2 polynomial through (0,y0), (1,y1), (2,y2) at point x.
__device__ FpExt lagrange_interp_012(const FpExt *vals, FpExt x) {
    // BabyBear p = 2013265921, inv(2) = (p+1)/2 = 1006632961
    const FpExt INV_2 = FpExt(Fp(1006632961u));
    FpExt p = (vals[2] - vals[1] - vals[1] + vals[0]) * INV_2;
    FpExt q = vals[1] - vals[0] - p;
    return (p * x + q) * x + vals[0];
}

// eq(a, b) for single variable: a*b + (1-a)*(1-b)
__device__ FpExt eval_eq_single(FpExt a, FpExt b) {
    FpExt one = FpExt(Fp(1u));
    return a * b + (one - a) * (one - b);
}

// Single-thread kernel that performs post-round processing on GPU:
// 1. Reconstruct s_evals from GPU-computed s'(1), s'(2)
// 2. Transcript observe (3 extension field elements = 12 BabyBear)
// 3. Transcript sample (1 extension field element = 4 BabyBear)
// 4. Update accumulators (eq_r_acc, prev_s_eval)
__global__ void gkr_round_postprocess_kernel(
    const FpExt *d_sum_evals,      // 2 EF values from compute_round kernel
    DeviceSpongeState *sponge,     // transcript sponge state (modified in-place)
    FpExt *d_prev_s_eval,          // scalar accumulator (in/out)
    FpExt *d_eq_r_acc,             // scalar accumulator (in/out)
    FpExt xi_j,                    // constant per call
    FpExt *d_challenge_out,        // output: sampled challenge r
    FpExt *d_s_evals_out           // output: 3 EF for round_polys_eval
) {
    FpExt eq_r_acc = *d_eq_r_acc;
    FpExt prev_s_eval = *d_prev_s_eval;
    FpExt one = FpExt(Fp(1u));

    // 1. Reconstruct s_evals (matches fractional.rs reconstruct_s_evals)
    FpExt sp[3];
    sp[1] = d_sum_evals[0] * eq_r_acc;
    sp[2] = d_sum_evals[1] * eq_r_acc;
    FpExt eq_xi_0 = one - xi_j;  // eq(xi_j, 0) = 1 - xi_j
    FpExt eq_xi_1 = xi_j;        // eq(xi_j, 1) = xi_j
    sp[0] = (prev_s_eval - eq_xi_1 * sp[1]) * inv(eq_xi_0);

    // Compute s_evals at {1, 2, 3}
    FpExt s_evals[3];
    for (int i = 0; i < 3; i++) {
        FpExt x = FpExt(Fp((uint32_t)(i + 1)));
        FpExt sp_eval;
        if (i < 2) {
            sp_eval = sp[i + 1];
        } else {
            sp_eval = lagrange_interp_012(sp, x);
        }
        s_evals[i] = eval_eq_single(xi_j, x) * sp_eval;
    }
    d_s_evals_out[0] = s_evals[0];
    d_s_evals_out[1] = s_evals[1];
    d_s_evals_out[2] = s_evals[2];

    // 2. Transcript observe: 3 EF = 12 BabyBear elements
    for (int i = 0; i < 3; i++) {
        for (int c = 0; c < 4; c++) {
            sponge_observe(*sponge, s_evals[i].elems[c]);
        }
    }

    // 3. Transcript sample: 1 EF = 4 BabyBear elements
    FpExt r;
    for (int c = 0; c < 4; c++) {
        r.elems[c] = sponge_sample(*sponge);
    }
    *d_challenge_out = r;

    // 4. Update accumulators
    FpExt eq_r_val = eval_eq_single(xi_j, r);
    *d_eq_r_acc = eq_r_acc * eq_r_val;
    *d_prev_s_eval = eq_r_val * lagrange_interp_012(sp, r);
}

} // namespace fractional_sumcheck_gkr

// Launcher function callable from Rust
extern "C" int _gkr_round_postprocess(
    const FpExt *d_sum_evals,
    DeviceSpongeState *sponge,
    FpExt *d_prev_s_eval,
    FpExt *d_eq_r_acc,
    FpExt xi_j,
    FpExt *d_challenge_out,
    FpExt *d_s_evals_out
) {
    fractional_sumcheck_gkr::gkr_round_postprocess_kernel<<<1, 1>>>(
        d_sum_evals,
        sponge,
        d_prev_s_eval,
        d_eq_r_acc,
        xi_j,
        d_challenge_out,
        d_s_evals_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return 0;
}
