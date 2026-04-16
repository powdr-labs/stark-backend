# Plan: GPU Stacked Reduction Polynomial Extraction

## Goal

Move the Stacked Reduction Round 0 polynomial reconstruction (`reconstruct_s0_from_g` + `ntt_multiply_and_add`) from CPU to GPU using a multi-kernel pipeline with the existing `batch_ntt_small` primitive. This eliminates per-bucket D2H + stream sync overhead (~3.6ms) and moves the NTT compute to GPU (~6-10ms CPU NTT → <1ms GPU NTT), for an estimated net ~10ms improvement on the 74ms Stacked Reduction span.

## Current Code Path

### Entry: `batch_sumcheck_uni_round0_poly` (crates/cuda-backend/src/stacked_reduction.rs:481-556)

Per-trace GPU kernel loop accumulates G0, G1, G2 evaluations into bucket buffers:
- `d_g_pos`: 3 × skip_domain EF elements (for n ≥ 0 traces)
- `d_g_neg[k]`: 3 × skip_domain EF elements per bucket (for |n| = k+1, k in 0..l_skip)

Then calls `reconstruct_s0_from_g` (lines 566-641) which for each active bucket:
1. `d_g_bucket.to_host()` — D2H + stream sync (~200µs per bucket, ~9 buckets × 2 segments)
2. Check all-zero, skip if so
3. Compute E polynomial coefficients on CPU
4. For n<0 buckets: multiply E by indicator polynomial via `poly_multiply_ntt`
5. Call `ntt_multiply_and_add` — CPU iDFT/DFT/pointwise-multiply/iDFT pipeline

### Key dimensions (l_skip=8)
- skip_domain = 256, large_uni_domain = 512
- s_0_deg = 2 × (256 − 1) = 510 → output has 511 coefficients
- Each G bucket: 3 × 256 = 768 EF elements
- Each E polynomial: ≤ 256 coefficients (padded to 512 for DFT)

### Why it's slow
- 9 D2H + stream sync per segment × 2 segments: ~18 × 200µs = **~3.6ms** from syncs
- CPU NTT work per segment: ~60-80 NTTs of 256-512 EF elements = **~6-8ms**
- Total addressable overhead: **~10-16ms**

## Changes

### Overview: Multi-kernel GPU pipeline

Replace the CPU `ntt_multiply_and_add` with a GPU pipeline using existing `batch_ntt_small` and two small new CUDA kernels (AoS↔SoA transpose, pointwise multiply+sum). No fused kernel — each step is a separate kernel launch for correctness and simplicity.

Pipeline per bucket:
1. G evaluations already on GPU (in d_g_pos / d_g_neg)
2. GPU: AoS → SoA transpose (FpExt → 4 × Fp layout)
3. GPU: `batch_ntt_small(is_intt=true)` on Fp data — iDFT of G evals → G coefficients
4. GPU: Zero-pad G from skip_domain to large_uni_domain
5. E evaluations pre-computed and pre-uploaded (see Step 6 in orchestration)
6. GPU: `batch_ntt_small(is_intt=false)` on Fp data — DFT of G coefficients → G evals
7. GPU: SoA→AoS of G via `transpose_fp_to_fpext_vec`
8. GPU: Pointwise EF multiply+sum kernel: s[j] = Σᵢ E_i[j] × G_i[j]
9. GPU: AoS→SoA of s via `split_ext_to_base_col_major_matrix`
10. GPU: `batch_ntt_small(is_intt=true)` on s — iDFT → product coefficients
11. GPU: SoA→AoS of result via `transpose_fp_to_fpext_vec`
12. GPU: Accumulate into d_s0_accum (element-wise add, no atomics — same stream)

Final: single D2H of d_s0_accum (511 EF elements = ~8KB).

### Step 1: AoS ↔ SoA conversion (reuse existing kernels)

Reuse existing codebase primitives for AoS↔SoA conversion:
- **AoS→SoA (EF → 4×Fp)**: `split_ext_to_base_col_major_matrix` in `crates/cuda-backend/src/cuda/matrix.rs:145`
- **SoA→AoS (4×Fp → EF)**: `transpose_fp_to_fpext_vec` in `crates/cuda-backend/src/cuda/poly.rs:56`

These are already used by the WHIR prover (`crates/cuda-backend/src/whir.rs:100-153`) for exactly this pattern: convert EF to Fp SoA, run `batch_ntt_small`, convert back. No new kernels needed for AoS↔SoA.

### Step 2: Pointwise EF multiply-sum kernel

**File**: `crates/cuda-backend/cuda/src/stacked_reduction.cu`

```cuda
// Pointwise multiply 3 pairs of EF vectors and sum: s[j] = Σᵢ E_i[j] × G_i[j]
// Input E and G are AoS FpExt format (already DFT'd to evaluation domain)
// Output s is AoS FpExt format
__global__ void ef_pointwise_mul3_sum(
    const FpExt *e_evals,   // [3 × domain_size]
    const FpExt *g_evals,   // [3 × domain_size]
    FpExt *s_out,           // [domain_size]
    uint32_t domain_size
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= domain_size) return;
    FpExt acc;
    for (int i = 0; i < 3; i++) {
        acc += e_evals[i * domain_size + j] * g_evals[i * domain_size + j];
    }
    s_out[j] = acc;
}
```

Grid: `(ceil(domain_size/256), 1)`, Block: `(256, 1)`. For domain_size=512: 2 blocks.

### Step 3: Element-wise EF accumulate kernel

```cuda
// s0_accum[i] += s_new[i] for i in 0..len
__global__ void ef_accumulate(
    FpExt *accum,
    const FpExt *addend,
    uint32_t len
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    accum[i] += addend[i];
}
```

No atomics needed — we process buckets sequentially on the same stream.

### Step 4: Rust FFI wrappers

**File**: `crates/cuda-backend/src/cuda/stacked_reduction.rs`

Add extern "C" declarations and safe Rust wrappers for the two new kernels:
- `ef_pointwise_mul3_sum`
- `ef_accumulate`

The AoS↔SoA conversions use existing `split_ext_to_base_col_major_matrix` and `transpose_fp_to_fpext_vec` wrappers. Zero-padding uses existing `batch_expand_pad`.

### Step 5: GPU pipeline function

**File**: `crates/cuda-backend/src/stacked_reduction.rs`

Add `gpu_ntt_multiply_and_add`:

```rust
fn gpu_ntt_multiply_and_add(
    d_g_evals: &DeviceBuffer<EF>,      // 3 * skip_domain EF elements on device
    e_coeffs: [&[EF]; 3],              // E polynomial coefficients (from CPU)
    skip_domain: usize,                 // 1 << l_skip
    large_uni_domain: usize,            // 2 * skip_domain
    l_skip: usize,
    d_s0_accum: &mut DeviceBuffer<EF>,  // accumulator, large_uni_domain elements
    // Pre-allocated work buffers:
    d_g_soa: &mut DeviceBuffer<F>,      // 4 * 3 * large_uni_domain Fp elements
    d_e_soa: &mut DeviceBuffer<F>,      // 4 * 3 * large_uni_domain Fp elements
    d_g_aos_padded: &mut DeviceBuffer<EF>, // 3 * large_uni_domain EF elements
    d_s_work: &mut DeviceBuffer<EF>,    // large_uni_domain EF elements
) -> Result<(), StackedReductionError>
```

Implementation (follows the WHIR prover pattern from `whir.rs:100-153`):

1. **AoS→SoA of G**: `split_ext_to_base_col_major_matrix` on d_g_evals (3 × skip_domain EF → SoA Fp buffer with 4 × 3 × skip_domain elements)

2. **iDFT of G (SoA)**: `batch_ntt_small(d_g_soa, l_skip, cnt_blocks=12, is_intt=true)` — 12 = 4 components × 3 vectors, each 2^l_skip Fp elements. Output is natural-order (confirmed by existing usage in sumcheck.rs:200-234)

3. **SoA→AoS of G coefficients**: `transpose_fp_to_fpext_vec` on d_g_soa → d_g_aos (3 × skip_domain EF)

4. **Zero-pad G**: Use existing `batch_expand_pad` (matrix.rs:234) to pad each skip_domain block to large_uni_domain with zeros

5. **AoS→SoA of padded G**: `split_ext_to_base_col_major_matrix` on d_g_padded (3 × large_uni_domain EF → SoA Fp)

6. **DFT of G (SoA)**: `batch_ntt_small(d_g_soa_large, l_skip+1, cnt_blocks=12, is_intt=false)`

7. **SoA→AoS of G evals**: `transpose_fp_to_fpext_vec` on d_g_soa_large → d_g_aos_evals

8. **Pointwise multiply**: Launch `ef_pointwise_mul3_sum(d_e_evals_bucket, d_g_aos_evals, d_s_work, large_uni_domain)` — E evaluations are pre-computed on CPU and pre-uploaded (AoS format)

9. **AoS→SoA of product**: `split_ext_to_base_col_major_matrix` on d_s_work (large_uni_domain EF → SoA Fp)

10. **iDFT of product (SoA)**: `batch_ntt_small(d_s_soa, l_skip+1, cnt_blocks=4, is_intt=true)` — 4 components of one vector

11. **SoA→AoS of result**: `transpose_fp_to_fpext_vec` on d_s_soa → d_s_work_aos

12. **Accumulate**: `ef_accumulate(d_s0_accum, d_s_work_aos, large_uni_domain)` — sequential on same stream, no atomics

### Step 6: Replace `reconstruct_s0_from_g`

**File**: `crates/cuda-backend/src/stacked_reduction.rs`

Replace `reconstruct_s0_from_g` (lines 566-641):

1. **Track active buckets**: During the per-trace kernel loop (already in `batch_sumcheck_uni_round0_poly`), maintain a bitmask of active buckets based on the `n` value computed at line 511.

2. **Pre-allocate work buffers** once (before the bucket loop):
   - d_g_soa, d_e_soa: 4 × 3 × large_uni_domain Fp elements each
   - d_g_aos_padded: 3 × large_uni_domain EF elements
   - d_s_work: large_uni_domain EF elements
   - d_s0_accum: large_uni_domain EF elements (zero-initialized)

3. **Pre-compute ALL E evaluations on CPU** (before GPU loop):
   For each active bucket, compute E polynomial coefficients, DFT them to the evaluation domain (large_uni_domain points) on CPU via `Radix2BowersSerial::dft`, and store in a flat `Vec<EF>`. Upload ALL E evaluation data in a single `to_device()` call. This removes per-bucket CPU→GPU sync points and allows the GPU pipeline to run without CPU interruption.

4. **For each active bucket** (n≥0, then each active n<0):
   - **GPU only**: Call `gpu_ntt_multiply_and_add(d_g_bucket, d_e_evals_for_bucket, ..., d_s0_accum, ...)`
   - No per-bucket CPU work or H2D transfers

5. **Single D2H**: `d_s0_accum.to_host()` → 511 EF elements
6. Truncate to s_0_deg+1 coefficients, return UnivariatePoly

### Step 7: Guard for l_skip > MAX_NTT_LEVEL

At the top of the new `reconstruct_s0_from_g`, check `l_skip + 1 > MAX_NTT_LEVEL (10)`. If so, fall back to the existing CPU implementation (which remains as a private helper method). For the current l_skip=8, the GPU path is always taken.

## Invariants

1. **Mathematical equivalence**: NTT of EF with base-field twiddles = 4 independent Fp NTTs. The `batch_ntt_small` kernel produces natural-order output (it applies bit-reversal internally). The pointwise multiply uses full EF multiplication (not component-wise), preserving correctness.

2. **Buffer lifecycle**: Work buffers (d_g_soa, d_e_soa, etc.) are allocated before the bucket loop and reused across buckets. The d_s0_accum persists across buckets for accumulation.

3. **Bucket ordering**: Buckets are processed sequentially on the same stream. No atomics needed — `ef_accumulate` runs after the previous bucket's pipeline completes (stream ordering).

4. **No protocol change**: The output polynomial s_0, its degree, and the transcript interaction are unchanged.

5. **batch_ntt_small contract**: It takes `cnt_blocks` contiguous blocks of `1 << l_skip` Fp elements, each independently NTT'd. The SoA layout ensures each Fp component of each G/E vector is a separate block.

## Measurement Plan

Run `run_pairing.sh` for APC {0, 100, 300}. Compare:
- **Primary**: STARK excl trace at APC 300 (expected: 1110ms → ~1100ms)
- **Component**: Stacked Reduction at APC 300 (expected: 74ms → ~64ms)
- **Regression check**: APC 0 (expected: no regression or slight improvement)

Verify correctness: all proofs verify. Profile with nsight to confirm D2H count reduction in the Stacked Reduction span.

## Rollback Criteria

Revert if any of:
1. Stacked Reduction at APC 300 improves by less than 5ms
2. APC 0 regresses by more than 10ms on STARK excl trace
3. Proof verification fails at any APC config
