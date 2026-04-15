# Plan: GPU-Side GKR Transcript Processing

## Goal

Eliminate per-round CPU-GPU roundtrips in the GKR fractional sumcheck by moving the `observe_and_update` function (D2H copy + arithmetic + Poseidon2 transcript observe/sample) to a GPU kernel. This targets the 14ms per-segment inter-kernel gap measured by nsight in the fractional sumcheck round loop (28ms total across 2 segments at APC 300).

## Current Code Path

### Call chain

```
fractional_sumcheck_gpu() @ fractional.rs:491
└─ for round in 1..total_rounds (line 655)
    ├─ eq_buffer = SqrtEqLayers::from_xi() (line 663) — CPU
    ├─ lambda = transcript.sample_ext() (line 670) — CPU Poseidon2 sponge
    ├─ do_sumcheck_round_and_revert() (line 698) — GPU kernel + observe_and_update
    └─ match backend:
         ├─ FoldEval: for xi_j { do_fused_sumcheck_round() } — GPU + observe_and_update per inner round
         └─ PrecomputeM: precompute_m → multifold → observe_and_update (with CPU eq_table deps)
```

### Per-round observe_and_update (fractional.rs:318-346)

Each call does:
1. **D2H sync**: `d_sum_evals.to_host()` — copies 2 EF (32 bytes) from GPU, BLOCKS until preceding GPU kernel completes
2. **CPU arithmetic**: `reconstruct_s_evals()` — ~12 EF field operations (inverse, interpolation)
3. **CPU transcript**: `transcript.observe_ext()` × 3 — absorbs 12 BabyBear elements, triggers 1-2 Poseidon2 permutations (~10μs each)
4. **CPU transcript**: `transcript.sample_ext()` — squeezes 4 BabyBear elements, triggers 0-1 Poseidon2 permutations
5. **CPU accumulator update**: `eq_r_acc *= ...`, `prev_s_eval = ...`
6. **Return**: challenge `r` used by next round's GPU kernel (passed as scalar argument)

### Why it's slow

nsight profiling of the fractional sumcheck kernel timeline shows:
- Segment 0: 80.5ms wall time, 14.2ms of inter-kernel gaps across ~41 major kernel transitions (mean gap: 0.35ms)
- Segment 1: 80.3ms wall time, 14.0ms of inter-kernel gaps
- Each gap is a CPU-GPU roundtrip: GPU idles while CPU does D2H + reconstruct + transcript + next kernel launch

### Existing GPU infrastructure

- **GPU Poseidon2 permutation**: `poseidon2_mix(Fp *cells)` in `crates/cuda-common/include/poseidon2.cuh:184-202`
- **GPU sponge functions**: `sponge_observe()` and `sponge_sample()` as `__device__` functions in `crates/cuda-backend/cuda/src/sponge.cu:13-42` (must be extracted to shared header)
- **DeviceSpongeState struct**: defined in both Rust (`sponge.rs:32-39`) and CUDA (`sponge.cu`), `#[repr(C)]` ABI-compatible
- **Host-device sync**: `DuplexSpongeGpu::sync_h2d()` at `sponge.rs:188` converts Challenger → DeviceSpongeState and uploads
- **FpExt arithmetic**: Extension field arithmetic already exists in CUDA for GKR kernels (`FracExt` type with multiply, add). Check `fpext.h` or equivalent for `reciprocal()`.

## Changes

### Scope: FoldEval path only (initial prototype)

The PrecomputeM path (fractional.rs:817+) uses per-round challenges in CPU-side `eval_mle_table` to build eq tables. Moving this to GPU requires additional scope (GPU-side eq table construction). The initial prototype targets the FoldEval path only. When the round strategy selects PrecomputeM, fall back to the existing CPU observe_and_update path.

### Change 1: Extract sponge functions to shared header

**File**: Create `crates/cuda-backend/cuda/include/sponge.cuh` (new header)

Extract from `sponge.cu`:
- `DeviceSpongeState` struct definition
- `sponge_observe()` device function
- `sponge_sample()` device function

Update `sponge.cu` to `#include "sponge.cuh"` and remove the inlined definitions.

### Change 2: New CUDA kernel `gkr_round_postprocess`

**File**: Add to `crates/cuda-backend/cuda/src/sumcheck.cu` (or new file)

```cuda
#include "sponge.cuh"
#include "fpext.h"  // FpExt arithmetic (add, sub, mul, inv)

// Lagrange interpolation of degree-2 polynomial through (0,y0), (1,y1), (2,y2) at point x.
// Uses the formula: p = halve(y2 - 2*y1 + y0), q = y1 - y0 - p, result = (p*x + q)*x + y0
// halve(x) = x * INV_2 where INV_2 = 2^30 mod (2^31 - 2^27 + 1) [BabyBear inverse of 2]
__device__ FpExt lagrange_interp_012(const FpExt *vals, FpExt x) {
    // Compute inv(2) as a compile-time constant for BabyBear
    // BabyBear p = 2013265921, inv(2) = (p+1)/2 = 1006632961
    const Fp INV_2 = Fp(1006632961u);
    FpExt p = (vals[2] - vals[1] - vals[1] + vals[0]) * FpExt(INV_2);
    FpExt q = vals[1] - vals[0] - p;
    return (p * x + q) * x + vals[0];
}

__global__ void gkr_round_postprocess(
    const FpExt *d_sum_evals,      // 2 EF values from compute_round kernel
    DeviceSpongeState *sponge,     // transcript sponge state (modified in-place)
    FpExt *d_prev_s_eval,          // scalar accumulator
    FpExt *d_eq_r_acc,             // scalar accumulator
    FpExt xi_j,                    // constant per call (passed by value, 16 bytes)
    FpExt *d_challenge_out,        // output: sampled challenge r
    FpExt *d_s_evals_out           // output: 3 EF for round_polys_eval
) {
    // Single thread kernel (<<<1,1>>>): all work is sequential, tiny data
    FpExt eq_r_acc = *d_eq_r_acc;
    FpExt prev_s_eval = *d_prev_s_eval;
    
    // 1. Reconstruct s_evals (matches fractional.rs:1126-1163)
    FpExt sp[3];
    sp[1] = d_sum_evals[0] * eq_r_acc;
    sp[2] = d_sum_evals[1] * eq_r_acc;
    FpExt eq_xi_0 = FpExt(1u) - xi_j;  // eq(xi_j, 0) = 1 - xi_j
    sp[0] = (prev_s_eval - xi_j * sp[1]) * inv(eq_xi_0);
    
    // s(X) = eq(xi_j, X) * sp(X), where eq(xi_j, x) = xi_j*x + (1-xi_j)*(1-x)
    FpExt s_evals[3];
    for (int i = 0; i < 3; i++) {
        FpExt x((uint32_t)(i + 1));
        FpExt sp_x = (i < 2) ? sp[i + 1] : lagrange_interp_012(sp, x);
        FpExt eq_val = xi_j * x + eq_xi_0 * (FpExt(1u) - x);
        s_evals[i] = eq_val * sp_x;
    }
    d_s_evals_out[0] = s_evals[0];
    d_s_evals_out[1] = s_evals[1];
    d_s_evals_out[2] = s_evals[2];
    
    // 2. Transcript observe: 3 EF = 12 BabyBear elements
    for (int i = 0; i < 3; i++) {
        for (int c = 0; c < 4; c++) {
            sponge_observe(*sponge, s_evals[i].elems[c]);  // .elems, not .coeffs
        }
    }
    
    // 3. Transcript sample: 1 EF = 4 BabyBear elements
    FpExt r;
    for (int c = 0; c < 4; c++) {
        r.elems[c] = sponge_sample(*sponge);
    }
    *d_challenge_out = r;
    
    // 4. Update accumulators
    FpExt eq_r_val = xi_j * r + eq_xi_0 * (FpExt(1u) - r);
    *d_eq_r_acc = eq_r_acc * eq_r_val;
    *d_prev_s_eval = eq_r_val * lagrange_interp_012(sp, r);
}
```

The `FpExt` type uses the existing `fpext.h` API: constructor `FpExt(uint32_t)`, operators `+`, `-`, `*`, and free function `inv()` for multiplicative inverse. Field element access is via `.elems[0..4]`.

### Change 3: FFI binding for gkr_round_postprocess

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs` (or new cuda module file)

```rust
extern "C" { fn _gkr_round_postprocess(...) -> i32; }

pub unsafe fn gkr_round_postprocess(
    d_sum_evals: &DeviceBuffer<EF>,
    d_sponge: &DeviceBuffer<DeviceSpongeState>,
    d_prev_s_eval: &DeviceBuffer<EF>,
    d_eq_r_acc: &DeviceBuffer<EF>,
    xi_j: EF,
    d_challenge_out: *mut EF,  // pointer into pre-allocated challenge array
    d_s_evals_out: *mut EF,    // pointer into pre-allocated s_evals array
) -> Result<(), CudaError> { ... }
```

### Change 4: Modify compute_round kernels to read challenge from device pointer

**Files**: 
- `crates/cuda-backend/cuda/src/sumcheck.cu` — kernel signatures
- `crates/cuda-backend/src/cuda/logup_zerocheck.rs` — FFI wrappers

For `frac_compute_round_and_fold` and `frac_compute_round_and_fold_inplace`:
- Add a `const FpExt *d_r_prev` parameter (device pointer to previous round's challenge)
- At kernel start: `FpExt r_prev = *d_r_prev;` (L1 cache hit, negligible latency)
- Remove the scalar `r_prev` parameter

Keep the existing scalar-argument versions as fallback (for PrecomputeM path and first round where challenge comes from CPU).

### Change 5: Modify FoldEval round loop in fractional_sumcheck_gpu

**File**: `crates/cuda-backend/src/logup_zerocheck/fractional.rs`

**Per-outer-round structure** (the key design: sync_h2d after round 0, GPU for inner rounds, replay before next outer round):

```rust
for round in 1..total_rounds {
    // -- CPU phase: eq_buffer, lambda sampling (uses CPU transcript) --
    let eq_buffer = SqrtEqLayers::from_xi(&xi_prev[1..]);
    let lambda = transcript.sample_ext();
    
    // -- Round 0: stays on CPU (do_sumcheck_round_and_revert → observe_and_update) --
    let r0 = do_sumcheck_round_and_revert(..., transcript, ...)?;
    // CPU transcript is now up-to-date through round 0
    
    // -- Allocate per-outer-round GPU buffers --
    let inner_count = round - 1;  // number of FoldEval inner rounds
    let d_challenges_inner = DeviceBuffer::<EF>::with_capacity(inner_count);
    let d_s_evals_inner = DeviceBuffer::<EF>::with_capacity(inner_count * 3);
    
    match backend {
        GkrRoundStrategy::FoldEval => {
            // Upload sponge state AFTER round 0 observe_and_update
            d_sponge.sync_h2d(transcript)?;
            // Upload current accumulators
            upload_scalar(&d_prev_s_eval, *prev_s_eval)?;
            upload_scalar(&d_eq_r_acc, *eq_r_acc)?;
            
            let mut challenge_idx = 0;
            let mut scheduler = BufferScheduler::new(max_work_size);
            for &xi_j in xi_prev.iter().skip(1) {
                let src_pq_size = pq_size;
                let post_fold_size = pq_size >> 1;
                
                // BufferScheduler routing is PRESERVED UNCHANGED:
                // Each match arm calls the same compute kernel variant as before,
                // but with r_prev passed as device pointer (Change 4) for challenge_idx > 0.
                let r_prev_arg = if challenge_idx == 0 {
                    RArg::Scalar(r0)  // First inner round: r0 from CPU
                } else {
                    RArg::DevicePtr(d_challenges_inner.as_ptr().add(challenge_idx - 1))
                };
                
                match scheduler.next_target(post_fold_size, last_outer_round) {
                    BufferTarget::LayerToWork => {
                        do_fused_sumcheck_round_gpu(&eq_buffer, &layer, &mut work_buffer,
                            src_pq_size, lambda, r_prev_arg, &mut d_sum_evals, &mut tmp_block_sums)?;
                    }
                    BufferTarget::WorkToLayer => {
                        do_fused_sumcheck_round_gpu(&eq_buffer, &work_buffer, &mut layer,
                            src_pq_size, lambda, r_prev_arg, &mut d_sum_evals, &mut tmp_block_sums)?;
                    }
                    BufferTarget::InPlaceLayer => {
                        do_fused_sumcheck_round_inplace_gpu(&eq_buffer, &mut layer,
                            src_pq_size, lambda, r_prev_arg, &mut d_sum_evals, &mut tmp_block_sums)?;
                    }
                    BufferTarget::InPlaceWork => {
                        do_fused_sumcheck_round_inplace_gpu(&eq_buffer, &mut work_buffer,
                            src_pq_size, lambda, r_prev_arg, &mut d_sum_evals, &mut tmp_block_sums)?;
                    }
                }
                eq_buffer.drop_layer();
                pq_size >>= 1;
                
                // GPU postprocess: replaces observe_and_update in ALL 4 match arms
                gkr_round_postprocess(
                    &d_sum_evals, &d_sponge, &d_prev_s_eval, &d_eq_r_acc,
                    xi_j,
                    d_challenges_inner.as_mut_ptr().add(challenge_idx),
                    d_s_evals_inner.as_mut_ptr().add(challenge_idx * 3),
                )?;
                challenge_idx += 1;
            }
            
            // -- Per-outer-round replay: sync CPU transcript --
            let gpu_s_evals = d_s_evals_inner.to_host()?;
            let gpu_challenges = d_challenges_inner.to_host()?;
            // Also D2H accumulators
            let new_eq_r_acc = download_scalar(&d_eq_r_acc)?;
            let new_prev_s_eval = download_scalar(&d_prev_s_eval)?;
            
            for i in 0..challenge_idx {
                let s = &gpu_s_evals[i*3..(i+1)*3];
                for &eval in s { transcript.observe_ext(eval); }
                let r_cpu = transcript.sample_ext();
                debug_assert_eq!(r_cpu, gpu_challenges[i], "sponge mismatch at round {i}");
                round_polys_eval.push([s[0], s[1], s[2]]);
                r_vec.push(gpu_challenges[i]);
            }
            *eq_r_acc = new_eq_r_acc;
            *prev_s_eval = new_prev_s_eval;
        }
        
        GkrRoundStrategy::PrecomputeM { .. } => {
            // Fallback: existing CPU path (observe_and_update per inner round)
            // CPU transcript is already up-to-date from round 0
            // No GPU sponge involvement
            /* existing code unchanged */
        }
    }
    // CPU transcript is now synchronized for next outer round's lambda sampling
}
```

**Key design points:**
- `sync_h2d` happens AFTER round 0 of each outer round (not before the outer loop)
- Per-outer-round replay synchronizes CPU transcript BEFORE the next outer round starts
- Buffer allocation is per-outer-round (matching existing `round_polys_eval: Vec::with_capacity(round)`)
- First FoldEval inner round receives `r0` (from CPU) as a scalar; subsequent rounds read from device pointer
- PrecomputeM falls back to existing CPU path with no GPU sponge involvement

### Change 6: Per-outer-round CPU transcript replay

**File**: `crates/cuda-backend/src/logup_zerocheck/fractional.rs`

The CPU transcript replay is integrated into the per-outer-round structure shown in Change 5. After each outer round's FoldEval inner loop completes:

1. **D2H**: Retrieve the inner round results (s_evals, challenges, accumulators) — shown in Change 5 code
2. **Replay**: For each inner round `i`, call `transcript.observe_ext()` on the 3 s_evals, then `transcript.sample_ext()`, and `debug_assert_eq!` the CPU-sampled challenge against the GPU-produced challenge
3. **Update CPU accumulators**: `eq_r_acc` and `prev_s_eval` from the D2H'd device values

This ensures the CPU transcript is synchronized BEFORE the next outer round's `lambda = transcript.sample_ext()` call. The replay cost is small: at most ~20 inner rounds per outer round × 16 sponge ops = ~320 sponge operations per outer round, taking ~30-50μs. Over ~20 outer rounds × 2 segments = ~1-2ms total replay overhead, which is negligible vs. the ~28ms savings.

The `debug_assert_eq!` serves as a built-in correctness check during development. It verifies that the GPU Poseidon2 sponge produces identical challenges to the CPU sponge, catching any implementation mismatch early. It can be left in place (zero cost in release builds).

### Change 7: Specialize fractional_sumcheck_gpu transcript to DuplexSpongeGpu

**File**: `crates/cuda-backend/src/logup_zerocheck/fractional.rs`

The `fractional_sumcheck_gpu` function is currently generic over `TS: FiatShamirTranscript<SC>`. Since this function is in the CUDA backend crate and only ever called with `DuplexSpongeGpu` as the transcript type, change the signature to accept `&mut DuplexSpongeGpu` directly instead of generic `TS`:

```rust
// Before:
fn fractional_sumcheck_gpu<SC, TS>(transcript: &mut TS, ...) -> ...
  where SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>

// After:
fn fractional_sumcheck_gpu(transcript: &mut DuplexSpongeGpu, ...) -> ...
```

This gives direct access to `transcript.sync_h2d()`, `transcript.device_ptr()`, and `transcript.host` (the DeviceSpongeState that implements observe/sample for replay).

The caller `prove_zerocheck_and_logup_gpu` (mod.rs:489) and its own callers also use generic TS. Two approaches:
- **(a)** Specialize the call chain down to `prove_zerocheck_and_logup_gpu` (simplest since this is CUDA-backend-specific code)
- **(b)** Add `gpu_sponge: Option<&mut DuplexSpongeGpu>` as an extra parameter alongside the generic `TS`, used only by fractional_sumcheck_gpu

Approach (a) is preferred: the entire prove_zerocheck_and_logup_gpu function is CUDA-specific (it's in the cuda-backend crate), so removing the generic TS in favor of the concrete DuplexSpongeGpu is idiomatic. The GenericGpuBackend (which calls this) already knows the concrete type.

**Files affected**: `fractional.rs` (signature change), `mod.rs` (caller signature change), possibly `gpu_backend.rs` (propagate concrete type). No changes to the trait definition or other backends.

## Invariants

1. **Transcript determinism**: The GPU Poseidon2 sponge must produce identical results to the CPU sponge. This is verified by the debug_assert_eq on replayed challenges (Change 6). The sponge functions in `sponge.cu` use the same constants and algorithm as `poseidon2.cuh`, which matches the CPU implementation.

2. **No protocol change**: Same challenges produced in same order. Only execution location changes.

3. **FoldEval-only scope**: PrecomputeM rounds use the existing CPU path (with sync overhead). This ensures the optimization is safe for the initial prototype. The PrecomputeM path accounts for a minority of inner rounds (typically the last few per outer round).

4. **Sponge state consistency**: After the GKR round loop, the CPU transcript is synchronized via replay (Change 6), not via sync_d2h. The replay produces the exact same state because the transcript is deterministic (same inputs → same outputs).

5. **No APC 0 regression**: At APC 0, fewer GKR rounds means less overhead to eliminate. The fixed cost of buffer allocation and post-loop replay is small (~0.1ms). The optimization should be neutral-to-positive.

## Measurement Plan

Build and run the standard benchmarks:
```bash
cd /home/georg/powdr/results/pairing
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"
PROVE_BIN=$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')/release/powdr_openvm_riscv
$PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics <path>/after_apc300.json --recursion
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics <path>/after_apc000.json --recursion
```

Analyze with `python3 /home/georg/spec.py <metrics_path> <name>`.

Run nsight profiling to verify inter-kernel gap reduction:
```bash
nsys profile --output <output> --force-overwrite true --trace cuda,nvtx,osrt --sample none --stats true -- $PROVE_BIN prove --artifact apc300.cbor --input 0 --recursion
```

### Expected results
- LogUp GKR at APC 300: 526ms → ~500ms (FoldEval rounds only; ~26ms from FoldEval gaps eliminated, PrecomputeM gaps remain)
- STARK excl trace at APC 300: 1296ms → ~1270ms
- Fractional sumcheck FoldEval inter-kernel gaps: 14ms → ~3ms per segment (residual from kernel launch overhead + PrecomputeM fallback gaps)
- No regression at APC 0
- All debug_assert_eq on replayed challenges pass

### Verification
- Proof verification must pass at APC 0, 100, and 300
- Debug assertions on GPU-vs-CPU challenge comparison must hold

## Rollback Criteria

Revert if:
- LogUp GKR improvement at APC 300 is less than 10ms
- Any regression at APC 0 exceeds 10ms on STARK excl trace
- Proof verification fails at any APC configuration
- GPU-produced challenges differ from CPU reference (debug_assert fires)
