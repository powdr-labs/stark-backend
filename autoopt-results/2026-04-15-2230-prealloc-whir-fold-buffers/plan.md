# Plan: Pre-allocate WHIR Fold Buffers

## Goal

Eliminate per-round GPU memory allocation churn in the WHIR opening proof's inner sumcheck loop by pre-allocating one extra pair of fold buffers and reusing them via a pingpong pattern for the first `k_whir - 1` inner rounds of each WHIR outer round. The last inner round of each outer round allocates correctly-sized output buffers to preserve `.len()` correctness for `w_moments_accumulate`, which reads `DeviceBuffer::len()` to determine its processing height (`crates/cuda-backend/src/cuda/whir.rs:148`).

At APC 300, WHIR takes 125ms (measured by `spec.py`). The inner sumcheck loop runs `k_whir × num_whir_rounds = 4 × 4 = 16` iterations (m=24, k_whir=4, log_final_poly_len=8). Each iteration currently allocates 2 fresh `DeviceBuffer<EF>` and frees 2 old ones. The first ~5 iterations allocate buffers of 128 MiB, 64 MiB, 32 MiB, 16 MiB, 16 MiB — exceeding the 16 MiB VPMM page threshold — causing `cuMemCreate` + `cuMemMap` overhead. Pre-allocating eliminates 24 of 32 per-round allocations, reducing VPMM page churn and stabilizing the GPU memory pool state.

## Current Code Path

**File:** `crates/cuda-backend/src/whir.rs`, function `prove_whir_opening_gpu` (line 62)

**Outer loop:** `for (whir_round, round_params) in whir_params.rounds.iter().enumerate()` (line 199) — iterates `num_whir_rounds = 4` times.

**Inner loop:** `for round in 0..k_whir` (line 202) — iterates 4 times per outer round (k_whir=4). Each iteration:

1. **Line 204:** `let f_height = 1 << (m - round);` — logical height, starts at 2^24 and halves each round.
2. **Line 218:** `let mut new_f_coeffs = DeviceBuffer::<EF>::with_capacity(output_height);` — allocates `output_height = f_height / 2` elements, where `sizeof(EF) = 16` bytes.
3. **Line 219:** `let mut new_w_moments = DeviceBuffer::<EF>::with_capacity(output_height);` — same size.
4. **Line 224:** `whir_sumcheck_coeff_moments_round(&f_coeffs, &w_moments, ...)` — reads from current buffers using explicit `height` parameter.
5. **Line 237:** `d_s_evals.to_host()` — D2H of 2 values for transcript.
6. **Line 255:** `whir_fold_coeffs_and_moments(&f_coeffs, &w_moments, &mut new_f_coeffs, &mut new_w_moments, alpha, f_height)` — reads current, writes new. Uses explicit `height` parameter, not `.len()`.
7. **Line 269-270:** `f_coeffs = new_f_coeffs; w_moments = new_w_moments;` — drops old, replaces.

**After inner loop (line 272+):** `f_coeffs` and `w_moments` are used for:
- `split_ext_to_base_col_major_matrix` (line 281) — uses explicit `f_height` parameter.
- `eval_poly_ext_at_point_from_base` (line 332) — uses explicit height parameter.
- `w_moments_accumulate` (line 501) — **reads `w_moments.len()` as height** (`cuda/whir.rs:148`). This is the critical constraint: after the inner loop, `w_moments.len()` MUST equal the logically valid element count (`1 << (m - k_whir)`), not the pre-allocated capacity.

**At outer loop end (line 513):** `m -= k_whir;`

**Allocation sizes at APC 300 (m=24, EF=16 bytes):**
- WHIR round 0, inner round 0: output_height = 2^23, size = 128 MiB → VPMM (8 pages)
- WHIR round 0, inner round 1: output_height = 2^22, size = 64 MiB → VPMM (4 pages)
- WHIR round 0, inner round 2: output_height = 2^21, size = 32 MiB → VPMM (2 pages)
- WHIR round 0, inner round 3: output_height = 2^20, size = 16 MiB → borderline (1 page or cudaMallocAsync)
- WHIR round 1, inner round 0: output_height = 2^19, size = 8 MiB → cudaMallocAsync
- All subsequent: ≤ 8 MiB → cudaMallocAsync

Total VPMM allocations eliminated by the optimization: inner rounds 0-2 of WHIR round 0 (3 rounds × 2 buffers = 6 allocs) plus some from later rounds. Plus corresponding frees.

## Changes

### Change 1: Pre-allocate one extra buffer pair before the WHIR outer loop

**File:** `crates/cuda-backend/src/whir.rs`

**Where:** After line 195 (`let mut d_sumcheck_tmp = ...`), before line 199 (outer loop start).

**What:** Allocate one pair of "scratch" buffers at the initial maximum size:

```rust
// Pre-allocate scratch buffers for pingpong fold reuse in inner rounds.
// Capacity = initial height (2^m). Only inner rounds 0..k_whir-2 use these;
// the last inner round allocates correctly-sized buffers for w_moments_accumulate.
let initial_height = 1 << m;
let mut scratch_f = DeviceBuffer::<EF>::with_capacity(initial_height);
let mut scratch_w = DeviceBuffer::<EF>::with_capacity(initial_height);
```

**Why:** Two buffers at 256 MiB each = 512 MiB total additional peak memory. At APC 300, the prove phase uses ~15 GiB of 24 GiB on RTX 4090. The 512 MiB increase is within budget (the WHIR phase runs after constraints, when peak memory from GKR intermediates has been freed). This pair persists across all WHIR outer rounds.

### Change 2: Replace per-round allocation with pingpong for inner rounds 0..k_whir-2

**File:** `crates/cuda-backend/src/whir.rs`

**What:** Replace the inner loop body (lines 202-271) with:

```rust
for round in 0..k_whir {
    let f_height = 1 << (m - round);
    debug_assert!(f_coeffs.len() >= f_height, ...);
    debug_assert!(w_moments.len() >= f_height);
    let output_height = f_height / 2;

    // Reuse d_sumcheck_tmp (existing pattern, unchanged)
    let tmp_buffer_capacity = unsafe { _whir_sumcheck_coeff_moments_required_temp_buffer_size(f_height as u32) };
    if d_sumcheck_tmp.len() < tmp_buffer_capacity as usize {
        d_sumcheck_tmp = DeviceBuffer::<EF>::with_capacity(tmp_buffer_capacity as usize);
    }

    // Sumcheck round: reads from f_coeffs, w_moments (unchanged)
    unsafe {
        whir_sumcheck_coeff_moments_round(
            &f_coeffs, &w_moments, &mut d_s_evals, &mut d_sumcheck_tmp, f_height as u32,
        ).map_err(|error| WhirProverError::SumcheckMleRound { error, whir_round, round })?;
    }
    let s_evals = d_s_evals.to_host()?;
    for &eval in &s_evals { transcript.observe_ext(eval); }
    whir_sumcheck_polys.push(s_evals.try_into().unwrap());

    folding_pow_witnesses.push(
        transcript.grind_gpu(whir_params.folding_pow_bits).map_err(WhirProverError::FoldingGrind)?,
    );
    let alpha = transcript.sample_ext();

    if round < k_whir - 1 {
        // Non-final inner round: fold into pre-allocated scratch buffers.
        // scratch_f/scratch_w have capacity >= output_height (initial_height >= any output_height).
        // The fold kernel uses explicit `f_height` param, not buffer .len().
        unsafe {
            whir_fold_coeffs_and_moments(
                &f_coeffs, &w_moments, &mut scratch_f, &mut scratch_w, alpha, f_height as u32,
            ).map_err(|error| WhirProverError::FoldMle { error, whir_round, round })?;
        }
        // Swap: scratch becomes current, old current becomes scratch for next round.
        std::mem::swap(&mut f_coeffs, &mut scratch_f);
        std::mem::swap(&mut w_moments, &mut scratch_w);
    } else {
        // Final inner round: allocate correctly-sized output buffers.
        // This preserves .len() == output_height, which w_moments_accumulate
        // reads at cuda/whir.rs:148 to determine the processing height.
        let mut new_f_coeffs = DeviceBuffer::<EF>::with_capacity(output_height);
        let mut new_w_moments = DeviceBuffer::<EF>::with_capacity(output_height);
        unsafe {
            whir_fold_coeffs_and_moments(
                &f_coeffs, &w_moments, &mut new_f_coeffs, &mut new_w_moments, alpha, f_height as u32,
            ).map_err(|error| WhirProverError::FoldMle { error, whir_round, round })?;
        }
        f_coeffs = new_f_coeffs;
        w_moments = new_w_moments;
    }
}
```

**Why:** 
- Inner rounds 0-2: fold output goes to pre-allocated scratch pair (no allocation). The swap makes scratch the "current" buffer for the next round. The old "current" (with oversized capacity) becomes the scratch for the round after.
- Inner round 3 (last): allocates correctly-sized buffers, preserving `.len() == output_height` for `w_moments_accumulate`.
- After the inner loop: `f_coeffs` and `w_moments` have `.len() == 1 << (m - k_whir)`, exactly as in the current code.
- The rest of the WHIR outer round body (lines 272-514) is unchanged.

### Change 3: No other changes needed

The `d_sumcheck_tmp` reuse pattern (lines 215-216) is already optimal. The g_coeffs, g_rs, and Merkle tree allocations are one-per-WHIR-round and not in the inner hot loop — they're not worth pre-allocating.

## Invariants

1. **`.len()` correctness for `w_moments_accumulate`:** After each inner loop, `w_moments.len() == 1 << (m - k_whir)` because the last inner round allocates a correctly-sized buffer via `with_capacity(output_height)`. This matches the current code's behavior exactly.

2. **Kernel correctness:** `whir_sumcheck_coeff_moments_round` and `whir_fold_coeffs_and_moments` both use the explicit `height` parameter for all indexing (`cuda/whir.rs` lines 85-104, 118-131). Pre-allocated oversized buffers (from the swap pattern) satisfy the debug_asserts `len >= height` and `len >= height/2` because `initial_height >= any f_height >= any output_height`.

3. **Transcript consistency:** All transcript operations (`observe_ext`, `grind_gpu`, `sample_ext`) are identical in both code paths.

4. **Memory safety:** Both scratch buffers are allocated before the loop and live until function return. The `std::mem::swap` only exchanges the `DeviceBuffer` values (pointer + length + capacity); it does not copy GPU memory. After the swap, `f_coeffs` holds the scratch buffer (oversized but valid) and `scratch_f` holds the old `f_coeffs` (also valid, just contains stale data that won't be read).

5. **APC 0 regression guard:** At APC 0, WHIR has the same structure (k_whir=4 inner rounds × num_whir_rounds outer rounds). The optimization eliminates the same allocation pattern. The additional 512 MiB peak memory is within the 24 GiB GPU budget.

## Measurement Plan

1. Run `run_pairing.sh` for APC {0, 100, 300} and capture metrics.json.
2. Compare using `spec.py`:
   - Primary: WHIR at APC 300 should decrease from 125ms
   - Secondary: STARK excl trace at APC 300 should decrease
   - Regression check: APC 0 STARK excl trace must not increase by >20ms
3. Run nsight profiling at APC 300 and compare `cuMemCreate` call count (should decrease).

## Rollback Criteria

- WHIR at APC 300 does not improve by at least 5ms (lowered from 10ms because realistic savings from ~12 VPMM operations may be 5-10ms, and pool state effects are hard to predict precisely)
- STARK excl trace at APC 300 does not improve or regresses
- APC 0 STARK excl trace regresses by more than 20ms
- Proof verification fails at any APC configuration
