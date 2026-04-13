# Plan: Parallelize Round 0 CPU Post-Processing

## Goal

Reduce Round 0 time at APC 300 from ~662ms to ~475ms by parallelizing the CPU-bound post-processing (transpose + iDFT + polynomial construction) across AIRs using rayon. The current implementation processes 623 AIRs sequentially, spending ~200ms on CPU-only work that is embarrassingly parallel. With 32 cores, this drops to ~7-13ms.

This addresses the core scaling problem: as APC count increases, AIR count grows (99 → 623) but per-AIR work shrinks. The per-AIR CPU overhead becomes dominant, accounting for ~30% of Round 0 time at APC 300 vs ~3% at APC 0.

## Current Code Path

### Entry point
`sumcheck_uni_round0_polys()` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:598-884`

### Pre-loop setup (lines 606-728)
- Lambda powers upload (line 610)
- Lambda combinations for AIRs with monomials (lines 617-624)
- eq_3b computation per trace (lines 628-671)
- Logup combinations per trace (lines 674-691)
- EqEvalLayers construction (lines 695-703)
- Selector matrices per trace (lines 705-718)

This section is unchanged by the optimization.

### The per-AIR loop (lines 732-880) — THIS IS WHAT CHANGES

For each of 623 AIRs, the loop currently performs:

**A. GPU constraint work (lines 744-790):**
1. Build `SymbolicConstraints` from proving key (line 744-745)
2. Compute `omega_root`, gather trace pointers (lines 757-768)
3. Call `evaluate_round0_constraints_gpu()` → returns `sum_buffer: DeviceBuffer<EF>` (lines 777-790)

**B. CPU constraint post-processing (lines 791-827):**
1. `sum_buffer.to_host()` — D2H sync point (line 792)
2. Transpose `q_evals` from coset-major to row-major (lines 795-801)
3. `UnivariatePoly::from_geometric_cosets_evals_idft()` — serial CPU iDFT (lines 802-806)
4. Polynomial coefficient assembly via `(Z^{2^l_skip} - 1) * q` (lines 810-818)
5. Store into `batch_sp_poly[2 * num_present_airs + trace_idx]` (line 826)

**C. GPU interaction work (lines 831-846):**
1. Call `evaluate_round0_interactions_gpu()` which internally:
   - Builds symbolic DAG for interactions (round0.rs:162-180)
   - Compiles to `SymbolicRulesGpu`, computes weights (round0.rs:181-206)
   - Uploads rules + weights to GPU (round0.rs:204-210)
   - Launches `logup_bary_eval_interactions_round0` kernel (round0.rs:250-272)
   - Returns `s_evals: DeviceBuffer<Frac<EF>>` (line 831)

**D. CPU interaction post-processing (lines 847-879):**
1. `sum.to_host()` — D2H sync point (line 848)
2. Unzip fractions into numerator/denominator vectors (line 849-850)
3. Optional normalization for negative `n` (lines 851-857)
4. Transpose numer/denom from coset-major to row-major (lines 858-867)
5. `from_geometric_cosets_evals_idft()` × 2 — serial CPU iDFT for numer and denom (lines 869-878)
6. Store into `batch_sp_poly[2*trace_idx]` and `batch_sp_poly[2*trace_idx + 1]` (lines 869, 874)

### Why it's slow

Steps B and D are CPU-only and executed **sequentially** for 623 AIRs. Each includes:
- 1-2 transpose operations: O(num_cosets × 2^l_skip)
- 1-3 iDFT calls: O(degree × log(degree)) each (tiny per AIR but 1,869 calls total)
- Polynomial coefficient assembly

Total CPU post-processing: ~200ms (measured: 662ms total - 462ms GPU kernel time from nsight)

The GPU is idle during CPU post-processing. With 623 AIRs, that's ~0.32ms per AIR × 623 = 200ms of GPU-idle time.

## Changes

### Change 1: No new dependency needed — use existing re-export

**File:** No Cargo.toml changes required.

`openvm-stark-backend` already re-exports `p3_maybe_rayon` at `crates/stark-backend/src/lib.rs:12`, and its `parallel` feature is already enabled by the powdr workspace. Due to Cargo feature unification, `p3_maybe_rayon` with rayon support is already active in the cuda-backend's dependency tree.

No new dependency or feature flag is needed. The cuda-backend accesses rayon via:
```rust
use openvm_stark_backend::p3_maybe_rayon::prelude::*;
```

**Why:** Adding a separate `parallel` feature to cuda-backend would require downstream consumers (powdr, openvm) to explicitly enable it. Currently none of them do, so `par_iter()` would silently fall back to serial `iter()`, making the optimization a no-op. Using the existing re-export avoids this entirely.

### Change 2: Define a struct for raw Round 0 per-AIR results

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add a local struct (private, inside the impl block or at module level) to hold raw D2H data and the metadata needed for CPU post-processing:

```rust
/// Raw per-AIR data collected during GPU phase of Round 0,
/// to be post-processed in parallel.
struct Round0AirRaw {
    trace_idx: usize,
    n: isize,
    local_constraint_deg: usize,
    omega_root: F,

    /// Raw D2H constraint evaluations, or None if no constraints
    constraint_evals: Option<Vec<EF>>,
    /// Raw D2H interaction evaluations, or None if no interactions
    interaction_evals: Option<Vec<Frac<EF>>>,
}
```

**Why:** Decouples GPU result collection from CPU post-processing. Each instance is self-contained with all metadata needed for independent processing.

### Change 3: Split the per-AIR loop into Phase 1 (GPU) and Phase 2 (CPU)

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines 730-880

Replace the single loop body with two phases:

**Phase 1 — GPU work + D2H collection (sequential):**

Keep the loop structure at lines 732-739. Inside the loop body:
- Keep lines 741-790 unchanged (constraint data prep + GPU kernel launch)
- At line 791-792: Do `to_host()` and store the raw `Vec<EF>` into `Round0AirRaw.constraint_evals` (but do NOT do the transpose/iDFT/polynomial construction)
- Keep lines 830-846 unchanged (interaction GPU kernel launch)
- At line 847-848: Do `to_host()` and store the raw `Vec<Frac<EF>>` into `Round0AirRaw.interaction_evals` (but do NOT do the transpose/iDFT/polynomial construction)
- Push the `Round0AirRaw` onto a `Vec<Round0AirRaw>`

This phase retains all GPU kernel launches and D2H copies in order. The only change is removing CPU post-processing from the loop body.

**Phase 2 — CPU post-processing (parallel with rayon):**

After the GPU loop completes:

```rust
// p3_maybe_rayon imported at file top via openvm_stark_backend::p3_maybe_rayon::prelude::*

let results: Vec<(usize, Option<UnivariatePoly<EF>>, Option<UnivariatePoly<EF>>, Option<UnivariatePoly<EF>>)> = 
    raw_results.into_par_iter().map(|raw| {
        let l_skip = l_skip; // captured from outer scope
        let num_present_airs = num_present_airs;
        
        // Constraint post-processing (former lines 792-827)
        let zc_poly = raw.constraint_evals.map(|q_evals| {
            let num_cosets_zc = raw.local_constraint_deg.saturating_sub(1);
            // Transpose
            let mut values = EF::zero_vec(num_cosets_zc << l_skip);
            for coset_idx in 0..num_cosets_zc {
                for i in 0..1 << l_skip {
                    values[i * num_cosets_zc + coset_idx] = q_evals[(coset_idx << l_skip) + i];
                }
            }
            let q = UnivariatePoly::from_geometric_cosets_evals_idft(
                RowMajorMatrix::new(values, num_cosets_zc),
                raw.omega_root,
                raw.omega_root,
            );
            // sp_0 = (Z^{2^l_skip} - 1) * q
            let sp_0_deg = sumcheck_round0_deg(l_skip, raw.local_constraint_deg);
            let coeffs = (0..=sp_0_deg)
                .map(|i| {
                    let mut c = -*q.coeffs().get(i).unwrap_or(&EF::ZERO);
                    if i >= 1 << l_skip { c += q.coeffs()[i - (1 << l_skip)]; }
                    c
                })
                .collect_vec();
            UnivariatePoly::new(coeffs)
        });
        
        // Interaction post-processing (former lines 847-879)
        let (numer_poly, denom_poly) = if let Some(evals) = raw.interaction_evals {
            let num_cosets_logup = raw.local_constraint_deg;
            let (mut numer, denom): (Vec<EF>, Vec<EF>) =
                evals.into_iter().map(|frac| (frac.p, frac.q)).unzip();
            if raw.n.is_negative() {
                let norm_factor = F::from_u32(1 << raw.n.unsigned_abs()).inverse();
                for s in &mut numer { *s *= norm_factor; }
            }
            let mut numer_values = EF::zero_vec(num_cosets_logup << l_skip);
            let mut denom_values = EF::zero_vec(num_cosets_logup << l_skip);
            for coset_idx in 0..num_cosets_logup {
                for i in 0..1 << l_skip {
                    let src = (coset_idx << l_skip) + i;
                    let dst = i * num_cosets_logup + coset_idx;
                    numer_values[dst] = numer[src];
                    denom_values[dst] = denom[src];
                }
            }
            let np = UnivariatePoly::from_geometric_cosets_evals_idft(
                RowMajorMatrix::new(numer_values, num_cosets_logup),
                raw.omega_root, F::ONE,
            );
            let dp = UnivariatePoly::from_geometric_cosets_evals_idft(
                RowMajorMatrix::new(denom_values, num_cosets_logup),
                raw.omega_root, F::ONE,
            );
            (Some(np), Some(dp))
        } else {
            (None, None)
        };
        
        (raw.trace_idx, numer_poly, denom_poly, zc_poly)
    }).collect();

// Scatter results into batch_sp_poly
for (trace_idx, numer_poly, denom_poly, zc_poly) in results {
    if let Some(p) = numer_poly {
        batch_sp_poly[2 * trace_idx] = p;
    }
    if let Some(p) = denom_poly {
        batch_sp_poly[2 * trace_idx + 1] = p;
    }
    if let Some(p) = zc_poly {
        batch_sp_poly[2 * num_present_airs + trace_idx] = p;
    }
}
```

**Why this is the critical change:** The `from_geometric_cosets_evals_idft` calls (3 per AIR × 623 AIRs = 1,869 calls) and the transpose operations are the CPU bottleneck. By moving them to `par_iter`, they execute across all 32 cores simultaneously. Expected speedup: 200ms / 32 ≈ 7ms (theoretical), ~13ms (realistic with rayon overhead).

### Change 4: Retain the debug_assert for zerocheck sum

The `debug_assert_eq!` at line 819-824 that checks the zerocheck sum is zero must be preserved inside the parallel closure. It accesses `ctx.per_trace[trace_idx].0` which is the `air_idx`. Include `air_idx` in `Round0AirRaw`:

```rust
struct Round0AirRaw {
    trace_idx: usize,
    air_idx: usize,  // for debug_assert in zerocheck sum check
    n: isize,
    ...
}
```

Inside the constraint post-processing closure (after computing `coeffs`), add:
```rust
debug_assert_eq!(
    coeffs.iter().step_by(1 << l_skip).copied().sum::<EF>(),
    EF::ZERO,
    "Zerocheck sum is not zero for air_id: {}",
    raw.air_idx
);
```

### Change 5: Add import for rayon prelude

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add at the top of the file:
```rust
use openvm_stark_backend::p3_maybe_rayon::prelude::*;
```

This import provides `par_iter()` / `into_par_iter()` when the `parallel` feature is active (which it already is through the stark-backend dependency chain). Without `parallel`, it falls back to serial `iter()` / `into_iter()`.

## Invariants

1. **Output equivalence:** `batch_sp_poly` must contain exactly the same polynomials in the same positions as the current code. Each element is computed from the same raw data using the same arithmetic.

2. **GPU kernel ordering:** All GPU kernels are still launched in the same order on the same CUDA stream. Phase 1 does not change GPU behavior.

3. **D2H sync correctness:** Each `to_host()` call still synchronizes the stream before returning data. The data is fully materialized before being stored in `Round0AirRaw`.

4. **No CUDA calls from rayon threads:** Phase 2 is pure CPU. No `DeviceBuffer`, `to_device()`, or CUDA FFI calls occur inside `par_iter`. This avoids CUDA context issues with rayon worker threads.

5. **Memory safety:** `Round0AirRaw` owns its `Vec<EF>` and `Vec<Frac<EF>>` data. No references to `DeviceBuffer` or GPU memory. All GPU buffers are dropped at the end of Phase 1.

6. **Deterministic output:** The polynomial operations (transpose, iDFT, coefficient assembly) are deterministic. Parallel execution does not change results because each AIR is processed independently and results are scattered by index.

## Measurement Plan

### Commands

Run in the powdr repo:
```bash
openvm-riscv/scripts/run_pairing.sh
```

For each APC config {0, 100, 300}, analyze with:
```bash
python spec.py <metrics_json_path> <experiment_name>
```

### Expected results

| Metric | Baseline (APC 300) | Expected After | Change |
|--------|-------------------|----------------|--------|
| Round 0 | 662ms | 462-490ms | -172 to -200ms (-26 to -30%) |
| STARK excl trace | 2464ms | 2264-2300ms | -164 to -200ms (-7 to -8%) |
| Round 0 (APC 0) | 177ms | 170-177ms | ~0ms (few AIRs, low parallel benefit) |

The improvement should be proportional to AIR count:
- APC 0 (99 AIRs): minimal improvement (CPU overhead is small)
- APC 100 (360 AIRs): moderate improvement  
- APC 300 (623 AIRs): largest improvement

### Verification

1. The prove+verify cycle must succeed for all three APC configs
2. No correctness regressions (proofs must verify)
3. No memory regressions (check `nvidia-smi` peak GPU memory)

## Rollback Criteria

- **Less than 10% improvement** in Round 0 at APC 300 (< 66ms reduction): ROLLBACK
- Correctness failure (proof doesn't verify): ROLLBACK immediately
- APC 0 regression > 5% on STARK excl trace: ROLLBACK
- GPU memory peak increases by > 10%: ROLLBACK
