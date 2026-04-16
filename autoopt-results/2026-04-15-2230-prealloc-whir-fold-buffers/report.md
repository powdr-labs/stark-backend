# Report: Pre-allocate WHIR Fold Buffers

## Description

Pre-allocate and reuse WHIR sumcheck fold buffers (`f_coeffs`, `w_moments`) via a pingpong pattern instead of allocating new DeviceBuffers each inner round. Each of the `k_whir × num_whir_rounds` inner sumcheck rounds allocates two fresh DeviceBuffers and frees the previous pair. For the first ~5 rounds at APC 300 (buffer sizes 128–16 MiB), these go through VPMM, creating page churn. The hypothesis was that eliminating per-round VPMM allocations/frees would save 5–10ms on WHIR and stabilize GPU memory pool state for subsequent phases.

## Implementation

**File:** `crates/cuda-backend/src/whir.rs`, function `prove_whir_opening_gpu`

Two changes were made:

1. **Pre-allocate scratch buffers** (after line 195, before the outer loop): Allocated one pair of scratch DeviceBuffers at initial maximum size `1 << m` (256 MiB each at APC 300). These persist across all WHIR outer rounds.

2. **Pingpong for non-final inner rounds**: For inner rounds `0..k_whir-2`, fold output goes to the pre-allocated scratch pair instead of newly allocated buffers, then `std::mem::swap` makes the scratch the "current" buffer. The final inner round (`k_whir-1`) still allocates correctly-sized buffers to preserve `.len() == output_height`, which `w_moments_accumulate` (cuda/whir.rs:148) reads to determine processing height.

No deviations from the plan. All 14 WHIR unit tests pass.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **STARK excl trace APC 300** | 2455ms | 1117ms | 1115ms | -1340ms, 2.20x lower | -2ms, 1.00x (noise) |
| **WHIR APC 300** | 100ms | 130ms | 126ms | +26ms, 1.26x higher | -4ms, 1.03x lower |
| **STARK excl trace APC 100** | 2155ms | 1389ms | 1312ms | -843ms, 1.64x lower | -77ms, 1.06x lower |
| **WHIR APC 100** | 138ms | 154ms | 156ms | +18ms, 1.13x higher | +2ms (noise) |
| **STARK excl trace APC 0** | 2153ms | 1784ms | 1810ms | -343ms, 1.19x lower | +26ms (noise) |
| **WHIR APC 0** | 220ms | 217ms | 221ms | +1ms (noise) | +4ms (noise) |

Second run at APC 300 confirmed: WHIR 125ms, STARK excl trace 1127ms — consistent with first run.

## Assessment

The optimization **failed** to meet the rollback criteria:

- WHIR at APC 300 improved by only ~4ms (threshold: 5ms)
- STARK excl trace at APC 300 showed no improvement (-2ms, within noise)
- No regression at APC 0 (within 26ms noise band, below 20ms systematic threshold)

The per-round VPMM allocation overhead at APC 300 is lower than estimated. At most 6 VPMM allocations are eliminated (inner rounds 0–2 of WHIR round 0, × 2 buffers), with subsequent rounds using cudaMallocAsync's pool cache (buffer sizes ≤ 8 MiB). The actual per-VPMM-operation overhead is ~0.5–1ms (not ~2ms as estimated), so maximum possible savings were ~3–6ms — and we observed ~4ms. The pool state hypothesis (that eliminating VPMM churn would benefit subsequent bandwidth-bound phases) did not materialize.

## Future Work

- WHIR at APC 300 is 126ms, which is already well-optimized. The dominant cost is GPU kernel execution in the 16 inner sumcheck rounds, not allocation overhead.
- The WHIR phase has regressed 26ms from baseline (100ms → 126ms) across previous optimizations. This regression is likely due to accumulated VPMM pool state changes from other optimizations (similar to patterns seen in `cache-codeword-buffer-across-segments` and `multistream-stacked-reduction-round0`), not allocation overhead within WHIR itself.
- Further WHIR improvements would require algorithmic changes (e.g., fusing sumcheck+fold into a single kernel, or reducing the number of inner rounds).
- The WHIR phase is now a small fraction (2.3%) of total proof time and (11.3%) of STARK excl trace — diminishing returns make further WHIR-specific optimization low priority.
