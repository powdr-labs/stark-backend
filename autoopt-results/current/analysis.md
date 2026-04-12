# Current Performance Analysis

## Status

The batch-round0-coset-parallel commit (74134b4e) appears to introduce a severe performance
regression on the pairing benchmark. The benchmark fails to complete within 2 minutes (expected
~6 seconds). This prevents fresh measurements.

Data below uses the most recent successful measurements:
- Post defer-round0-d2h-sync: from powdr/results/pairing/apc{100,300}/metrics.json (Apr 10)
- Post stacked-reduction: from log.md entries (measured by that agent)
- Baseline: from autoopt-results/baseline/

## STARK excl trace (ms)

| Config | Baseline | Post d2h-sync | Post stacked-red. | vs Baseline |
|--------|----------|---------------|---------------------|-------------|
| APC 0  | 2149     | 2146          | ~2146*              | -3ms        |
| APC 100| 2159     | 2077          | ~2077*              | -82ms       |
| APC 300| 2478     | 2358          | 2195                | -283ms      |

*Estimated (stacked reduction has negligible effect at APC 0/100)

## APC 300 STARK excl trace breakdown (estimated current = post stacked-red.)

| Component         | Baseline | Current est. | vs Baseline |
|-------------------|----------|-------------|-------------|
| Round 0           | 663ms    | ~569ms      | -94ms       |
| LogUp GKR         | 818ms    | ~791ms      | -27ms       |
| MLE Rounds        | 180ms    | ~180ms      | 0ms         |
| Trace Commit      | 403ms    | ~403ms      | 0ms         |
| Stacked Reduction | 308ms    | ~147ms      | -161ms      |
| WHIR              | 101ms    | ~100ms      | -1ms        |
| Other             | 5ms      | ~5ms        | 0ms         |
| **Total**         | **2478** | **~2195**   | **-283ms**  |

## Scaling ratio (APC 300 / APC 0)

| Metric | Baseline | Current est. |
|--------|----------|-------------|
| STARK excl trace | 1.153 | ~1.014 |

## Per-component scaling (APC 300 vs APC 0, current est.)

| Component         | APC 0   | APC 300 | Ratio | Direction |
|-------------------|---------|---------|-------|-----------|
| Round 0           | 175ms   | 569ms   | 3.25x | WORSE     |
| LogUp GKR         | 1007ms  | 791ms   | 0.79x | Better    |
| MLE Rounds        | 118ms   | 180ms   | 1.53x | WORSE     |
| Trace Commit      | 519ms   | 403ms   | 0.78x | Better    |
| Stacked Reduction | 113ms   | 147ms   | 1.30x | WORSE     |
| WHIR              | 218ms   | 100ms   | 0.46x | Better    |

Round 0 is the dominant scaling offender: +394ms at APC 300 vs APC 0.

## GPU Kernel Analysis (from baseline nsight, APC 300)

Round 0 kernels: ~479ms GPU time total
- zerocheck coset_parallel<true>: 221ms (539 instances)
- logup r0 coset_parallel<true>: 196ms (396 instances)
- Non-coset-parallel variants: ~62ms

Round 0 CPU overhead: 663 - 479 = 184ms (baseline), ~90ms post d2h-sync

LogUp GKR kernels: ~666ms GPU time
- evaluate_interactions_gkr<true>: 374ms (486 instances)
- evaluate_interactions_gkr<false>: 54ms (305 instances)
- fractional_sumcheck_gkr: ~238ms

LogUp GKR CPU overhead: 818 - 666 = 152ms
