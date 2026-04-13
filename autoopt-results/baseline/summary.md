# Baseline Measurements

Date: 2026-04-12
Branch: v2-powdr-07-04 (commit 2123b6a7)
GPU: NVIDIA RTX 4090

## STARK (excl. trace) — Target Metric

| Config | STARK excl trace | Constraints | LogUp GKR | Round 0 | MLE Rounds | Openings | WHIR | Stacked Red. | Trace Commit |
|--------|-----------------|-------------|-----------|---------|------------|----------|------|-------------|-------------|
| APC 0  | 2153 ms         | 1292 ms     | 993 ms    | 178 ms  | 118 ms     | 336 ms   | 220 ms | 113 ms    | 518 ms      |
| APC 100| 2155 ms         | 1393 ms     | 775 ms    | 464 ms  | 150 ms     | 342 ms   | 138 ms | 202 ms    | 418 ms      |
| APC 300| 2455 ms         | 1634 ms     | 790 ms    | 662 ms  | 180 ms     | 413 ms   | 100 ms | 311 ms    | 406 ms      |

## Workload Scaling (APC 0 → APC 300)

| Metric                    | APC 0       | APC 300     | Ratio  |
|--------------------------|-------------|-------------|--------|
| Cells                    | 1,904M      | 811M        | 2.35x  |
| Constraint Instances     | 1,126M      | 505M        | 2.23x  |
| Bus Interaction Messages | 965M        | 448M        | 2.16x  |
| AIR Instances            | 99          | 623         | 0.16x (6.3x MORE) |
| Columns                  | 3,919       | 106,842     | 0.04x (27x MORE) |
| STARK (excl. trace)      | 2153 ms     | 2455 ms     | 0.88x (14% WORSE) |

## Key Observation

Despite >2x reduction in cells/constraints/messages, STARK time INCREASES by 14%.
The main culprits: Round 0 (+3.7x), Stacked Reduction (+2.75x), MLE Rounds (+53%).
LogUp GKR and Trace Commit do improve somewhat, but are offset by the regressions.
