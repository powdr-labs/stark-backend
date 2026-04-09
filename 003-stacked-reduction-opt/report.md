# Report: Stacked Reduction Deferred Sync (003)

## Idea

The stacked reduction sumcheck had per-window GPU-CPU synchronization barriers in `batch_sumcheck_poly_eval()`. For each of ~3250 windows per MLE round, the code:
1. Uploaded eq_ub values (H2D)
2. Zeroed accumulator buffer
3. Launched kernel
4. Transferred results to host (D2H sync barrier)

With 10 MLE rounds, this produced ~32,500 sync barriers total. APC300 has ~3x more windows than APC000, causing 2.75x worse performance.

## Implementation

Cherry-picked commit c868e4b from the `002-parallel-streams` branch (a previous optimization attempt that was already developed but not merged to the base branch). The fix restructures `batch_sumcheck_poly_eval()` to:
1. Upload all eq_ub values once (single H2D)
2. Zero accumulator once (not per-window)
3. Launch ALL window kernels without intermediate D2H syncs
4. Single D2H copy after all kernels complete

This works because the kernels use warp-aggregated atomic accumulation into the same output buffer, automatically summing results across windows.

## Results (Combined: Round 0 Parallel Streams + Stacked Reduction Deferred Sync)

| Phase | APC000 Base | APC000 Opt | Delta | APC100 Base | APC100 Opt | Delta | APC300 Base | APC300 Opt | Delta |
|-------|-------------|------------|-------|-------------|------------|-------|-------------|------------|-------|
| **STARK excl. trace** | 2176ms | 2134ms | -42ms (1.9%) | 2179ms | 2043ms | -136ms (6.2%) | 2491ms | 2088ms | **-403ms (16.2%)** |
| Stacked Reduction | 113ms | 83ms | **-30ms (26.5%)** | 205ms | 99ms | **-106ms (51.7%)** | 311ms | 126ms | **-185ms (59.5%)** |
| Openings | 334ms | 305ms | -29ms | 341ms | 237ms | -104ms | 412ms | 228ms | **-184ms (44.7%)** |
| Round 0 | 179ms | 174ms | -5ms | 470ms | 253ms | -217ms | 670ms | 284ms | **-386ms (57.6%)** |
| LogUp GKR | 1010ms | 1005ms | -5ms | 783ms | 970ms | +187ms* | 800ms | 982ms | +182ms* |
| MLE Rounds | 118ms | 118ms | 0ms | 149ms | 150ms | +1ms | 181ms | 184ms | +3ms |
| Trace Commit | 519ms | 521ms | +2ms | 418ms | 426ms | +8ms | 408ms | 407ms | -1ms |
| **Total** | 5127ms | 5043ms | -84ms | 6106ms | 5858ms | **-248ms** | 6981ms | 6460ms | **-521ms (7.5%)** |

*GKR variance is high between runs (800-990ms). GKR runs before the optimized phases and is not causally affected.

### APC Scaling After Optimization

| Metric | APC000 | APC300 | APC300/APC000 |
|--------|--------|--------|---------------|
| Cells | 1.90B | 811M | 0.43x |
| STARK excl. trace | 2134ms | 2088ms | **0.98x** |
| Expected (proportional to cells) | 2134ms | 907ms | 0.43x |

STARK excl. trace is now nearly constant across APC configs (within 2% of APC000). The gap to perfect cell-proportional scaling remains ~1.2s, dominated by GKR (~1s, roughly constant) and Trace Commit (~400ms, scales moderately).

## Future Work

1. **MLE Rounds scaling:** 118ms → 184ms (APC000→APC300). Per-AIR overhead in MLE evaluation. Batching AIRs could help.
2. **Round 0 residual:** 284ms for APC300 (still not scaling). Full kernel batching could reduce further.
3. **GKR input eval parallelism:** 167ms for APC300 segment 0. Parallel streams could save ~40-80ms.
4. **GKR fractional sumcheck:** ~640ms, the single largest phase. Algorithmic optimization needed.
