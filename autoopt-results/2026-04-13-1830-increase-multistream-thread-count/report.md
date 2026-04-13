# Report: Increase Multi-Stream Thread Count from 4 to 8

## Description

Increased the number of OS threads (each with a per-thread CUDA stream via `cudaStreamPerThread`) from 4 to 8 for both Round 0 constraint evaluation and GKR input evaluation. At APC 300, ~310 AIRs per segment launch tiny kernels (1-4 SMs each) on the RTX 4090's 128 SMs. With 4 threads, nsys profiling showed only 1.7-2.1x effective parallelism, leaving >90% of SMs idle. Doubling to 8 threads was expected to improve concurrent kernel occupancy, reduce per-chunk load imbalance, and provide more CPU/GPU overlap.

## Implementation

Two constants changed, no logic changes:

1. **`crates/cuda-backend/src/logup_zerocheck/mod.rs` line 93**: `NUM_ROUND0_STREAMS` from 4 to 8
2. **`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs` line 30**: `NUM_GKR_INPUT_STREAMS` from 4 to 8

Updated the comment on `NUM_ROUND0_STREAMS` to reflect the new value.

No deviations from the plan. The existing memory budget division (`memory_limit_bytes / NUM_ROUND0_STREAMS`) automatically adjusts, giving each of 8 threads ~1 GiB instead of ~2 GiB — sufficient with >20x headroom for the largest AIRs.

The `>= 100 AIRs` threshold for multi-threading is unchanged, so APC 0 (~20 AIRs/segment) always uses the sequential path.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455 ms | 1578 ms | 1517 ms | -938ms, 1.62x lower | -61ms, 1.04x lower |
| Constraints | 1634 ms | 1103 ms | 1039 ms | -595ms, 1.57x lower | -64ms, 1.06x lower |
| Round 0 | 662 ms | 348 ms | 295 ms | -367ms, 2.24x lower | -53ms, 1.18x lower |
| LogUp GKR | 790 ms | 573 ms | 559 ms | -231ms, 1.41x lower | -14ms, 1.03x lower |
| MLE Rounds | 180 ms | 181 ms | 183 ms | +3ms, 1.02x higher | +2ms, 1.01x higher |
| Openings | 413 ms | 226 ms | 227 ms | -186ms, 1.82x lower | +1ms, unchanged |
| Trace Commit | 406 ms | 246 ms | 250 ms | -156ms, 1.62x lower | +4ms, unchanged |

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155 ms | 1879 ms | 1848 ms | -307ms, 1.17x lower | -31ms, 1.02x lower |
| Round 0 | 464 ms | 286 ms | 244 ms | -220ms, 1.90x lower | -42ms, 1.17x lower |
| LogUp GKR | 775 ms | 859 ms | 864 ms | +89ms, 1.11x higher | +5ms, unchanged |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153 ms | 2175 ms | 2143 ms | -10ms, 1.00x lower | -32ms, 1.01x lower |
| Round 0 | 178 ms | 177 ms | 177 ms | -1ms, unchanged | 0ms, unchanged |
| LogUp GKR | 993 ms | 1028 ms | 998 ms | +5ms, unchanged | -30ms, 1.03x lower |

### Success Criteria Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Round 0 (APC 300) | < 300ms | 295ms | PASS |
| LogUp GKR (APC 300) | < 530ms | 559ms | FAIL |
| STARK excl trace (APC 300) | < 1500ms | 1517ms | FAIL (close) |
| STARK excl trace (APC 0) | < 2250ms (no regression) | 2143ms | PASS |
| APC 100 Round 0 | < 250ms | 244ms | PASS |

### Rollback Criteria Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| STARK excl trace APC 300 improvement | > 60ms | 61ms | PASS (borderline) |
| STARK excl trace APC 0 regression | < 65ms increase | -32ms (improved) | PASS |
| Correctness | prove+verify passes | All configs pass | PASS |
| GPU OOM | No OOM | No OOM | PASS |
| APC 100 STARK regression | < 5% regression | -1.6% (improved) | PASS |

## Assessment

The optimization achieved a **clear improvement in Round 0** — 15-17% faster at both APC 100 and APC 300 — confirming that kernel concurrency was a bottleneck for that phase. Round 0 at APC 300 is now 2.24x lower than baseline, up from 1.89x before this task.

The **GKR input eval improvement was minimal** (14ms at APC 300), suggesting that with 4 threads it was already closer to its concurrency ceiling, possibly due to higher per-AIR allocation overhead or memory manager mutex contention becoming the bottleneck before SM utilization.

Overall STARK excl trace improved by 61ms at APC 300, barely above the 60ms rollback threshold. The optimization is kept because:
1. The Round 0 improvement is unambiguous and consistent across configs
2. No regressions at any config
3. The change is trivial (two constant changes) with minimal complexity cost
4. Cumulative improvement vs baseline is now 1.62x, up from 1.54x

## Future Work

- **Per-thread buffer pools**: The MemoryManager mutex is likely the bottleneck preventing GKR input eval from scaling further with more threads. Pre-allocating per-thread reusable buffers would eliminate ~3,100 lock acquisitions per segment.
- **Adaptive thread count**: Instead of a fixed constant, the thread count could be tuned based on the number of AIRs and their height distribution. For segments with fewer large AIRs, fewer threads may be optimal; for many small AIRs, even more threads could help.
- **Work stealing**: Replace the static chunking (sorted by descending height) with a work-stealing queue to better balance load across threads. The current approach still creates tail imbalance when the largest AIRs are concentrated in the first chunk.
- **Beyond 8 threads**: 12 or 16 threads could be tested, though diminishing returns from mutex contention are expected. This would be more impactful after the per-thread buffer pool optimization.
- **Kernel fusion**: Rather than launching more concurrent tiny kernels, fusing per-AIR kernels into batch kernels (similar to the stacking scatter optimization) could eliminate launch overhead entirely.
