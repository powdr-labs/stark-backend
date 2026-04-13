# Report: Round 0 Interleaved Work Balance

## Description

Replace contiguous work chunking in Round 0's multi-threaded dispatch with interleaved (round-robin) assignment. With 8 threads and ~310 AIRs sorted by descending trace height, contiguous chunking (`chunks(39)`) assigns all large AIRs to thread 0 and all small AIRs to thread 7. The hypothesis was that this creates load imbalance, with the critical thread accumulating more wall-clock time from a combination of GPU kernel waits and CPU post-processing.

The expected improvement was 15-30ms at APC 300, based on reducing the gap between the fastest and slowest threads.

## Implementation

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines 950-985

### Deviation from plan

The plan specified greedy LPT (Longest Processing Time first) assignment using `height` as the cost proxy. Diagnostic instrumentation revealed this approach is counterproductive:

- **LPT with height-only cost model** assigned only 1 AIR to each of threads 0-2 (the 3 largest-height AIRs) and 49-78 AIRs to threads 3-7. The 3 large AIRs finished in 10-16ms each while threads with 66+ small AIRs took 117-124ms. The critical path *increased* from 109ms to 124ms.
- **Root cause:** `height` is not a good cost proxy because it ignores per-AIR fixed overhead (~1.8ms per AIR for kernel launch, D2H copy, CPU post-processing). A single 2^20-height AIR takes ~13ms, but 66 tiny AIRs take ~118ms despite having less total height.

Instead, implemented **round-robin (interleaved) assignment** on the sorted list: item `i` goes to thread `i % num_threads`. This naturally balances both per-AIR count (equal ±1) and height-weighted cost (each thread gets the kth-largest, kth+N largest, kth+2N largest, etc.) without needing a calibrated cost model.

### Key changes

1. Replaced `work_items.chunks(chunk_size)` with round-robin index assignment into `Vec<Vec<usize>>` per thread
2. Each thread references `work_items` by index, processing its assigned items via `process_air_round0(&items[idx])`
3. Added `tracing::debug!` per-thread elapsed time logging (debug level only, zero overhead at default log level)
4. Single-threaded path (APC 0, <100 AIRs) is unchanged

### Thread balance improvement (APC 300, segment 1)

| Metric | Contiguous Chunking | Round-Robin |
|--------|-------------------|-------------|
| Fastest thread | 83ms | 93ms |
| Slowest thread | 109ms | 100ms |
| Spread | 26ms | 7ms |
| Items per thread | 33-39 (even) | 38-39 (even) |

## Results

### APC 300 (primary target, 3 stability runs)

| Metric | Baseline | Before Task | After Task (avg of 3) | vs Baseline | vs Before |
|--------|----------|-------------|----------------------|-------------|-----------|
| STARK excl trace | 2455ms | 1433ms | 1411ms | -1044ms, 1.74x lower | -22ms, 1.02x lower |
| Round 0 | 662ms | 313ms | 277ms | -385ms, 2.39x lower | -36ms, 1.13x lower |
| Constraints | 1634ms | 1008ms | 983ms | -651ms, 1.66x lower | -25ms, 1.03x lower |
| LogUp GKR | 790ms | 523ms | 531ms | -259ms, 1.49x lower | +8ms (noise) |
| MLE Rounds | 180ms | 170ms | 172ms | -8ms, 1.05x lower | +2ms (noise) |
| Trace Commit | 406ms | 248ms | 249ms | -157ms, 1.63x lower | +1ms (noise) |
| Openings | 413ms | 176ms | 176ms | -237ms, 2.35x lower | 0ms |

After stability runs: 1410ms, 1402ms, 1420ms (Round 0: 277ms, 277ms, 277ms)

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155ms | 1573ms | 1578ms | -577ms, 1.37x lower | +5ms (noise) |
| Round 0 | 464ms | 241ms | 230ms | -234ms, 2.02x lower | -11ms, 1.05x lower |

### APC 000 (no-regression check)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2133ms | 2142ms | -11ms (noise) | +9ms (noise) |
| Round 0 | 178ms | 178ms | 179ms | +1ms (noise) | +1ms (noise) |

APC 0 uses single-threaded path (<100 AIRs), confirming no regression from the interleaved change.

## Assessment

The optimization achieved its goal. Round 0 improved by 36ms (1.13x) at APC 300, within the plan's 15-30ms target range (slightly above). STARK excl trace improved by 22ms on average, passing the 15ms rollback threshold.

The improvement came from reducing the max-thread wall time by ~9ms per segment (2 segments at APC 300), with additional savings from better GPU stream overlap. The cumulative STARK excl trace improvement vs baseline is now 1.74x.

At APC 100, Round 0 improved by 11ms (1.05x), consistent with fewer AIRs (360 vs 623) giving less room for load imbalance. APC 0 showed no regression as expected.

**Worth the complexity?** Yes — the change is 15 lines of straightforward code (round-robin index assignment) replacing 3 lines (contiguous chunking). The improvement is modest but consistent, and the code is arguably clearer about its intent.

## Future Work

- **Cost-model-aware LPT:** A calibrated cost model of `height * α + β` (where β captures per-AIR fixed overhead) could achieve even better balance than round-robin, especially if the AIR height distribution is highly skewed. The diagnostic data suggests β ≈ 1.8ms and α ≈ 11ns/cell.
- **Apply to GKR input evaluation:** The same contiguous chunking pattern exists in the multi-stream GKR input evaluation path. Round-robin assignment there could yield similar 5-10% improvements.
- **Dynamic thread count:** Instead of a fixed threshold (>=100 AIRs → 8 threads), adaptively choose thread count based on total work. For example, APC 100 has only 360 AIRs across 3 segments — some segments may have too few AIRs to benefit from 8 threads.
- **Profile-guided cost model:** Record actual per-AIR processing times from previous runs and use them as cost weights for LPT in future runs. This would give optimal balance without needing a parametric cost model.
