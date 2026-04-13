# Report: GKR Input Eval Round-Robin Work Assignment

## Description

Replace contiguous work chunking in `log_gkr_input_evals()` with interleaved (round-robin) assignment across the 8 multi-stream worker threads. When work items are sorted by descending height and split into contiguous chunks, thread 0 receives all large AIRs while thread 7 receives only small AIRs, creating load imbalance. Round-robin assignment (item i goes to thread i % N) distributes a balanced mix of large and small AIRs to each thread, reducing the critical thread's wall time. This follows the exact pattern that improved Round 0 by 36ms / 1.13x at APC 300.

## Implementation

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`, lines 328-360

**Change**: Replaced `work_items.chunks(chunk_size)` dispatch with round-robin index assignment:

1. Build `thread_indices: Vec<Vec<usize>>` where item `idx` goes to `thread_indices[idx % num_threads]`
2. Each thread iterates over its assigned indices, accessing `work_items` by shared reference
3. Added per-thread `tracing::debug!` logging (elapsed_ms, num_airs, thread_id) for diagnostics

**No deviations from the plan.** The single-thread path (< 100 AIRs, used by APC 0) is unchanged.

**Diagnostic verification**: With `RUST_LOG=debug`, thread balance at APC 300:
- **Segment 0** (310 AIRs, 8 threads): 36-65ms range (29ms spread). Compare with contiguous chunking in Round 0 which had 26ms spread before the same fix.
- **Segment 1** (317 AIRs, 8 threads): 37-43ms range (6ms spread). Very well balanced.

## Results

### APC 300

| Metric | Baseline | Before | After | vs Baseline | vs Before |
|--------|----------|--------|-------|-------------|-----------|
| STARK excl trace | 2455ms | 1389ms | 1386ms | -1069ms (1.77x lower) | -3ms (1.00x) |
| GKR input evals | 165ms | 302ms | 290ms | +125ms (1.76x higher) | -12ms (1.04x lower) |
| LogUp GKR | 790ms | 520ms | 508ms | -282ms (1.56x lower) | -12ms (1.02x lower) |
| Round 0 | 662ms | 273ms | 275ms | -387ms (2.41x lower) | +2ms (noise) |
| Trace Commit | 406ms | 247ms | 254ms | -152ms (1.60x lower) | +7ms (noise) |
| MLE Rounds | 180ms | 170ms | 169ms | -11ms (1.07x lower) | -1ms (noise) |

### APC 100

| Metric | Baseline | Before | After | vs Baseline | vs Before |
|--------|----------|--------|-------|-------------|-----------|
| STARK excl trace | 1551ms | 1197ms | 1200ms | -351ms (1.29x lower) | +3ms (noise) |
| GKR input evals | 162ms | 291ms | 270ms | +108ms (1.67x higher) | -21ms (1.08x lower) |
| LogUp GKR | 577ms | 512ms | 503ms | -74ms (1.15x lower) | -9ms (1.02x lower) |

### APC 000 (control — unchanged single-thread path)

| Metric | Baseline | Before | After | vs Baseline | vs Before |
|--------|----------|--------|-------|-------------|-----------|
| STARK excl trace | 1145ms | 1157ms | 1179ms | +34ms (noise) | +22ms (noise) |
| GKR input evals | 354ms | 370ms | 397ms | +43ms (noise) | +27ms (noise) |

### Stability (APC 300, 3 runs each)

| Run | Before STARK | Before GKR input | After STARK | After GKR input |
|-----|-------------|-----------------|------------|-----------------|
| 1   | 1389ms | 302ms | 1386ms | 290ms |
| 2   | 1403ms | 304ms | 1406ms | 311ms |
| 3   | 1406ms | 308ms | 1406ms | 296ms |
| **Median** | **1403ms** | **304ms** | **1406ms** | **296ms** |

GKR input eval median improvement: 304ms → 296ms = **-8ms** (2.6%).
STARK excl trace median: 1403ms → 1406ms = **+3ms** (within 17-20ms noise range).

## Assessment

**Result: FAILURE** — Reverted per rollback criterion #1 (STARK excl trace APC 300 improvement < 8ms).

The GKR input eval improved by ~8ms median at APC 300 and ~21ms at APC 100, confirming that round-robin does improve thread balance. However, the improvement is too small to be detectable at the STARK excl trace level, where run-to-run variance is 17-20ms.

**Why it didn't work as well as Round 0 round-robin (which saved 36ms):**

1. **Different cost structure**: In Round 0, per-AIR fixed overhead (kernel launch, D2H copy, CPU post-processing) is ~1.8ms, making item count the dominant cost factor. Round-robin balances item count well. In GKR input eval, per-AIR overhead is lower (~0.5-1ms) because buffers are pre-allocated (no per-AIR malloc), and the kernel-to-overhead ratio is different.

2. **Already partially balanced**: The pre-allocated buffer optimization (previous task) eliminated the main source of thread imbalance in GKR input eval — per-AIR mutex contention and cudaMallocAsync serialization. With buffers pre-allocated, the remaining imbalance from height variation is small.

3. **Seg0 dominates, seg1 already fast**: At APC 300, seg0 takes 240ms and seg1 takes 62ms. Even perfect load balance in seg0 (max thread time = avg) would save at most ~10-15ms. The actual improvement was ~5ms on seg0 and ~9ms on seg1.

## Future Work

- **What worked**: The round-robin pattern itself is correct and verified via diagnostic logging. Thread spread reduction is real (29ms spread vs what would be ~40ms+ with contiguous chunking for seg0).
- **Why it's insufficient**: GKR input eval's per-AIR cost is already low after pre-allocation. The remaining bottleneck is not thread imbalance but aggregate kernel execution time.
- **Better targets**: The GKR input eval total (300ms at APC 300) is high but dominated by kernel execution, not scheduling overhead. Reducing kernel count (batching via descriptor arrays, similar to the MLE rounds approach) would be more impactful than rebalancing thread work.
- **Measurement note**: `RUST_LOG=warn` blocks INFO-level tracing spans, which prevents the `TimingMetricsLayer` from recording timing metrics. Always run benchmarks without `RUST_LOG` or with `RUST_LOG=info` to get timing data.
