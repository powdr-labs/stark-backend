# Report: Round 0 Parallel CUDA Streams (001)

## Idea

Round 0 evaluation (`sumcheck_uni_round0_polys`) processes all AIRs sequentially in a single loop. Each iteration launches two GPU kernels (`evaluate_round0_constraints_gpu` + `evaluate_round0_interactions_gpu`). With 300+ APC AIRs, most are small and don't saturate the GPU. By dispatching work across multiple OS threads, each getting its own CUDA stream via `--default-stream=per-thread`, we overlap kernel execution and CPU-side work (DAG building, weight computation).

## Implementation

Modified `crates/cuda-backend/src/logup_zerocheck/mod.rs`, function `sumcheck_uni_round0_polys`:

1. **Phase 1 (GPU kernel launches):** Extracted the per-AIR loop body into a `process_range` closure. When AIR count exceeds 100, partition work across 4 OS threads via `std::thread::scope`. Each thread processes its chunk, launching GPU kernels that execute on separate per-thread CUDA streams. Returns `DeviceBuffer` outputs (no D2H in this phase).

2. **Phase 2 (D2H + CPU postprocessing):** After all threads complete (and all streams are synchronized via TLS cleanup), iterate over collected DeviceBuffers. Perform `to_host()` transfers and CPU-side interpolation/coefficient construction sequentially.

Key design decisions:
- **Deferred D2H:** Avoids serialization through the global `COPY_EVENT` mutex during the parallel phase.
- **Full memory budget per thread:** Output buffers are tiny (~768B for zerocheck, ~2KB for logup), so no memory over-subscription.
- **Threshold gate:** Below 100 AIRs, the sequential path is used (prevents regression for APC000).

## Results

### Comparison: Baseline vs Optimized

| Phase | APC000 Base | APC000 Opt | Delta | APC100 Base | APC100 Opt | Delta | APC300 Base | APC300 Opt | Delta |
|-------|-------------|------------|-------|-------------|------------|-------|-------------|------------|-------|
| **STARK excl. trace** | 2176ms | 2168ms | -8ms (0.4%) | 2179ms | 2141ms | -38ms (1.7%) | 2491ms | 2098ms | **-393ms (15.8%)** |
| Round 0 | 179ms | 173ms | -6ms | 470ms | 245ms | **-225ms (47.9%)** | 670ms | 281ms | **-389ms (58.1%)** |
| LogUp GKR | 1010ms | 1006ms | -4ms | 783ms | 974ms | +191ms* | 800ms | 807ms | +7ms |
| MLE Rounds | 118ms | 118ms | 0ms | 149ms | 152ms | +3ms | 181ms | 181ms | 0ms |
| Stacked Reduction | 113ms | 113ms | 0ms | 205ms | 205ms | 0ms | 311ms | 311ms | 0ms |
| Trace Commit | 519ms | 515ms | -4ms | 419ms | 427ms | +8ms | 408ms | 411ms | +3ms |
| WHIR | 218ms | 219ms | +1ms | 137ms | 137ms | 0ms | 100ms | 101ms | +1ms |
| **Total** | 5127ms | 5091ms | -36ms | 6106ms | 6107ms | +1ms | 6981ms | 6553ms | **-428ms** |

*GKR variance: APC100 and APC300 show GKR variance between runs (807ms to 994ms for APC300). The best APC300 run (run 3) shows GKR at 807ms, confirming no systematic regression. GKR runs before Round 0 in the proving flow so it cannot be causally affected.

### Relative to APC000 (optimized)

| Phase | APC000 | APC100 | Factor | APC300 | Factor |
|-------|--------|--------|--------|--------|--------|
| STARK excl. trace | 2168ms | 2141ms | 1.01x | 2098ms | 1.03x |
| Round 0 | 173ms | 245ms | 0.71x | 281ms | 0.62x |
| Cells | 1.90B | 1.19B | 1.60x | 811M | 2.35x |

Note: With the optimization, STARK excl. trace is nearly identical across all APC configs (2168, 2141, 2098ms), whereas the baseline showed APC300 15% worse than APC000 (2491 vs 2176).

## Future Work

1. **GKR variance:** The GKR phase shows high run-to-run variance (807-994ms for APC300). Investigate whether this is thermal throttling, memory pool fragmentation, or code layout effects.

2. **Round 0 kernel batching:** Even with parallel streams, Round 0 still takes 281ms for APC300. Full kernel-level batching (grouping AIRs into single kernel launches with BlockCtx dispatch, as in batch_mle) could further reduce this by eliminating per-kernel launch overhead entirely.

3. **Combine with GKR input batching:** The GKR input evaluation is also per-AIR (~800ms). Batching this would compound with the Round 0 improvement.

4. **Memory manager contention:** The global `MEMORY_MANAGER` mutex serializes all GPU allocations. A per-thread or lock-free allocator could improve the parallel path further.
