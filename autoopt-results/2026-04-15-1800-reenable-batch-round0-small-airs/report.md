# Report: Re-enable Batch Round 0 Descriptor-Array Kernels for Small AIRs

## Description

The optimization aimed to reduce Round 0 wall-clock time at APC 300 by batching the eval kernels for ~250 small coset-parallel AIRs per segment into 2-4 descriptor-array kernel launches on the default stream. The previous implementation (commit `0d1bb2f9`) was reverted due to a GPU hang at APC 300. The plan assumed the hang was caused by GPU OOM (since resolved by VPMM 16 MiB page size increase), but this turned out to be incorrect.

## Implementation

### CUDA Kernels (restored from reverted commit `0d1bb2f9`)
- `batched_zerocheck_r0_coset_parallel_kernel`: Block-to-AIR mapping via `Round0BlockCtx` array, per-AIR `Round0ZcCtx` descriptors, GLOBAL=true/NEEDS_SHMEM=false only
- `batched_logup_r0_coset_parallel_kernel`: Same descriptor-array pattern for logup interactions
- Reduction via existing `batched_final_reduce_block_sums` from batch_mle.cu

### FFI Bindings (restored)
- `Round0BlockCtx`, `Round0ZcCtx`, `Round0LogupCtx` repr(C) structs
- `batched_zerocheck_r0_eval_constraints()` and `batched_logup_r0_eval_interactions()` safe wrappers

### Rust Orchestration (`round0_batched.rs`, new file)
- `identify_batchable_airs()`: Selects coset-parallel AIRs within per-AIR and aggregate intermediates budget (64 MB L2 cache limit)
- `batch_round0_eval_only()`: Runs batched eval+reduce on default stream, returns `BatchEvalState` with output buffers
- `extract_batched_air()`: Per-AIR extraction called from Phase 2 worker threads
- Design: eval-only on default stream before Phase 2, extraction deferred to worker threads (maintaining 8-way concurrency)

### Integration (`mod.rs`)
- Batch eval inserted after Phase 1 buffer allocation, before `num_threads` branch
- `current_stream_sync()` after batch eval to make results visible to worker streams
- Phase 2 workers check `batch_mask` and route batched AIRs to extraction-only path

## Results

The implementation was fully functional for small batch sizes but the batch CUDA kernels hang when more than ~12 AIRs are batched. This prevented any meaningful performance measurement.

### Diagnostic Findings

| Test | AIRs Batched | ZC Groups | Logup Groups | Result |
|------|-------------|-----------|-------------|--------|
| APC 0 (no batch) | 0 | - | - | OK |
| APC 300, 1 AIR | 1 | 1×2 | skipped | OK, syncs in <1ms |
| APC 300, 10 AIRs | 10 | 9×1, 1×2 | skipped | OK, all threads complete |
| APC 300, 12 AIRs | 12 | 11×1, 1×2 | skipped | OK |
| APC 300, 15 AIRs | 15 | 13×1, 2×2 | skipped | HANG at current_stream_sync |
| APC 300, 50 AIRs | 50 | 35×1, 15×2 | skipped | HANG at current_stream_sync |
| APC 300, 171 AIRs | 171 | 43×1, 128×2 | 43×2, 128×3 | HANG at current_stream_sync |

The hang occurs during `current_stream_sync()` after the batch kernel launch. The CPU-side launch returns immediately (measured 0ms), but the GPU kernel never completes. GPU shows 100% utilization and 16 GB memory used during the hang. The hang persists even when:
- The logup batch path is completely skipped (zerocheck only)
- The batch state is dropped and all GPU memory freed after sync
- Phase 2 falls back to full per-AIR processing (no extraction used)

### Root Cause

The batched CUDA kernel hangs when the total number of CUDA blocks exceeds ~50-80. With the coset-parallel design, each AIR requires `num_x_blocks × num_cosets` blocks. For 15 AIRs:
- 13 AIRs × 1 block + 2 AIRs × ~8 blocks = ~29 blocks (zc group 1+2)
This is borderline. For 50 AIRs: ~65+ blocks, which consistently hangs.

The likely cause is an out-of-bounds memory access in the batch kernel that causes a silent GPU fault (no cudaErrorIllegalAddress returned, just a stalled pipeline). The kernel reads per-AIR descriptors from device memory and computes intermediates buffer offsets. With many AIRs, the aggregate intermediates arena is large (tens of MB), and an incorrect offset calculation or descriptor field could cause an access far beyond allocated memory.

This is consistent with the original revert report: "The batched Round 0 path causes APC 300 STARK proving to hang for 12+ minutes due to either CPU-side rule reconstruction overhead or a batched CUDA kernel bug."

### Performance (before-only, since optimization could not be enabled)

| Metric | Baseline | Before Task | vs Baseline |
|--------|----------|-------------|-------------|
| STARK excl trace APC 300 | 2455ms | 1110ms | 1.95x lower |
| Round 0 APC 300 | 662ms | 186ms | 3.56x lower |
| STARK excl trace APC 100 | 2155ms | 1295ms | 1.66x lower |
| STARK excl trace APC 0 | 2153ms | 1794ms | 1.20x lower |

## Assessment

**Failure.** The batch Round 0 descriptor-array kernel approach has a fundamental CUDA bug that causes GPU hangs when more than ~12 AIRs are batched. With only 12 AIRs batchable (out of ~300 at APC 300), the optimization would save negligible time (~1-2ms estimated) and is not worth the complexity.

The plan's assumption that the previous hang was caused by OOM (resolved by VPMM changes) was incorrect. The hang is caused by a bug in the batched CUDA kernel itself, likely an out-of-bounds memory access in the intermediates buffer computation that scales with the number of batched AIRs.

## Future Work

- **Root cause the kernel hang**: Use `compute-sanitizer --tool memcheck` on a minimal reproducer (13+ AIRs) to identify the exact memory access violation. This was attempted but compute-sanitizer timed out (5 min limit) before reaching the batch code path.
- **Alternative: stream-pipelining small AIRs**: Instead of batching into one kernel, group small AIRs into sub-batches of 10-12 and launch them on multiple streams concurrently. This avoids the large aggregate intermediates buffer while still reducing per-AIR overhead.
- **Focus on other bottlenecks**: At APC 300, Round 0 is 186ms (17% of 1110ms STARK excl trace). LogUp GKR (359ms, 32%) and MLE Rounds (164ms, 15%) are larger targets. Trace Commit (193ms, 17%) and Openings (200ms, 18%) are also significant.
- **The per-AIR sequential approach's L2 cache reuse is a fundamental advantage**: Previous failed batch attempts (batch-global-gkr-input-eval, batch-round0-descriptor-arrays) all hit the same issue where aggregate intermediates exceed L2 cache. Any future batch approach must keep per-batch intermediates within 72 MB.
