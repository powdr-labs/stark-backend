# Report: 2026-04-13-1430-batch-mle-main-ptrs-upload

## Description

Replace per-trace `main_ptrs.to_device()` calls in MLE sumcheck rounds with a single batched device upload per round. Each MLE round iterates over ~310 traces (at APC 300) and calls `main_ptrs.to_device()` individually, issuing a separate `cudaMallocAsync` + `cudaMemcpyAsync` for a tiny buffer (~16-48 bytes) per trace. The plan estimated ~3000-4000 unnecessary allocation+copy+free triplets across all rounds and segments, each also acquiring the global `MEMORY_MANAGER` mutex. Consolidating all per-round main_ptrs into a single flat vector and uploading once was expected to save 20-40ms from the ~170ms MLE Rounds span at APC 300.

## Implementation

### Files changed

- **`crates/cuda-backend/src/logup_zerocheck/batch_mle.rs`**: Changed `TraceCtx.main_ptrs_dev: DeviceBuffer<MainMatrixPtrs<EF>>` to `main_ptrs_ptr: *const MainMatrixPtrs<EF>`. Updated all consumers (`ZerocheckMleBatchBuilder::new`, `LogupMleBatchBuilder::new`, `evaluate_zerocheck_batched`, `evaluate_single_logup`) to use `t.main_ptrs_ptr` instead of `t.main_ptrs_dev.as_ptr()`.

- **`crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs`**: Updated `ZerocheckMonomialBatch::new`, `ZerocheckMonomialParYBatch::new`, `LogupMonomialBatch::new` to use `t.main_ptrs_ptr` instead of `t.main_ptrs_dev.as_ptr()`. Updated doc comment referencing `main_ptrs_dev`.

- **`crates/cuda-backend/src/logup_zerocheck/mle_round.rs`**: Changed `evaluate_mle_constraints_gpu` and `evaluate_mle_interactions_gpu` parameter from `&DeviceBuffer<MainMatrixPtrs<EF>>` to `*const MainMatrixPtrs<EF>`, removing the `.as_ptr()` indirection inside.

- **`crates/cuda-backend/src/logup_zerocheck/mod.rs`**: In `sumcheck_polys_batch_eval()`:
  - **Case A (late_eval)**: Collect all main_ptrs into `all_late_main_ptrs: Vec<MainMatrixPtrs<EF>>` with per-trace offsets during the loop. After the loop, do a single `to_device()` and backfill pointers into each `TraceCtx.main_ptrs_ptr`. Keep the `DeviceBuffer` alive via `_keepalive_late_main_ptrs`.
  - **Case B (early_eval)**: Same pattern — collect into `all_early_main_ptrs` with offsets, single `to_device()` after Phase 3 loop, backfill pointers, keep alive via `_keepalive_early_main_ptrs`.

- **`crates/cuda-backend/src/tests.rs`**: Updated `test_monomial_vs_dag_equivalence` to use `main_ptrs_ptr: _main_ptrs_dev.as_ptr()` and keep the `DeviceBuffer` alive via `_main_ptrs_dev`.

### Key decisions

- Used the backfill approach: push TraceCtx with `main_ptrs_ptr: null()`, then overwrite after the batch upload. This avoids restructuring the existing loop logic.
- No deviations from the plan.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1417ms | 1421ms | -1034ms, 1.73x lower | +4ms, 1.00x (noise) |
| MLE Rounds | 180ms | 170ms | 165ms | -15ms, 1.09x lower | -5ms, 1.03x lower |
| Round 0 | 662ms | 295ms | 296ms | -366ms, 2.24x lower | +1ms, 1.00x (noise) |
| LogUp GKR | 790ms | 518ms | 533ms | -257ms, 1.48x lower | +15ms (noise) |

### APC 100

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2155ms | 1583ms | 1593ms | -562ms, 1.35x lower | +10ms, 1.01x (noise) |
| MLE Rounds | 150ms | 140ms | 135ms | -15ms, 1.11x lower | -5ms, 1.04x lower |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2147ms | 2150ms | -3ms, 1.00x | +3ms, 1.00x (noise) |
| MLE Rounds | 118ms | 114ms | 113ms | -5ms, 1.04x lower | -1ms, 1.01x lower |

### CUDA API call counts (from nsys)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| H2D copies | 30,035 | 24,436 | -5,599 |

## Assessment

**The optimization did NOT meet its success criteria.** MLE Rounds at APC 300 improved by only 5ms (170ms → 165ms), well below the plan's 10ms rollback threshold. STARK excl trace showed no net improvement (+4ms, within noise).

The nsys data confirms the batching is mechanically correct: ~5,600 fewer H2D copy calls. However, the per-call overhead of these small uploads was much lower than estimated:
- **Estimated**: ~15-20μs per call (mutex + cudaMallocAsync + cudaMemcpyAsync + cudaFreeAsync + HashMap ops)
- **Actual**: ~1μs per call (5ms improvement / 5600 eliminated calls ≈ 0.9μs each)

The discrepancy is likely because:
1. `cudaMallocAsync` with small allocations may be served from a pool cache, avoiding actual allocation
2. The MEMORY_MANAGER mutex is uncontended in the single-threaded MLE rounds path (unlike multi-stream Round 0/GKR)
3. `cudaMemcpyAsync` for 16-48 bytes is essentially free on the PCIe bus

No regression at APC 0 (MLE Rounds -1ms, STARK excl trace +3ms, both noise).

Per rollback criterion #1 (MLE Rounds improvement < 10ms), this optimization is **reverted**.

## Future Work

- **What worked**: The batching pattern is sound and the code change is minimal/clean. The `_keepalive` pattern for shared device buffers with raw pointer offsets is reusable.
- **Root cause of small impact**: The MLE rounds path is single-threaded, so mutex contention is near-zero. The dominant MLE Rounds cost is GPU kernel execution, not CUDA API overhead.
- **More impactful MLE Rounds targets**: The remaining 165ms is dominated by actual kernel execution time across ~16 invocations of `sumcheck_polys_batch_eval`. Further improvement would require:
  - Reducing the number of MLE rounds (algorithmic change)
  - Making the per-round GPU kernels faster (kernel optimization)
  - Batching the evaluation kernels themselves across traces (not just the upload)
- **Potential combination**: This optimization could be combined with multi-stream MLE rounds if that approach is pursued, where mutex contention would be higher and the per-call overhead would increase.
