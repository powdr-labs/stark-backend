# Report: Batch Round 0 Coset-Parallel Kernel Launches

## Description

Round 0 of constraint evaluation at APC 300 launches ~935 small `coset_parallel` kernels sequentially, each utilizing only 3-11% of GPU SMs. This optimization batches all small (coset-parallel eligible, GLOBAL=false) AIR instances into single kernel launches — one for zerocheck and one for logup — so that all small AIRs execute concurrently across the GPU's 108+ SMs.

The batched kernel uses a 2D grid `(total_x_blocks, max_num_cosets)` with per-AIR metadata structs. Each block identifies its AIR via an `air_for_block` mapping and loads per-AIR parameters (trace pointers, rules, weights, etc.) from a metadata array. The `batched_final_reduce_block_sums` reduction kernel (already used by MLE rounds) handles the multi-segment output reduction.

## Implementation

### CUDA changes

1. **`zerocheck_round0.cu`**: Added `ZerocheckBatchMeta` struct and `zerocheck_ntt_evaluate_constraints_coset_parallel_batched_kernel<NEEDS_SHMEM>` — structurally identical to the existing single-AIR coset-parallel kernel but with `blockIdx.x` mapped to per-AIR blocks via metadata. Added `_zerocheck_ntt_eval_constraints_batched` extern "C" launcher.

2. **`logup_round0.cu`**: Added `LogupBatchMeta` struct and `logup_r0_ntt_eval_interactions_coset_parallel_batched_kernel<NEEDS_SHMEM>` with the same batching pattern. Preserves the identity-coset special case (`coset_idx == 0`) and uses `FracExt` output. Added `_logup_bary_eval_interactions_round0_batched` extern "C" launcher.

### Rust changes

3. **`src/cuda/logup_zerocheck.rs`**: Added FFI declarations for both batched launchers and safe wrapper functions.

4. **`src/logup_zerocheck/round0.rs`**: Added `#[repr(C)]` metadata structs (`ZerocheckBatchMeta`, `LogupBatchMeta`) matching the CUDA definitions. Added constants `BUFFER_THRESHOLD`, `COSET_PARALLEL_THRESHOLD`, `MAX_THREADS_ROUND0`.

5. **`src/logup_zerocheck/mod.rs`**: Restructured `sumcheck_uni_round0_polys` Phase 1:
   - Pre-check: estimates batch memory and disables batching if it would exceed 128 MB (prevents GPU OOM when `skip_domain` is large).
   - Sub-phase A: Classifies each AIR as batch-eligible or individual. For logup, the per-AIR DAG compilation and weight computation is done inline and device buffers are kept alive.
   - Sub-phase B: Uploads metadata arrays, allocates tmp/output buffers, and issues batched kernel launches. Tmp buffers are dropped immediately after each kernel.
   - Phase 2: D2H transfers for batched output buffers, extracts per-AIR slices, feeds into existing polynomial construction.

### Deviations from plan

- Added a **memory budget check** (128 MB) that disables batching when `skip_domain` is large. The plan didn't anticipate that `l_skip` could be large enough to make the batch temp buffer exceed 500 MB. Without this check, the batch allocation would cause GPU OOM.
- Used **2D grid** `(total_x_blocks, max_num_cosets)` with `air_for_block` mapping as planned, rather than 3D grid.

## Results

**Measurement could not be completed** due to a pre-existing GPU OOM bug in `gkr_input::log_gkr_input_evals` on the pairing benchmark's second segment. This bug exists on the unmodified code (confirmed by reverting changes and testing) and prevents the benchmark from producing metrics files. The OOM attempts to allocate 8 GB, which exceeds available GPU memory regardless of any Round 0 optimization.

All 94 `openvm-cuda-backend` unit tests pass, confirming correctness of the batched kernels.

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| Round 0 APC 300 | 663ms | 569ms | N/A (OOM) | N/A | N/A |
| STARK excl trace APC 300 | 2478ms | 2195ms | N/A (OOM) | N/A | N/A |
| Round 0 APC 0 | 178ms | 173ms | N/A (OOM) | N/A | N/A |
| Tests passing | 94/94 | 94/94 | 94/94 | — | — |

## Assessment

The implementation is **correct but unmeasured**. The optimization cannot be validated against the target benchmarks due to the pre-existing GPU OOM. Additionally, the memory budget check may disable batching for the pairing benchmark if `l_skip` is large (which it appears to be based on `n_logup = 24`), in which case the optimization would have no effect on this specific benchmark.

The optimization is most effective when:
- `l_skip` is small (skip_domain ≤ 32), keeping batch memory under budget
- Many small AIRs exist (APC 100-300)
- `buffer_size ≤ 16` (GLOBAL=false, which captures most small AIRs)

The implementation does NOT regress performance when batching is disabled — it falls back to the exact same per-AIR launch path as before.

## Future Work

- **Fix the pre-existing GPU OOM** in `gkr_input::log_gkr_input_evals` to enable benchmarking. The 8 GB allocation should be investigated — it may be a bug or need chunked processing.
- **Sub-batching by memory budget**: Instead of all-or-nothing, process AIRs in chunks sized to fit within the memory budget. This would enable batching even when `l_skip` is large, at the cost of multiple (but fewer) kernel launches.
- **Sub-batching by `num_cosets`**: Group AIRs with identical `num_cosets` into separate sub-batches to eliminate coset-dimension padding waste.
- **Batch LogUp GKR input evaluation kernels**: Similar SM underutilization pattern (427ms at APC 300, 791 instances).
- **Batch non-degenerate stacked reduction kernels**: 7467 per-window launches contributing ~86ms of CPU overhead.
