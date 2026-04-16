## Goal

Fix the batch Round 0 descriptor-array optimization so it works at APC 300 without the 12+ minute regression introduced by commit 0d1bb2f9. Round 0 is the worst-scaling component of STARK excl trace: 177ms at APC 0 vs 187ms at APC 300 (0.95x, no scaling despite 2.35x fewer cells). The batched approach eliminates ~500 per-AIR kernel launches for small AIRs while preserving the multi-stream path for large AIRs. Expected improvement: 5-15ms on Round 0 at APC 300, primarily from eliminating per-AIR kernel launch and H2D overhead for small AIRs, with possible additional GPU occupancy benefits from batched execution.

## Current Code Path

### Round 0 evaluation flow (mod.rs, `sumcheck_uni_round0_polys`)

1. **Phase 1** (single-threaded, lines 900-1190): Build work items, compute eq_xis, prepare selectors, allocate d_batch_array.

2. **Phase 1b** (single-threaded, lines 1194-1210): The batch path from commit 0d1bb2f9:
   - `identify_batchable_airs()` (round0_batched.rs:42) — marks AIRs with coset-parallel mode and intermediates < 16M Fp as batchable; requires >= 50 qualifying AIRs.
   - `batch_round0_small_airs()` (round0_batched.rs:192) — for each batchable AIR, calls `build_logup_rules_for_air()` (round0_batched.rs:96), which:
     - Calls `SymbolicConstraints::from(&single_pk.vk.symbolic_constraints)` to convert compact VK representation to full expression trees
     - Builds a `SymbolicDagBuilder` and DAG
     - Creates `SymbolicRulesGpu::new(&dag, true)` — computes rules and buffer sizes
     - Computes numer/denom weights
     - Uploads rules and weights to GPU (3 H2D transfers per AIR)
   - Groups by num_cosets, launches batched zerocheck and logup kernels
   - Runs per-AIR extraction (writes to d_batch_array via GPU kernels)

3. **Phase 2** (multi-threaded, lines 1212-1400): For each non-batchable AIR, calls `process_air_round0()` (mod.rs:234), which:
   - Also calls `SymbolicConstraints::from()` per AIR (line 241) — this is needed for both zerocheck (via `evaluate_round0_constraints_gpu`) and logup (via `evaluate_round0_interactions_gpu`)
   - Builds logup DAG and rules in `evaluate_round0_interactions_gpu()` (round0.rs:162)
   - Launches per-AIR zerocheck + logup kernels
   - Runs per-AIR GPU extraction

### Root cause of the regression

The 12+ minute APC 300 hang has two possible causes, which must be disambiguated:

**(a) CPU-side rule construction overhead**: `batch_round0_small_airs()` runs single-threaded, calling `SymbolicConstraints::from()` + DAG + `SymbolicRulesGpu::new()` + weight computation + 3 H2D uploads per batchable AIR. However, the non-batched path processes all ~300 AIRs across 8 threads in ~187ms, implying per-AIR rule construction is ~0.1ms. At 300 AIRs single-threaded, this would be ~30ms — not 12 minutes.

**(b) Batched CUDA kernel hang**: The more likely cause. After rule construction, `batch_round0_small_airs` launches batched zerocheck and logup CUDA kernels via `evaluate_batched_zerocheck_group` and `evaluate_batched_logup_group`. If these batched kernels have a bug for certain APC-generated AIR configurations (e.g., incorrect grid dimensions, out-of-bounds intermediates access, or infinite loop in DAG evaluation), the GPU would hang while the host waits.

The tracing timestamps added in Step 5 will immediately identify which cause applies. If CPU setup completes in < 1s and the hang follows, the batched kernel is the root cause and must be debugged (check `total_blocks`, `intermediates_size`, kernel termination conditions with `compute-sanitizer`).

### Why the same per-AIR work is done twice

Both the batch path (`build_logup_rules_for_air`) and the non-batched path (`evaluate_round0_interactions_gpu`) independently build SymbolicDagBuilder, SymbolicRulesGpu, and compute numer/denom weights from scratch. Neither reuses the other's output. The `AirDataGpu.logup_round0_buffer_size` field (pkey.rs:35) stores only the buffer size, not the full rules.

## Changes

### Step 1: Revert commit 0d1bb2f9

- File: git operation
- Action: `git revert 0d1bb2f9` to cleanly revert "Batch Round 0 descriptor arrays with PRUNE_SHMEM_KERNELS"
- This restores the vpmm-bulk-page-creation state (last known working state)
- Verify: APC 0 and APC 300 benchmarks produce correct proofs with expected timing (~1806ms / ~1123ms STARK excl trace)

### Step 2: Re-apply PRUNE_SHMEM_KERNELS separately

- File: `crates/cuda-backend/build.rs`
- Action: Add `.flag("-DPRUNE_SHMEM_KERNELS")` to the cc::Build chain (line 26 in current code)
- File: `crates/cuda-backend/cuda/include/utils.cuh`
- Action: Add the `#ifdef PRUNE_SHMEM_KERNELS` guard for `DISPATCH_BOOL_PAIR` and `DEFINE_DISPATCH_N_B1_B2` macros (same changes as in commit 0d1bb2f9, lines 50-76)
- Why: Reduces cubin from ~9.3M to ~6.5M, which was needed to avoid GPU OOM at APC 100/300 after CUDA rebuilds. Safe because l_skip=4 gives skip_domain=16 <= WARP_SIZE=32.
- Verify: APC 0 and APC 300 benchmarks still produce correct proofs with same timing

### Step 3: Hoist logup rule construction from per-AIR to Phase 1

Instead of building logup rules inside `batch_round0_small_airs` (single-threaded for ~300 AIRs) or inside `evaluate_round0_interactions_gpu` (per-AIR in Phase 2), pre-build them once during Phase 1 work-item preparation.

- File: `crates/cuda-backend/src/logup_zerocheck/mod.rs`
- Action: Add a `logup_rules: Option<LogupRulesOnDevice>` field to `Round0AirWorkItem` (line 209-232). Define `LogupRulesOnDevice` as:
  ```rust
  pub(super) struct LogupRulesOnDevice {
      pub d_rules: DeviceBuffer<u128>,  // encoded rules (RuleWithFlag<F>::Encoded = u128)
      pub rules_len: usize,
      pub buffer_size: u32,
      pub d_numer_weights: DeviceBuffer<EF>,
      pub d_denom_weights: DeviceBuffer<EF>,
      pub denom_sum_init: EF,
  }
  ```
- Action: During Phase 1 work-item construction (the loop at ~lines 1000-1100), after computing `eq_3bs`, build the logup rules for each AIR. The `SymbolicConstraints::from` result is used as a local temporary and can be dropped after rule construction:
  ```rust
  let logup_rules = if !eq_3bs.is_empty() {
      let symbolic = SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
      // symbolic is consumed here and dropped at end of block.
      // LogupRulesOnDevice captures only encoded device buffers and weights.
      Some(build_logup_rules_on_device(&symbolic, &beta_pows, &eq_3bs)?)
  } else {
      None
  };
  ```
  Store in the work item.
- Why: Phase 1 is already single-threaded and builds per-AIR data (eq_xis, selectors, lambda_pows). Adding logup rules here avoids redundant computation in the batch path.
- Note: `process_air_round0` (mod.rs:241) still calls `SymbolicConstraints::from` per AIR because the zerocheck constraint path (`evaluate_round0_constraints_gpu`) also requires `symbolic`. Only the logup interaction rule construction is eliminated from Phase 2.

### Step 4: Modify `evaluate_round0_interactions_gpu` to accept precomputed rules

- File: `crates/cuda-backend/src/logup_zerocheck/round0.rs`
- Action: Change `evaluate_round0_interactions_gpu` to accept an optional `precomputed_rules: Option<&LogupRulesOnDevice>` parameter. When `Some`, skip DAG construction and weight computation (lines 193-239) and use the precomputed `d_rules`, `d_numer_weights`, `d_denom_weights`, `buffer_size`, and `denom_sum_init` directly. When `None`, fall back to the current behavior (build DAG from `symbolic`).
- Action: Update `process_air_round0` in `mod.rs` to pass `w.logup_rules.as_ref()`.
- Why: This eliminates per-AIR logup DAG construction from the multi-threaded Phase 2 while keeping the fallback path for callers that don't have precomputed rules.

### Step 5: Re-implement batch path using precomputed rules

- File: `crates/cuda-backend/src/logup_zerocheck/round0_batched.rs`
- Action: Replace the `build_logup_rules_for_air()` call in `batch_round0_small_airs()` (lines 255-259) with reading from the precomputed `w.logup_rules` field. Update the `BatchAirDesc` struct to store device pointers to the precomputed rule data instead of owning separate copies:
  ```rust
  // In the descriptor loop:
  let logup_rules_ref = w.logup_rules.as_ref(); // Already built in Phase 1
  // Store d_rules.as_raw_ptr(), d_numer_weights.as_ptr(), etc. in BatchAirDesc
  ```
- Action: Update `BatchAirDesc.logup_rules` field type from `Option<LogupRulesForAir>` (owning) to pointers/references from `LogupRulesOnDevice`.
- Action: Remove the `build_logup_rules_for_air()` function entirely.
- Action: Add `tracing::info!` timestamps before and after the batched kernel launches in `batch_round0_small_airs` to diagnose if any remaining slowness is CPU-side or GPU-side.
- Why: Eliminates all redundant CPU-side DAG construction and device uploads from the batch path. If the batched CUDA kernels themselves are the source of the hang (not CPU setup), the timestamps will reveal this immediately.

## Invariants

1. **Correctness**: The logup rules (DAG, encoded rules, numer/denom weights) must be bit-identical whether computed in Phase 1 or per-AIR. The DAG construction is deterministic given the same `VkSymbolicConstraints`, `beta_pows`, and `eq_3bs` inputs.

2. **Device pointer lifetime**: The `LogupRulesOnDevice` device buffers must live as long as the batch kernel or per-AIR kernel that references them. Since they are stored in `Round0AirWorkItem` which lives for the entire scope of `sumcheck_uni_round0_polys`, this is guaranteed.

3. **Pointer safety in DAG construction**: `SymbolicDagBuilder.expr_to_idx` uses raw pointers (`*const SymbolicExpression`) as keys. The `SymbolicConstraints` value from which these pointers derive must remain alive during the entire `build_logup_rules_on_device` call. The code snippet in Step 3 creates `symbolic` as a local that lives for the duration of the rule construction block, satisfying this requirement.

4. **Batch/non-batch equivalence**: The batch path and per-AIR path must produce identical results in `d_batch_array`. This is testable by running with the batch threshold set to 0 (force all AIRs through non-batched path) and comparing metrics.

5. **No APC 0 regression**: At APC 0 (~20 AIRs per segment), the batch path is inactive (< 50 qualifying AIRs). The only change affecting APC 0 is that logup rules are precomputed in Phase 1 instead of per-AIR in Phase 2. Since Phase 1 is single-threaded in both cases, the overhead is the same.

6. **PRUNE_SHMEM_KERNELS safety**: With l_skip=4, skip_domain=16 <= WARP_SIZE=32, the NEEDS_SHMEM=true variants are never dispatched. This applies to both batched and per-AIR kernels.

7. **`SymbolicConstraints::from` still used per-AIR for zerocheck**: `process_air_round0` continues to call `SymbolicConstraints::from` per AIR (mod.rs:241) because the zerocheck path requires it. This optimization only eliminates the redundant logup rule construction, not the `SymbolicConstraints::from` call itself. The per-AIR `SymbolicConstraints::from` overhead is ~0.1ms per AIR, distributed across 8 threads.

## Measurement Plan

1. After Step 1 (revert): Run full benchmark suite (APC 0, 100, 300). Verify results match pre-batch-round0 state (~1806ms / ~1301ms / ~1123ms).

2. After Step 2 (PRUNE_SHMEM_KERNELS): Run full benchmark suite. Verify no regression.

3. After Steps 3-5 (re-implemented batch path): Run full benchmark suite.
   - **APC 0**: Expect no regression (batch path inactive). STARK excl trace ~1806ms.
   - **APC 300**: Expect Round 0 improvement of 5-15ms (187ms → ~172-182ms). STARK excl trace ~1108-1118ms. Primary source of improvement: elimination of ~500 per-AIR kernel launches and ~900 H2D transfers. Additional potential improvement from batched kernel GPU occupancy benefits.
   - **APC 100**: Expect Round 0 improvement of 3-10ms.

4. Run `cargo nextest run -p openvm-cuda-backend --test-threads=4` to verify all tests pass.

5. Run nsight profile on APC 300 to verify kernel launch count reduction (expect ~500 fewer logup+zerocheck launches per segment for small AIRs).

## Rollback Criteria

- If APC 300 Round 0 does not improve measurably (within noise): revert Steps 3-5, keep Steps 1-2. The revert and PRUNE_SHMEM_KERNELS changes are independently valuable.
- If APC 0 regresses by more than 20ms: revert all changes.
- If the `tracing::info!` timestamps in Step 5 reveal the batched CUDA kernel itself hangs (CPU work completes quickly but kernel blocks for minutes): do not re-apply the batch path. Stop after Step 4 (precomputed logup rules for the non-batched path), document the kernel bug, and investigate separately.
- If the PRUNE_SHMEM_KERNELS flag (Step 2) alone causes any benchmark regression or test failure: do not apply it; document the failure and proceed with Steps 3-5 without it.
