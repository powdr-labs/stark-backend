# Plan: Re-enable Batch Round 0 Descriptor-Array Kernels for Small AIRs

## Goal

Reduce Round 0 wall-clock time at APC 300 by batching the eval kernels for ~250 small coset-parallel AIRs per segment into 2-4 descriptor-array kernel launches on the default stream, then letting Phase 2 worker threads handle extraction for batched AIRs alongside full processing of the remaining ~60 large AIRs. This eliminates ~500 eval kernel launches per segment plus the per-AIR CPU overhead (`SymbolicConstraints::from`, DAG build, rule encode) for batched AIRs, without serializing extraction onto the critical path.

Current: Round 0 = 179ms (90ms/seg). Expected after: Round 0 = 145-165ms (73-83ms/seg). Estimated net savings: 15-35ms on STARK excl trace (1106ms → 1071-1091ms).

Previous implementations (commits `0c699f47`, `bcb1c0d2`, `0d1bb2f9`) were reverted due to GPU OOM at APC 100/300. The OOM has since been resolved (VPMM 16 MiB page size + PRUNE_SHMEM_KERNELS), confirmed by the current APC 300 run at 1106ms.

## Current Code Path

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

### Phase 1 (single-threaded, lines 1022-1198)
Builds `Vec<Round0AirWorkItem>` for all AIRs with proving key refs, trace pointers, pre-computed buffer sizes, eq_3bs, and `Round0ExtractTables`. Items sorted by descending height. Percentile buffer sizes computed for per-thread pre-allocation.

### Phase 1.5 (overlapped background, lines 1341-1382)
Background thread uploads `d_eq_3b_per_trace` and precomputes `logup_combinations`, concurrent with Phase 2.

### Phase 2 (multi-threaded, lines 1319-1413)
8 OS threads, per-thread CUDA streams, round-robin assignment. Per AIR: `process_air_round0()` at `round0.rs` does:
1. `evaluate_round0_zerocheck_gpu()` → zerocheck eval + reduction (~2 kernel launches)
2. `evaluate_round0_interactions_gpu()` → CPU rule construction (`SymbolicConstraints::from`, DAG, encode, H2D) + logup eval + reduction (~2 kernel launches)
3. `_round0_extract_zerocheck_poly()` → GPU extraction to `d_batch_array` (1 kernel launch)
4. `_round0_extract_logup_polys()` → GPU extraction to `d_batch_array` (1 kernel launch)

At APC 300: ~312 AIRs/seg, ~6 kernel launches/AIR = ~1872 launches/seg. Wall clock: ~90ms/seg.

### Why it's slow at APC 300
The ~250 small AIRs (coset-parallel, height ≤ 256) each take ~0.2ms CPU work + ~0.1ms GPU eval + ~0.1ms extraction = ~0.4ms total. Across 8 threads: 31 per thread × 0.4ms ≈ 12ms. The remaining ~60 large AIRs take ~3-10ms each, ~8 per thread, total ~24-40ms. But measured per-segment Round 0 is ~90ms, with the difference attributable to CUDA driver overhead, stream scheduling, and memory pool contention across 1872 kernel launches.

## Changes

### Change 0: Diagnose the previous GPU hang

**Before any code changes**, reproduce and diagnose the 12-minute GPU hang from commit `0d1bb2f9`:
1. Cherry-pick `0d1bb2f9` onto a throwaway branch
2. Build with debug symbols
3. Run: `compute-sanitizer --tool memcheck -- $PROVE_BIN prove --artifact apc300.cbor --input 0 --recursion`
4. If memcheck identifies an OOB access: note the location and fix in the new implementation
5. If no memcheck error: the hang was likely caused by the OOM (now resolved) — this de-risks the approach
6. Revert the cherry-pick

### Change 1: Store logup Round 0 rules in proving key at keygen time

**File:** `crates/cuda-backend/src/pkey.rs`

Add a `LogupRound0Rules` struct to `AirDataGpu` containing challenge-independent data only:
- `d_rules: DeviceBuffer<u128>` — encoded DAG rules on GPU
- `rules_len: usize` — number of rules
- `buffer_size: u32` — logup intermediates buffer size (already present as `logup_round0_buffer_size`)
- `dag_idx_to_rule_idx: Vec<usize>` — mapping from interaction index to rule index

**NOT included** (challenge-dependent, computed per-proof):
- Interaction weight vectors (`numer_weights`, `denom_weights`, `denom_sum_init`) — these depend on `beta_pows` and `eq_3bs` and must be computed at proving time

These rules are used **ONLY by the batch path** (Change 4). Phase 2's `evaluate_round0_interactions_gpu` is unchanged and continues to reconstruct rules per-AIR, preserving the GPU pacing.

### Change 2: CUDA Batched Kernels

**Files:**
- `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`
- `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

Replicate the descriptor-array kernel design from reverted commit `0d1bb2f9`:

- **`batched_zerocheck_r0_coset_parallel_kernel`**: Block-to-AIR mapping via `Round0BlockCtx` array. Each `Round0ZcCtx` descriptor has per-AIR parameters (main/preprocessed pointers, constraint config, eq_xis, sels, intermediates offset, output offset). Only `GLOBAL=true, NEEDS_SHMEM=false` instantiation. Uses the existing single-AIR kernel body with a descriptor lookup preamble.

- **`batched_logup_r0_coset_parallel_kernel`**: Same pattern with `Round0LogupCtx` descriptors. Uses keygen-time rules from Change 1 (device pointers in descriptor) plus proving-time interaction weights (computed in Change 4).

- **`batched_final_reduce_block_sums`**: Reuse the existing reduction pattern from `batch_mle.cu`.

**Null intermediates safety:** Descriptors for AIRs with `buffer_size == 0` get the arena base pointer as their intermediates pointer. The kernel guards intermediates writes with `if (buffer_size > 0)`. The `cudaErrorIllegalAddress` from commit `0c699f47` was caused by unconditional intermediates dereference; this guard prevents it.

### Change 3: FFI Bindings

**File:** `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `#[repr(C)]` Rust structs:
- `Round0BlockCtx { air_idx: u32, local_block_idx_x: u32 }`
- `Round0ZcCtx` — main_trace_ptr, preprocessed_ptr, buffer_size, num_x, skip_domain, num_cosets, eq_xis_ptr, sels_ptr, intermediates_offset, output_offset
- `Round0LogupCtx` — trace ptrs, d_rules_ptr, rules_len, d_numer_weights_ptr, d_denom_weights_ptr, buffer_size, d_eq_3b_ptr, intermediates_offset, output_offset

Add extern "C" declarations and safe Rust wrappers.

### Change 4: Batch Orchestration Module

**File:** `crates/cuda-backend/src/logup_zerocheck/round0_batched.rs` (new file)

All public functions `#[inline(never)]`.

**`identify_batchable_airs(work_items, l_skip) -> Vec<bool>`**
- Batchable if: (a) coset-parallel (`num_x * skip_domain < 32768`), (b) intermediates within budget (`< 16M Fp` per AIR), (c) AIR has zerocheck constraints or logup interactions
- Returns all-false if fewer than `MIN_BATCHABLE_AIRS = 50` qualify
- **Also** computes and logs the aggregate intermediates size for the reviewer's L2 cache concern: `info!("Batch Round 0: {} AIRs, aggregate intermediates: {} MB", count, total_bytes / 1_000_000)`

**`batch_round0_eval_only(work_items, batch_mask, pk, d_batch_ptr, ...) -> Result<BatchEvalState>`**
This function runs on the default stream and performs **only the eval kernels**, NOT extraction. Returns a `BatchEvalState` containing per-AIR result pointers and metadata needed for extraction.

Steps:
1. Group batchable AIRs by `num_cosets_zc` (constraint_degree − 1)
2. For each group:
   a. Compute total intermediates size (sum of per-AIR `zc_intermed_cap + logup_intermed_cap`)
   b. Allocate shared intermediates arena `DeviceBuffer`
   c. Build `Round0ZcCtx` descriptors with arena offsets
   d. Build `Round0BlockCtx` mapping
   e. Upload descriptors to GPU
   f. Launch `batched_zerocheck_r0_coset_parallel_kernel` + reduction
   g. Compute proving-time interaction weights per AIR (CPU: ~10μs/AIR using `beta_pows` and `eq_3bs`)
   h. Upload `d_numer_weights`, `d_denom_weights` to GPU
   i. Build `Round0LogupCtx` descriptors using keygen-time rules from `pk.per_air[air_idx].other_data.logup_round0_rules`
   j. Launch `batched_logup_r0_coset_parallel_kernel` + reduction
3. Return `BatchEvalState` that owns:
   - Per-group `DeviceBuffer<EF>` for zerocheck output (post-reduction results)
   - Per-group `DeviceBuffer<Frac<EF>>` for logup output (post-reduction results)
   - Per-AIR metadata: which group it belongs to, its local index within the group, and the stride (so extraction can index into the group's output buffer at `base + local_idx * stride`)
   
   Note: the intermediates arena is scratch space freed after eval+reduce. The output buffers are separate and must stay alive until all Phase 2 extraction is complete. Since `BatchEvalState` is created before `std::thread::scope` and dropped after the scope exits (all worker threads joined), the output buffers remain valid for all extraction launches.

**Key design: extraction is NOT done here.** Extraction is deferred to Phase 2 worker threads (Change 5).

### Change 5: Integration

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Insert the batch eval between Phase 1 (after `current_stream_sync()` at line 1202) and the `num_threads` branch:

```rust
// Batch eval for small AIRs (default stream)
let batch_mask = round0_batched::identify_batchable_airs(&work_items, l_skip);
let batch_eval_state = if batch_mask.iter().any(|&b| b) {
    let state = round0_batched::batch_round0_eval_only(
        &work_items, &batch_mask, self.pk,
        d_batch_array.as_mut_ptr(), l_skip,
        &self.eq_3b_per_trace, &self.beta_pows,
    )?;
    current_stream_sync()?; // batch eval results visible to all streams
    Some(state)
} else {
    None
};
```

In Phase 2 worker threads, handle batched AIRs with extraction-only:
```rust
for &item_idx in &thread_items[thread_id] {
    if batch_mask[item_idx] {
        // Batch eval already done; only run extraction
        let w = &work_items[item_idx];
        round0_batched::extract_batched_air(
            w, &batch_eval_state.as_ref().unwrap(),
            batch_ptr_val as *mut EF, &extract_tables,
        )?;
    } else {
        // Full per-AIR processing (eval + extraction, unchanged)
        process_air_round0(&work_items[item_idx], &mut bufs, batch_ptr_val as *mut EF, &extract_tables)?;
    }
}
```

**Single-threaded path** (`num_threads <= 1`, lines 1280-1318): The batch path is skipped because `identify_batchable_airs` returns all-false (< 50 AIRs at APC 0). If somehow triggered, the single-threaded path would also check `batch_mask` — but this is a no-op since the mask is all-false.

### Change 6: Visibility

Widen `Round0AirWorkItem` and `Round0ExtractTables` to `pub(super)` and place `round0_batched.rs` as a submodule of `logup_zerocheck`.

## Savings Breakdown (per segment at APC 300)

| Component | Current | After | Change | Source |
|-----------|---------|-------|--------|--------|
| Batch eval (default stream) | 0ms | 5-8ms | +5-8ms | New: descriptor build + batched kernel |
| Small AIR CPU work in Phase 2 | ~6ms/thread | 0ms | −6ms | Eliminated: skip SymbolicConstraints::from, rule build |
| Small AIR eval launches in Phase 2 | ~5ms/thread | 0ms | −5ms | Eliminated: ~31 AIRs × 6 launches × 10μs |
| Small AIR extraction in Phase 2 | ~3ms/thread | ~3ms/thread | 0ms | Unchanged: extraction runs on worker threads |
| Large AIR processing in Phase 2 | ~40ms (max thread) | ~40ms (max thread) | 0ms | Unchanged |
| CUDA overhead (scheduling, pool) | ~36ms | ~20-30ms | −6-16ms | Fewer kernel launches = less driver overhead |
| **Total per segment** | **~90ms** | **~68-81ms** | **−9-22ms** | |
| **Total (2 segments)** | **179ms** | **135-162ms** | **17-44ms** | |

The improvement is driven by: (1) eliminating ~11ms/thread of small-AIR CPU+launch overhead, and (2) reducing systemic CUDA driver overhead from fewer total kernel launches (~1872 → ~1372 per segment, a 27% reduction).

## Invariants

1. **Correctness**: Batched eval produces the same coset-parallel NTT + constraint evaluation as per-AIR eval. Extraction uses the identical per-AIR kernels and writes to the same `d_batch_array` offsets.

2. **APC 0 guard**: `MIN_BATCHABLE_AIRS = 50` disables batching (APC 0 has ~20 AIRs/seg).

3. **Phase 2 pacing preserved**: Phase 2 is unchanged for non-batched (large) AIRs. Per-AIR `SymbolicConstraints::from()` + rule building remain as GPU pacing for large AIRs.

4. **Stream ordering**: Batch eval on default stream → `current_stream_sync()` → Phase 2 on per-thread streams. Batch eval results are visible before any extraction runs.

5. **Extraction concurrency preserved**: Extraction for batched AIRs runs on Phase 2 worker threads (not the default stream), maintaining 8-way concurrent extraction just like the current code.

6. **L2 cache budget**: The aggregate intermediates arena for ~250 small AIRs is logged at runtime. Expected size: 10-50MB (well within 72MB L2). If > 72MB, the L2 thrashing documented in `batch-global-gkr-input-eval` would apply and we'd see a regression — caught by measurement.

7. **Null pointer safety**: Descriptors with `buffer_size == 0` get arena base pointer + kernel guards writes with `if (buffer_size > 0)`.

## Measurement Plan

1. **Step 0**: Run `compute-sanitizer --tool memcheck` on cherry-picked `0d1bb2f9` at APC 300
2. Build: `cd /home/georg/powdr && cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"`
3. APC 0: verify Round 0 ≤ 180ms, STARK excl trace ≤ 1810ms
4. APC 300: expect Round 0 135-162ms (from 179ms), STARK excl trace 1062-1089ms
5. APC 100 for completeness
6. If Round 0 savings < 10ms: add tracing spans inside `batch_round0_eval_only` to identify bottleneck (descriptor build vs eval vs weight computation)

## Rollback Criteria

- **Hard revert**: APC 0 STARK excl trace regresses >20ms vs 1784ms
- **Hard revert**: Verification failure at any APC config
- **Hard revert**: GPU OOM, `cudaErrorIllegalAddress`, or GPU hang at any APC config
- **Soft revert**: APC 300 Round 0 improvement <10ms
- **Soft revert**: APC 300 STARK excl trace improvement <15ms
