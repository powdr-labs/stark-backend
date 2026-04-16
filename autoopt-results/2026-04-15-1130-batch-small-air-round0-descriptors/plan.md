# Plan: Batch Small AIR Round 0 Descriptors

## Goal

Reduce Round 0 wall time at APC 300 from ~180ms to ~140-155ms by batching small AIRs into descriptor-array CUDA kernel launches, eliminating per-AIR kernel launch overhead and per-AIR H2D copies for the majority of AIRs. This targets the worst-scaling component of STARK excl trace (0.98x scaling from APC 0 to APC 300).

## Current Code Path

Round 0 is orchestrated by `sumcheck_uni_round0_polys` in `crates/cuda-backend/src/logup_zerocheck/mod.rs` (function signature at line 922, body phases from line 1022 onward). It is called from `prove_zerocheck_and_logup_gpu` at line 510 of the same file.

### Phase 1: Work item preparation (lines 1022-1152)
For each present AIR, build a `Round0AirWorkItem` with:
- Pre-computed buffer sizes via `_zerocheck_r0_intermediates_buffer_size()` (line 1045) and `_logup_r0_intermediates_buffer_size()` (line 1067)
- Extract transform tables per unique constraint degree
- Batch array offsets for output polynomials

### Phase 2: Multi-stream dispatch (lines 1204-1420)
When AIR count >= 100 (APC 300 has ~312 per segment):
- Pre-allocate per-thread `Round0ThreadBuffers` at 95th percentile sizes (lines 1229-1273)
- Round-robin work distribution across 8 threads (line 1330)
- Background precompute thread for logup combinations (lines 1341-1382)
- Each worker thread calls `process_air_round0()` per AIR sequentially (lines 1385-1406)

### Per-AIR processing in `process_air_round0` (lines 244-389)
1. Zerocheck evaluation: launch `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel` (line 291)
2. Zerocheck extraction: launch `_round0_extract_zerocheck_poly` GPU kernel (line 314)
3. Logup evaluation: build interaction DAG on CPU (lines 193-239), launch `logup_r0_ntt_eval_interactions_coset_parallel_kernel` (line 346)
4. Logup extraction: launch `_round0_extract_logup_polys` GPU kernel (line 373)

### Why Round 0 doesn't scale
At APC 300: 623 AIRs (312 per segment), 8 threads, ~78 AIRs per thread.
Per-AIR overhead: 2-4 kernel launches + H2D copies (`d_main_parts.to_device()`, weight vectors).
Per-segment wall time at APC 300: ~90ms with 8 threads. Per-segment at APC 0: ~36ms with 1 thread (20 AIRs).
The multi-stream overhead is ~30-40ms per segment beyond ideal GPU kernel time, from kernel launch gaps, H2D serialization, and imperfect SM sharing between streams.

## Step 0: Template Memory Audit (before implementation)

Before writing any batched kernel code, quantify the GPU memory cost of additional CUDA template instantiations:
1. Compile the current cuda-backend and measure .cubin total size via `cuobjdump --dump-elf-section`.
2. Add placeholder batched kernel instantiations (skeleton only) and re-measure.
3. Compare against GPU memory headroom at APC 300 (current peak: ~14.2 GiB in pool, 24 GiB total).
4. If memory headroom < 1 GiB after adding batched instantiations, prune `NEEDS_SHMEM=true` variants first (the benchmark uses `l_skip=4`, so `skip_domain=16 <= WARP_SIZE` and `needs_shmem` is always false).

**If the memory audit shows OOM is unavoidable even after pruning NEEDS_SHMEM=true, stop here.** The optimization cannot proceed on this GPU. Document findings and select a different optimization.

## Changes

### Change 1: CUDA batched Round 0 kernels

**Files**: `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`, `logup_round0.cu`

Add batched descriptor-array kernel variants. Reference reverted commits `0c699f47` and `bcb1c0d2` for the kernel design:

**Zerocheck descriptor struct** (`Round0ZcCtx`):
```c
struct Round0ZcCtx {
    const Fp *selectors_cube;
    const Fp *preprocessed;
    const Fp *const *main_parts;
    const FpExt *eq_cube;
    const FpExt *d_lambda_pows;
    const Fp *public_values;
    const Rule *d_rules;
    size_t rules_len;
    const size_t *d_used_nodes;
    size_t used_nodes_len;
    size_t lambda_len;
    uint32_t buffer_size;
    Fp *d_intermediates;     // Points into shared intermediates arena
    uint32_t skip_domain;
    uint32_t num_x;
    uint32_t height;
    Fp g_shift;
};
```

**Logup descriptor struct** (`Round0LogupCtx`): same pattern with logup-specific fields (numer_weights, denom_weights, denom_sum_init).

**Grid design**: use the `BlockCtx` mapping pattern from the MLE batched kernels. Grid is `dim3(total_blocks, num_cosets)` where `total_blocks = sum(blocks_per_air)` across all batchable AIRs. Each AIR may use 1 or more blocks depending on `num_x`:
- Small AIRs with `num_x <= 256` (height <= skip_domain * 256): 1 block
- Larger batchable AIRs: `ceil(num_x * skip_domain / block_size)` blocks

Each block reads `BlockCtx { air_idx, local_block_idx_x }` to find its descriptor and work slice. Per-AIR partial sums are reduced by `batched_final_reduce_block_sums` (already in `sumcheck.cuh`) when an AIR uses multiple blocks. For single-block AIRs, the block writes its output directly (no reduction needed). The block count per AIR is stored in `air_block_offsets` for the reduction kernel.

**Batched extraction**: a new `batched_round0_extract_kernel` where grid.x = num_batchable_airs, each block applies its AIR's transform matrix to convert evaluation sums to polynomial coefficients. Each block reads its transform matrix pointer and output offset from a descriptor array.

**Why this helps**: eliminates ~500 kernel launches per segment (250 zerocheck + 250 logup → 2-4 batched launches per coset group + 2-4 batched extraction launches). Also eliminates per-AIR `d_main_parts.to_device()` H2D copies.

**L2 cache**: for batchable AIRs with intermediates < threshold, the GPU processes blocks in waves of ~128 (one per SM). Each wave's combined intermediates working set must fit in L2 cache. With a per-AIR intermediates limit of 64KB and 128 SMs: 128 × 64KB = 8MB per wave, well within the 72MB L2. The total arena size (sum of all batchable AIRs' intermediates) may exceed L2, but at any given time only one wave's worth is hot. This is the same sequential-reuse pattern that makes the existing per-AIR approach work — the batch kernel just removes the launch gaps between AIRs.

### Change 2: Rust FFI bindings

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `#[repr(C)]` Rust structs matching the CUDA descriptors: `Round0ZcCtx`, `Round0LogupCtx`, `Round0BlockCtx` (reuse the existing `BlockCtx` type from MLE batching). Add safe wrapper functions for the batched kernel launchers and the batched extraction kernel.

### Change 3: Batched orchestration module

**File**: `crates/cuda-backend/src/logup_zerocheck/round0_batched.rs` (new file)

Functions:
1. `identify_batchable_airs(work_items) -> Vec<bool>`: Mark AIRs as batchable if BOTH their zerocheck and logup intermediates sizes are below 64KB (chosen so 128 concurrent blocks' intermediates fit in L2 cache). Non-batchable criteria: intermediates > 64KB, or AIR has no zerocheck constraints (checked via `zc_intermed_cap == 0`, which is the existing proxy used at mod.rs lines 276-282).

2. `batch_round0_small_airs(batchable_items, d_batch_ptr, tables) -> Result<()>`:
   - Group batchable AIRs by `num_cosets` (constraint_degree determines coset count; same coset count required for grid.y dimension)
   - For each coset group:
     a. Allocate shared intermediates arena: `sum(per_air_intermediates_size)` in one `DeviceBuffer`, with per-AIR offset tracking
     b. Build per-AIR logup interaction rules: call `SymbolicDagBuilder` + `SymbolicRulesGpu::new` for each batchable AIR, concatenate rules into a single device buffer with per-AIR offsets. (Note: per-AIR DAG building is ~0.05ms/AIR per the precompute-logup-round0-interaction-rules report, so this is not a performance-critical path — it's architecturally cleaner to batch the upload.)
     c. Build `Round0ZcCtx` descriptors on host with per-AIR pointers into the arena and rule buffers
     d. Build `BlockCtx` array mapping blocks to AIRs
     e. Upload descriptor arrays + block mappings via single H2D copy
     f. Launch batched zerocheck kernel
     g. Launch batched zerocheck extraction kernel
     h. Build `Round0LogupCtx` descriptors (reuse intermediates arena)
     i. Upload logup descriptors
     j. Launch batched logup kernel
     k. Launch batched logup extraction kernel
   - All descriptors write output to the shared `d_batch_array` at the same offsets the per-AIR path would use (`w.zc_batch_offset`, `w.numer_batch_offset`, `w.denom_batch_offset`).

3. `merge_skip_mask(batchable, work_items) -> Vec<bool>`: Create skip mask for existing multi-stream path so it only processes non-batched (large) AIRs.

All functions marked `#[inline(never)]` to prevent Rust optimizer interference with the existing `sumcheck_uni_round0_polys` function.

### Change 4: Caller-level routing

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Modify `prove_zerocheck_and_logup_gpu` (line 510 of mod.rs, the caller of `sumcheck_uni_round0_polys`) to:
1. Before calling `sumcheck_uni_round0_polys`, call `identify_batchable_airs()` on the work items
2. If any AIRs are batchable, call `batch_round0_small_airs()` for the batched subset
3. Pass a `skip_traces: &[bool]` parameter to `sumcheck_uni_round0_polys` so it skips already-processed AIRs
4. After both paths complete, polynomial results are already in the shared `d_batch_array`

Minimal changes to `sumcheck_uni_round0_polys`: add a single `.filter()` on the work items iterator using the skip mask (lines ~1110, ~1330). This is the same approach that succeeded in the `batch-round0-caller-routing` task (commit `bcb1c0d2`).

### Change 5: Prune unused CUDA template instantiations

**Files**: `zerocheck_round0.cu`, `logup_round0.cu`

Guard `NEEDS_SHMEM=true` template instantiations behind a compile-time `#ifdef ENABLE_SHMEM_KERNELS` flag, defaulting to off. The benchmark uses `l_skip=4` (skip_domain=16 ≤ WARP_SIZE), so `needs_shmem` is always false. This reduces .cubin size and GPU memory consumption, providing headroom for the new batched kernel instantiations.

Only perform this step if the Step 0 audit shows memory pressure.

## Invariants

1. **Correctness**: Batched path must produce identical polynomial coefficients as the per-AIR path. Verify by running both paths and comparing the `d_batch_array` output.
2. **No APC 0 regression**: APC 0 has ~20 AIRs per segment (below the 100-AIR multi-threading threshold). The batched path should be skipped entirely at APC 0. Additionally, at APC 100 with ~120 AIRs per segment, the batched path activates for small AIRs — no regression is acceptable.
3. **Memory safety**: The shared intermediates arena must have correct per-AIR offsets. Each AIR's intermediates pointer must not overlap with another AIR's region. Use a cumulative offset counter during arena layout.
4. **Coset grouping**: All AIRs in a batched kernel launch must share the same `num_cosets` value (determines grid.y dimension). Different coset groups get separate launches.
5. **VPMM pool state**: The batched arena allocation is a single large `DeviceBuffer` that is freed before the multi-stream path or GKR begins. This minimizes disruption to the VPMM free-region layout.
6. **BlockCtx correctness**: The `air_block_offsets` array must correctly delimit per-AIR block ranges so `batched_final_reduce_block_sums` produces correct per-AIR sums.

## Measurement Plan

1. **Memory audit** (Step 0): compile batched kernel skeletons, verify .cubin size and GPU memory fit
2. **Build**: `cargo check -p openvm-cuda-backend` to verify compilation
3. **Test**: `cargo nextest run -p openvm-cuda-backend --test-threads=4` to verify existing tests pass
4. **Correctness check**: Add a `#[cfg(debug_assertions)]` comparison that runs both batched and per-AIR paths and asserts polynomial equality at APC 300
5. **Benchmark APC 0**: verify no regression (Round 0 should be unchanged, single-threaded path)
6. **Benchmark APC 300**: measure STARK excl trace and Round 0 component
7. **Benchmark APC 100**: verify improvement scales with AIR count
8. **Profile with nsight at APC 300**: confirm reduced kernel launch count in Round 0

Expected results:
- Round 0 at APC 300: 180ms → 140-155ms (25-40ms improvement)
- STARK excl trace at APC 300: 1114ms → 1074-1089ms (2-4% improvement)
- APC 0: no change
- APC 100: proportional improvement (fewer small AIRs than APC 300, but some)

The 25-40ms estimate is based on: the batched path eliminates ~500 kernel launches per segment (at ~10-15μs overhead each = 5-7.5ms per segment) plus ~500 per-AIR H2D copies and buffer checks (~3-5ms per segment), for ~8-12ms per segment × 2 segments = ~16-24ms. The remaining gain comes from improved GPU utilization during small-AIR processing (continuous execution vs. launch gaps). The total improvement could exceed this if large AIRs benefit from reduced multi-stream thread count (fewer threads, better per-thread scheduling of remaining large AIRs).

## Rollback Criteria

Revert if any of:
- Round 0 at APC 300 improves by less than 15ms (below noise threshold for this component)
- APC 0 STARK excl trace regresses by more than 20ms
- Any APC configuration produces incorrect proofs
- GPU OOM prevents measurement at APC 300 (indicates template instantiation bloat — should be caught by Step 0 audit)
