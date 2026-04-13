# Plan: Batch Round 0 via Caller-Level Routing

## Goal

Reduce Round 0 evaluation time at APC 300 from 301ms to ~100-150ms by batching small AIRs' kernel launches into single descriptor-array kernels, while keeping the existing `sumcheck_uni_round0_polys` function's critical multi-stream code path effectively unchanged.

The previous attempt (2026-04-13-1600) implemented the CUDA batched kernels and FFI but failed during Rust orchestration integration: adding routing code inside `sumcheck_uni_round0_polys` changed Rust's LLVM optimizer behavior for the existing multi-threaded path, causing `cudaErrorIllegalAddress` even when the batched code was never executed.

**Key difference in this attempt**: route from the CALLER (`prove_zerocheck_and_logup_gpu`), not from inside the function. Add only a minimal `skip_traces` filter parameter to the existing function. Keep all batched logic in a separate `#[inline(never)]` module.

## Current Code Path

### Entry point
`crates/cuda-backend/src/logup_zerocheck/mod.rs:393` — call to `prover.sumcheck_uni_round0_polys(ctx, lambda)` from within `prove_zerocheck_and_logup_gpu`.

### sumcheck_uni_round0_polys (mod.rs:767-983, 217 lines)

**Phase 1 (lines 772-926) — Precomputation + work item building (CPU)**:
- Lines 772-898: Compute `lambda_pows`, `lambda_combinations`, `eq_3b_per_trace`, `d_eq_3b_per_trace`, `logup_combinations`, `eq_xis`, `sels_per_trace_base` for ALL traces. These are stored on `self` and needed by both the per-AIR and batched paths.
- Lines 899-926: Build `Vec<Round0AirWorkItem>` — one per present trace. Work items hold read-only references into `self`'s precomputed data and `ctx`'s trace buffers.
- Line 929: Sort by descending height for load balance.

**Phase 2 (lines 935-966) — Multi-stream per-AIR processing (GPU + CPU)**:
- Lines 939-943: Choose `num_threads = 8` if ≥100 work items, else 1.
- Lines 944-966: `std::thread::scope` distributes work items across threads. Each thread calls `process_air_round0(w)` per AIR sequentially.

**Phase 3 (lines 968-982) — Scatter results (CPU)**:
- Maps `Round0AirResult` back into `batch_sp_poly` output array.

### process_air_round0 (mod.rs:140-272, 133 lines)

Per AIR:
1. Build `main_parts` pointer array, upload to device (`to_device()`) — 1 H2D
2. `evaluate_round0_constraints_gpu()` (round0.rs:31-124) — 3 buffer allocs, 1 kernel launch, 1 D2H, CPU transpose + iDFT
3. `evaluate_round0_interactions_gpu()` (round0.rs:132-276) — rebuild interaction DAG from scratch (CPU), 3 H2D (weights, rules), 3 buffer allocs, 1 kernel launch, 1 D2H, CPU unzip + transpose + iDFT

### Overhead analysis (APC 300, 2 segments, ~310 AIRs/segment)

Per segment, the current code executes:
- ~620 kernel launches (310 AIRs × 2 evals, each eval + reduction)
- ~2000+ `DeviceBuffer` allocations via global `MemoryManager` mutex
- ~930 H2D copies (main_parts + rules + weights per AIR)
- ~620 D2H copies
- ~310 CPU DAG constructions (`SymbolicDagBuilder`)

Nsys data: Round 0 GPU kernel time totals ~505ms across all 8 streams. With perfect 8-way parallelism the floor would be ~63ms. Actual = 301ms. Overhead = ~238ms.

### CUDA kernels (used by the per-AIR path)
- `zerocheck_round0.cu`: `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel` — 539 instances, 237ms total
- `logup_round0.cu`: `logup_r0_ntt_eval_interactions_coset_parallel_kernel` — 735 instances (396 `<true>` + 339 `<false>`), 247ms total

These coset-parallel kernels are used when `num_x * skip_domain < 32768` (COSET_PARALLEL_THRESHOLD). At APC 300, all small APC-generated AIRs meet this criterion. Large original AIRs (height >= 2^15 with l_skip=3, so num_x >= 4096) use the lockstep/NTT kernel instead.

### FFI layer
`crates/cuda-backend/src/cuda/logup_zerocheck.rs` — extern "C" declarations for per-AIR zerocheck and logup launchers (lines 351-373, 311-332), with safe Rust wrappers (lines 962-1058).

## Changes

### Change 1: Minimal modification to `sumcheck_uni_round0_polys`

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add a `skip_traces: Option<&[bool]>` parameter to the function signature. In the work item building loop (line 901), add a `.filter()` to exclude skipped traces:

```rust
fn sumcheck_uni_round0_polys(
    &mut self,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    lambda: EF,
    skip_traces: Option<&[bool]>,  // NEW: traces handled by batched path
) -> Result<Vec<UnivariatePoly<EF>>, LogupZerocheckError> {
    // ... lines 772-898 unchanged (precomputation for ALL traces) ...

    // Phase 1: Build work items — only for non-skipped traces
    let mut work_items: Vec<Round0AirWorkItem<HS>> = izip!(...)
        .enumerate()
        .filter(|(trace_idx, _)| {
            skip_traces.map_or(true, |s| !s[*trace_idx])
        })
        .map(|(trace_idx, ...)| Round0AirWorkItem { ... })
        .collect();

    // ... lines 928-982 unchanged (sort, sync, multi-stream, scatter) ...
}
```

Total change: ~5 lines added to this function. Phase 2 (the sensitive multi-stream code at lines 935-966) is UNCHANGED — it processes whatever work items Phase 1 built.

**Why this is safe**: The previous failure was caused by adding ~200 lines of orchestration code (batched path routing, descriptor building, kernel launch logic) inside this function. Adding a single `.filter()` predicate to the existing iterator chain does not change:
- Function size significantly (5 lines vs 200)
- The multi-stream Phase 2 code
- The optimizer's analysis of the thread spawning / CUDA kernel launch path
- Register pressure or inlining decisions for `process_air_round0`

### Change 2: Re-introduce batched CUDA kernels from reverted commit

**Source**: Git commit `0c699f47` (reverted by `b617e156`)

Cherry-pick the CUDA kernel additions from this commit:

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`

Add:
- `Round0ZcCtx` struct — per-AIR descriptor with base-field pointers matching `NttEvalContext<1>`:
  ```cuda
  struct Round0ZcCtx {
      const Fp *selectors_cube;     // [3][num_x]
      const Fp *preprocessed;       // preprocessed trace (nullable)
      const Fp *const *main_parts;  // array of trace buffer pointers
      const FpExt *eq_cube;         // eq(x) evaluations
      const FpExt *lambda_pows;     // lambda challenge powers
      const Fp *public_values;      // per-AIR public inputs
      const Rule *d_rules;          // constraint DAG rules
      const size_t *d_used_nodes;   // constraint node indices
      size_t rules_len, used_nodes_len, lambda_len;
      uint32_t buffer_size;
      Fp *d_intermediates;          // offset into shared intermediates buffer
      uint32_t buffer_stride;       // = num_cosets * num_x_blocks * block_x
      uint32_t num_x, height;
      Fp g_shift;
  };
  ```
- `Round0BlockCtx` struct: `{ uint32_t local_block_idx; uint32_t air_idx; }`
- `batched_zerocheck_r0_coset_parallel_kernel` — 1D grid, each block loads its `BlockCtx` and `Round0ZcCtx`, computes coset_idx and x_block from `local_block_idx`, evaluates constraints using `acc_constraints<1, false>()` (GLOBAL mode only), writes to shared `tmp_sums_buffer`.
- `_batched_zerocheck_r0_eval_constraints` launcher — launches eval kernel + `batched_final_reduce_block_sums` for per-AIR reduction.

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

Add:
- `Round0LogupCtx` struct — like `Round0ZcCtx` but with `numer_weights`, `denom_weights`, `denom_sum_init` fields instead of `lambda_pows`/`d_used_nodes`.
- `batched_logup_r0_coset_parallel_kernel` — same 1D grid pattern; handles identity coset (coset_idx=0 → `g_coset = Fp::one()`, `skip_ntt = true`); outputs `FracExt`.
- `_batched_logup_r0_eval_interactions` launcher — uses `reinterpret_cast<FpExt*>` on `FracExt` buffers and `d = 2 * num_cosets * skip_domain` for the reduction kernel (matching the stacked reinterpret pattern from existing non-batched logup launchers at logup_round0.cu:579,651).

### Change 3: Re-introduce FFI bindings from reverted commit

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `#[repr(C)]` Rust structs matching the CUDA descriptors:
```rust
#[repr(C)]
pub struct Round0BlockCtx {
    pub local_block_idx: u32,
    pub air_idx: u32,
}

#[repr(C)]
pub struct Round0ZcCtx { /* fields matching CUDA struct */ }

#[repr(C)]
pub struct Round0LogupCtx { /* fields matching CUDA struct */ }
```

Add extern "C" declarations and safe wrapper functions for both batched launchers.

### Change 4: New batched processing module

**New file**: `crates/cuda-backend/src/logup_zerocheck/round0_batched.rs`

This module contains ALL batched orchestration logic, completely isolated from the existing per-AIR code. All public functions are marked `#[inline(never)]` to prevent optimizer cross-contamination.

#### 4a: `identify_batchable_airs()`

```rust
#[inline(never)]
pub fn identify_batchable_airs<HS: GpuHashScheme>(
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    l_skip: usize,
) -> Vec<bool>
```

Returns a `Vec<bool>` of length `ctx.per_trace.len()`. A trace is batchable if:
1. `num_x * skip_domain < 32768` (uses the coset-parallel kernel path)
2. The total number of batchable traces >= 50 (enough to justify batching overhead)

If fewer than 50 traces qualify, returns all-false (no batching).

#### 4b: `batch_round0_small_airs()`

```rust
#[inline(never)]
pub fn batch_round0_small_airs<HS: GpuHashScheme>(
    prover: &mut LogupZerocheckGpu<HS>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    skip_mask: &[bool],
) -> Result<Vec<Round0AirResult>, LogupZerocheckError>
```

Called AFTER `sumcheck_uni_round0_polys` returns (so all precomputed data in `prover` is populated).

**Step 0 — Build per-AIR main_parts device pointers (GPU, batched)**:
For each batchable trace, build the `main_parts` pointer array (same as process_air_round0 lines 152-162): construct the array of `*const F` pointers to `cached_mains` and `common_main` columns. Instead of individual `to_device()` per AIR, concatenate all AIRs' pointer arrays into a single flat `Vec<*const F>` with tracked per-AIR offsets. Upload once via a single `to_device()`. Each AIR's `Round0ZcCtx.main_parts` and `Round0LogupCtx.main_parts` point to its offset within this shared device buffer. This replaces ~600 individual H2D copies with one.

**Step 1 — Build zerocheck descriptor groups (CPU)**:
For each batchable trace (where `skip_mask[trace_idx] == true`):
- Compute `num_cosets_zc = constraint_degree - 1`, `num_x = 1 << max(0, n)`
- Group by `num_cosets_zc` (expected 2-4 groups)
Within each group:
- `block_x = min(128, skip_domain * ceil_to_multiple(max_num_x, skip_domain))` (matching MAX_THREADS=128 from zerocheck_round0.cu:472), ensure multiple of `skip_domain`
- `x_per_block = block_x / skip_domain`
- For each AIR: compute `num_x_blocks = ceil(num_x / x_per_block)`, `blocks_per_air = num_x_blocks * num_cosets_zc`
- Build `Round0BlockCtx` entries and `Round0ZcCtx` descriptors, with `main_parts` pointing into the shared device buffer from Step 0
- Compute intermediates size per AIR via `_zerocheck_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets_zc, usize::MAX)` — pass `usize::MAX` as `max_temp_bytes` to prevent eval_config from reducing grid.x
- Set `buffer_stride = num_cosets_zc * num_x_blocks * block_x` per AIR
- Track cumulative `air_block_offsets` for the reduction kernel

**Step 2 — Build logup descriptor groups (CPU)**:
For each batchable trace with interactions:
- Build the interaction DAG from scratch using `SymbolicDagBuilder` and `SymbolicRulesGpu::new()` (same as existing round0.rs:161-206). Use rayon `par_iter` to parallelize across AIRs.
- Compute dynamic weights (`numer_weights`, `denom_weights`, `denom_sum_init`) from `prover.eq_3b_per_trace` and `prover.beta_pows` using the DAG's rule-to-interaction mapping.
- Group by `num_cosets_logup = constraint_degree` (= `num_cosets_zc + 1`)
- Build `Round0LogupCtx` descriptors with pointers to cached rules (from step above) and weights.

**Step 3 — Upload and launch per group (GPU)**:
For each zerocheck group:
1. Memory budget check: if total intermediates > 2GB, split into sub-groups
2. Allocate shared `intermediates: DeviceBuffer<F>`, `tmp_sums: DeviceBuffer<EF>`, `output: DeviceBuffer<EF>`
3. Backfill `d_intermediates` pointer offsets in context structs
4. Upload: `d_block_ctxs`, `d_zc_ctxs`, `d_air_offsets` (one H2D each)
5. Launch `_batched_zerocheck_r0_eval_constraints` (eval kernel + reduction)
6. D2H: `output.to_host_on_current_stream()`

For each logup group:
1. Same allocation pattern (with `Frac<EF>` output buffers)
2. Upload descriptors + flat weight arrays
3. Launch `_batched_logup_r0_eval_interactions` (eval kernel + stacked FracExt reduction)
4. D2H results

**Step 4 — CPU post-processing (parallel via rayon)**:
For each batchable AIR:
1. Extract zerocheck result slice from group output (offset = `air_idx * num_cosets_zc * skip_domain`)
2. Transpose + iDFT → zerocheck polynomial (identical to existing process_air_round0 lines 185-208)
3. Extract logup result slice, unzip Frac into numer/denom arrays
4. **Negative-n normalization**: If the AIR's `n` is negative (height < skip_domain, i.e., height 1, 2, or 4 with l_skip=3), multiply all numerator values by `F::from_u32(1 << n.unsigned_abs()).inverse()` before polynomial construction (matching process_air_round0 lines 235-239). This is critical for correctness — small APC-generated AIRs with very small trace heights are exactly the ones being batched.
5. Transpose + iDFT → logup numer/denom polynomials (identical to lines 240-261)
6. Build `Round0AirResult`

#### 4c: `merge_results()`

```rust
pub fn merge_results(
    large_polys: Vec<UnivariatePoly<EF>>,
    small_results: Vec<Round0AirResult>,
    num_present_airs: usize,
) -> Vec<UnivariatePoly<EF>>
```

Takes the `batch_sp_poly` from the existing function (with slots for large AIRs filled, small AIR slots empty) and fills in the small AIR results from the batched path.

### Change 5: Wire up routing in the caller

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

At line 393, replace:
```rust
let sp_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda)?;
```
With:
```rust
let skip_mask = round0_batched::identify_batchable_airs::<HS>(ctx, l_skip);
let sp_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda, Some(&skip_mask))?;
let small_results = round0_batched::batch_round0_small_airs(&mut prover, ctx, &skip_mask)?;
let sp_0_polys = round0_batched::merge_results(sp_0_polys, small_results, num_present_airs);
```

Note: `sumcheck_uni_round0_polys` runs first. This ensures all precomputed per-trace data (`eq_3b_per_trace`, `eq_xis`, `sels_per_trace_base`, etc.) is populated on `prover` before the batched function reads it. The batched function runs second and uses `&mut prover` to access precomputed data and the memory tracker.

### Change 6: Register the module

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add `mod round0_batched;` to the module declarations.

## Invariants

1. **Correctness**: The batched kernel produces identical results to the per-AIR kernel for every batchable AIR. The per-block computation is identical to one block of the existing coset-parallel kernel. Verified by running prove+verify for all APC configs.

2. **Existing function untouched in its critical path**: Phase 2 (lines 935-966, multi-stream processing) has zero modifications. The only change is a `.filter()` in Phase 1's iterator chain (5 lines). Phase 1 precomputation (lines 772-898) is unchanged — it runs for ALL traces regardless of skip_mask.

3. **No regression at APC 0**: At APC 0, ~99 AIRs, most large. Fewer than 50 batchable AIRs → `identify_batchable_airs` returns all-false → skip_mask is all-false → existing function processes all AIRs as before. Zero code path change.

4. **Memory safety**: All device pointers in descriptor structs are valid for the kernel launch duration. Shared intermediates buffer is sized to hold all AIRs in a sub-group simultaneously. The 2GB cap prevents the OOM that sank the prealloc-round0-buffers attempt (which allocated 1GB × 8 threads = 8GB).

5. **Numerical equivalence**: Batched kernel uses GLOBAL intermediates mode only (shared memory variant unnecessary for small AIRs). For all batchable AIRs, buffer_size is small enough that GLOBAL mode was already used by the per-AIR path.

6. **Identity coset**: Batched logup kernel handles coset_idx=0 as identity (g_coset = 1, skip_ntt = true), matching existing logup_round0.cu:361-436.

7. **Uniform reduction per group**: All AIRs in a group share `num_cosets`, so the reduction kernel's `d = num_cosets * skip_domain` (or `2 * num_cosets * skip_domain` for logup's FracExt stacking) is uniform. Uses `batched_final_reduce_block_sums` with `air_block_offsets` for per-AIR block ranges.

8. **Intermediates sizing**: All `_zerocheck_r0_intermediates_buffer_size` calls use `max_temp_bytes = usize::MAX` to get the unreduced buffer size, ensuring `buffer_stride` is consistent between allocation and kernel indexing.

## Measurement Plan

1. Run `run_pairing.sh` for APC {0, 100, 300} before and after.
2. Analyze with `spec.py` for STARK excl trace and Round 0 sub-metrics.
3. Run nsys for APC 300 to verify:
   - Kernel launch count reduction (from ~1500 to ~12-20 per segment)
   - H2D copy count reduction (from ~30K to ~25K)
4. Verify prove+verify passes for all 3 APC configs.

Expected results:
- Round 0 APC 300: 301ms → 100-150ms (2.0-3.0x improvement)
- Round 0 APC 0: 178ms → ≤ 178ms (no regression; batched path inactive)
- STARK excl trace APC 300: 1429ms → 1229-1279ms (1.12-1.16x improvement, cumulative 1.92-2.00x vs baseline)

## Rollback Criteria

1. Round 0 at APC 300 improves by less than 60ms
2. STARK excl trace at APC 0 regresses by more than 20ms
3. Prove+verify fails for any APC configuration
4. `cudaErrorIllegalAddress` or other CUDA errors (the critical risk from the previous attempt)
5. GPU OOM during batched kernel launch
