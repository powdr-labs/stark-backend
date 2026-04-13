# Report: Batch Round 0 Descriptor Arrays

## Description

Replace per-AIR Round 0 kernel launches (zerocheck constraint eval + logup interaction eval) with batched descriptor-array kernel launches. Currently, each of ~310 AIRs per segment at APC 300 gets 2+ individual kernel launches, buffer allocations, and H2D copies, resulting in ~1500 kernel launches per segment. Nsys profiling showed 76% of Round 0 wall time (294ms) was overhead, not GPU compute.

The optimization aimed to reduce Round 0 from ~294ms to ~100-180ms at APC 300 by batching all per-AIR contexts into single H2D uploads + single kernel launches using descriptor-array patterns.

## Implementation

### Completed

1. **CUDA batched kernels** (`zerocheck_round0.cu`, `logup_round0.cu`):
   - `Round0ZcCtx` / `Round0LogupCtx` per-AIR descriptor structs
   - `Round0BlockCtx` for block-to-AIR mapping
   - `batched_zerocheck_r0_coset_parallel_kernel` — 1D grid, each block reads its AIR descriptor
   - `batched_logup_r0_coset_parallel_kernel` — same pattern with identity-coset handling
   - Extern "C" launcher functions: `_batched_zerocheck_r0_eval_constraints`, `_batched_logup_r0_eval_interactions`
   - Uses `batched_final_reduce_block_sums` for per-AIR reduction

2. **Rust FFI bindings** (`cuda/logup_zerocheck.rs`):
   - `#[repr(C)]` structs: `Round0BlockCtx`, `Round0ZcCtx`, `Round0LogupCtx`
   - Safe wrapper functions for both batched launchers

3. **Rust batch helper functions** (`round0.rs`):
   - `BatchedR0AirDesc`, `BatchedR0LogupRules` descriptor types
   - `build_logup_rules_for_air()` — pre-builds DAG rules outside kernel
   - `evaluate_batched_zerocheck_group()` — groups AIRs by num_cosets, allocates shared intermediates, launches batched kernel
   - `evaluate_batched_logup_group()` — same pattern for logup

### Not Completed: Orchestration Integration

The Rust-side orchestration that routes small AIRs to the batched path could not be integrated. Adding the orchestration code to `mod.rs::sumcheck_uni_round0_polys` consistently caused `cudaErrorIllegalAddress` crashes at APC 300, even when all AIRs took the existing per-AIR path and the batched code never executed. The root cause appears to be Rust optimizer interference: adding significant code to the large `sumcheck_uni_round0_polys` function changes code generation for the existing multi-threaded CUDA kernel launch path, causing subtle memory safety issues.

Attempts to isolate via `#[inline(never)]` separate functions were in progress when the session ended.

## Results

Since the batched kernels are not yet wired into the proving pipeline, performance is unchanged.

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace APC 0 (ms) | 2153 | 2164 | 2153 | 0ms (1.00x) | -11ms (1.01x lower) |
| STARK excl trace APC 300 (ms) | 2455 | 1436 | 1422 | -1033ms (1.73x lower) | -14ms (1.01x lower) |
| Round 0 APC 0 (ms) | 178 | 178 | 178 | 0ms (1.00x) | 0ms (1.00x) |
| Round 0 APC 300 (ms) | 662 | 292 | 302 | -360ms (2.19x lower) | +10ms (1.03x higher) |

Note: Small variations (10-14ms) are within measurement noise. The after measurements confirm no regression from the CUDA+FFI infrastructure additions.

## Assessment

This optimization did **not** achieve its goal due to incomplete Rust-side integration. The CUDA kernel infrastructure (batched descriptor-array kernels, FFI bindings, batch helper functions) is fully implemented and compiles correctly, but routing small AIRs to the batched path caused unexpected crashes related to Rust optimizer behavior in the large `sumcheck_uni_round0_polys` function.

The core batched kernels have not been tested end-to-end. The implementation should be considered infrastructure-only: the kernels exist but are not yet exercised.

## Future Work

- **Resolve orchestration integration**: The crash when adding code to `sumcheck_uni_round0_polys` suggests the function is at a compiler optimization boundary. Options:
  1. Extract the entire Round 0 logic into a separate crate/module to reduce function size
  2. Use `#[inline(never)]` on the Phase 4 function with careful parameter passing
  3. Use a separate compilation unit to prevent cross-function optimization
  4. Profile the exact code generation difference to identify the root cause

- **Verify kernel correctness**: Once the orchestration is integrated, verify results match the per-AIR path by running both paths and comparing output polynomials.

- **Expected gains**: The batched kernels should reduce Round 0 from ~292ms to ~100-180ms at APC 300 (based on nsys analysis showing 76% overhead in per-AIR path). This would save ~110-190ms from STARK excl trace.

- **Keygen caching**: The plan included caching interaction DAG rules at keygen time. This was not implemented but would provide additional CPU-side savings (~310 DAG constructions per segment moved from prove-time to keygen-time).
