# Plan: Right-size GKR input eval kernel grid for small AIRs

## Goal

Reduce GKR input evaluation wall time at APC 300 by fixing a grid-sizing inefficiency: the CUDA launcher for `evaluate_interactions_gkr_kernel<GLOBAL=true>` always launches `TASK_SIZE/256 = 256` blocks regardless of the AIR's actual `permutation_height`. For the majority of AIRs at APC 300 where height << 65536, over 98% of launched threads exit at the `if (row < permutation_height)` boundary check, but the 256-block kernel still saturates all 128 SMs on the RTX 4090, preventing concurrent kernel execution from the 8 multi-stream worker threads. The fix uses `min(TASK_SIZE, permutation_height)` as the thread count, shrinking the grid proportionally for small AIRs and freeing SMs for inter-stream concurrency.

## Current Code Path

### Entry point
`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:215` — `log_gkr_input_evals()`

1. Allocates `leaves` buffer for all interactions: `DeviceBuffer<Frac<EF>>::with_capacity(total_leaves)` (line 228)
2. Builds work items sorted by descending height (lines 232-252)
3. Pre-computes max buffer sizes, pre-allocates per-thread `GkrThreadBuffers` (lines 254-321)
4. Dispatches to 8 OS threads (when ≥100 AIRs). Each thread iterates its work items calling `process_gkr_input_air()` (lines 328-348)

### Per-AIR processing
`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:112` — `process_gkr_input_air()`

1. Copies `partition_ptrs` and optionally `public_values` to pre-allocated device buffers (lines 141-156)
2. Computes `is_global = buffer_size > 10` (line 160)
3. Computes `num_rows_per_tile = height.div_ceil(TASK_SIZE).max(1)` (line 162)
4. Calls `logup_gkr_input_eval(is_global, ..., height, num_rows_per_tile)` (line 174)

### CUDA launcher
`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu:216` — `_logup_gkr_input_eval()`

```cpp
auto count = is_global ? TASK_SIZE : permutation_height;  // BUG: always 65536 for GLOBAL
auto [grid, block] = kernel_launch_params(count, 256);    // grid = 256 blocks
```

For GLOBAL mode: always `grid = {256, 1, 1}`, `block = {256, 1, 1}` = 65536 threads. For a height=1024 AIR, threads 1024-65535 do nothing but still occupy SMs.

### Kernel
`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu:66` — `evaluate_interactions_gkr_kernel<GLOBAL>()`

```cpp
uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t task_stride = gridDim.x * blockDim.x;
FpExt *intermediates_ptr = (FpExt *)d_intermediates + task_offset;  // GLOBAL
uint32_t intermediate_stride = task_stride;                          // GLOBAL
for (uint32_t j = 0; j < num_rows_per_tile; j++) {
    uint32_t row = task_offset + j * task_stride;
    if (row < permutation_height) { /* evaluate DAG */ }
}
```

The intermediates buffer is indexed as `intermediates_ptr + node_idx * intermediate_stride`. With reduced grid, `intermediate_stride = min(TASK_SIZE, height)`, and the buffer access stays within `height * buffer_size` elements ≤ the pre-allocated `TASK_SIZE * max_buffer_size`.

### Why current code is slow

nsys profiling shows `evaluate_interactions_gkr_kernel<true>` has 486 instances totaling 456ms GPU time. Each GLOBAL kernel launches 256 blocks. On the RTX 4090 with 128 SMs and heavy register pressure (~2 blocks/SM occupancy), a single GLOBAL kernel fills all SMs. When 8 threads each launch GLOBAL kernels concurrently, the GPU serializes them — measured concurrency for seg0 is only 1.37x despite 8 streams. Per-segment metrics confirm: seg0 GKR input eval = 241ms, seg1 = 62ms. Seg0 is the single largest sub-component of STARK excl trace.

## Changes

### Change 1: CUDA launcher grid sizing

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`, line 231

**What**: Change the `count` computation for GLOBAL mode from `TASK_SIZE` to `min(TASK_SIZE, permutation_height)`.

**Before**:
```cpp
auto count = is_global ? TASK_SIZE : permutation_height;
```

**After**:
```cpp
auto count = is_global ? min((uint32_t)TASK_SIZE, permutation_height) : permutation_height;
```

**Why**: For height=1024, this changes the grid from 256 blocks to 4 blocks. The 4 blocks occupy 4 SMs instead of 128, freeing 124 SMs for concurrent kernels from other streams. No computation changes — the kernel's loop body is identical, and idle threads (which did no work anyway) are simply not launched.

**Correctness**: The kernel accesses `intermediates[task_offset + node_idx * task_stride]` where `task_stride = gridDim.x * blockDim.x = count`. For small AIRs, `count = height`, so the maximum intermediates index is `(height - 1) + (buffer_size - 1) * height = height * buffer_size - 1`. The pre-allocated buffer has capacity `TASK_SIZE * max_buffer_size ≥ height * buffer_size`. All accesses are in bounds.

This is the only code change. The Rust-side pre-allocation (`max_intermediates_len = TASK_SIZE * buffer_size`) is left unchanged because at APC 300 at least one AIR has height >= TASK_SIZE, so the pre-allocated maximum stays the same. Aligning the Rust pre-allocation to use `min(TASK_SIZE, height)` per AIR can be done as a follow-up but is a no-op for this benchmark.

## Invariants

1. **Correctness**: The kernel computation is unchanged. All threads that previously did useful work still do the same work. The only change is that idle threads (row ≥ permutation_height) are not launched.
2. **Intermediates buffer bounds**: For any height ≤ TASK_SIZE, `height * buffer_size ≤ TASK_SIZE * max_buffer_size` (the pre-allocated capacity). For height > TASK_SIZE, the grid is unchanged (count = TASK_SIZE).
3. **num_rows_per_tile**: This is computed on the Rust side as `height.div_ceil(TASK_SIZE).max(1)`. For height ≤ TASK_SIZE: `num_rows_per_tile = 1`, which is correct — each thread processes at most 1 row. For height > TASK_SIZE: unchanged.
4. **APC 0 no-regression**: At APC 0, there are only 99 AIRs (below the 100-AIR multi-threading threshold). Processing is single-threaded, so concurrency improvements don't apply. Large AIRs at APC 0 already use height ≥ TASK_SIZE, so no grid change. No regression expected.
5. **Output layout**: The `d_fracs` output pointer is per-AIR and writes to the correct offset in the shared `leaves` buffer. No change to output addressing.

## Measurement Plan

### Commands
```bash
# In powdr repo:
PROVE_BIN="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')/release/powdr_openvm_riscv"

# Build (includes CUDA rebuild)
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"

# Run benchmarks (3 runs each for stability)
for i in 1 2 3; do
  $PROVE_BIN prove --artifact results/pairing/apc300.cbor --input 0 --metrics results/pairing/apc300/metrics_after_$i.json --recursion
  $PROVE_BIN prove --artifact results/pairing/apc000.cbor --input 0 --metrics results/pairing/apc000/metrics_after_$i.json --recursion
done

# Profile with nsys
nsys profile --output results/pairing/nsys_after_apc300 --force-overwrite true --trace cuda,nvtx,osrt --sample none --stats true -- $PROVE_BIN prove --artifact results/pairing/apc300.cbor --input 0 --recursion
```

### Expected results
- **GKR input eval seg0**: 241ms → 110-160ms (1.5-2.2x improvement)
- **GKR input eval seg1**: 62ms → 40-55ms
- **STARK excl trace APC 300**: 1372ms → 1220-1300ms (5-11% improvement)
- **APC 0 no-regression**: STARK excl trace ≈ 2131ms ± 20ms
- **nsys verification**: Total blocks for `evaluate_interactions_gkr_kernel<true>` should decrease significantly (from ~124K to ~20-40K total across all phases)

### Key metrics to compare
- `prover.rap_constraints.logup_gkr.input_evals_time_ms` per segment
- `prover.rap_constraints.logup_gkr_time_ms` per segment
- Overall `STARK (excl. trace)` via spec.py

## Rollback Criteria

1. **STARK excl trace APC 300 improvement < 30ms** (median of 3 runs): The fix targets a 100-150ms improvement. If the improvement is under 30ms, either the analysis was wrong about GPU saturation being the bottleneck, or other serialization points dominate.
2. **APC 0 regression > 20ms**: Any regression at APC 0 indicates a correctness or performance problem with the grid change for large AIRs.
3. **Correctness failure**: Any verification error or incorrect proof output at any APC configuration.
