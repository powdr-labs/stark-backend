# Plan: Batch MLE Interpolation

## Goal

Replace per-AIR `interpolate_columns_gpu` kernel launches in MLE sumcheck rounds with a single batched descriptor-array kernel launch per round. This targets the MLE Rounds span under **Constraints** (`prover.rap_constraints.mle_rounds`), which is 180ms at APC 300 — completely unoptimized since baseline. Note: this is a DIFFERENT metric from the Stacked Reduction MLE rounds under Openings (`prover.openings.stacked_reduction.mle_rounds`) that was optimized in task `2026-04-12-2330` and task `2026-04-13-2300`. The Constraints MLE Rounds has never been targeted.

Nsys profiling shows 4993 `interpolate_columns_kernel` launches with only 20ms total GPU compute time. Each of the ~310 per-round launches incurs: (1) a `cudaMallocAsync` for the interpolated output buffer, (2) a `to_device()` H2D copy for column pointers, (3) a CUDA kernel launch. At ~15μs overhead per call × 4993 calls = ~75ms of pure overhead, which dominates the 180ms span.

A secondary source of overhead is per-trace `main_ptrs.to_device()` calls (~5000 `cudaMallocAsync` + `cudaMemcpyAsync`), adding ~10-15ms. These are NOT batched by this plan because `TraceCtx.main_ptrs_dev` is a `DeviceBuffer<MainMatrixPtrs<EF>>` consumed both as `.as_ptr()` in batch builders and as `&DeviceBuffer` in non-batch fallback paths (`evaluate_mle_constraints_gpu`, `evaluate_mle_interactions_gpu`). Changing the `TraceCtx` type would cascade through all evaluator code. This residual overhead is noted as follow-up work.

## Current Code Path

### Per-round interpolation loop (the target)

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines 1128-1293

In `sumcheck_polys_batch_eval()`, called once per MLE round (line 515), the function iterates over all traces. For each trace in Case B (round <= n_lift, line 1209):

1. **Line 1216-1226**: Build `columns: Vec<*const EF>` — collects pointers to all columns (selectors + preprocessed + mains) from the current trace's `DeviceMatrix` buffers.
2. **Line 1227**: `DeviceMatrix::<EF>::with_capacity(sp_deg * num_y, columns.len())` — allocates a new device buffer for the interpolated output. This calls `cudaMallocAsync`.
3. **Line 1228**: `columns.to_device()` — allocates a small `DeviceBuffer<*const EF>` and copies the pointer array H2D. This calls `cudaMallocAsync` + `cudaMemcpyAsync`.
4. **Line 1229-1231**: `interpolate_columns_gpu(...)` — launches one `interpolate_columns_kernel` invocation.
5. **Line 1275**: `_keepalive_interpolated.push(interpolated)` — stores the interpolated buffer to keep it alive for downstream evaluation kernels.
6. **Line 1278-1292**: Builds `TraceCtx` with pointers into the interpolated buffer.

At APC 300 with ~310 AIRs per segment and ~8 rounds of interpolation on average, this produces ~2500 iterations per segment × 2 segments = ~5000 total kernel launches.

### The CUDA kernel

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`, lines 71-92

```cuda
__global__ void interpolate_columns_kernel(
    FpExt *interpolated, const FpExt *const *columns,
    uint32_t s_deg, uint32_t num_y, uint32_t num_columns
) {
    // Each thread handles one (col, y) pair
    // Reads t0=column[2*y], t1=column[2*y+1]
    // Writes s_deg interpolated values: t0 + (t1-t0) * Fp(x+1) for x in 0..s_deg
}
```

**Launcher**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`, lines 190-201

```c
extern "C" int _interpolate_columns(
    FpExt *interpolated, const FpExt *const *columns,
    size_t s_deg, size_t num_y, size_t num_columns
) {
    auto [grid, block] = kernel_launch_params(num_y * num_columns, 512);
    interpolate_columns_kernel<<<grid, block>>>(interpolated, columns, s_deg, num_y, num_columns);
    return CHECK_KERNEL();
}
```

**Rust wrapper**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`, lines 518-531

### Downstream consumers

The interpolated buffer is consumed via `TraceCtx` pointers by:
- `evaluate_logup_batched()` (line 1342-1353)
- `evaluate_zerocheck_batched()` (line 1375-1383)
- `ZerocheckMonomialParYBatch` (line 1392-1406)
- `ZerocheckMonomialBatch` (line 1416-1423)

These evaluation functions only read from the interpolated data via device pointers stored in `TraceCtx.sels_ptr`, `TraceCtx.prep_ptr`, and `TraceCtx.main_ptrs_dev`. They do not depend on the interpolated data being in individual `DeviceMatrix` objects — only the raw device pointers matter.

## Changes

### Change 1: Add `InterpColDesc` struct to CUDA source

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

Add after line 70 (before `interpolate_columns_kernel`):

```cuda
struct InterpColDesc {
    FpExt* output;              // Base pointer for this trace's interpolated output
    uint32_t columns_offset;    // Offset into the flattened columns array
    uint32_t num_y;             // Number of y-values for this trace
    uint32_t num_columns;       // Number of columns for this trace
    uint32_t total_threads;     // = num_y * num_columns (precomputed for efficiency)
};
```

### Change 2: Add batched interpolation kernel to CUDA source

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

Add after the existing `interpolate_columns_kernel`:

```cuda
__global__ void batched_interpolate_columns_kernel(
    const InterpColDesc* descs,
    const FpExt *const * all_columns,
    uint32_t s_deg,
    uint32_t num_descs
) {
    uint32_t desc_idx = blockIdx.x;
    if (desc_idx >= num_descs) return;

    const InterpColDesc& d = descs[desc_idx];
    for (uint32_t tidx = threadIdx.x; tidx < d.total_threads; tidx += blockDim.x) {
        uint32_t y = tidx % d.num_y;
        uint32_t col_local = tidx / d.num_y;
        if (col_local >= d.num_columns) continue;

        const FpExt *column = all_columns[d.columns_offset + col_local];
        auto t0 = column[y << 1];
        auto t1 = column[(y << 1) | 1];
        FpExt *this_out = d.output + col_local * s_deg * d.num_y;

        for (int x = 0; x < s_deg; x++) {
            this_out[x * d.num_y + y] = t0 + (t1 - t0) * Fp(x + 1u);
        }
    }
}
```

Design: one block per descriptor (= per trace). Each block's threads cooperatively process all (column, y) pairs for that trace, looping if `total_threads > blockDim.x`. This keeps the output layout identical to the existing kernel.

### Change 3: Add batched launcher to CUDA source

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

Add after the existing `_interpolate_columns` launcher:

```c
extern "C" int _batched_interpolate_columns(
    const InterpColDesc* descs,
    const FpExt *const * all_columns,
    size_t s_deg,
    size_t num_descs
) {
    if (num_descs == 0) return 0;
    dim3 grid(num_descs);
    dim3 block(512);
    batched_interpolate_columns_kernel<<<grid, block>>>(
        descs, all_columns, s_deg, num_descs
    );
    return CHECK_KERNEL();
}
```

### Change 4: Add `InterpColDesc` Rust struct and FFI bindings

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add the Rust descriptor struct:

```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InterpColDesc {
    pub output: *mut EF,
    pub columns_offset: u32,
    pub num_y: u32,
    pub num_columns: u32,
    pub total_threads: u32,
}
unsafe impl Send for InterpColDesc {}
unsafe impl Sync for InterpColDesc {}
```

Add the extern declaration alongside the existing `_interpolate_columns`:

```rust
fn _batched_interpolate_columns(
    descs: *const InterpColDesc,
    all_columns: *const *const EF,
    s_deg: usize,
    num_descs: usize,
) -> i32;
```

Add a safe wrapper:

```rust
pub unsafe fn batched_interpolate_columns_gpu(
    descs: &DeviceBuffer<InterpColDesc>,
    all_columns: &DeviceBuffer<*const EF>,
    s_deg: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_batched_interpolate_columns(
        descs.as_ptr(),
        all_columns.as_ptr(),
        s_deg,
        descs.len(),
    ))
}
```

### Change 5: Restructure `sumcheck_polys_batch_eval` to batch interpolation

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Replace the per-trace interpolation loop (lines 1128-1293) with a three-phase approach:

**Phase 1: Collect metadata and compute buffer sizes (CPU-only, no CUDA calls)**

First pass over all traces, identical case-splitting logic as today. For each Case B trace:
- Compute `num_columns` (selectors width + all matrix widths)
- Compute `num_y = 1 << (n_lift - round)`
- Compute `interpolated_size = sp_deg * num_y * num_columns` (elements)
- Accumulate total buffer size across all Case B traces
- Record per-trace metadata: (trace_idx, air_idx, num_columns, num_y, interpolated_offset, column_start_idx, has_preprocessed, need_rot, main matrix widths)
- Collect all column pointers into a single flat `Vec<*const EF>`

For Case A traces, handle exactly as before (push to `late_eval`).

**Phase 2: Batch allocate and launch (minimal CUDA calls)**

1. Allocate one big `DeviceBuffer<EF>` of size = total interpolated elements across all traces
2. Build `Vec<InterpColDesc>`, one per Case B trace, with `output` pointing to the trace's offset within the big buffer
3. Upload descriptors and column pointers: `descs.to_device()`, `all_columns.to_device()` — 2 H2D copies total
4. Launch `batched_interpolate_columns_gpu` — one kernel call for all traces

This replaces ~310 × (cudaMallocAsync + columns.to_device + kernel launch) with 1 × (cudaMallocAsync + 2 × to_device + kernel launch).

**Phase 3: Build TraceCtx (per-trace main_ptrs.to_device still needed)**

For each Case B trace, compute `sels_ptr`, `prep_ptr` as pointer offsets into the big interpolated buffer. The `main_ptrs_dev` per-trace `to_device()` call remains because `TraceCtx.main_ptrs_dev` is `DeviceBuffer<MainMatrixPtrs<EF>>` consumed both as `.as_ptr()` in batch builders and as `&DeviceBuffer` in fallback evaluator functions. Changing this would require modifying `TraceCtx` and all downstream evaluator code — deferred to follow-up.

The `_keepalive_interpolated` vector is replaced by keeping the single big `DeviceBuffer` alive for the duration of the evaluation phase.

### Change 6: Choose block size for the batched kernel

Use `threads_per_block = 512` (matching the existing `_interpolate_columns` launcher which uses `kernel_launch_params(num_y * num_columns, 512)`). For traces where `total_threads > 512`, the thread loop (`for tidx = threadIdx.x; tidx < total_threads; tidx += blockDim.x`) handles the excess. For traces where `total_threads < 512`, excess threads exit early via the `if (col_local >= num_columns) continue` guard.

**Load imbalance note**: One-block-per-trace creates load imbalance because small APCs have `total_threads` as low as 1, while large AIRs may have 50K+. A flat-thread-grid with binary search would give better GPU utilization but adds implementation complexity (prefix-sum array upload, per-thread descriptor lookup). Since the optimization targets **launch overhead** (not GPU compute time), and the kernel's total GPU compute is only 20ms, the load imbalance in compute is acceptable. The alternative flat-grid design can be explored as follow-up if GPU compute becomes the bottleneck after launch overhead is eliminated.

## Invariants

1. **Output layout is identical**: Each trace's interpolated data has the same column-major layout as before: `interpolated[col * sp_deg * num_y + x * num_y + y]`. The `TraceCtx` pointers reference the same memory layout.

2. **Downstream evaluation is unchanged**: The `evaluate_logup_batched`, `evaluate_zerocheck_batched`, monomial batch, and monomial par-y batch functions receive the same `TraceCtx` with the same device pointers. No changes needed to evaluation or fold paths.

3. **Late traces (Case A) are unchanged**: The late_eval path (lines 1154-1207) does not call `interpolate_columns_gpu` and remains untouched.

4. **Correctness of interpolation**: The batched kernel computes the same per-thread formula as the original: `t0 + (t1 - t0) * Fp(x + 1)`. Same indexing into column data, same output stride.

5. **Memory lifetime**: The big interpolated buffer must remain live until all evaluation kernels complete. Storing it in a local variable that outlives the evaluation calls (same scope as the current `_keepalive_interpolated`) suffices.

6. **s_deg is uniform**: All traces in a given round use the same `s_deg` (= `self.constraint_degree`), so it can be passed as a scalar to the batched kernel.

## Measurement Plan

### Before changes

Run the benchmark for APC {0, 100, 300} and record `spec.py` output. Save as `before_apc{000,100,300}.json`.

### After changes

Run the same benchmark suite. Save as `after_apc{000,100,300}.json`.

### Key metrics to compare

1. **MLE Rounds** (`prover.rap_constraints.mle_rounds`): Primary target. Expect 180ms → ~130-150ms at APC 300 (17-28% improvement). This eliminates ~5000 interpolation kernel launches + ~5000 cudaMallocAsync for interpolation buffers + ~5000 columns.to_device() calls, but ~5000 main_ptrs.to_device() calls remain as residual overhead.
2. **STARK excl trace**: Expect 1466ms → ~1420-1440ms at APC 300 (2-3% improvement).
3. **APC 0 STARK excl trace**: Expect no regression (within noise of 2155ms). At APC 0, only ~99 AIRs, so the per-AIR overhead reduction is smaller.
4. **Correctness**: `prove+verify` must pass for all three configs.

### Profiling

Run nsys on APC 300 after the change. Verify:
- `interpolate_columns_kernel` launches drop from ~4993 to ~16 (one batched launch per round × 2 segments)
- Total `interpolate_columns` GPU time should remain ~20ms (same computation, different launch pattern)
- H2D copy count should decrease significantly

## Rollback Criteria

Revert if ANY of the following:
- MLE Rounds at APC 300 improves less than 20ms (below noise floor)
- STARK excl trace at APC 0 regresses by more than 50ms
- `prove+verify` fails for any config
- GPU OOM on RTX 4090 (24GB)
- STARK excl trace at APC 300 improves less than 15ms
