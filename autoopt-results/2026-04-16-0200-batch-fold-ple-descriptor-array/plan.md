# Plan: Batch fold_ple_from_evals Kernel Launches via Descriptor Array

## Goal

Replace the per-trace sequential `fold_ple_from_evals` GPU kernel launches in `LogupZerocheckGpu::fold_ple_evals` (mod.rs:1478-1594) with a single batched descriptor-array kernel. nsight shows `fold_ple_from_evals_kernel<false>`: 797 instances, 9.5ms total at APC 300. The per-trace loop launches one kernel per matrix per trace, serializing on the GPU stream. A batched launch enables concurrent block scheduling across traces.

## Current Code Path

### Entry: `LogupZerocheckGpu::fold_ple_evals` (mod.rs:1478-1594)

Called once per segment at line 619, inside the Round 0 tracing span (after `sumcheck_uni_round0_polys`, before MLE rounds begin). Runs on the default stream, single-threaded (NOT inside the multi-streamed section).

The function iterates all traces and launches `fold_ple_from_evals` per matrix:

```rust
self.mat_evals_per_trace = ctx.per_trace.iter().map(|(air_idx, air_ctx)| {
    let mut results: Vec<DeviceMatrix<EF>> = Vec::new();
    // Preprocessed (if exists): fold_ple_evals_rotate → 1-2 kernel launches
    if let Some(committed) = &air_pk.preprocessed_data {
        results.push(fold_ple_evals_rotate(l_skip, ..., trace, ..., need_rot)?);
    }
    // Cached mains: fold_ple_evals_rotate → 1-2 launches each
    for committed in &air_ctx.cached_mains { ... }
    // Common main: fold_ple_evals_rotate → 1-2 launches
    results.push(fold_ple_evals_rotate(l_skip, ..., trace, ..., need_rot)?);
    Ok(results)
}).collect()?;
```

### `fold_ple_evals_rotate` (fold_ple.rs:16-55)

Calls `fold_ple_evals_gpu` once (rotate=false), and if `need_rot`, once more (rotate=true). Each call allocates a separate output `DeviceBuffer`, then launches the kernel.

### `fold_ple_evals_gpu` → `fold_ple_from_evals` (fold_ple.rs:67-102)

FFI call to `_fold_ple_from_evals` in `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`. Kernel: `fold_ple_from_evals_kernel<ROTATE>`.

Grid: `(ceil(num_x / chunks_per_block), width)`, block: `max(256, skip_domain)`.

Per-thread: barycentric interpolation of `min(height, skip_domain)` source values into one output EF value. When `ROTATE=true`, source values are shifted by one position (modular rotation).

### Parameters per launch

- `src`: matrix data pointer (`*const F`)
- `dst`: output pointer (`*mut EF`)
- `omega_skip_pows`: shared across all traces
- `inv_lagrange_denoms`: shared across all traces
- `height`, `width`, `l_skip`, `num_x`: per-trace dimensions
- `rotate`: bool (template parameter)

### Why it's slow at APC 300

797 kernel launches at avg 12μs each = 9.5ms GPU time. Two costs:
1. **Kernel launch overhead**: ~2-3μs × 797 = ~2-2.4ms of CUDA driver overhead.
2. **Serial SM utilization**: Small traces (low width × low num_x) produce few blocks, leaving most of the 128 SMs idle while each small kernel runs to completion.

At APC 0: ~99 AIRs × ~1.3 matrices = ~130 launches across 5 segments, ~26 per segment. Savings proportionally smaller.

## Changes

### Step 1: Define descriptor struct (CUDA side)

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

Follow the existing `InterpColDesc` / `InterpTraceDescM` pattern in utils.cu (lines 77, 92) by embedding `block_start` directly in the descriptor:

```c
struct FoldPleDesc {
    const Fp *src;       // source matrix data pointer
    FpExt *dst;          // output buffer pointer (non-overlapping per trace)
    uint32_t height;     // source matrix height
    uint32_t width;      // source matrix width
    uint32_t num_x;      // = max(height, skip_domain) / skip_domain
    uint32_t block_start; // first 1D block index for this descriptor
};
```

### Step 2: Add batched kernel using block_start + binary search

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

Follow the existing `batched_interpolate_columns_kernel` pattern in the same file (line 118):

```c
template <bool ROTATE>
__global__ void batched_fold_ple_from_evals_kernel(
    const FoldPleDesc *descs,
    uint32_t num_descs,
    const Fp *omega_skip_pows,
    const FpExt *inv_lagrange_denoms,
    uint32_t skip_domain
) {
    // Binary search on block_start to find descriptor for this block
    uint32_t lo = 0, hi = num_descs;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (descs[mid].block_start <= blockIdx.x) lo = mid + 1;
        else hi = mid;
    }
    uint32_t desc_idx = lo - 1;
    auto desc = descs[desc_idx];
    uint32_t local_block = blockIdx.x - desc.block_start;

    // Decompose local_block into (row_block, col_idx) using desc dimensions
    uint32_t chunks_per_block = blockDim.x / skip_domain;
    uint32_t blocks_per_col = (desc.num_x + chunks_per_block - 1) / chunks_per_block;
    uint32_t col_idx = local_block / blocks_per_col;
    uint32_t row_block = local_block % blocks_per_col;

    // Same body as existing fold_ple_from_evals_kernel:
    // - chunk_in_block = threadIdx.x / skip_domain
    // - tid_in_chunk = threadIdx.x % skip_domain
    // - row_idx = row_block * chunks_per_block + chunk_in_block
    // - Barycentric interpolation from desc.src → writes to desc.dst
}
```

### Step 3: Add C launcher

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`

```c
extern "C" int _batched_fold_ple_from_evals(
    const FoldPleDesc *descs,
    uint32_t num_descs,
    uint32_t total_blocks,
    const Fp *omega_skip_pows,
    const FpExt *inv_lagrange_denoms,
    uint32_t l_skip,
    bool rotate
) {
    if (total_blocks == 0) return 0;
    uint32_t skip_domain = 1u << l_skip;
    uint32_t block_size = std::max(256u, skip_domain);
    size_t smem = (skip_domain > WARP_SIZE) ?
        (block_size / WARP_SIZE) * sizeof(FpExt) : 0;
    
    if (rotate) {
        batched_fold_ple_from_evals_kernel<true><<<total_blocks, block_size, smem>>>(
            descs, num_descs, omega_skip_pows, inv_lagrange_denoms, skip_domain);
    } else {
        batched_fold_ple_from_evals_kernel<false><<<total_blocks, block_size, smem>>>(
            descs, num_descs, omega_skip_pows, inv_lagrange_denoms, skip_domain);
    }
    return CHECK_KERNEL();
}
```

### Step 4: Add Rust FFI bindings

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `#[repr(C)]` struct `FoldPleDesc` and the extern "C" + safe wrapper for `batched_fold_ple_from_evals`.

### Step 5: Add batched dispatch function in fold_ple.rs

**File**: `crates/cuda-backend/src/logup_zerocheck/fold_ple.rs`

Add a new function that batches the kernel launches but keeps per-trace output allocation:

```rust
/// Batch-launches fold_ple_from_evals for multiple (src, dst) pairs.
/// Each pair gets its own output range; the kernel processes all pairs concurrently.
pub unsafe fn batched_fold_ple_evals_gpu(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    items: &[(/*src*/ &DeviceBuffer<F>, /*src_height*/ usize, /*src_width*/ usize,
              /*dst*/ *mut EF, /*num_x*/ usize)],
    d_inv_lagrange_denoms: &DeviceBuffer<EF>,
    rotate: bool,
) -> Result<(), FoldPleError>
```

This function:
1. Builds `Vec<FoldPleDesc>` from the `items` slice, computing `block_start` as cumulative block count.
2. Uploads the descriptor array to device.
3. Calls `batched_fold_ple_from_evals` with `total_blocks = last.block_start + last_blocks`.
4. For 0 items, returns immediately.

Then modify `fold_ple_evals_rotate` to collect its 1-2 fold calls into a batch rather than launching individually. This keeps the mod.rs per-trace loop structure and mem_limit accounting completely unchanged.

### Step 6: Wire fold_ple_evals_rotate to use batched path

**File**: `crates/cuda-backend/src/logup_zerocheck/fold_ple.rs`

Modify `fold_ple_evals_rotate` (or add a new `batched_fold_ple_evals_rotate` that takes multiple traces) to collect all the (src, dst, height, width, num_x) tuples and call `batched_fold_ple_evals_gpu` twice: once for rotate=false (all traces), once for rotate=true (traces with need_rot).

In mod.rs `fold_ple_evals`, collect all the trace matrix information in a first pass, then call the batch function(s), then distribute the results. The per-trace output `DeviceBuffer` allocation and `DeviceMatrix` construction remain in fold_ple.rs. The `mem_limit` accounting at line 1533 applies to the same per-trace buffer and is unchanged.

## Invariants

1. **Output equivalence**: Identical computation per block. The descriptor provides per-trace parameters.
2. **Destination non-overlap**: Each descriptor's `dst` points to a separately allocated `DeviceBuffer`.
3. **Shared parameters**: `omega_skip_pows`, `inv_lagrange_denoms`, `skip_domain` are the same for all traces.
4. **Block size**: Same as existing: `max(256, skip_domain)`. Shared memory allocation unchanged.
5. **Per-trace output allocation preserved**: Downstream `mat_evals_per_trace` receives the same per-trace `DeviceMatrix` types.
6. **Height=0 or width=0 traces**: Filtered out during descriptor building (matching fold_ple.rs:78-80).
7. **save_memory accounting**: `mem_limit` subtraction for common_main (line 1533) uses per-trace buffer sizes — unchanged since per-trace allocation is preserved.

## Measurement Plan

1. Run benchmarks for APC {0, 100, 300}.
2. Primary metric: Round 0 at APC 300 — expect ~3-5ms reduction (181ms → 176-178ms) from faster fold_ple_evals.
3. STARK excl trace at APC 300 — expect same ~3-5ms reduction (1129ms → 1124-1126ms).
4. APC 0: expect no regression (batched path active with ~26 per-segment launches, overhead negligible).
5. nsight: `fold_ple_from_evals_kernel<false>` instances should drop from ~797 to ~2-4.

## Rollback Criteria

- Round 0 improvement at APC 300 < 3ms
- APC 0 regression > 5ms on STARK excl trace
- Correctness failure at any APC config
