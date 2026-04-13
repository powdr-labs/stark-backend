# Plan: Batch Stacking Scatter Kernel

## Goal

Replace the per-column loop in `stack_traces_into_expanded()` (`crates/cuda-backend/src/stacked_pcs.rs:145-179`) with a single batched scatter kernel. The current loop issues ~106K individual CUDA API calls at APC 300 (one `cudaMemcpyAsync` or one `batch_expand_pad_wide` kernel launch per trace column). Each API call has ~1-5us CPU overhead, creating an estimated ~100-150ms of CPU-side stalling across 2 segments that doesn't exist at APC 0 (~4K columns). A single kernel launch per invocation eliminates this overhead.

The 106K columns are split across 2 segments (~53K per segment). The function is called once per segment, so the savings are approximately ~50-75ms per segment for a total of ~100-150ms.

## Current Code Path

### Entry point

`stacked_commit()` at `crates/cuda-backend/src/stacked_pcs.rs:48` calls either:
- Path A (`cache_stacked_matrix=true`): `stack_traces()` → `stack_traces_into_expanded(layout, traces, &mut q_evals, height)` (line 126)
- Path B (`cache_stacked_matrix=false`): `rs_code_matrix()` → `stack_traces_into_expanded(layout, traces, &mut codewords, codeword_height)` (line 219)

Both paths call the same function with different `padded_height`.

**Benchmark configuration**: The default `GpuProverConfig` has `cache_stacked_matrix: false`, so the benchmark uses Path B. With Path B, `stack_traces_into_expanded` is called inside `rs_code_matrix` with `padded_height = codeword_height` (typically `height * 2^log_blowup`). Path A's `stack_traces()` is not called. The optimization benefits Path B directly.

### The loop (`stack_traces_into_expanded`, lines 134-181)

```rust
buffer.fill_zero()?;                              // single kernel: zeros entire stacked buffer
for (mat_idx, j, s) in &layout.sorted_cols {       // iterates over ALL columns from ALL AIRs
    let start = s.col_idx * padded_height + s.row_idx;
    let trace = traces[*mat_idx];
    let s_len = s.len(l_skip);
    if s.log_height() >= l_skip {
        // D2D memcpy: copy s_len elements from trace column j to stacked position
        cuda_memcpy::<true, true>(dst, src, s_len * size_of::<F>())?;
    } else {
        // Strided expansion: write trace.height() elements with stride into stacked position
        batch_expand_pad_wide(dst, src, trace.height() as u32, stride as u32, 1)?;
    }
}
```

At APC 300: `layout.sorted_cols` has ~106K entries total (106,842 columns across 623 AIR instances, split across 2 segments — ~53K per segment). Each iteration issues one CUDA API call:
- ~95K `cudaMemcpyAsync` calls total (D2D, ~1us CPU overhead each)
- ~11K `batch_expand_pad_wide` kernel launches total (~5us CPU overhead each)
- Total CPU overhead across both segments: ~95K × 1us + 11K × 5us ≈ ~150ms
- Per segment: ~75ms

Note on `batch_expand_pad_wide` parameter usage: the stacking code "abuses" this function with `width=trace.height()`, `padded_height=stride`, `height=1` — treating each source element as a separate "column" to create a strided output pattern. The scatter kernel replaces this with a simple `dst[i * stride] = src[i]` loop.

At APC 0: ~4K entries total → ~4ms overhead (negligible).

### Why this is slow

The GPU work per column is tiny (copying 8-256 elements = 32-1024 bytes). The actual GPU compute time for all copies combined is ~3ms. The bottleneck is purely CPU-side: iterating over 106K entries and making 106K CUDA runtime API calls.

## Changes

### Change 1: New CUDA kernel — `stack_columns_kernel`

**File**: `crates/cuda-backend/cuda/src/matrix.cu`

Add a new kernel and its extern "C" wrapper:

```cuda
struct StackColDesc {
    const Fp* src;       // source pointer (device memory)
    Fp* dst;             // destination pointer (device memory)
    uint32_t height;     // number of source elements to copy
    uint32_t stride;     // 1 for plain copy, >1 for strided expansion
};

__global__ void stack_columns_kernel(
    const StackColDesc* descs,
    uint32_t num_descs
) {
    uint32_t desc_idx = blockIdx.x;
    if (desc_idx >= num_descs) return;

    const StackColDesc& d = descs[desc_idx];
    for (uint32_t i = threadIdx.x; i < d.height; i += blockDim.x) {
        d.dst[i * d.stride] = d.src[i];
    }
}

extern "C" int _stack_columns(
    const StackColDesc* descs,
    uint32_t num_descs
) {
    if (num_descs == 0) return 0;
    dim3 grid(num_descs);
    dim3 block(256);
    stack_columns_kernel<<<grid, block>>>(descs, num_descs);
    return CHECK_KERNEL();
}
```

**Why**: Each CUDA block handles one column descriptor. Thread 0..255 within the block cooperatively copy elements from `src` to `dst` with the given stride. For a column with `height=64` and `stride=1`, 64 of the 256 threads do work. For `height=8` and `stride=4`, 8 threads do work. The kernel is launch-overhead-dominated (not compute-limited), so 256 threads per block is sufficient.

**Grid dimension limit**: `num_descs` can be up to ~53K per segment. CUDA grid.x supports up to 2^31-1, so this is fine.

### Change 2: Rust FFI wrapper

**File**: `crates/cuda-backend/src/cuda/matrix.rs`

Add FFI binding and safe wrapper:

```rust
#[repr(C)]
pub struct StackColDesc {
    pub src: *const F,
    pub dst: *mut F,
    pub height: u32,
    pub stride: u32,
}

// SAFETY: StackColDesc contains device pointers that are only dereferenced on GPU
unsafe impl Send for StackColDesc {}
unsafe impl Sync for StackColDesc {}

extern "C" {
    fn _stack_columns(descs: *const StackColDesc, num_descs: u32) -> i32;
}

pub unsafe fn stack_columns(descs: &DeviceBuffer<StackColDesc>) -> Result<(), CudaError> {
    CudaError::from_result(_stack_columns(descs.as_ptr(), descs.len() as u32))
}
```

**Why**: The `#[repr(C)]` struct matches the CUDA struct layout. The DeviceBuffer ensures descriptors are uploaded to GPU memory before the kernel reads them. Send+Sync are needed because the struct holds raw pointers that are only used on the GPU side.

### Change 3: Restructure `stack_traces_into_expanded()`

**File**: `crates/cuda-backend/src/stacked_pcs.rs`

Replace the per-column loop (lines 145-179) with a two-phase approach:

```rust
pub(crate) fn stack_traces_into_expanded(
    layout: &StackedLayout,
    traces: &[&DeviceMatrix<F>],
    buffer: &mut DeviceBuffer<F>,
    padded_height: usize,
) -> Result<(), StackTracesError> {
    let l_skip = layout.l_skip();
    debug_assert_eq!(padded_height % layout.height(), 0);
    debug_assert_eq!(buffer.len() % padded_height, 0);
    debug_assert_eq!(buffer.len() / padded_height, layout.width());
    buffer.fill_zero().map_err(StackTracesError::FillZero)?;

    // Phase 1: Build descriptor array on host (CPU-only, no CUDA API calls)
    let mut descs: Vec<StackColDesc> = Vec::with_capacity(layout.sorted_cols.len());
    for (mat_idx, j, s) in &layout.sorted_cols {
        let start = s.col_idx * padded_height + s.row_idx;
        let trace = traces[*mat_idx];
        let s_len = s.len(l_skip);
        debug_assert_eq!(trace.height(), 1 << s.log_height());
        let (src, height, stride) = if s.log_height() >= l_skip {
            debug_assert_eq!(trace.height(), s_len);
            let src = unsafe { trace.buffer().as_ptr().add(*j * s_len) };
            (src, s_len as u32, 1u32)
        } else {
            let stride = s.stride(l_skip);
            debug_assert_eq!(stride * trace.height(), s_len);
            let src = unsafe { trace.buffer().as_ptr().add(*j * trace.height()) };
            (src, trace.height() as u32, stride as u32)
        };
        let dst = unsafe { buffer.as_mut_ptr().add(start) };
        descs.push(StackColDesc { src, dst, height, stride });
    }

    // Phase 2: Upload descriptors to GPU and launch single scatter kernel
    if !descs.is_empty() {
        let d_descs = descs.to_device().map_err(StackTracesError::DescriptorUpload)?;
        unsafe { stack_columns(&d_descs).map_err(StackTracesError::StackColumns)? };
    }
    Ok(())
}
```

**Why**: Phase 1 iterates on the CPU to build the descriptor Vec — this has ~0 CUDA overhead per iteration (just pointer arithmetic and Vec::push). Phase 2 uploads the descriptor array to GPU with a single H2D copy (~2.5MB for ~53K descriptors per segment) and launches one kernel.

**Memory safety for H2D copy**: `to_device()` uses `cudaMemcpyAsync` on pageable (non-pinned) host memory. For pageable memory, the CUDA runtime internally stages the data before returning, so the host Vec can safely go out of scope after the call. The `fill_zero`, H2D copy, and scatter kernel all run on `cudaStreamPerThread`, so they execute in order on the GPU.

### Change 4: Add error variants

**File**: `crates/cuda-backend/src/error.rs` (where `StackTracesError` is defined, lines 130-137)

Add two new variants to `StackTracesError`:
- `DescriptorUpload(MemCopyError)` — for the H2D upload of the descriptor array (uses explicit `.map_err()`, not `#[from]`)
- `StackColumns(CudaError)` — for the scatter kernel launch

The existing `MemCopy(#[from] MemCopyError)` and `BatchExpandPadWide(CudaError)` variants become dead code after this change. Remove them to avoid confusion.

### Change 5: Update imports

**File**: `crates/cuda-backend/src/stacked_pcs.rs`

- Add `use openvm_cuda_common::copy::MemCopyH2D;` to import the `to_device()` method on `[StackColDesc]`.
- Add `use crate::cuda::matrix::{StackColDesc, stack_columns};` to import the new FFI types.
- Remove unused imports: `cuda_memcpy` from line 5, `batch_expand_pad_wide` from line 16 (both are no longer called).

The `MemCopyH2D` impl for `[T]` requires `T: Copy`. `StackColDesc` with `#[repr(C)]` and all `Copy` fields derives `Copy` automatically. If `MemCopyH2D` requires `T: Pod` or similar, implement the necessary trait for `StackColDesc`.

## Invariants

1. **Correctness**: The stacked buffer must contain identical data after the optimization. Each column's source data must be written to the same `(col_idx * padded_height + row_idx)` offset with the same stride.
2. **Zero padding**: The `fill_zero()` call must remain. The scatter kernel only writes non-zero elements; zeros between stacked columns are provided by fill_zero.
3. **Memory safety**: All source pointers (`src`) come from `traces[mat_idx].buffer()` which are valid GPU allocations. All destination pointers (`dst`) point into `buffer` which is a single contiguous GPU allocation. No overlapping writes (the stacking layout guarantees non-overlapping column positions).
4. **Ordering**: The kernel does not require any ordering between descriptors. All writes are to non-overlapping regions, so concurrent execution is safe.
5. **Existing callers**: Both Path A (`stack_traces()`, line 126) and Path B (`rs_code_matrix()`, line 219) call `stack_traces_into_expanded()` and will automatically benefit.

## Measurement Plan

### Before/After comparison

Run `openvm-riscv/scripts/run_pairing.sh` for APC {0, 100, 300} before and after the change.

### Primary metrics (from spec.py)

The 404ms "Trace Commit" at APC 300 is the sum across 2 segments (233ms + 166ms from `stacked_commit_time_ms`). Each segment calls `stack_traces_into_expanded()` once with ~53K columns. The savings are per-segment: ~50-75ms per segment, ~100-150ms total.

| Metric | Target |
|--------|--------|
| Trace Commit (APC 300) | < 330ms (18%+ improvement from 404ms) |
| STARK excl trace (APC 300) | < 1660ms (4%+ improvement from 1731ms) |
| STARK excl trace (APC 0) | No regression (< 2200ms, within noise of 2148ms) |
| All other sub-metrics (APC 300) | No regression (within ±5% of current) |

### Validation

Run `run_pairing.sh` for all three APC configs. All must complete prove+verify successfully.

### Profiling

Run with nsight to verify:
- `batch_expand_pad_wide_kernel` instances drop from ~12K to 0
- New `stack_columns_kernel` appears with ~1-2 instances per segment
- Total kernel launch count within Trace Commit is significantly reduced

## Rollback Criteria

Revert if ANY of the following:
- Trace Commit at APC 300 improves by less than 10% (< 40ms savings)
- STARK excl trace at APC 0 regresses by more than 3% (> 64ms increase)
- Any APC config fails prove+verify
- Memory usage increases by more than 10MB (descriptor array overhead)
