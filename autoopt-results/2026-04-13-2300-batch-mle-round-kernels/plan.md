# Plan: Batch MLE Round Kernels

## Goal

Replace the per-AIR kernel launch loop in `batch_sumcheck_poly_eval()` with batched kernel launches using descriptor arrays. This eliminates ~15K individual kernel launches per benchmark at APC 300, improving GPU utilization and reducing host-side overhead.

**Target**: Reduce MLE Rounds from 182ms to ~80-130ms at APC 300 (save 50-100ms). No regression at APC 0.

## Current Code Path

### Rust entry point
`crates/cuda-backend/src/stacked_reduction.rs:789` — `batch_sumcheck_poly_eval()`

Per MLE round (rounds 1..=n_stack):
1. **Line 831**: `fill_zero(d_accum)` — zeros the 8-element u64 atomic accumulator
2. **Line 828**: `eq_ub_per_trace.copy_to(d_eq_ub_all)` — bulk H2D upload (once per round)
3. **Lines 835-888**: Loop over `ht_diff_idxs.windows(2)`:
   - Compute per-window pointers: `unstacked_cols_ptr = d_unstacked_cols.as_ptr().add(window[0])`, `lambda_pows_ptr = d_lambda_pows.as_ptr().add(2 * window[0])`
   - **Line 845**: Branch on `log_height < l_skip + round` → degenerate vs non-degenerate
   - **Degenerate (line 852)**: `stacked_reduction_sumcheck_mle_round_degenerate()` — 1 block, min(window_len, 256) threads
   - **Non-degenerate (line 873)**: `stacked_reduction_sumcheck_mle_round()` — grid.x × grid.y blocks with auto-tuned stride
4. **Line 891**: `d_accum.to_host()` — single D2H copy + CPU reduction

### CUDA kernels
`crates/cuda-backend/cuda/src/stacked_reduction.cu`

**Degenerate kernel** (line 306): Single block per AIR. Each thread iterates over window columns at stride=blockDim.x. Loads per-column q_eval from global memory, multiplies by eq/lambda factors, accumulates via `block_reduce_sum` + `atomic_add_fpext_to_u64` to shared output.

**Non-degenerate kernel** (line 228): 2D grid (y_dim blocks × window_stride blocks). Each thread handles one y-point, iterates over window columns at stride=gridDim.y. Same atomic accumulation to shared output.

Both kernels accumulate into the SAME `d_accum` buffer via atomics → all per-AIR launches within a round are data-independent and can execute concurrently.

### FFI bindings
`crates/cuda-backend/src/cuda/stacked_reduction.rs`
- Lines 53-64: `_stacked_reduction_sumcheck_mle_round` extern declaration
- Lines 66-78: `_stacked_reduction_sumcheck_mle_round_degenerate` extern declaration
- Lines 209-231: Safe wrapper `stacked_reduction_sumcheck_mle_round()`
- Lines 245-275: Safe wrapper `stacked_reduction_sumcheck_mle_round_degenerate()`

### Why it's slow

At APC 300 with 2 segments (~311 AIRs/segment, ~17 MLE rounds):
- **10,348 degenerate kernel launches**: avg 3.9μs each, 40ms total GPU time
- **5,076 non-degenerate kernel launches**: avg 4.3μs each, 22ms total GPU time
- **Total**: 15,424 launches, 62ms GPU time, **182ms wall time**
- **120ms overhead** (66%) from: kernel launch latency (~30ms), GPU pipeline gaps between tiny launches (~40ms), CPU loop overhead (~20ms), sequential execution on single stream preventing SM utilization (~30ms)

Each degenerate launch uses 1 block on 1 SM. The RTX 4090 has 128 SMs. Effective utilization: ~0.8% per launch.

## Changes

### Change 1: Add descriptor structs (CUDA side)

**File**: `crates/cuda-backend/cuda/src/stacked_reduction.cu`

Add two new structs after the existing `UnstackedSlice` (line 34):

```c
struct DegenMleDesc {
    uint32_t col_offset;    // = window[0], indexes into unstacked_cols, eq_ub, lambda_pows
    uint32_t window_len;    // = window[1] - window[0]
    FpExt eq_r;             // from eq_stable[log_height]
    FpExt k_rot_r;          // from k_rot_stable[log_height]
};

struct NonDegenMleDesc {
    uint32_t col_offset;    // = window[0]
    uint32_t window_len;    // = window[1] - window[0]
    uint32_t num_y;         // = 1 << (log_height - l_skip - round)
    uint32_t stride;        // auto-tuned grid.y from existing heuristic
    uint32_t blocks_x;      // = ceil(num_y / 256)
};
```

### Change 2: Add batched degenerate kernel (CUDA side)

**File**: `crates/cuda-backend/cuda/src/stacked_reduction.cu`

Add new kernel after the existing degenerate kernel (after line 368):

```cuda
__global__ void batched_degenerate_mle_round_kernel(
    const DegenMleDesc *__restrict__ descs,
    const FpExt *__restrict__ const *__restrict__ q_evals,
    const FpExt *__restrict__ eq_ub_base,
    const UnstackedSlice *__restrict__ unstacked_cols_base,
    const FpExt *__restrict__ lambda_pows_base,
    uint64_t *__restrict__ output,
    uint32_t q_height,
    uint32_t shift_factor)
```

**Grid**: `(num_degenerate_airs, 1)`, **Block**: `(256, 1)`

Each block reads `descs[blockIdx.x]` to get its col_offset, window_len, eq_r, k_rot_r. Computes base pointers: `eq_ub_ptr = eq_ub_base + desc.col_offset`, `unstacked_cols = unstacked_cols_base + desc.col_offset`, `lambda_pows = lambda_pows_base + 2 * desc.col_offset`. Body is identical to existing degenerate kernel (lines 328-358) but using descriptor fields instead of kernel args. Threads with `threadIdx.x >= desc.window_len` contribute zero to the reduction (handled naturally by the for-loop bounds).

### Change 3: Add batched non-degenerate kernel (CUDA side)

**File**: `crates/cuda-backend/cuda/src/stacked_reduction.cu`

Add new kernel after the batched degenerate kernel:

```cuda
__global__ void batched_nondegen_mle_round_kernel(
    const NonDegenMleDesc *__restrict__ descs,
    const uint32_t *__restrict__ block_prefix_sums,
    uint32_t num_descs,
    const FpExt *__restrict__ const *__restrict__ q_evals,
    const FpExt *__restrict__ eq_r_ns,
    const FpExt *__restrict__ k_rot_ns,
    const UnstackedSlice *__restrict__ unstacked_cols_base,
    const FpExt *__restrict__ lambda_pows_base,
    uint64_t *__restrict__ output,
    uint32_t q_height)
```

**Grid**: `(total_blocks, 1)` where `total_blocks = sum of (blocks_x * stride) for all non-degenerate AIRs`, **Block**: `(256, 1)`

Each block uses binary search on `block_prefix_sums` to find its AIR index (`air_idx`), then computes its local block index within that AIR. Decomposes local index into y-block and stride-block indices:
- `y_block = local_block % desc.blocks_x` — corresponds to original `blockIdx.x` (y-dimension blocks)
- `stride_block = local_block / desc.blocks_x` — corresponds to original `blockIdx.y` (window stride)
- `y_int = y_block * 256 + threadIdx.x` — same as original `blockIdx.x * blockDim.x + threadIdx.x`
- `window_idx_base = stride_block` — same as original `blockIdx.y`

Computes base pointers: `unstacked_cols = unstacked_cols_base + desc.col_offset`, `lambda_pows = lambda_pows_base + 2 * desc.col_offset`. Rest of computation (lines 254-291 of existing kernel): iterate over window columns at stride `desc.stride`, load q_evals, compute eq/k_rot products, block reduce, atomic add to output.

Binary search is O(log N) per block — at most ~10 comparisons for 600 AIRs. Negligible overhead.

### Change 4: Add launcher functions (CUDA side)

**File**: `crates/cuda-backend/cuda/src/stacked_reduction.cu`

Add two new extern "C" launcher functions after the existing launchers (after line 570):

`_batched_stacked_reduction_sumcheck_mle_round_degenerate`:
- Receives: descs ptr, num_descs, shared array pointers, q_height, shift_factor
- Grid: `(num_descs, 1)`, Block: `(256, 1)`
- shmem: `div_ceil(256, WARP_SIZE) * sizeof(FpExt) = 8 * 16 = 128 bytes`

`_batched_stacked_reduction_sumcheck_mle_round`:
- Receives: descs ptr, block_prefix_sums ptr, num_descs, total_blocks, shared array pointers, q_height
- Grid: `(total_blocks, 1)`, Block: `(256, 1)`
- shmem: same 128 bytes
- The Rust side (Changes 6-7) pre-computes stride and blocks_x per descriptor and uploads them via the descriptor array. The CUDA launcher simply reads from descriptors.

### Change 5: Add Rust descriptor types and FFI bindings

**File**: `crates/cuda-backend/src/cuda/stacked_reduction.rs`

Add `#[repr(C)]` Rust structs matching the CUDA descriptors:

```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DegenMleDesc {
    col_offset: u32,
    window_len: u32,
    eq_r: EF,
    k_rot_r: EF,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NonDegenMleDesc {
    col_offset: u32,
    window_len: u32,
    num_y: u32,
    stride: u32,
    blocks_x: u32,
}
```

Add extern declarations for the new launcher functions:

```rust
fn _batched_stacked_reduction_sumcheck_mle_round_degenerate(
    descs: *const DegenMleDesc,
    num_descs: u32,
    q_evals: *const *const EF,
    eq_ub_base: *const EF,
    unstacked_cols_base: *const UnstackedSlice,
    lambda_pows_base: *const EF,
    output: *mut u64,
    q_height: u32,
    shift_factor: u32,
) -> i32;

fn _batched_stacked_reduction_sumcheck_mle_round(
    descs: *const NonDegenMleDesc,
    block_prefix_sums: *const u32,
    num_descs: u32,
    total_blocks: u32,
    q_evals: *const *const EF,
    eq_r_ns: *const EF,
    k_rot_ns: *const EF,
    unstacked_cols_base: *const UnstackedSlice,
    lambda_pows_base: *const EF,
    output: *mut u64,
    q_height: u32,
) -> i32;
```

Add safe Rust wrappers. The non-degenerate wrapper extracts raw pointers from `EqEvalSegments` via `.buffer.as_ptr()` (matching the existing pattern at `crates/cuda-backend/src/cuda/stacked_reduction.rs:225-226`).

### Change 6: Modify batch_sumcheck_poly_eval to use batched launches

**File**: `crates/cuda-backend/src/stacked_reduction.rs`

Replace the per-window loop (lines 835-888) with:

**Phase A — Build descriptors** (CPU-only, no CUDA calls):
```rust
let mut degen_descs: Vec<DegenMleDesc> = Vec::new();
let mut nondegen_descs: Vec<NonDegenMleDesc> = Vec::new();
let mut nondegen_prefix_sums: Vec<u32> = Vec::new();
let mut total_nondegen_blocks: u32 = 0;

for window in self.ht_diff_idxs.windows(2) {
    let col_offset = window[0] as u32;
    let window_len = (window[1] - window[0]) as u32;
    let log_height = self.unstacked_cols[window[0]].log_height as usize;

    if log_height < l_skip + round {
        degen_descs.push(DegenMleDesc {
            col_offset,
            window_len,
            eq_r: self.eq_stable[log_height],
            k_rot_r: self.k_rot_stable[log_height],
        });
    } else {
        let hypercube_dim = log_height - l_skip - round;
        let num_y = 1u32 << hypercube_dim;
        let (blocks_x, stride) = compute_mle_launch_params(num_y, window_len, self.sm_count);
        nondegen_prefix_sums.push(total_nondegen_blocks);
        total_nondegen_blocks += blocks_x * stride;
        nondegen_descs.push(NonDegenMleDesc {
            col_offset, window_len, num_y, stride, blocks_x,
        });
    }
}
```

**Phase B — Upload descriptors and launch batched kernels**:
```rust
let stacked_height = self.stacked_height(round);
let shift_factor = (l_skip + round) as u32;

if !degen_descs.is_empty() {
    let d_degen_descs = degen_descs.to_device()?;
    batched_stacked_reduction_sumcheck_mle_round_degenerate(
        &d_degen_descs,
        &self.d_q_eval_ptrs,
        self.d_eq_ub_all.as_ptr(),
        self.d_unstacked_cols.as_ptr(),
        self.d_lambda_pows.as_ptr(),
        &mut self.d_accum,
        stacked_height,
        shift_factor,
    )?;
}

if !nondegen_descs.is_empty() {
    let d_nondegen_descs = nondegen_descs.to_device()?;
    let d_block_offsets = nondegen_prefix_sums.to_device()?;
    batched_stacked_reduction_sumcheck_mle_round(
        &d_nondegen_descs,
        &d_block_offsets,
        total_nondegen_blocks,
        &self.d_q_eval_ptrs,
        self.eq_r_ns.buffer.as_ptr(),   // raw EF pointer from EqEvalSegments
        self.k_rot_ns.buffer.as_ptr(),  // raw EF pointer from DeviceBuffer
        self.d_unstacked_cols.as_ptr(),
        self.d_lambda_pows.as_ptr(),
        &mut self.d_accum,
        stacked_height,
    )?;
}
```

### Change 7: Add stride computation helper

**File**: `crates/cuda-backend/src/cuda/stacked_reduction.rs`

Add a Rust function that exactly replicates the auto-tuning heuristic from `stacked_reduction.cu` lines 505-524:

```rust
const MAX_GRID_DIM: u32 = 65535;
const WAVES_TARGET: u32 = 4;
const ITERS_MIN: u32 = 4;
const ITERS_MAX: u32 = 16;

/// Replicates the stride auto-tuning from _stacked_reduction_sumcheck_mle_round.
/// Returns (blocks_x, stride) matching the CUDA-side grid dimensions.
pub(crate) fn compute_mle_launch_params(num_y: u32, window_len: u32, sm_count: u32) -> (u32, u32) {
    let blocks_x = num_y.div_ceil(256);

    let stride_occ = (sm_count * WAVES_TARGET).div_ceil(blocks_x);
    let stride_loop_lo = window_len.div_ceil(ITERS_MAX);
    let stride_loop_hi = window_len.div_ceil(ITERS_MIN);

    let lo = 1.max(stride_occ.max(stride_loop_lo));
    let hi = window_len.min(MAX_GRID_DIM).min(stride_loop_hi);

    let stride = if lo <= hi {
        lo
    } else {
        lo.min(window_len.min(MAX_GRID_DIM))
    };

    (blocks_x, stride)
}
```

This keeps the auto-tuning logic accessible from Rust without an FFI call, needed because we build descriptors on the CPU before any kernel launch. Constants and branch structure match the CUDA code exactly.

### Change 8: Pre-allocate descriptor buffers (optional optimization)

**File**: `crates/cuda-backend/src/stacked_reduction.rs`

Add fields to `StackedReductionGpu`:
```rust
d_degen_descs: DeviceBuffer<DegenMleDesc>,
d_nondegen_descs: DeviceBuffer<NonDegenMleDesc>,
d_block_offsets: DeviceBuffer<u32>,
```

Pre-allocate these with capacity equal to `ht_diff_idxs.len() - 1` during construction. Use `copy_to()` instead of `to_device()` per round to avoid per-round allocation.

This eliminates ~34 per-round DeviceBuffer allocations (17 rounds × 2 descriptor types). This is a secondary optimization; correctness and the main speedup come from the batched kernels themselves.

## Invariants

1. **Correctness**: The batched kernels must produce identical atomic accumulations to the per-AIR kernels. All kernel arguments and data access patterns must match exactly. The only difference is how blocks are mapped to AIRs.

2. **Atomic accumulator semantics**: All blocks across both batched kernels atomically add to the same `d_accum` buffer. The u64 overflow guarantee holds: at APC 300, total atomic adds = ~608 degenerate blocks + ~2000 non-degenerate blocks = ~2608 adds per accumulator entry, well below the 2^33 overflow limit.

3. **APC 0 path unchanged**: With <100 AIR instances at APC 0, most windows are non-degenerate with large num_y. The batched kernels still handle this correctly — the non-degenerate batch will have fewer descriptors but each with more blocks, and the degenerate batch may be empty.

4. **Descriptor layout matches**: `#[repr(C)]` on Rust structs ensures memory layout matches the CUDA structs. Field order and types must be identical.

5. **Block-to-AIR mapping**: The prefix-sum array for non-degenerate batching must be computed correctly. Each entry `block_prefix_sums[i]` = sum of blocks for descriptors 0..i (exclusive prefix sum). A block with `blockIdx.x == block_prefix_sums[i]` belongs to descriptor `i`.

6. **Stride auto-tuning**: The Rust-side stride computation must exactly replicate the CUDA-side heuristic (lines 505-524 of stacked_reduction.cu) to ensure equivalent parallelism and correctness.

7. **fill_zero and to_host ordering**: `fill_zero(d_accum)` must complete before any batched kernel launch. Both batched kernels must complete before `to_host(d_accum)`. This is guaranteed by same-stream ordering (all on the default stream).

## Measurement Plan

### Before implementation
Run `openvm-riscv/scripts/run_pairing.sh` in the powdr repo. Save metrics as `before_apc{000,100,300}.json`.

### After implementation
Run the same benchmark. Save metrics as `after_apc{000,100,300}.json`.

### Verification
1. All three APC configs must prove + verify successfully (the benchmark script includes verification)
2. Use `spec.py` to compare before/after metrics
3. Key metrics to check:
   - MLE Rounds at APC 300: expect 80-130ms (down from ~182ms)
   - STARK excl trace at APC 300: expect ~1390-1440ms (down from ~1493ms)
   - APC 0 STARK excl trace: must not regress >5% (i.e., stay under 2260ms)

### Profiling
Run nsys on APC 300 after implementation. Verify:
- Degenerate kernel launches reduced from ~10K to ~17 per benchmark
- Non-degenerate kernel launches reduced from ~5K to ~17 per benchmark
- GPU utilization during MLE phase improved

## Rollback Criteria

Revert if ANY of the following:
1. **MLE Rounds at APC 300 does not improve by at least 30ms** (less than ~17% improvement)
2. **STARK excl trace at APC 0 regresses by more than 65ms** (>3% regression)
3. **GPU OOM** on any APC config
4. **Correctness failure**: prove+verify fails on any APC config
5. **STARK excl trace at APC 300 does not improve by at least 25ms** (less than ~1.7% overall improvement)
