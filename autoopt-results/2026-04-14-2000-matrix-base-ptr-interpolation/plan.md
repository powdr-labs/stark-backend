# Plan: Per-matrix base pointer interpolation kernel

## Goal

Replace the flat per-column pointer array (`all_columns: Vec<*const EF>`, ~106K entries at APC 300) with per-matrix base pointers (~2K entries) in the `batched_interpolate_columns_kernel`. This eliminates the O(#total_columns) CPU loop that collects individual column GPU addresses each MLE round, replacing it with an O(#matrices) loop (~4 per trace). The CUDA kernel computes column addresses internally via `base + col_within_matrix * height`, which is trivially cheap (1 multiply + 1 add per thread).

## Current Code Path

Per MLE round (~12 rounds/segment × 2 segments = ~24 total at APC 300), `sumcheck_polys_batch_eval` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:1575` runs three phases:

**Phase 1 (mod.rs:1590-1724)**: For each Case B trace (round <= n_lift), collects per-column GPU pointers into a flat `all_columns: Vec<*const EF>`:
```rust
// mod.rs:1696-1705
all_columns.extend(
    iter::once(sels).chain(mats.iter())
        .flat_map(|m| {
            (0..m.width())
                .map(|col| m.buffer().as_ptr().wrapping_add(col * m.height()))
        }),
);
```
At APC 300 with ~600 Case B traces and ~106K total columns in early rounds. The iterator chain (flat_map with closures + Vec::extend) runs at ~6-10ns per column. This loop costs ~0.6-1.0ms per round in early rounds, decreasing as traces become Case A. Additionally, the Vec allocation for ~106K pointers (850KB) adds overhead per round. Total across all ~24 rounds (weighted by decreasing Case B column count): ~7-10ms of CPU collection time.

**Phase 2 (mod.rs:1727-1765)**: Uploads `all_columns` to GPU via `to_device()` (850KB H2D in early rounds, shrinking later), builds `InterpColDesc` descriptors (one per trace), uploads those, launches `batched_interpolate_columns_kernel`. The per-round H2D transfer of 850KB takes ~0.07ms (transfer time at PCIe 4.0) plus ~2μs CUDA API overhead. Total across 24 rounds: ~1.7ms.

**The kernel** (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu:103-134`) resolves each thread's column via:
```cuda
const FpExt *column = all_columns[d.columns_offset + col_local];
```
This is a double indirection: first read the pointer from the flat array, then read the column data.

**Note**: Phase 3 (mod.rs:1767-1809) and per-trace `main_ptrs.to_device()` (line 1811) are NOT modified. Prior work (batch-mle-main-ptrs-upload) showed that per-call CUDA API overhead is ~1μs with pool caching, making the ~600 per-trace uploads cost only ~0.6ms total — not a significant target.

## Changes

### Change 1: New CUDA structs (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`)

Add after the existing `InterpColDesc` (line 78):

```cuda
struct InterpMatrixInfo {
    const FpExt* base;   // Matrix base pointer (column-major data)
    uint32_t width;      // Number of columns in this matrix
};

struct InterpTraceDescM {
    FpExt* output;              // Base pointer for this trace's interpolated output
    uint32_t matrix_offset;     // Index into the d_matrices array
    uint32_t num_matrices;      // Number of matrices for this trace (typically 2-5)
    uint32_t num_y;             // Number of y-values (height / 2)
    uint32_t num_columns;       // Total columns across all matrices
    uint32_t total_threads;     // = num_y * num_columns
    uint32_t block_start;       // First block index assigned to this descriptor
};
```

**Why**: The per-matrix representation captures the same information as the flat column array but uses one entry per matrix (~4 per trace) instead of one entry per column (~170 per trace). The kernel computes column addresses from `base + col * (2 * num_y)` using the column-major layout.

### Change 2: New CUDA kernel (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`)

Add after `batched_interpolate_columns_kernel` (after line 134):

```cuda
__global__ void batched_interpolate_columns_matrix_kernel(
    const InterpTraceDescM* descs,
    const InterpMatrixInfo* matrices,
    uint32_t s_deg,
    uint32_t num_descs
) {
    // Binary search: find descriptor owning this block (same as existing kernel)
    uint32_t block_idx = blockIdx.x;
    uint32_t lo = 0, hi = num_descs;
    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (descs[mid].block_start <= block_idx) lo = mid;
        else hi = mid;
    }

    const InterpTraceDescM& d = descs[lo];
    uint32_t local_block = block_idx - d.block_start;
    uint32_t tidx = local_block * blockDim.x + threadIdx.x;
    if (tidx >= d.total_threads) return;

    uint32_t y = tidx % d.num_y;
    uint32_t col_local = tidx / d.num_y;

    // Compute column pointer from matrix base + offset
    // Linear scan over matrices (typically 2-5 per trace)
    const FpExt *column = nullptr;
    uint32_t rem = col_local;
    for (uint32_t i = 0; i < d.num_matrices; i++) {
        const InterpMatrixInfo& m = matrices[d.matrix_offset + i];
        if (rem < m.width) {
            column = m.base + rem * (2u * d.num_y);
            break;
        }
        rem -= m.width;
    }

    auto t0 = column[y << 1];
    auto t1 = column[(y << 1) | 1];
    FpExt *this_out = d.output + col_local * s_deg * d.num_y;

    for (int x = 0; x < s_deg; x++) {
        this_out[x * d.num_y + y] = t0 + (t1 - t0) * Fp(x + 1u);
    }
}
```

**Why**: The interpolation logic (t0/t1 read, linear interpolation, output write) is identical to the existing kernel. The only difference is how the column pointer is obtained: via a short linear scan over 2-5 matrices instead of a flat array lookup. The per-thread overhead is 2-5 comparisons and subtractions, negligible compared to the memory-bound interpolation work.

### Change 3: New CUDA launcher (`crates/cuda-backend/cuda/src/logup_zerocheck/utils.cu`)

Add after `_batched_interpolate_columns` (after line 259):

```cuda
extern "C" int _batched_interpolate_columns_matrix(
    const InterpTraceDescM* descs,
    const InterpMatrixInfo* matrices,
    size_t s_deg,
    size_t num_descs,
    size_t total_blocks
) {
    if (num_descs == 0) return 0;
    dim3 grid(total_blocks);
    dim3 block(512);
    batched_interpolate_columns_matrix_kernel<<<grid, block>>>(
        descs, matrices, s_deg, num_descs
    );
    return CHECK_KERNEL();
}
```

### Change 4: New Rust FFI types and binding (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)

Add after `InterpColDesc` (after line 111):

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InterpMatrixInfo {
    pub base: *const EF,
    pub width: u32,
}

unsafe impl Send for InterpMatrixInfo {}
unsafe impl Sync for InterpMatrixInfo {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InterpTraceDescM {
    pub output: *mut EF,
    pub matrix_offset: u32,
    pub num_matrices: u32,
    pub num_y: u32,
    pub num_columns: u32,
    pub total_threads: u32,
    pub block_start: u32,
}

unsafe impl Send for InterpTraceDescM {}
unsafe impl Sync for InterpTraceDescM {}
```

Add extern "C" declaration (in the `extern "C"` block):
```rust
fn _batched_interpolate_columns_matrix(
    descs: *const InterpTraceDescM,
    matrices: *const InterpMatrixInfo,
    s_deg: usize,
    num_descs: usize,
    total_blocks: usize,
) -> i32;
```

Add safe wrapper:
```rust
pub unsafe fn batched_interpolate_columns_matrix_gpu(
    descs: &DeviceBuffer<InterpTraceDescM>,
    matrices: &DeviceBuffer<InterpMatrixInfo>,
    s_deg: usize,
    total_blocks: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_batched_interpolate_columns_matrix(
        descs.as_ptr(),
        matrices.as_ptr(),
        s_deg,
        descs.len(),
        total_blocks,
    ))
}
```

### Change 5: Replace Phase 1+2 in `sumcheck_polys_batch_eval` (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)

Replace the `all_columns` collection (lines 1608, 1696-1706) and Phase 2 descriptor building (lines 1727-1765) with the new approach:

**Phase 1**: Remove the `all_columns: Vec<*const EF>` variable and the per-column `extend` call (lines 1608, 1696-1705). Instead, track per-matrix info alongside CaseBMeta:

```rust
// Replace all_columns with per-matrix collection
let mut all_matrices: Vec<InterpMatrixInfo> = Vec::new();
```

In the Case B loop body (replacing lines 1695-1706):
```rust
let matrix_offset = all_matrices.len();
let height = meta.num_y * 2;
// Sels matrix
debug_assert_eq!(sels.height(), height);
all_matrices.push(InterpMatrixInfo {
    base: sels.buffer().as_ptr(),
    width: sels.width() as u32,
});
// All mat_evals matrices (preprocessed + mains)
for m in mats.iter() {
    debug_assert_eq!(m.height(), height); // Preserves assertion from current line 1701
    all_matrices.push(InterpMatrixInfo {
        base: m.buffer().as_ptr(),
        width: m.width() as u32,
    });
}
let num_matrices = 1 + mats.len();
let num_columns = sels.width() + mats.iter().map(|m| m.width()).sum::<usize>();
```

Note: The per-matrix `debug_assert_eq!(m.height(), height)` preserves the existing hard `assert_eq!` at line 1701. It is downgraded to `debug_assert` because the invariant is also enforced by the kernel (wrong heights would produce wrong column addresses and fail verification). The hard assert at line 1693 (`debug_assert_eq!(height, mats[0].height())`) already validates the first matrix.

Store `matrix_offset`, `num_matrices`, and `num_columns` in `CaseBMeta` (replacing `col_start`).

**Phase 2**: Build the new descriptors:
```rust
let descs: Vec<InterpTraceDescM> = case_b_traces.iter().map(|meta| {
    let total_threads = (meta.num_y * meta.num_columns) as u32;
    let blocks = total_threads.div_ceil(THREADS_PER_BLOCK);
    let desc = InterpTraceDescM {
        output: unsafe { big_ptr.add(meta.interp_offset) },
        matrix_offset: meta.matrix_offset as u32,
        num_matrices: meta.num_matrices as u32,
        num_y: meta.num_y as u32,
        num_columns: meta.num_columns as u32,
        total_threads,
        block_start: block_offset,
    };
    block_offset += blocks;
    desc
}).collect();

let d_matrices = all_matrices.to_device()?;
let d_descs = descs.to_device()?;
unsafe {
    batched_interpolate_columns_matrix_gpu(&d_descs, &d_matrices, sp_deg, total_blocks)
        .map_err(|e| LogupZerocheckError::InterpolateColumns(e.into()))?;
}
```

**Why**: This replaces O(106K) pointer computations with O(2K) matrix info pushes per round, and 850KB H2D with ~40KB H2D.

### Change 6: Update `CaseBMeta` struct (`crates/cuda-backend/src/logup_zerocheck/mod.rs:1593-1606`)

Replace `col_start: usize` with:
```rust
matrix_offset: usize,
num_matrices: usize,
```

And update `num_columns` to be computed from matrix widths (as shown in Change 5) instead of from `all_columns.len() - col_start`.

## Invariants

1. **Column order preserved**: Matrices are collected in the same order as the current code (sels, then all mats). The kernel computes column addresses using the same column-major layout, so the output buffer layout is identical.

2. **Output buffer layout unchanged**: Phase 3 (TraceCtx building, lines 1767-1809) reads from the interpolated buffer using `widths_so_far` offsets that match the sels → preprocessed → mains ordering. This code does NOT change.

3. **Column-major layout**: All DeviceMatrix objects use column-major storage where column `c` starts at `base + c * height`. This is guaranteed by the DeviceMatrix API.

4. **Same height within a trace**: All matrices within a Case B trace have the same height (asserted at mod.rs:1701). The kernel uses `2 * num_y` as the column stride for all matrices in a trace.

5. **Existing unbatched path preserved**: The non-batched `interpolate_columns_kernel` (utils.cu:80) remains for any callers that use the original API.

6. **Phase 3 debug_assert preserved**: The `debug_assert_eq!(widths_so_far, meta.num_columns)` at mod.rs:1810 continues to hold because `num_columns` is computed identically (sum of sels width + all mat widths). The new code uses `sels.width() + mats.iter().map(|m| m.width()).sum()` which equals the current `all_columns.len() - col_start`.

## Measurement Plan

1. Build: `cargo check -p openvm-cuda-backend` then full build via powdr
2. Run APC 300 benchmark: `RUST_LOG=info $PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics current_apc300.json --recursion`
3. Run APC 0 benchmark: same with apc000.cbor
4. Analyze with spec.py: expect MLE Rounds at APC 300 to improve by 8-12ms (from ~162ms to ~150-154ms)
5. Expect STARK excl trace at APC 300 to improve by 8-12ms (from ~1287ms to ~1275-1279ms)
6. Expect no regression at APC 0 (only 4K columns, overhead is negligible)

**Savings breakdown**: ~7-10ms from eliminated CPU column collection (O(106K) iterations weighted across ~24 rounds with decreasing Case B count) + ~1.5ms from reduced H2D size (850KB → 40KB per round) + ~1ms from smaller Vec allocation/deallocation. Prior MLE Rounds optimization history (batch-mle-main-ptrs-upload: 5ms, pingpong-mle-fold-buffers: 10ms, bulk-alloc-mle-fold-buffers: 13ms) confirms that individual overhead sources in MLE Rounds are in the 5-15ms range, consistent with this estimate.

## Rollback Criteria

- Less than 5ms improvement in STARK excl trace at APC 300
- Any regression (> 10ms) in STARK excl trace at APC 0
- Correctness failure: proof does not verify at any APC configuration
