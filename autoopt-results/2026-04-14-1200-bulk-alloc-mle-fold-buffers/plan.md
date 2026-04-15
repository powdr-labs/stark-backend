# Plan: Bulk-Allocate MLE Fold Output Buffers

## Goal

Eliminate ~37,000 per-round `cudaMallocAsync`/`cudaFreeAsync` calls in MLE sumcheck fold rounds by replacing per-matrix individual allocations with a single bulk allocation per fold group per round. This targets the main-thread allocation overhead visible in nsight (47K mallocs for 38ms + 49K frees for 21ms = 59ms total), which is the primary cause of MLE Rounds reverse-scaling at APC 300 (173ms vs 114ms at APC 0 despite 2x less total work).

**Expected saving**: 40–55ms at APC 300 (MLE Rounds 173ms → ~118–133ms).

**Scope**: Only the MLE fold loop (`fold_mle_evals`, rounds 1..n_max). The one-time `fold_ple_evals` (round 0, ~2500 allocs, ~2ms overhead) is NOT changed — it continues to use `DeviceMatrix<EF>` and is handled by a transitional type in the first MLE fold round.

## Current Code Path

### MLE round loop (`mod.rs:631–648`)

Each round calls:
1. `sumcheck_polys_batch_eval(round, r[round-1])` — evaluates sumcheck polynomials for all traces
2. `compute_batch_s_poly(...)` — CPU batching
3. `fold_mle_evals(round, r_round)` — halves all trace matrix heights via GPU kernel

### `fold_mle_evals` (`mod.rs:2000–2091`)

The inner `batch_fold` closure (line 2002–2048):
1. **Counts foldable matrices**: `partition_point(|mat| mat.height() > 1)` → `num_matrices`
2. **Allocates per-matrix output buffers** (line 2014):
   ```rust
   let output_mat = DeviceMatrix::<EF>::with_capacity(output_height, width);
   ```
   At APC 300: ~1246 matrices for mat_evals + ~623 for sels = **~1869 individual allocations per round**.
3. **Collects pointer arrays**, uploads 4 small H2D buffers (input_ptrs, output_ptrs, heights, widths)
4. **Single `batch_fold_mle` kernel launch** — one kernel folds all matrices
5. **Appends unfoldable matrices** (height=1) unchanged

Over ~20 MLE rounds: **~37,380 allocs + ~37,380 frees = ~74,760 CUDA memory API calls**.

### Consumers of `mat_evals_per_trace`

`mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>` is accessed at:
- `sumcheck_polys_batch_eval` (line 1574): `.iter()` for metadata loop — reads `buffer().as_ptr()`, `height()`, `width()` per matrix
- `sumcheck_polys_batch_eval` (line 1730): `&self.mat_evals_per_trace[meta.trace_idx]` — indexed access for Case B
- `fold_mle_evals` (line 2051–2066): `std::mem::take`, flatten, fold, restructure
- `into_column_openings` (line 2107): `std::mem::take`, D2H copy via `transport_matrix_d2h_col_major(&mat)`
- Memory accounting (line 2069): `.buffer().len()` sum

`sels_per_trace: Vec<DeviceMatrix<EF>>` has the same access patterns in `batch_fold` and `sumcheck_polys_batch_eval`.

### Why it's slow at APC 300

At APC 300 (623 AIR instances, ~1869 matrices), each fold round allocates ~1869 output buffers via `cudaMallocAsync`. Even though the CUDA pool allocator caches memory (making per-call overhead ~0.8μs), the aggregate over 20 rounds is ~30K alloc calls + ~30K free calls = ~50ms. Combined with CPU-side metadata iteration over 623 traces per round, the total MLE per-round overhead is ~5.5ms at APC 300 vs ~1.5ms at APC 0 (99 AIRs, ~200 matrices per fold).

Quantitative estimate from nsight: the fold accounts for ~(37K/47K) × 38ms allocs + ~(37K/49K) × 21ms frees ≈ 31ms + 16ms = **~47ms** of pure CUDA memory API overhead. Additional savings from reduced memory pool fragmentation bring the total to ~40–55ms.

## Changes

### Change 1: Add `ArenaMatrix<T>` type (`crates/cuda-backend/src/base.rs`)

A lightweight non-owning matrix type that stores a raw pointer + dimensions:

```rust
#[derive(Clone, Copy)]
pub struct ArenaMatrix<T> {
    ptr: *mut T,
    height: usize,
    width: usize,
}

unsafe impl<T> Send for ArenaMatrix<T> {}
unsafe impl<T> Sync for ArenaMatrix<T> {}
```

Methods: `new(ptr, height, width)`, `as_ptr() -> *const T`, `as_mut_ptr() -> *mut T`, `height()`, `width()`, `buffer_len() -> usize` (returns `height * width`).

Add a `to_host(&self) -> Result<Vec<T>, MemCopyError>` method that performs a D2H copy of `height * width` elements from `ptr`. This requires:
1. Calling `cudaMemcpyAsync` directly (import from `openvm_cuda_common::stream::{cudaStreamPerThread, cudaStream_t}`), since there is no backing `DeviceBuffer` to delegate to.
2. Synchronizing after the async copy using the same `COPY_EVENT.lock().unwrap().record_and_wait(cudaStreamPerThread)` pattern used in `DeviceBuffer::to_host` (`crates/cuda-common/src/copy.rs:101-125`), ensuring the host Vec contents are valid before returning.

Derive `Copy + Clone` — all fields are `Copy`-safe. This is needed for `extend_from_slice` on unfoldable height-1 matrices in `batch_fold`.

**Why `DeviceMatrixView` is not used**: The existing `DeviceMatrixView<'a, T>` in `base.rs:117-156` has a lifetime parameter `'a`. Using it for `mat_evals_per_trace` would require adding a lifetime to `LogupZerocheckGpu`, which would create a self-referential borrow (the arena and the views are both fields on the same struct). `ArenaMatrix` avoids this by omitting the lifetime — safety is upheld structurally because `FoldArena` (the owner) and `mat_evals_per_trace` (the views) are fields on the same struct, with the struct's `into_column_openings` consuming all views before drop.

### Change 2: Add `MatrixRef<T>` enum (`crates/cuda-backend/src/base.rs`)

To handle the transition from `DeviceMatrix` (output of `fold_ple_evals`) to `ArenaMatrix` (output of MLE fold), use an enum:

```rust
#[derive(Clone)]
pub enum MatrixRef<T> {
    Owned(DeviceMatrix<T>),
    Arena(ArenaMatrix<T>),
}
```

Methods (delegating to inner type): `as_ptr()`, `height()`, `width()`, `buffer_len()`, `to_host()`.

**Why**: `fold_ple_evals` (round 0) produces `DeviceMatrix<EF>` via existing `fold_ple_evals_rotate` calls. The first MLE fold round reads these `Owned` matrices and produces `Arena` matrices. From round 2 onward, all matrices are `Arena`. This avoids modifying `fold_ple_evals` entirely.

### Change 3: Add `FoldArena<T>` type (`crates/cuda-backend/src/logup_zerocheck/mod.rs`)

```rust
struct FoldArena<T> {
    /// Big buffer(s) backing ArenaMatrix views. Grows by one buffer per
    /// allocate_bulk call. Older buffers are kept alive because height-1
    /// ArenaMatrix entries from earlier rounds still point into them.
    /// Total memory: bounded at ~2× the first round's allocation
    /// (geometric sum 1 + 1/2 + 1/4 + ... < 2).
    buffers: Vec<DeviceBuffer<T>>,
}
```

Methods:
- `new() -> Self`
- `allocate_bulk(total_cells: usize) -> *mut T`: Allocates one `DeviceBuffer` of `total_cells`, pushes to `self.buffers`, returns `as_mut_ptr()`.
- When `FoldArena` drops, all buffers are freed via their normal `DeviceBuffer::drop`.

### Change 4: Change storage types in `LogupZerocheckGpu`

```rust
// Before:
mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>,
sels_per_trace: Vec<DeviceMatrix<EF>>,

// After:
mat_evals_per_trace: Vec<Vec<MatrixRef<EF>>>,
sels_per_trace: Vec<MatrixRef<EF>>,
fold_arena: FoldArena<EF>,
```

`fold_ple_evals` (unchanged) populates these with `MatrixRef::Owned(DeviceMatrix<EF>)` values. The MLE fold loop transitions them to `MatrixRef::Arena(ArenaMatrix<EF>)` values from round 1 onward.

### Change 5: Rewrite `batch_fold` in `fold_mle_evals` (`mod.rs:2002–2048`)

Replace per-matrix allocation with arena-based bulk allocation. The closure now accepts `MatrixRef<EF>` inputs:

```rust
let batch_fold = |input_mats: Vec<MatrixRef<EF>>,
                  arena: &mut FoldArena<EF>|
    -> Result<Vec<MatrixRef<EF>>, LogupZerocheckError>
{
    let num_matrices = input_mats.partition_point(|mat| mat.height() > 1);

    // Compute total output cells and per-matrix metadata
    let mut total_cells = 0usize;
    let mut offsets = Vec::with_capacity(num_matrices);
    let mut log_output_heights = Vec::with_capacity(num_matrices);
    let mut widths_u32 = Vec::with_capacity(num_matrices);
    let mut max_output_cells = 0usize;

    for mat in input_mats.iter().take(num_matrices) {
        let h = mat.height() >> 1;
        let w = mat.width();
        let cells = h * w;
        offsets.push(total_cells);
        total_cells += cells;
        max_output_cells = max(max_output_cells, cells);
        log_output_heights.push(h.ilog2() as u8);
        widths_u32.push(w as u32);
    }

    if total_cells == 0 {
        // All matrices already at height 1
        return Ok(input_mats);
    }

    // ONE allocation instead of ~1869
    let arena_ptr = arena.allocate_bulk(total_cells);

    // Build output matrices as arena views
    let output_arena_mats: Vec<ArenaMatrix<EF>> = offsets.iter().enumerate().map(|(i, &off)| {
        let h = input_mats[i].height() >> 1;
        let w = input_mats[i].width();
        ArenaMatrix::new(unsafe { arena_ptr.add(off) }, h, w)
    }).collect();

    // Collect pointers for CUDA kernel (reads from input, writes to arena)
    let input_ptrs: Vec<_> = input_mats.iter().take(num_matrices)
        .map(|mat| mat.as_ptr()).collect();
    let output_ptrs: Vec<_> = output_arena_mats.iter()
        .map(|mat| mat.as_mut_ptr()).collect();

    // Upload and launch (same kernel as before)
    let d_input_ptrs = input_ptrs.to_device()?;
    let d_output_ptrs = output_ptrs.to_device()?;
    let d_log_output_heights = log_output_heights.to_device()?;
    let d_widths = widths_u32.to_device()?;

    unsafe {
        batch_fold_mle(
            &d_input_ptrs, &d_output_ptrs, &d_widths,
            num_matrices.try_into().unwrap(),
            &d_log_output_heights,
            max_output_cells.try_into().unwrap(),
            r_round,
        ).map_err(LogupZerocheckError::BatchFoldMle)?;
    }

    // Build output: arena matrices for folded + carried forward for height-1
    let mut output_mats: Vec<MatrixRef<EF>> = output_arena_mats
        .into_iter()
        .map(MatrixRef::Arena)
        .collect();
    // Carry forward unfoldable matrices unchanged (Copy/Clone via MatrixRef)
    for mat in &input_mats[num_matrices..] {
        output_mats.push(mat.clone());
    }
    Ok(output_mats)
};
```

**Allocation reduction**: 1 bulk `cudaMallocAsync` per fold call instead of ~1869 individual calls. Over 20 rounds × 2 groups (mat_evals + sels) = **~40 arena allocs + ~160 auxiliary allocs (pointer arrays) = ~200 total**, down from ~37,380.

### Change 6: Update `sumcheck_polys_batch_eval` consumers (`mod.rs:1536–1922`)

Mechanical replacements. All access through `MatrixRef` delegate methods:
- `mats[i].buffer().as_ptr()` → `mats[i].as_ptr()`
- `m.buffer().as_ptr().wrapping_add(col * m.height())` → `m.as_ptr().wrapping_add(col * m.height())`
- `mats[i].height()` → `mats[i].height()` (unchanged)
- `mats[i].width()` → `mats[i].width()` (unchanged)

These are ~15 call sites that access buffer pointers.

### Change 7: Update `into_column_openings` (`mod.rs:2098–2172`)

Replace `transport_matrix_d2h_col_major(&mat)` with `MatrixRef::to_host()`:

```rust
let host_data = mat.to_host()?;
let mat_host = ColMajorMatrix::new(host_data, mat.width());
```

At this point all matrices have height=1, so these are tiny copies (width elements each). The `to_host()` method on `MatrixRef` dispatches to either `DeviceMatrix::to_host()` (Owned) or `ArenaMatrix::to_host()` (Arena).

### Change 8: Update `save_memory` accounting (`mod.rs:2067–2075`)

Replace `m.buffer().len() * size_of::<EF>()` with `m.buffer_len() * size_of::<EF>()`. `MatrixRef::buffer_len()` returns `height * width`, matching the current `DeviceBuffer::len()` semantics.

## Invariants

1. **No double-free**: `ArenaMatrix` is `Copy` — it stores a raw pointer but does NOT free memory on drop. Only `FoldArena::drop` frees the big buffers. The `DeviceMatrix` values from `fold_ple_evals` are dropped when they're replaced in the first MLE fold round — their individual `Arc<DeviceBuffer>` handles cleanup normally.

2. **Arena lifetime**: `FoldArena` is a field of `LogupZerocheckGpu` and outlives all `ArenaMatrix` references (stored in `mat_evals_per_trace` and `sels_per_trace` on the same struct). `into_column_openings` consumes `self`, so arena data is read before the struct is dropped. Drop order within a Rust struct is field-declaration order, but since `into_column_openings` takes `self` by value and explicitly processes the matrices before returning, there's no use-after-free regardless of field order.

3. **Arena memory bound**: The arena accumulates buffers over ~20 rounds. Each round's total is approximately half the previous (since fold halves heights). Total arena memory: `S × (1 + 1/2 + 1/4 + ...) < 2S` where S is the first-round total. This matches the current code's memory usage (old `DeviceMatrix` values for height-1 entries are kept alive by `Arc` until replaced).

4. **Correctness**: The CUDA `batch_fold_mle` kernel receives the same pointer arrays as before. The kernel doesn't care whether output pointers come from individual allocations or a single bulk allocation. The fold algorithm is unchanged.

5. **Unfoldable matrices**: Matrices that reach height=1 are carried forward by value via `MatrixRef::clone()`. For `MatrixRef::Owned`, this clones the `Arc`. For `MatrixRef::Arena`, this copies the raw pointer (which is `Copy`). Both cases are correct.

6. **APC 0 behavior**: At APC 0 (~99 AIRs, ~200 matrices), the bulk allocation still applies but the savings are minimal (~3ms). No regression expected since the big buffer allocation amortizes the overhead of 200 individual allocations.

## Measurement Plan

**Benchmark commands** (in powdr repo):
```bash
RUST_LOG=info APC=300 openvm-riscv/scripts/run_pairing.sh  # Target metric
RUST_LOG=info APC=0 openvm-riscv/scripts/run_pairing.sh    # Regression check
```

**Analysis**:
```bash
python3 /home/georg/spec.py results/pairing/apc300/metrics.json after_apc300
python3 /home/georg/spec.py results/pairing/apc000/metrics.json after_apc000
```

**Expected results**:
- MLE Rounds at APC 300: 173ms → 118–133ms (23–32% reduction)
- STARK excl trace at APC 300: 1322ms → ~1267–1282ms
- APC 0: No significant change (±5ms noise)

**Nsight verification**:
- Main-thread `cudaMallocAsync` call count should drop from ~47K to ~10K (eliminating ~37K fold allocs)
- Main-thread `cudaFreeAsync` call count should drop similarly

## Rollback Criteria

Revert if:
- MLE Rounds improvement at APC 300 is < 20ms (below noise threshold for a 173ms metric)
- STARK excl trace improvement at APC 300 is < 30ms
- APC 0 regresses by > 15ms
- Correctness failures (proof verification fails at any APC config)
