# Plan: Ping-Pong MLE Fold Buffers

## Goal

Eliminate ~67K per-benchmark DeviceMatrix allocations and ~67K frees in the constraints-phase MLE fold (`fold_mle_evals`) by replacing them with two pre-allocated contiguous device buffers used in a ping-pong pattern. MLE Rounds at APC 300 take 173ms, of which ~15-20ms is CUDA memory pool API overhead from these allocations (confirmed by the bulk-alloc-mle-fold-buffers task which recovered 13ms with a heavier ArenaMatrix approach). This cleaner implementation avoids the ArenaMatrix/MatrixRef/FoldArena type complexity of the previous attempt.

## Current Code Path

### MLE Rounds loop
`crates/cuda-backend/src/logup_zerocheck/mod.rs:631-648`
```rust
for round in 1..=n_max {
    let sp_round_evals = prover.sumcheck_polys_batch_eval(round, r[round - 1])?;
    // ... transcript observe, sample r_round ...
    prover.fold_mle_evals(round, r_round)?;
}
```

### fold_mle_evals
`crates/cuda-backend/src/logup_zerocheck/mod.rs:1999-2091`

Contains `batch_fold` closure (lines 2002-2048):
1. Counts foldable matrices (height > 1) via `partition_point` (line 2003)
2. Allocates output: `DeviceMatrix::<EF>::with_capacity(output_height, width)` per foldable matrix (line 2014) → `cudaMallocAsync`
3. Collects input/output pointer Vecs (lines 2019-2027)
4. Uploads 4 pointer arrays to device (lines 2029-2032)
5. Launches single `batch_fold_mle` kernel (lines 2034-2044)
6. Returns new output matrices, dropping old inputs → `cudaFreeAsync` each

Called twice per round:
- `mat_evals_per_trace` (lines 2050-2066): ~800 matrices at APC 300
- `sels_per_trace` (line 2078): ~623 matrices

Per round: ~1400 allocs + ~1400 frees = ~2800 CUDA API calls.
Per benchmark (24 rounds × 2 segments): ~134K API calls (~67K allocs, ~67K frees).
At 0.3-0.5μs per call: **20-33ms total overhead** (previous task confirmed 13ms recovery, suggesting ~15-20ms was actual overhead).

### sumcheck_polys_batch_eval
`crates/cuda-backend/src/logup_zerocheck/mod.rs:1536-1922`

Reads `self.mat_evals_per_trace[trace_idx][mat_idx]` and `self.sels_per_trace[trace_idx]` to get `.buffer().as_ptr()`, `.height()`, `.width()`. These are the DeviceMatrix objects being replaced.

### into_column_openings
`crates/cuda-backend/src/logup_zerocheck/mod.rs:2098+`

Calls `m.to_host()` on each final DeviceMatrix to get column openings. Must still work after replacement.

## Changes

### 1. Add ping-pong buffer state to LogupZerocheckGpu

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add fields to `LogupZerocheckGpu`:
```rust
/// Ping-pong buffer A for mat_evals fold (current round's data)
mat_fold_buf_a: DeviceBuffer<EF>,
/// Ping-pong buffer B for mat_evals fold (next round's output)
mat_fold_buf_b: DeviceBuffer<EF>,
/// Per-matrix (height, width) — updated each fold (height halves)
mat_fold_sizes: Vec<(usize, usize)>,
/// Per-matrix byte offset within the current buffer
mat_fold_offsets: Vec<usize>,
/// Per-trace count of matrices (to reconstruct nested structure)
mat_per_trace_counts: Vec<usize>,
/// Same for sels
sel_fold_buf_a: DeviceBuffer<EF>,
sel_fold_buf_b: DeviceBuffer<EF>,
sel_fold_sizes: Vec<(usize, usize)>,
sel_fold_offsets: Vec<usize>,
```

### 2. Initialize ping-pong buffers before MLE loop

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

After `fold_ple_evals` (line 617) and before the MLE loop (line 631), compute layout metadata and allocate buffer B only (buffer A is the original DeviceMatrix data, used in round 1):

```rust
// Compute per-matrix metadata for ping-pong layout.
struct MatSlice { offset: usize, height: usize, width: usize }

let mat_slices: Vec<MatSlice> = self.mat_evals_per_trace.iter()
    .flatten()
    .map(|m| MatSlice { offset: 0 /* filled after round 1 */, height: m.height(), width: m.width() })
    .collect();
let mat_per_trace_counts: Vec<usize> = self.mat_evals_per_trace.iter()
    .map(|v| v.len()).collect();
// Prefix sums for per-trace indexing into flat array
let mat_per_trace_prefix: Vec<usize> = mat_per_trace_counts.iter()
    .scan(0, |acc, &c| { let v = *acc; *acc += c; Some(v) }).collect();
let mat_total: usize = mat_slices.iter().map(|s| s.height * s.width).sum();

let sel_slices: Vec<MatSlice> = self.sels_per_trace.iter()
    .map(|m| MatSlice { offset: 0, height: m.height(), width: m.width() }).collect();
let sel_total: usize = sel_slices.iter().map(|s| s.height * s.width).sum();

// Allocate ONE ping-pong buffer (B). Round 1 reads from original DeviceMatrix data,
// writes into buffer B. Then drop originals and allocate buffer A for round 2 output.
let mut mat_buf_b = DeviceBuffer::<EF>::with_capacity(mat_total);
let mut sel_buf_b = DeviceBuffer::<EF>::with_capacity(sel_total);
```

**D2D copy primitive**: For all D2D copies throughout this plan, use the existing `cuda_memcpy::<true, true>(dst as *mut c_void, src as *const c_void, count * size_of::<EF>())` from `crates/cuda-common/src/copy.rs:42`. No new helper function needed.

Pack mat_evals and sels into their respective buffer A via D2D copies:

```rust
let mut mat_offsets = Vec::with_capacity(mat_slices.len());
let mut off = 0usize;
for m in self.mat_evals_per_trace.iter().flatten() {
    let bytes = m.buffer().len() * std::mem::size_of::<EF>();
    unsafe { cuda_memcpy::<true, true>(
        mat_buf_a.as_mut_ptr().add(off) as *mut c_void,
        m.buffer().as_ptr() as *const c_void, bytes); }
    mat_offsets.push(off);
    off += m.buffer().len();
}
// Update mat_slices with computed offsets
for (s, &o) in mat_slices.iter_mut().zip(&mat_offsets) { s.offset = o; }

// Same for sels into sel_buf_a
let mut sel_offsets = Vec::with_capacity(sel_slices.len());
off = 0;
for m in &self.sels_per_trace {
    let bytes = m.buffer().len() * std::mem::size_of::<EF>();
    unsafe { cuda_memcpy::<true, true>(
        sel_buf_a.as_mut_ptr().add(off) as *mut c_void,
        m.buffer().as_ptr() as *const c_void, bytes); }
    sel_offsets.push(off);
    off += m.buffer().len();
}
for (s, &o) in sel_slices.iter_mut().zip(&sel_offsets) { s.offset = o; }

// Free original DeviceMatrix vectors (reclaim GPU memory)
drop(std::mem::take(&mut self.mat_evals_per_trace));
drop(std::mem::take(&mut self.sels_per_trace));
```

**Why**: The initial D2D pack costs ~2-3ms but enables uniform code for all rounds (no round-1 special case). The ~2-3ms is amortized across 24 rounds of zero-alloc folding, netting 10-17ms improvement.

### 3. Replace fold_mle_evals with ping-pong implementation

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

New method `fold_mle_evals_pingpong` replaces the old `fold_mle_evals`:

```rust
fn fold_mle_evals_pingpong(
    &mut self, round: usize, r_round: EF,
    mat_sizes: &mut [(usize, usize)], mat_offsets: &mut [usize],
    buf_a: &mut DeviceBuffer<EF>, buf_b: &mut DeviceBuffer<EF>,
    sel_sizes: &mut [(usize, usize)], sel_offsets: &mut [usize],
    sel_a: &mut DeviceBuffer<EF>, sel_b: &mut DeviceBuffer<EF>,
) -> Result<(), LogupZerocheckError> {
    // Helper: fold one set of matrices (mat_evals or sels)
    let fold_set = |sizes: &mut [(usize, usize)], offsets: &mut [usize],
                     src: &DeviceBuffer<EF>, dst: &mut DeviceBuffer<EF>| -> Result<(), _> {
        let num_foldable = sizes.partition_point(|(h, _)| *h > 1);
        if num_foldable == 0 { return Ok(()); }

        let input_ptrs: Vec<_> = (0..num_foldable)
            .map(|i| unsafe { src.as_ptr().add(offsets[i]) })
            .collect();
        let mut out_off = 0usize;
        let output_ptrs: Vec<_> = (0..num_foldable)
            .map(|i| {
                let ptr = unsafe { dst.as_mut_ptr().add(out_off) };
                out_off += (sizes[i].0 / 2) * sizes[i].1;
                ptr
            })
            .collect();
        let log_heights: Vec<u8> = (0..num_foldable)
            .map(|i| (sizes[i].0 / 2).ilog2() as u8)
            .collect();
        let widths: Vec<u32> = (0..num_foldable)
            .map(|i| sizes[i].1 as u32)
            .collect();
        let max_cells = (0..num_foldable)
            .map(|i| (sizes[i].0 / 2) * sizes[i].1)
            .max()
            .unwrap_or(0);

        let d_in = input_ptrs.to_device()?;
        let d_out = output_ptrs.to_device()?;
        let d_h = log_heights.to_device()?;
        let d_w = widths.to_device()?;

        unsafe {
            batch_fold_mle(&d_in, &d_out, &d_w, num_foldable as u32, &d_h,
                           max_cells as u32, r_round)?;
        }

        // Update sizes and offsets for next round
        out_off = 0;
        for i in 0..num_foldable {
            sizes[i].0 /= 2;
            offsets[i] = out_off;
            out_off += sizes[i].0 * sizes[i].1;
        }
        // Copy non-foldable (height=1) matrices to maintain contiguous layout
        for i in num_foldable..sizes.len() {
            let cells = sizes[i].0 * sizes[i].1;
            unsafe {
                cuda_memcpy_d2d(dst.as_mut_ptr().add(out_off),
                               src.as_ptr().add(offsets[i]), cells);
            }
            offsets[i] = out_off;
            out_off += cells;
        }
        Ok(())
    };

    fold_set(mat_sizes, mat_offsets, buf_a, buf_b)?;
    std::mem::swap(buf_a, buf_b);

    fold_set(sel_sizes, sel_offsets, sel_a, sel_b)?;
    std::mem::swap(sel_a, sel_b);

    // eq_xis, eq_ns, eq_sharp_ns updates — unchanged from current code (lines 2080-2089)
    for tree in self.eq_xis.values_mut() {
        if tree.layers.len() > 1 { tree.layers.pop(); }
    }
    let xi = self.xi[self.l_skip + round - 1];
    let eq_r = eval_eq_mle(&[xi], &[r_round]);
    self.eq_ns.push(self.eq_ns[round - 1] * eq_r);
    self.eq_sharp_ns.push(self.eq_sharp_ns[round - 1] * eq_r);
    Ok(())
}
```

### 4. Update sumcheck_polys_batch_eval to use buffer views

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Use `MatSlice` and prefix sums to index into the ping-pong buffer. For each `trace_idx`, the flat range of matrices is `mat_per_trace_prefix[trace_idx]..mat_per_trace_prefix[trace_idx] + mat_per_trace_counts[trace_idx]`.

All call sites that access `self.mat_evals_per_trace[trace_idx][mat_idx]` or `self.sels_per_trace[trace_idx]` need updating:

**Case A.1 (late eval, lines 1601-1638)**: Accesses `mats[0]` for preprocessed and `mats[first_main_idx..]` for mains:
```rust
let flat_base = mat_per_trace_prefix[trace_idx];
// Preprocessed (mat index 0 if has_preprocessed)
let prep_ptr = if has_preprocessed {
    let s = &mat_slices[flat_base];
    MainMatrixPtrs { data: unsafe { buf.as_ptr().add(s.offset) }, air_width: air_width_for_mat(need_rot, s.width) }
} else { ... };
// Mains (indices first_main_idx..)
let main_ptrs: Vec<_> = (flat_base + first_main_idx..flat_base + mat_per_trace_counts[trace_idx])
    .map(|i| {
        let s = &mat_slices[i];
        MainMatrixPtrs { data: unsafe { buf.as_ptr().add(s.offset) }, air_width: air_width_for_mat(need_rot, s.width) }
    }).collect();
```

**Case B (early eval, lines 1649-1685)**: Column pointer collection iterates columns:
```rust
// For sels:
let sel = &sel_slices[trace_idx];
let sel_ptr = unsafe { sel_buf.as_ptr().add(sel.offset) };
for col in 0..sel.width {
    all_columns.push(unsafe { sel_ptr.add(col * sel.height) });  // column-major stride
}
// For each mat in trace:
for i in flat_base..flat_base + mat_per_trace_counts[trace_idx] {
    let s = &mat_slices[i];
    let mat_ptr = unsafe { buf.as_ptr().add(s.offset) };
    for col in 0..s.width {
        all_columns.push(unsafe { mat_ptr.add(col * s.height) });
    }
}
```

**Key invariant**: Within each matrix's region in the ping-pong buffer, the column-major layout is preserved exactly (columns spaced by `height` elements) because the D2D/fold copies raw bytes in the same layout.

**Phase 3 width accesses (lines 1749-1767)**: Replace `mats[0].width()` with `mat_slices[flat_base].width` and iteration over `mats[first_main_idx..]` with flat-index iteration using `mat_slices[flat_base + first_main_idx..].iter().map(|s| s.width)`.

### 5. Update into_column_openings

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

The current `into_column_openings` (line 2098) does more than simple D2H: it splits doubled-width matrices for `need_rot` AIRs, reorders common_main to front, and interleaves plain/rotated columns. This logic MUST be preserved.

**Approach**: Before calling the existing `into_column_openings`, reconstruct `self.mat_evals_per_trace` and `self.sels_per_trace` from the ping-pong buffer as host-side data, then let the existing logic operate unchanged.

```rust
// Bulk D2H the entire ping-pong buffer
let host_mat_buf = {
    let total_cells = mat_slices.iter().map(|s| s.height * s.width).sum::<usize>();
    let view = std::mem::ManuallyDrop::new(
        unsafe { DeviceBuffer::<EF>::from_raw_parts(mat_buf_a.as_ptr() as *mut EF, total_cells) }
    );
    view.to_host().map_err(LogupZerocheckError::MemCopy)?
};

// Reconstruct per-trace DeviceMatrix from host data (for the existing into_column_openings logic).
// The existing code calls m.to_host() anyway, so we can short-circuit by providing host-side
// ColMajorMatrix views directly. Modify into_column_openings to accept either:
//   (a) Pre-fetched host data slices, or
//   (b) Reconstruct DeviceMatrix with ManuallyDrop from the ping-pong buffer.
//
// Simplest: reconstruct self.mat_evals_per_trace as ManuallyDrop<DeviceMatrix> views.
// Since DeviceMatrix wraps Arc<DeviceBuffer>, and we need non-owning views,
// the cleanest path is to do a per-matrix D2H at this point using ManuallyDrop<DeviceBuffer>:

let mut flat_idx = 0;
self.mat_evals_per_trace = mat_per_trace_counts.iter().map(|&count| {
    (0..count).map(|_| {
        let s = &mat_slices[flat_idx];
        flat_idx += 1;
        // Slice host buffer into a DeviceMatrix-compatible form.
        // Use ManuallyDrop to create a non-owning DeviceBuffer view for D2H.
        let ptr = unsafe { mat_buf_a.as_ptr().add(s.offset) } as *mut EF;
        let view = std::mem::ManuallyDrop::new(
            unsafe { DeviceBuffer::<EF>::from_raw_parts(ptr, s.height * s.width) }
        );
        // Construct DeviceMatrix from the non-owning buffer. Since into_column_openings
        // immediately calls to_host(), the DeviceMatrix is short-lived.
        DeviceMatrix::<EF>::from_non_owning(view, s.height, s.width)
    }).collect()
}).collect();
// Same for sels_per_trace...
```

**Preferred approach (per final review)**: Since `DeviceMatrix` wraps `Arc<DeviceBuffer>` which prevents non-owning construction, the cleanest path is:
1. Do a single bulk D2H of the entire ping-pong buffer to host memory
2. Slice the host data per-matrix using `mat_slices` offsets/sizes
3. Extract the split/reorder/need_rot logic from the existing `into_column_openings` into a helper that operates on `(host_data: &[EF], height, width, need_rot)` tuples
4. Call the helper per-trace to produce the final column openings

The existing `into_column_openings` logic is ~50 lines of straightforward host-side work (matrix transpose, width splitting, column interleaving). Extracting it avoids the `Arc<DeviceBuffer>` ownership issue entirely while preserving correctness.

### 6. Memory accounting for save_memory path

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Lines 2067-2075 adjust `memory_limit_bytes` based on mat_evals buffer sizes. Replace with:
```rust
if self.save_memory {
    let current_cells: usize = mat_sizes.iter().map(|(h,w)| h * w).sum();
    self.memory_limit_bytes = self.gkr_mem_contribution
        .saturating_sub(current_cells * size_of::<EF>());
}
```

## Invariants

1. **Fold kernel correctness**: batch_fold_mle receives identical (input_ptr, output_ptr, height, width) data. Only the source of pointers changes.

2. **Buffer capacity**: Buffer A starts at total_cells. After round 1, data is in B at total_cells/2. After swap, A holds total_cells/2 data. B (now old A, full capacity) is always large enough for the output.

3. **Non-foldable matrices**: D2D-copied from src to dst each round to maintain contiguous layout. Cost: sum of (width) elements for all height-1 matrices. Even with 1400 matrices of average width 50, total copy is ~70K elements × 16 bytes = 560KB — well within a single async D2D latency (~0.1ms).

4. **Pointer array uploads**: Still 4 H2D uploads per fold call (same as current code). These are small Vecs (~800 elements × 8 bytes = 6.4KB).

5. **No APC 0 regression**: APC 0 has fewer matrices (~99 traces × 1-3 each ≈ 150 matrices). Ping-pong setup D2D is proportionally smaller. Allocation savings are proportionally smaller but still positive.

6. **into_column_openings correctness**: The existing split/reorder/need_rot logic is preserved by reconstructing `mat_evals_per_trace` and `sels_per_trace` from the ping-pong buffer before calling the existing `into_column_openings` method. `ManuallyDrop<DeviceBuffer>` prevents double-free on sub-buffer pointers. If `DeviceMatrix` can't be constructed non-owningly, the fallback refactors `into_column_openings` to accept `(ptr, height, width)` tuples.

## Measurement Plan

1. Build and run benchmark for APC {0, 300}
2. Compare MLE Rounds time at APC 300: 173ms → expect ~155-160ms
3. Compare STARK excl trace: 1302ms → expect ~1282-1287ms
4. Verify APC 0 no regression
5. Verify correctness (proof verification succeeds)

## Rollback Criteria

- MLE Rounds improvement at APC 300 <10ms
- STARK excl trace improvement at APC 300 <10ms
- APC 0 regression >10ms
- Correctness failure
