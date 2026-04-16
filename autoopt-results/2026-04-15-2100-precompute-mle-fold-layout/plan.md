# Plan: Pre-compute MLE Fold Layout

## Goal

Reduce per-round overhead in MLE fold by: (1) pre-computing fold descriptor arrays for all rounds and bulk-uploading once before the MLE loop, and (2) fusing the two per-round `fold_pingpong` calls (mat_evals + selectors) into a single kernel launch. This replaces 112 per-round `to_device()` calls and 28 kernel launches per segment with 8 bulk uploads and 14 launches.

**Savings estimate**: 5-15ms at APC 300 based on measured per-call overhead of ~0.3-0.5μs (from `bulk-alloc-mle-fold-buffers` report) and ~0.3ms inter-kernel gap per launch. The fold descriptor overhead is a small fraction of MLE Rounds (166ms), which is dominated by kernel execution time (~113ms). This optimization targets the remaining ~53ms overhead budget.

**Why this is worth doing despite modest savings**: The MLE Rounds anti-scaling (0.71x ratio, 166ms at APC 300 vs 118ms at APC 0) is the single largest scaling bottleneck. Even a 10ms improvement compounds: it brings the APC 0/APC 300 ratio from 1.61x to 1.62x. More importantly, eliminating per-round fold overhead creates a cleaner baseline for future MLE round optimizations (which have been hard to isolate due to noise from this overhead).

## Current Code Path

### Pingpong buffer allocation (mod.rs:622-647)
After Round 0 completes, 4 `DeviceBuffer<EF>` are allocated:
- `fold_mat_buf_a`, `fold_mat_buf_b` — pingpong pair for evaluation matrices
- `fold_sel_buf_a`, `fold_sel_buf_b` — pingpong pair for selectors

### MLE rounds loop (mod.rs:660-677)
Each round calls `fold_mle_evals(round, r_round)`.

### fold_mle_evals (mod.rs:2051-2204)
Calls `fold_pingpong` twice:
1. Lines 2147-2168: Fold `mat_evals_per_trace` (flattened `Vec<Vec<DeviceMatrix<EF>>>`)
2. Lines 2179-2191: Fold `sels_per_trace` (`Vec<DeviceMatrix<EF>>`)

Between these, lines 2169-2177 update `memory_limit_bytes` when `save_memory` is true.

### fold_pingpong (mod.rs:2058-2145)
Each invocation:
1. **CPU collection** (lines 2069-2097): Iterates foldable matrices to build 4 arrays: `input_ptrs`, `output_ptrs`, `log_heights`, `widths`.
2. **H2D upload** (lines 2099-2102): 4× `to_device()` — allocates DeviceBuffer (cudaMallocAsync) + copies.
3. **Kernel launch** (lines 2104-2114): `batch_fold_mle(...)`.
4. **Buffer swap** (line 2118): `std::mem::swap(buf_a, buf_b)`.
5. **View reconstruction** (lines 2122-2142): Non-owning `DeviceMatrix` views into swapped buf_a.
6. **Implicit cleanup**: 4 DeviceBuffers dropped (cudaFreeAsync).

### Per-segment costs at APC 300
- 14 rounds × 2 fold calls × (4 to_device + 1 kernel + 4 drops) = **112 to_device + 28 kernels + 112 drops = 252 CUDA API calls per segment**

### CUDA kernel (sumcheck.cu:192-207, 304-318)
`batch_fold_mle_kernel` uses `blockIdx.y` for matrix index. Reads per-matrix pointers and dimensions from device arrays. Launcher uses `fold_mle_launch_params` for 2D grid sizing.

### Pingpong buffer invariant
After `fold_pingpong`, buf_a always contains the latest output. Physically:
- Round R output goes to phys_B (the pre-swap buf_b), then swap makes buf_a = phys_B.
- So input for round R+1 is always in the physical buffer that round R wrote to.
- Parity: round R writes to phys_{B if R odd, A if R even}.

## Changes

### Change 1: Pre-compute fold descriptor arrays (Rust)

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

After the pingpong buffer allocation (line 647), add a pre-computation step:

```rust
// Pre-compute fold descriptors for all rounds.
// For each round R in 1..=n_max and for each fold group (mat, sel):
//   - Which matrices are foldable (height >> (R-1) > 1)
//   - Input pointers (round 1: original matrix ptrs; round R>=2: offset in pingpong buf)
//   - Output offsets in the alternate pingpong buffer
//   - log_heights and widths

struct PrecomputedFold {
    /// Flat buffer of input pointers for all rounds, indexed by round_offsets.
    all_input_ptrs: Vec<*const EF>,      // round 1: absolute ptrs; R>=2: cast from buf base + offset
    all_output_ptrs: Vec<*mut EF>,
    all_log_heights: Vec<u8>,
    all_widths: Vec<u32>,
    /// Per-round: (start_index, num_foldable, max_output_cells) for mat and sel groups.
    round_meta: Vec<FoldRoundMeta>,
    /// Physical base pointers captured at build time.
    mat_phys_a: *mut EF,
    mat_phys_b: *mut EF,
    sel_phys_a: *mut EF,
    sel_phys_b: *mut EF,
}

struct FoldRoundMeta {
    mat_start: usize,
    mat_count: u16,
    mat_max_cells: u32,
    sel_start: usize,
    sel_count: u16,
    sel_max_cells: u32,
}
```

The builder iterates over rounds 1..=n_max. For each round:
- Determines which matrices are foldable based on initial heights
- For round 1: input_ptrs = absolute device pointers from current DeviceMatrix objects
- For round R ≥ 2: input_ptrs = phys_buf_base + cumulative offset matching previous round's output layout
- Output_ptrs = phys_buf_base + cumulative offset for this round's foldable set
- The output offset for trace i at round R = `sum_{j<i, foldable at R} (width_j × (initial_height_j >> R))`

**Critical invariant**: The output offsets at round R must match the input offsets at round R+1 exactly. Both use the same iteration order (sorted by height descending, matching the existing code's `partition_point` ordering).

After building the CPU arrays, upload them in 4 bulk to_device() calls (one each for input_ptrs, output_ptrs, log_heights, widths). These are indexed by round offset during the MLE loop.

### Change 2: Fuse mat+sel fold into single kernel launch

**File:** `crates/cuda-backend/cuda/src/sumcheck.cu`

No new kernel needed. Instead, concatenate the mat and sel descriptor entries into the same arrays:
- For round R: entries [0..mat_count) are mat_evals, entries [mat_count..mat_count+sel_count) are selectors
- The existing `batch_fold_mle_kernel` processes all entries uniformly (it doesn't care about the semantic group)
- Each entry has the correct input/output pointers pointing to the appropriate physical buffer

The key insight: the existing kernel already processes matrices by index (`blockIdx.y`). By concatenating mat and sel entries, one kernel launch handles both groups.

### Change 3: Modify fold_mle_evals to use pre-computed arrays

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Replace the two `fold_pingpong` calls with:

```rust
fn fold_mle_evals_precomputed(&mut self, round: usize, r_round: EF) -> Result<(), ...> {
    let plan = self.fold_plan.as_ref().unwrap();
    let meta = &plan.round_meta[round - 1];
    let total = meta.mat_count as usize + meta.sel_count as usize;
    if total == 0 { return Ok(()); }

    let combined_start = meta.mat_start; // mat and sel are concatenated
    let max_cells = max(meta.mat_max_cells, meta.sel_max_cells);

    // Single kernel launch for both mat + sel
    unsafe {
        batch_fold_mle(
            &plan.d_all_input_ptrs.slice(combined_start, total),
            &plan.d_all_output_ptrs.slice(combined_start, total),
            &plan.d_all_widths.slice(combined_start, total),
            total as u16,
            &plan.d_all_log_heights.slice(combined_start, total),
            max_cells,
            r_round,
        )?;
    }

    // Swap both buffer pairs (matches the fold_pingpong behavior)
    if meta.mat_count > 0 {
        if let (Some(a), Some(b)) = (&mut self.fold_mat_buf_a, &mut self.fold_mat_buf_b) {
            std::mem::swap(a, b);
        }
    }
    if meta.sel_count > 0 {
        if let (Some(a), Some(b)) = (&mut self.fold_sel_buf_a, &mut self.fold_sel_buf_b) {
            std::mem::swap(a, b);
        }
    }

    // Reconstruct views using pre-computed offsets (same logic as fold_pingpong lines 2122-2142)
    // ... rebuild mat_evals_per_trace and sels_per_trace views from swapped buf_a
    // using the pre-computed output offsets for this round

    // save_memory update (same as lines 2169-2177)
    if self.save_memory { ... }

    // eq_xis trim + eq_ns update (same as lines 2193-2203)
    ...
}
```

**Note on DeviceBuffer::slice**: The pre-uploaded arrays are indexed by offset. A `slice(start, len)` view into the DeviceBuffer provides the correct sub-range without new allocation. If `DeviceBuffer` doesn't support slicing, pass base pointer + offset as raw pointers.

### Change 4: Fallback path

Keep the existing `fold_pingpong` code for when `fold_plan.is_none()` (n_max = 0 or no foldable matrices).

## Invariants

1. **Fold arithmetic**: `output[j] = input[j] + r × (input[j + half_h × w] - input[j])` preserved exactly — same kernel, same inputs, same outputs.

2. **Buffer pointer identity**: The pre-computed plan captures physical buffer pointers BEFORE any swaps. The `std::mem::swap` on the Rust side swaps `DeviceBuffer` wrappers but physical GPU addresses don't change. The plan's pointers always refer to physical addresses.

3. **Offset thread-through**: Output offsets at round R define the data layout. Input offsets at round R+1 must match exactly. Both are computed from the same sorted iteration order (height descending) with the same foldability criteria.

4. **Non-foldable view preservation**: Matrices that have reached height=1 are not in any round's descriptor entries. Their `DeviceMatrix` views remain unchanged across rounds, pointing to wherever they last landed (matching the existing `extend_from_slice(&input_mats[num_foldable..])` behavior).

5. **save_memory ordering**: The `memory_limit_bytes` update (lines 2169-2177) must see the NEW mat_evals views before computing sizes. In the fused approach, both mat and sel fold happen in one kernel, so the view reconstruction for mat_evals MUST precede the save_memory update. This matches the current sequencing (mat fold → save_memory → sel fold) because the save_memory update only reads mat_evals, not sel.

6. **Proof correctness**: Only the fold parameter `r_round` changes per round. All pre-computed data depends on static trace metadata (initial heights, widths, pointer addresses), not on transcript challenges.

## Measurement Plan

1. Run `run_pairing.sh` benchmarks for APC {0, 100, 300}
2. Compare with spec.py:
   - **Primary metric**: MLE Rounds at APC 300 (current: 166ms, expect: 155-161ms)
   - **Secondary metric**: STARK excl trace at APC 300 (current: 1119ms)
   - **Regression check**: STARK excl trace at APC 0 (current: 1800ms, must not increase by >20ms)
3. Verify: all configs prove + verify with `--recursion`
4. Profile with nsight: confirm reduced kernel launch count in fold phase (14 per segment vs 28)

## Rollback Criteria

- MLE Rounds improvement at APC 300 < 3ms (below measurement noise floor, indicating zero real impact)
- STARK excl trace at APC 0 regresses by > 20ms
- Any proof verification failure
- Any CUDA error
