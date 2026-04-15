# Plan: Batch MLE Per-Trace H2D Copies

## Goal

Replace ~5000 per-trace `main_ptrs.to_device()` H2D copies per segment in MLE `sumcheck_polys_batch_eval` with ~24 bulk per-round uploads. This reduces CUDA API overhead from ~25ms per segment to ~1ms per segment, targeting a 40-50ms improvement on MLE Rounds at APC 300 (currently 162ms).

## Current Code Path

The bottleneck is in `LogupZerocheckGpu::sumcheck_polys_batch_eval()` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:1575-1961`. This function is called 24 times per segment (once per MLE round) at APC 300.

### Call chain
```
MLE Rounds loop (mod.rs:659)
  └─ sumcheck_polys_batch_eval(round, r_prev) (mod.rs:1575)
       ├─ Phase 1: Classify all 623 traces (mod.rs:1590-1725)
       │    ├─ Case A.1 (round == n_lift+1): build Vec<MainMatrixPtrs>, to_device() per trace (mod.rs:1652-1659)
       │    ├─ Case A.2 (round > n_lift+1): scale tilde evals only (mod.rs:1678-1686)
       │    └─ Case B (round <= n_lift): collect column pointers (mod.rs:1688-1724)
       ├─ Phase 2: Batched interpolation kernel (mod.rs:1727-1765) — already 1 H2D per round
       ├─ Phase 3: Build TraceCtx for Case B traces (mod.rs:1767-1828)
       │    └─ Per trace: build Vec<MainMatrixPtrs>, to_device() (mod.rs:1799-1811) ← BOTTLENECK
       └─ Phase 4: Batch evaluations (mod.rs:1830-1958)
```

### Why it's slow

- Phase 3 calls `main_ptrs.to_device()` once per Case B trace per round.
- At round 1, all 623 traces are Case B → 623 H2D copies.
- At round 12, ~300 traces are still Case B → 300 H2D copies.
- Across 24 rounds for segment 0, nsight measures **5168 H2D copies** totaling 40MB.
- Each call invokes `cudaMemcpyAsync` (confirmed median ~1.8μs per call from nsight `cuda_api_sum`) plus host-side Vec allocation.
- Phase 1 Case A.1 adds another ~623 H2D copies across all rounds (one per trace at its transition round).
- Total per segment: ~5800 H2D API calls contributing ~25ms of CPU-side overhead.

### Data structures

`TraceCtx` (batch_mle.rs:89-105) holds:
```rust
pub main_ptrs_dev: DeviceBuffer<MainMatrixPtrs<EF>>,
```
Each trace's `main_ptrs_dev` is an independently-allocated device buffer containing 1-5 `MainMatrixPtrs` structs (16 bytes each: pointer + width). Consumers access it via:
- `t.main_ptrs_dev.as_ptr()` — used in batch builder context structs (batch_mle_monomial.rs:184,407,672; batch_mle.rs:188,335)
- `&t.main_ptrs_dev` — passed by reference to `evaluate_mle_constraints_gpu` and `evaluate_mle_interactions_gpu` (mle_round.rs:28,82), which only call `.as_ptr()` and `.len()` on it.

## Changes

### Change 1: Batch Phase 3 (Case B) main_ptrs uploads

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Currently (lines 1767-1828):
```rust
for meta in &case_b_traces {
    // ... compute pointers ...
    let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..].iter().map(|m| { ... }).collect();
    let main_ptrs_dev = main_ptrs.to_device()?;
    early_eval.push(TraceCtx { ..., main_ptrs_dev, ... });
}
```

Change to:
```rust
// Step 1: Collect ALL MainMatrixPtrs for all Case B traces into one Vec, tracking per-trace offsets
let mut all_main_ptrs: Vec<MainMatrixPtrs<EF>> = Vec::new();
let mut main_ptrs_offsets: Vec<(usize, usize)> = Vec::new(); // (offset, count) per case_b trace

for meta in &case_b_traces {
    let mats = &self.mat_evals_per_trace[meta.trace_idx];
    let first_main_idx = usize::from(meta.has_preprocessed);
    let offset = all_main_ptrs.len();
    all_main_ptrs.extend(mats[first_main_idx..].iter().map(|m| {
        let interpolated_height = sp_deg * meta.num_y;
        MainMatrixPtrs {
            data: base_ptr.wrapping_add(widths_so_far * interpolated_height),
            air_width: air_width_for_mat(meta.need_rot, m.width()),
        }
        // NOTE: widths_so_far tracking must be done in the loop body, same as current code
    }));
    let count = all_main_ptrs.len() - offset;
    main_ptrs_offsets.push((offset, count));
}

// Step 2: Single bulk H2D upload
let d_all_main_ptrs = if !all_main_ptrs.is_empty() {
    Some(all_main_ptrs.to_device()?)
} else {
    None
};

// Step 3: Create TraceCtx with non-owning views into the bulk buffer
for (meta, &(offset, count)) in case_b_traces.iter().zip(main_ptrs_offsets.iter()) {
    let main_ptrs_dev = unsafe {
        DeviceBuffer::non_owning(
            d_all_main_ptrs.as_ref().unwrap().as_mut_ptr().add(offset),
            count,
        )
    };
    // ... compute prep_ptr, sels_ptr, etc. (unchanged) ...
    early_eval.push(TraceCtx { ..., main_ptrs_dev, ... });
}
```

The `d_all_main_ptrs` buffer is kept alive alongside `_keepalive_interpolated` until the end of the function. All non-owning views in `early_eval` TraceCtx remain valid.

### Change 2: Batch Phase 1 Case A.1 main_ptrs uploads

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Currently (lines 1639-1675): Each Case A.1 trace does `main_ptrs.to_device()` individually.

Apply the same pattern:
```rust
// After the Phase 1 loop: collect all Case A.1 main_ptrs, bulk upload, create non-owning views
let mut all_late_main_ptrs: Vec<MainMatrixPtrs<EF>> = Vec::new();
let mut late_offsets: Vec<(usize, usize)> = Vec::new();

// During Phase 1 Case A.1, instead of to_device(), append to all_late_main_ptrs and record offset
```

Restructure the Phase 1 loop to collect Case A.1 main_ptrs into a Vec (alongside their other TraceCtx fields stored in a temporary struct), then do a single bulk upload, then create TraceCtx objects with non-owning views.

This requires splitting the Phase 1 Case A.1 branch into two passes:
1. First pass: collect main_ptrs data + temporary metadata into `Vec<CaseA1Meta>` (no H2D)
2. After loop: single bulk `all_late_main_ptrs.to_device()` 
3. Second pass: create `TraceCtx` objects with non-owning views

### Change 3: Keep bulk buffers alive

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add `_keepalive_early_main_ptrs` and `_keepalive_late_main_ptrs` variables (similar pattern to existing `_keepalive_interpolated`) to ensure the bulk DeviceBuffers outlive all non-owning views.

Both variables must be declared before the TraceCtx vectors and dropped after all batch evaluation functions complete (end of `sumcheck_polys_batch_eval`).

### Change 4: DeviceBuffer::as_mut_ptr() for offset arithmetic

**File**: `crates/cuda-common/src/d_buffer.rs`

Check if `as_mut_ptr()` exists on DeviceBuffer. If not, add it (trivially: return `self.ptr`). Needed for computing non-owning view pointers with `.add(offset)`.

Alternatively, use `as_ptr() as *mut T` since the non_owning constructor takes `*mut T`.

## Invariants

1. **Lifetime safety**: All non-owning DeviceBuffer views in TraceCtx must be consumed (batch evaluations complete, including D2H copies) before the backing bulk buffer is dropped. The existing function structure guarantees this: bulk buffers are declared at function scope, TraceCtx vectors are consumed by Phase 4 batch evaluations, and the function returns after Phase 4.

2. **Pointer correctness**: Non-owning view pointers must point to the correct sub-region of the bulk buffer. The offset arithmetic (`base_ptr.add(offset)`) must match the order in which MainMatrixPtrs were appended to the bulk Vec.

3. **No APC 0 regression**: At APC 0 (99 AIRs, 5 segments, ~20 AIRs per segment), the overhead reduction is proportionally smaller but the bulk allocation overhead is also smaller. The optimization should be neutral or slightly positive at APC 0.

4. **Correctness of evaluation results**: The batch evaluation functions (LogupMonomialBatch, ZerocheckMonomialBatch, etc.) access `t.main_ptrs_dev.as_ptr()` to read MainMatrixPtrs from device memory. The non-owning view provides the same pointer, so GPU kernels see identical data. The `.len()` return value must also match (the non-owning constructor sets len correctly).

5. **No change to CUDA kernels**: This optimization is pure Rust-side; no CUDA kernel code changes.

## Measurement Plan

Run the standard benchmark:
```bash
cd /home/georg/powdr/results/pairing
/home/georg/powdr/target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics <output_path> --recursion
```

For APC 0:
```bash
/home/georg/powdr/target/release/powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics <output_path> --recursion
```

Analyze with `python3 /home/georg/spec.py <metrics_path> <name>`.

Run nsight profiling for APC 300 to verify H2D count reduction:
```bash
nsys profile --output <output> --force-overwrite true --trace cuda,nvtx,osrt --sample none --stats true -- <binary> prove --artifact apc300.cbor --input 0 --recursion
```

### Expected results
- MLE Rounds at APC 300: 162ms → 112-125ms (25-40% reduction)
- STARK excl trace at APC 300: 1296ms → 1246-1260ms
- H2D copy count in MLE window: ~5168 → ~200-400 per segment
- No regression at APC 0 (MLE Rounds stays ~113ms)

## Rollback Criteria

Revert if:
- MLE Rounds improvement at APC 300 is less than 15ms (below measurement noise threshold)
- Any regression at APC 0 exceeds 10ms on STARK excl trace
- Proof verification fails at any APC configuration
