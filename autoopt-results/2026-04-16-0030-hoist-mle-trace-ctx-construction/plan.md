# Plan: Hoist MLE Trace Context Construction

## Goal

Reduce MLE Rounds overhead by pre-allocating per-trace device pointer buffers (`main_ptrs_dev`) ONCE before the MLE round loop and reusing them via `copy_to()` instead of `to_device()` (which allocates + copies) each round. At APC 300, MLE Rounds = 163ms with ~113ms kernel time and ~50ms overhead across 28 round iterations (~1.79ms/round). A previous optimization ("batch-mle-main-ptrs-upload", 2026-04-13-1430) measured the total per-trace upload overhead at 5ms by eliminating ~5,600 H2D calls. The measured per-call overhead was ~0.9μs. This plan targets the ALLOCATION subset of that overhead: each `to_device()` call does `cudaMallocAsync` + `cudaMemcpyAsync`, and the buffer is freed via `cudaFreeAsync` on drop. Pre-allocating eliminates the alloc+free pair, retaining only the copy.

**Expected savings**: 3-5ms at APC 300, based on measured per-cycle overhead. This is a conservative, data-backed estimate.

## Current Code Path

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, function `sumcheck_polys_batch_eval()` (called once per MLE round)

### Per-round trace context construction (lines ~1780-1841):

For each Case B ("early") trace (num_y > 1), the function:

1. Computes `sels_ptr`, `prep_ptr` from the interpolation buffer offset
2. Builds `main_ptrs: Vec<MainMatrixPtrs<EF>>` from per-matrix pointers into the interpolation buffer (lines 1812-1822)
3. Calls `main_ptrs.to_device()` (line 1824) → allocates DeviceBuffer + H2D copy
4. Pushes `TraceCtx` to `early_eval` vector (lines 1826-1840)

For each Case A ("late") trace (num_y = 1), similar construction with `main_ptrs.to_device()`.

### Allocation pattern at APC 300:

- ~312 traces per segment, ~14 rounds per segment
- Early rounds: most traces are Case B → ~300 `to_device()` calls per round
- Late rounds: most traces transition to Case A → ~300 `to_device()` calls still
- Total per segment: ~300 × 14 = ~4,200 `to_device()` calls (each = cudaMallocAsync + cudaMemcpyAsync)
- Total per proof: ~8,400 allocation+upload cycles
- Each cycle: ~1.5μs (cudaMallocAsync ~1μs + cudaMemcpyAsync ~0.3μs + cudaFreeAsync ~0.2μs on drop)
- Total overhead: ~8,400 × 1.5μs ≈ 12.6ms

### `to_device()` implementation (from `crates/cuda-common/src/copy.rs` line 89):
```rust
fn to_device(&self) -> Result<DeviceBuffer<T>, MemCopyError> {
    let mut dst = DeviceBuffer::with_capacity(self.len());  // cudaMallocAsync
    self.copy_to(&mut dst)?;                                  // cudaMemcpyAsync
    Ok(dst)
}
```

Each `to_device()` allocates a fresh DeviceBuffer. On drop (when TraceCtx is dropped at end of round), `cudaFreeAsync` is called. Pre-allocating the buffer and using `copy_to()` directly eliminates the alloc+free pair.

## Changes

### Change 1: Pre-allocate per-trace main_ptrs DeviceBuffers before the MLE loop

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**Where**: Inside `prove_zerocheck_and_logup_gpu`, near the existing pingpong buffer pre-allocation block (approximately lines 622-646). The pool should be allocated alongside `fold_mat_buf_a`/`fold_mat_buf_b` and stored as a local variable or prover struct field that persists across all MLE round iterations.

**What**: For each trace, allocate a `DeviceBuffer<MainMatrixPtrs<EF>>` at the maximum needed size (number of main matrices for that trace). Store in a `Vec<DeviceBuffer<MainMatrixPtrs<EF>>>` indexed by trace_idx. The maximum needed size per trace is known from `self.trace_metas` — it's the count of matrices in `mat_evals_per_trace[trace_idx]` minus the preprocessed matrix (if any).

```rust
// Pre-allocate device buffers for per-trace main_ptrs, indexed by trace_idx
// Use air_indices_per_trace and mat_evals_per_trace to determine sizes
let d_main_ptrs_pool: Vec<DeviceBuffer<MainMatrixPtrs<EF>>> = (0..num_airs_present)
    .map(|trace_idx| {
        let air_idx = self.air_indices_per_trace[trace_idx];
        let has_preprocessed = self.pk.per_air[air_idx].preprocessed_data.is_some();
        let first_main_idx = usize::from(has_preprocessed);
        let num_main_mats = self.mat_evals_per_trace[trace_idx].len() - first_main_idx;
        DeviceBuffer::with_capacity(num_main_mats.max(1))
    })
    .collect();
```

**Why**: Each DeviceBuffer is allocated once (total ~312 allocations for APC 300). Across 14 MLE rounds, this replaces ~4,200 allocation+free pairs per segment with ~312 one-time allocations + ~4,200 copy-only operations. Based on measured per-call overhead of ~0.9μs (from batch-mle-main-ptrs-upload), the alloc+free portion is ~0.4-0.6μs per cycle. Net savings: ~8,400 cycles × 0.5μs ≈ 3-5ms per proof.

### Change 2: Replace to_device() with copy_to() in TraceCtx construction

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines ~1812-1824

**What**: Instead of `let main_ptrs_dev = main_ptrs.to_device()?;`, use `main_ptrs.copy_to(&mut d_main_ptrs_pool[meta.trace_idx])?;` and then reference `&d_main_ptrs_pool[meta.trace_idx]` in the TraceCtx.

**Before**:
```rust
let main_ptrs_dev = main_ptrs.to_device()?;  // allocate + copy
early_eval.push(TraceCtx {
    // ...
    main_ptrs_dev,
    // ...
});
```

**After**:
```rust
main_ptrs.copy_to(&mut d_main_ptrs_pool[meta.trace_idx])?;  // copy only
early_eval.push(TraceCtx {
    // ...
    main_ptrs_dev: unsafe {
        DeviceBuffer::non_owning(
            d_main_ptrs_pool[meta.trace_idx].as_mut_ptr(),
            main_ptrs.len(),
        )
    },
    // ...
});
```

The non-owning view ensures TraceCtx doesn't free the pre-allocated buffer when dropped.

**Why**: The copy_to() operation is just `cudaMemcpyAsync` (~0.3μs for 40 bytes), eliminating the `cudaMallocAsync` + `cudaFreeAsync` pair.

### Change 3: Apply the same pattern to Case A (late) trace context construction

**File**: Same file, similar code block for late_eval traces

**What**: Same transformation — use pre-allocated d_main_ptrs_pool instead of fresh to_device() calls.

### Change 4: Ensure d_main_ptrs_pool lifetime spans the entire MLE loop

**File**: Same file

**What**: The `d_main_ptrs_pool` Vec must be declared BEFORE the MLE round loop and dropped AFTER it. The non-owning DeviceBuffers in TraceCtx must not outlive the pool. Since early_eval/late_eval vectors are rebuilt (and their old entries dropped) each round, this is satisfied as long as the pool outlives the loop.

## Invariants

1. **Pointer correctness**: The `main_ptrs` host vectors are rebuilt each round with correct pointers (into the interpolation buffer for Case B traces, into fold buffers for Case A traces). The `copy_to()` uploads the fresh pointer values to the pre-allocated device buffer. The CUDA kernel reads the updated pointers.

2. **Buffer lifetime**: `d_main_ptrs_pool` must outlive all non-owning TraceCtx views. The pool is declared before the loop and dropped after, so this is guaranteed.

3. **No double-free**: TraceCtx contains non-owning DeviceBuffers (`owns_memory: false`). When TraceCtx is dropped at end of round, the device memory is NOT freed. The pool frees it after the loop.

4. **Size consistency**: Each pre-allocated buffer has capacity = num_main_matrices for that trace. This doesn't change between rounds (the number of matrices per trace is fixed). The `copy_to()` copies exactly `main_ptrs.len()` elements, which equals num_main_matrices.

5. **No aliasing**: Each trace has its own entry in d_main_ptrs_pool. Different traces are never assigned the same buffer.

## Measurement Plan

Run `openvm-riscv/scripts/run_pairing.sh` from the powdr repo for APC {0, 100, 300}.

Key metrics:
- MLE Rounds at APC 300: currently 163ms, target ≤ 158ms (5ms improvement)
- STARK excl trace at APC 300: currently 1106ms, target ≤ 1101ms
- No regression at APC 0 (MLE: 118ms, STARK: 1797ms)

Due to the 3ms threshold being near the noise floor, run each APC configuration 2-3 times and average before comparing. The improvement should be consistently above 0 across runs.

Nsight verification:
- cudaMallocAsync call count should drop by ~8,000-10,000
- cudaFreeAsync call count should drop similarly
- cudaMemcpyAsync count should stay similar (copies still happen, just without allocation)

## Rollback Criteria

Revert if:
- MLE Rounds at APC 300 improves by less than 3ms
- STARK excl trace at APC 300 shows no measurable improvement (within noise)
- Any APC 0 regression exceeds 20ms on STARK excl trace
- Proof verification fails at any APC configuration

Note: The previous "batch-mle-main-ptrs-upload" (2026-04-13-1430) measured total per-trace upload overhead at 5ms (0.9μs per call). This plan targets a subset of that overhead (alloc+free cycles, not the copy itself). The expected savings of 3-5ms are conservative. If improvement is below 3ms, the allocation pool is even faster than measured and further MLE overhead reduction requires targeting kernel execution time.
