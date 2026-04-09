# Plan: Round 0 Parallel CUDA Streams

## Claimed Bottleneck

Round 0 evaluation in `sumcheck_uni_round0_polys` (file: `crates/cuda-backend/src/logup_zerocheck/mod.rs:598`) processes all AIRs sequentially in a single loop (lines 732-880). Each iteration launches two GPU kernels (`evaluate_round0_constraints_gpu` + `evaluate_round0_interactions_gpu`) plus D2H transfers and CPU-side interpolation. With 300+ APC AIRs, most are small and don't saturate the GPU. All kernels execute on the same per-thread CUDA stream, serialized.

Previous Report 1 measured -280ms for APC300 with this approach. Previous Report 2 used kernel-level batching (-419ms) but had register pressure issues.

## Proposed Mechanism

Use `std::thread::scope` to partition per-AIR Round 0 work across N OS threads. Since the codebase compiles CUDA with `--default-stream=per-thread` (set in `crates/cuda-builder/src/lib.rs`), each OS thread automatically gets its own CUDA stream. Kernels from different threads execute concurrently on the GPU.

### Implementation Details

**Target function:** `sumcheck_uni_round0_polys` in `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**Step 1:** After the setup phase (lines 598-728), collect all loop inputs into a Vec of work items:
```rust
struct Round0WorkItem<'a> {
    trace_idx: usize,
    air_idx: usize,
    air_ctx: &'a AirProvingContext<...>,
    n: i32,
    selectors_cube: &'a DeviceMatrix<F>,
    public_values: &'a DeviceBuffer<F>,
    eq_3bs: &'a [EF],
}
```

**Step 2:** Determine parallelism. Gate on number of AIRs:
```rust
const PARALLEL_STREAMS_THRESHOLD: usize = 100;
const NUM_ROUND0_THREADS: usize = 4;
let num_threads = if work_items.len() > PARALLEL_STREAMS_THRESHOLD { NUM_ROUND0_THREADS } else { 1 };
```

**Step 3:** Extract read-only references from `self` into local variables:
- `pk: &DeviceMultiStarkProvingKey` (self.pk)
- `eq_xis: &FxHashMap` (self.eq_xis)
- `memory_limit_bytes: usize` (divided by num_threads)
- `beta_pows: &[EF]` (self.beta_pows)
- `d_lambda_pows: &DeviceBuffer<EF>` (already a local)
- `constraint_degree: usize` (self.constraint_degree)
- `l_skip: usize` (already a local)

**Step 4:** Use `std::thread::scope` to process work items in parallel:
```rust
// Each work item produces: (trace_idx, Option<zc_coeffs>, Option<numer_poly>, Option<denom_poly>)
type Round0Result = (usize, Option<Vec<EF>>, Option<UnivariatePoly<EF>>, Option<UnivariatePoly<EF>>);

let results: Vec<Round0Result> = if num_threads <= 1 {
    // Sequential path (unchanged behavior)
    work_items.iter().map(|item| process_round0_item(item, ...)).collect()
} else {
    // Parallel path
    let chunk_size = (work_items.len() + num_threads - 1) / num_threads;
    let chunks: Vec<_> = work_items.chunks(chunk_size).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks.into_iter().map(|chunk| {
            s.spawn(|| {
                chunk.iter().map(|item| process_round0_item(item, ...)).collect::<Vec<_>>()
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
};
```

**Step 5:** Scatter results back into `batch_sp_poly`:
```rust
for (trace_idx, zc_coeffs, numer_poly, denom_poly) in results {
    if let Some(coeffs) = zc_coeffs {
        batch_sp_poly[2 * num_present_airs + trace_idx] = UnivariatePoly::new(coeffs);
    }
    if let Some(poly) = numer_poly {
        batch_sp_poly[2 * trace_idx] = poly;
    }
    if let Some(poly) = denom_poly {
        batch_sp_poly[2 * trace_idx + 1] = poly;
    }
}
```

**Step 6:** Extract the loop body (lines 741-879) into a helper function `process_round0_item(...)` that takes only shared references. All GPU allocations (intermediates, temp_sums_buffer) are local to this function and freed on return.

### Deferred D2H (addressing review finding #1)

The `to_host()` implementation uses a global `COPY_EVENT` mutex, serializing D2H transfers across threads. To maximize GPU kernel overlap:

1. Each thread's `process_round0_item` returns `DeviceBuffer` outputs (not host vectors).
2. After `thread::scope` completes, iterate over collected DeviceBuffers and do `to_host()` + CPU postprocessing sequentially.

Output buffers are tiny (sp_evals: ~48 EF elements = 768 bytes; s_evals: ~64 Frac<EF> = 2KB), so keeping them alive across all AIRs is negligible memory impact.

### Memory Handling

Each thread keeps the full `memory_limit_bytes` budget. The limit is advisory (code already warns but doesn't error when exceeded, see round0.rs:86). Small APC AIRs use tiny temporary buffers regardless of the limit. The large temporary buffers (intermediates, temp_sums_buffer) are allocated and freed within each evaluate function call, so they don't accumulate.

### Thread Safety

- All `self` fields used in the loop are read-only during the loop. The `&mut self` is only needed for the setup phase before the loop and `mem.emit_metrics` after.
- `DeviceBuffer` allocations go through the Rust-side `MEMORY_MANAGER` mutex, so are thread-safe but serialized.
- `to_device()` uses `cudaMemcpyAsync` on the per-thread stream, thread-safe.
- `evaluate_round0_interactions_gpu` builds a new `SymbolicDagBuilder` locally (round0.rs:163) - no shared mutable state.
- `d_main_parts` is built inside each iteration (line 768), contained within `process_round0_item`.
- `SymbolicConstraints::from` (line 744-745) creates a local copy per AIR, no sharing.

### Fallback Path

When `work_items.len() <= PARALLEL_STREAMS_THRESHOLD`, the sequential path is used, preserving existing behavior for small numbers of AIRs.

## Expected Savings

- APC300 (623 AIRs): ~200-300ms reduction in Round 0 phase
- APC100 (~200 AIRs): ~50-100ms reduction
- APC000 (99 AIRs): minimal change (below threshold, sequential path used)

## Implementation Checklist

1. [ ] Add `process_round0_item` helper function extracted from the loop body (lines 741-879)
2. [ ] Add `Round0WorkItem` struct to hold per-AIR inputs
3. [ ] Add parallel dispatch using `std::thread::scope` with gating threshold
4. [ ] Add memory limit division by thread count
5. [ ] Scatter results back into `batch_sp_poly`
6. [ ] Run `cargo check -p openvm-cuda-backend` to verify compilation
7. [ ] Run pairing benchmark for APC 0, 100, 300 to measure improvement
8. [ ] Profile with nsight to verify concurrent stream execution
9. [ ] Verify proof still passes verification
