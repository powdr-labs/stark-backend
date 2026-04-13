# Plan: Multi-stream GKR Input Evaluation

## Goal

Reduce LogUp GKR time at APC 300 by parallelizing the GKR input evaluation across multiple OS threads with per-thread CUDA streams. This targets the single largest component of STARK excl trace (808ms / 41%), specifically the `log_gkr_input_evals()` function which launches 791 sequential per-AIR kernel calls totaling 428ms of GPU time. With 4 concurrent streams, we expect to reduce input eval wall clock from ~450ms to ~180-220ms, yielding a ~200-250ms reduction in LogUp GKR and a ~10-13% improvement in STARK excl trace at APC 300.

## Current Code Path

### Entry point

`prove_zerocheck_and_logup_gpu()` in `crates/cuda-backend/src/logup_zerocheck/mod.rs:274` calls `log_gkr_input_evals()` at line 344.

### Target function

`log_gkr_input_evals()` in `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:89-221`:

1. **Line 102-103**: Allocates `leaves` buffer (`DeviceBuffer<Frac<EF>>` with capacity `total_leaves`) and calls `fill_zero()`.
2. **Lines 106-107**: Creates shared mutable buffers `d_partition_ptrs` and `tmp` (reused across iterations).
3. **Lines 108-216**: Sequential `for meta in trace_interactions.iter().flatten()` loop. For each AIR with interactions:
   - **Lines 112-121**: Gathers preprocessed matrix and partitioned main matrices (read-only pointers).
   - **Lines 130-133**: H2D copy of `public_values` (small, per-AIR).
   - **Lines 138-145**: H2D copy of `partition_ptrs` into `d_partition_ptrs` (reused buffer, small).
   - **Lines 147-154**: Allocates `intermediates` buffer (per-iteration, freed at loop end).
   - **Lines 158-176**: Computes `dst_offset` into `leaves`, determines if lifting is needed, gets `trace_output` pointer (either `leaves_ptr` or `tmp`).
   - **Lines 177-191**: Launches `logup_gkr_input_eval` GPU kernel.
   - **Lines 193-214**: If lifting needed, launches `frac_vector_scalar_multiply_ext_fp` and `frac_matrix_vertically_repeat` GPU kernels.

### Why it is slow

Nsight profiling at APC 300 shows:
- `evaluate_interactions_gkr_kernel<true>`: 374.9ms across 486 instances (avg 771us each)
- `evaluate_interactions_gkr_kernel<false>`: 53.6ms across 305 instances (avg 176us each)
- Total: 428.5ms GPU kernel time, ~450ms wall clock

Each kernel uses 1-4 SMs on the RTX 4090's 128 SMs. Running sequentially on a single CUDA stream (`cudaStreamPerThread`), >95% of GPU compute capacity is idle during each kernel launch. The per-AIR operations are fully independent — they read from different trace partitions and write to non-overlapping offsets in the `leaves` buffer.

### Successful precedent

The Round 0 multi-stream optimization (task `2026-04-13-0100-multistream-round0-parallel`) achieved 1.89x improvement at APC 300 using the same pattern: distributing independent per-AIR work across 4 OS threads, each with `cudaStreamPerThread`. The kernel size distribution is similar (Round 0 avg ~600us, GKR input avg ~540us).

## Changes

### Change 1: Add `GkrInputWorkItem` struct

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

Add a struct that holds all read-only references needed to process one AIR's GKR input evaluation:

```rust
struct GkrInputWorkItem<'a, HS: GpuHashScheme> {
    meta: &'a TraceInteractionMeta,
    air_ctx: &'a AirProvingContext<GenericGpuBackend<HS>>,
    pk_air: &'a DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    l_skip: usize,
    d_challenges: &'a DeviceBuffer<EF>,
    /// Raw pointer into the pre-allocated, zero-filled `leaves` buffer at this AIR's offset.
    leaves_ptr: *mut Frac<EF>,
}
```

**Why**: Encapsulates per-AIR data so that the worker function can process one AIR independently, without mutable shared state. The `leaves_ptr` is a raw pointer because different AIRs write to non-overlapping regions of the same buffer; Rust's borrow checker cannot express this directly, but it is safe because the stacked layout guarantees non-overlapping offsets.

`leaves_ptr` must be wrapped in a newtype that implements `Send`:

```rust
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
```

### Change 2: Add `process_gkr_input_air()` function

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

Extract the loop body (lines 109-215) into a standalone function:

```rust
fn process_gkr_input_air<HS: GpuHashScheme>(
    w: &GkrInputWorkItem<HS>,
    d_partition_ptrs: &mut DeviceBuffer<u64>,
    tmp: &mut DeviceBuffer<Frac<EF>>,
) -> Result<(), InteractionGpuError> {
    let null_preprocessed = DeviceBuffer::<F>::new();
    // ... same logic as the current loop body, but using w.leaves_ptr.0
    // instead of indexing into a shared leaves buffer.
    //
    // d_partition_ptrs and tmp are passed in as grow-only buffers per thread
    // (matching the current sequential code's reuse pattern at lines 106-107, 142-144, 170-172).
    //
    // Each call creates its own:
    //   - intermediates (GPU buffer, per-iteration as in current code)
    //   - d_public_values (H2D copy, small)
    //   - null_preprocessed (zero-cost, just a null DeviceBuffer)
}
```

**Why**: A standalone function can be called from any thread. Each thread will get its own `cudaStreamPerThread`, so GPU kernel launches from different threads execute on different CUDA streams concurrently.

The `d_partition_ptrs` and `tmp` buffers follow the current code's grow-only pattern (lines 106-107, 142-144, 170-172): allocated once per thread, grown as needed within the thread's chunk. This avoids repeated malloc/free cycles and reduces `MemoryManager` mutex contention compared to per-AIR allocation.

The `null_preprocessed` is created per call as `DeviceBuffer::new()` (null pointer, zero cost — no GPU allocation). The `intermediates` buffer is allocated per-AIR as in the current code (freed at function return). The `d_public_values` H2D copy is small (a few field elements).

Per-thread peak memory is bounded: `d_partition_ptrs` is tiny (2-3 pointers = 16-24 bytes), `intermediates` capacity is at most `TASK_SIZE * buffer_size` (typically a few MB), and `tmp` is `height * num_interactions` (varies but bounded by the largest AIR in the chunk).

### Change 3: Restructure `log_gkr_input_evals()` into 3 phases

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

Replace the sequential loop with a three-phase structure:

**Phase 1 (CPU, main thread)**: Build work items and sort by descending height for load balance.

```rust
let mut work_items: Vec<GkrInputWorkItem<HS>> = trace_interactions
    .iter()
    .flatten()
    .map(|meta| {
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];
        let slice = meta.layout_slices.first().unwrap();
        assert_eq!(slice.col_idx, 0);
        let dst_offset = slice.row_idx;
        let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };
        GkrInputWorkItem {
            meta,
            air_ctx,
            pk_air,
            l_skip,
            d_challenges,
            leaves_ptr: SendPtr(leaves_ptr),
        }
    })
    .collect();

work_items.sort_by(|a, b| b.air_ctx.height().cmp(&a.air_ctx.height()));
```

**Sync barrier**: Call `current_stream_sync()` to ensure the `leaves.fill_zero()` and any prior GPU work (trace uploads, pk data) is visible to worker thread streams.

**Phase 2 (GPU, multi-threaded)**: Process AIRs in parallel.

```rust
const NUM_GKR_INPUT_STREAMS: usize = 4;

let num_threads = if work_items.len() >= 100 {
    NUM_GKR_INPUT_STREAMS.min(work_items.len())
} else {
    1
};

if num_threads <= 1 {
    let mut d_partition_ptrs = DeviceBuffer::<u64>::new();
    let mut tmp = DeviceBuffer::<Frac<EF>>::new();
    for w in &work_items {
        process_gkr_input_air(w, &mut d_partition_ptrs, &mut tmp)?;
    }
} else {
    let chunk_size = work_items.len().div_ceil(num_threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = work_items
            .chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || -> Result<(), InteractionGpuError> {
                    let mut d_partition_ptrs = DeviceBuffer::<u64>::new();
                    let mut tmp = DeviceBuffer::<Frac<EF>>::new();
                    for w in chunk {
                        process_gkr_input_air(w, &mut d_partition_ptrs, &mut tmp)?;
                    }
                    // Explicit stream sync to ensure all GPU writes from this
                    // thread's stream are globally visible before thread exit.
                    // This makes the synchronization invariant explicit rather
                    // than relying on the CudaThreadCleanup TLS destructor.
                    current_stream_sync().map_err(InteractionGpuError::from)?;
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok::<_, InteractionGpuError>(())
    })?;
}
```

**Why**: Same approach as the successful Round 0 multi-stream. `std::thread::scope` with N threads, each with `cudaStreamPerThread` for concurrent kernel execution. The threshold of 100 applies to `work_items.len()`, which is the count of AIRs-with-interactions (the flatten count from `trace_interactions`). At APC 0 (99 total AIR instances across 5 segments, ~20 per segment), the flatten count will be at most 99 (and likely less since some AIRs may lack interactions), so APC 0 uses the sequential path. This avoids the memory pool contention issue observed in the Round 0 task. At APC 100 (~360 AIR instances) and APC 300 (~623 AIR instances), the flatten count is well above 100.

**Phase 3**: None needed. Unlike Round 0, the results are written directly into the `leaves` buffer via raw pointers during Phase 2 — there is no CPU-side scatter step.

### Change 4: Reuse `NUM_ROUND0_STREAMS` constant or define `NUM_GKR_INPUT_STREAMS`

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

Define a constant `const NUM_GKR_INPUT_STREAMS: usize = 4;` for the number of worker threads. This matches `NUM_ROUND0_STREAMS` in `mod.rs:93` and can be tuned independently if needed.

**Why**: Keeps the GKR input parallelism configuration independent from Round 0, in case they need different tuning (e.g., if GKR input has different memory pressure characteristics).

## Invariants

1. **Correctness**: Each AIR writes to a non-overlapping region of `leaves`, determined by `dst_offset = meta.layout_slices.first().unwrap().row_idx`. The stacked layout guarantees these regions do not overlap. The `fill_zero()` call happens before the sync barrier, ensuring all regions start zeroed.

2. **Memory safety**: `leaves_ptr` raw pointers are derived from a single `DeviceBuffer` that outlives all worker threads (scoped threads join before `leaves` is returned). No thread reads from another thread's write region.

3. **GPU memory budget**: Each thread allocates its own `intermediates` (at most `TASK_SIZE * buffer_size * sizeof(EF)` ≈ few MB) and `tmp` (at most `height * num_interactions * sizeof(Frac<EF>)`). With 4 threads, peak additional GPU memory is ~4x one thread's allocation. This is bounded because the largest AIRs (which have the biggest buffers) are distributed across different threads by the height-descending sort.

4. **APC 0 no-regression**: The 100-AIR threshold ensures APC 0 (99 AIR instances across 5 segments, ~20 per segment) uses the sequential path. This avoids the memory pool contention issue observed in the Round 0 task.

5. **Determinism**: The `leaves` buffer content is deterministic regardless of thread scheduling because each AIR writes to a fixed, non-overlapping offset. The function returns the same `(leaves, alpha_logup)` as before.

6. **Stream visibility**: `current_stream_sync()` before spawning threads ensures all prior GPU operations (trace uploads, pk data, `fill_zero()`) are visible to worker streams. Each worker thread calls `current_stream_sync()` as its last operation, which calls `cudaStreamSynchronize(cudaStreamPerThread)` to drain the worker's CUDA stream and ensure all GPU writes are globally visible. `std::thread::scope` guarantees all spawned threads (including their `current_stream_sync()` calls) complete before returning, so the main thread sees all worker GPU writes. This explicit sync avoids depending on the `CudaThreadCleanup` TLS destructor (`crates/cuda-common/src/stream.rs:101-108`), making the correctness invariant visible in the code.

## Measurement Plan

### Before measurement
Run the prove command for APC {0, 100, 300} on the current code (already done — results in `measure_apc{000,100,300}/`).

### After measurement
Run the same prove commands after implementing the changes:

```bash
cd /home/georg/powdr/results/pairing
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics after_apc000.json --recursion
$PROVE_BIN prove --artifact apc100.cbor --input 0 --metrics after_apc100.json --recursion
$PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics after_apc300.json --recursion
```

Analyze with `spec.py` and compare:

| Metric | Target |
|--------|--------|
| LogUp GKR (APC 300) | < 650ms (currently 808ms, >20% improvement) |
| STARK excl trace (APC 300) | < 1830ms (currently 1986ms, >8% improvement) |
| STARK excl trace (APC 0) | No regression (within 3% noise, currently 2118ms) |
| Round 0 (APC 300) | No regression (currently 349ms) |

### Profiling
Run nsight on APC 300 to verify concurrent kernel execution:

```bash
nsys profile --output profile_after --trace cuda,nvtx --sample none -- $PROVE_BIN prove --artifact apc300.cbor --input 0 --recursion
```

Verify that `evaluate_interactions_gkr_kernel` instances show overlapping time ranges across different CUDA streams in the nsight timeline.

## Rollback Criteria

Revert if ANY of:
- LogUp GKR at APC 300 improves by less than 10% (< 80ms improvement from 808ms)
- STARK excl trace at APC 0 regresses by more than 3% (> 2182ms)
- Any APC config fails prove+verify
- Peak GPU memory increases by more than 2GB at any APC config
