# Plan: Pre-allocate per-thread GPU buffer pools for GKR input evaluation

## Goal

Eliminate per-AIR GPU memory allocation/deallocation inside the multi-threaded GKR input evaluation loop. Currently, each AIR in `process_gkr_input_air` allocates an `intermediates` DeviceBuffer and a `d_public_values` DeviceBuffer through the global `Mutex<MemoryManager>`, which also triggers `cudaMallocAsync`/`cudaFreeAsync` on the shared CUDA memory pool. With 8 worker threads, this creates two serialization points per allocation: (1) the Rust Mutex and (2) implicit CUDA pool cross-stream synchronization when reclaiming blocks. Pre-allocating these buffers before spawning threads and reusing them across AIRs should remove these serialization points and enable actual kernel concurrency on the 8 per-thread CUDA streams.

## Current Code Path

### Entry point

`log_gkr_input_evals()` at `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:215-297`:
1. Allocates `leaves` buffer (line 228) — single allocation, not per-AIR
2. Builds work items from `trace_interactions`, sorts by descending height (lines 232-252)
3. Syncs main stream (line 255)
4. Spawns 8 OS threads via `std::thread::scope` (lines 272-291)
5. Each thread creates fresh `d_partition_ptrs` and `tmp` buffers (lines 277-278)
6. Each thread calls `process_gkr_input_air` sequentially for its chunk of AIRs

### Per-AIR function

`process_gkr_input_air()` at `gkr_input.rs:105-210`:

**Allocations that go through MemoryManager per AIR:**
1. `d_public_values = air_ctx.public_values.to_device()` (line 135) — calls `DeviceBuffer::with_capacity(len)` + H2D copy. Only when `public_values` is non-empty.
2. `intermediates = DeviceBuffer::<EF>::with_capacity(TASK_SIZE * buffer_size)` (line 153) — when `buffer_size > 10` (i.e., `is_global = true`). For `is_global = false`, allocates with length 1 (16 bytes, trivial but still takes the Mutex).
3. `d_partition_ptrs` resize (line 145) — only if current length is too small. Already reused across AIRs within a thread.
4. `tmp` resize (line 167) — only if current length is too small AND `height != lifted_height`. Already reused across AIRs within a thread.

**Deallocations per AIR (via Drop):**
- `intermediates` drops at end of `process_gkr_input_air` → `d_free()` → `Mutex<MemoryManager>::lock()`
- `d_public_values` drops at end of `process_gkr_input_air` → `d_free()` → `Mutex<MemoryManager>::lock()`

Note: `DeviceBuffer` has no separate capacity/length distinction. `with_capacity(n)` allocates and sets `len = n` (`d_buffer.rs:60-82`). The `len()` method returns this allocated length.

### Lock contention path

`d_malloc` at `crates/cuda-common/src/memory_manager/mod.rs:141-145`:
```rust
pub fn d_malloc(size: usize) -> Result<*mut c_void, MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc(size)  // Lock held for entire operation
}
```

For small allocations (< 2MB page_size): calls `cudaMallocAsync(stream)` while holding the Rust Mutex.
For large allocations (>= 2MB): uses VirtualMemoryPool with potential defragmentation while holding the Mutex.

`d_free` at `mod.rs:150-154` has the same lock pattern.

### Why multi-streaming doesn't help

With 8 threads, each thread processes ~38 AIRs per segment (310 AIRs / 8). Per AIR, there are ~2 alloc + ~2 free calls through the Mutex. Total: ~310 × 4 = ~1240 Mutex acquisitions per segment under 8-way contention.

Additionally, `cudaMallocAsync` on per-thread streams shares a global CUDA memory pool. When one stream allocates, the CUDA runtime may synchronize with other streams to reclaim freed blocks, preventing true kernel concurrency.

Evidence: GKR input eval GPU kernel time = 471ms total (sum of all `evaluate_interactions_gkr_kernel` instances across both segments). Wall time = 529ms (267ms + 262ms per segment). Per-segment: ~236ms GPU kernel time in ~267ms wall time = effectively sequential (1.0x parallelism) despite 8 threads. Compare with Round 0 where the same multi-stream architecture achieves 1.62x — Round 0 has fewer per-AIR allocations (reuses `intermediates` via the `max_temp_bytes` budget rather than allocating per-AIR).

## Changes

### Change 1: Compute max buffer sizes before spawning threads

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
**Function**: `log_gkr_input_evals`
**Where**: After building and sorting work items (after line 252), before the sync barrier (line 255)

Add a pre-computation loop over all work items:

```rust
let mut max_intermediates_len: usize = 1;
let mut max_public_values_len: usize = 0;
let mut max_partition_ptrs_len: usize = 0;
let mut max_tmp_len: usize = 0;

for w in &work_items {
    let rules = &w.pk_air.other_data.interaction_rules;
    let buffer_size = rules.inner.buffer_size as usize;
    let is_global = buffer_size > 10;
    if is_global {
        max_intermediates_len = max_intermediates_len.max(TASK_SIZE as usize * buffer_size);
    }

    max_public_values_len = max_public_values_len.max(w.air_ctx.public_values.len());

    let num_partitions = w.air_ctx.cached_mains.len() + 1;
    max_partition_ptrs_len = max_partition_ptrs_len.max(num_partitions);

    let height = w.air_ctx.height();
    let lifted_height = height.max(1 << w.l_skip);
    if height != lifted_height {
        let num_interactions = w.pk_air.vk.symbolic_constraints.interactions.len();
        max_tmp_len = max_tmp_len.max(height * num_interactions);
    }
}
```

This is pure CPU work on data already available in the work items — no GPU calls.

### Change 2: Pre-allocate per-thread buffer pools

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
**Function**: `log_gkr_input_evals`
**Where**: After the sync barrier (line 255), before spawning threads (line 272)

Define a struct to hold per-thread buffers:
```rust
struct GkrThreadBuffers {
    intermediates: DeviceBuffer<EF>,
    public_values: DeviceBuffer<F>,
    partition_ptrs: DeviceBuffer<u64>,
    tmp: DeviceBuffer<Frac<EF>>,
}
```

Allocate `num_threads` instances. For each buffer, use `DeviceBuffer::with_capacity(max_len)` if the max is > 0, or `DeviceBuffer::new()` if 0:

```rust
let thread_buffers: Vec<GkrThreadBuffers> = (0..num_threads)
    .map(|_| GkrThreadBuffers {
        intermediates: DeviceBuffer::with_capacity(max_intermediates_len),
        public_values: if max_public_values_len > 0 {
            DeviceBuffer::with_capacity(max_public_values_len)
        } else {
            DeviceBuffer::new()
        },
        partition_ptrs: DeviceBuffer::with_capacity(max_partition_ptrs_len),
        tmp: if max_tmp_len > 0 {
            DeviceBuffer::with_capacity(max_tmp_len)
        } else {
            DeviceBuffer::new()
        },
    })
    .collect();
```

These allocations happen on the main thread's CUDA stream. All `num_threads` buffer sets are allocated sequentially (no contention), before any worker thread is spawned.

**Memory budget check**: Before allocating, compute `total_bytes = num_threads * (max_intermediates_len * size_of::<EF>() + max_public_values_len * size_of::<F>() + max_partition_ptrs_len * size_of::<u64>() + max_tmp_len * size_of::<Frac<EF>>())` (concretely: EF=16 bytes, F=4 bytes, u64=8 bytes, Frac\<EF\>=32 bytes). If this exceeds 2 GB, reduce `num_threads` (halving until it fits) to avoid OOM. Log a warning if reduction occurs. This handles the edge case where a few AIRs with very large `buffer_size` values make per-thread pre-allocation expensive.

### Change 3: Pass pre-allocated buffers to worker threads

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
**Function**: `log_gkr_input_evals`

Modify the thread-spawning code (lines 272-291). Pair each `GkrThreadBuffers` with its chunk of work items using `into_iter().zip()`:

```rust
let chunk_size = work_items.len().div_ceil(num_threads);
std::thread::scope(|s| {
    let handles: Vec<_> = thread_buffers
        .into_iter()
        .zip(work_items.chunks(chunk_size))
        .map(|(mut bufs, chunk)| {
            s.spawn(move || -> Result<(), InteractionGpuError> {
                for w in chunk {
                    process_gkr_input_air(w, &mut bufs)?;
                }
                current_stream_sync().map_err(InteractionGpuError::from)?;
                Ok(())
            })
        })
        .collect();
    // ... join handles as before
});
```

For the single-thread path (`num_threads <= 1`): use `thread_buffers.into_iter().next().unwrap()` and pass to the sequential loop.

### Change 4: Modify `process_gkr_input_air` to use pre-allocated buffers

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
**Function**: `process_gkr_input_air`

Change the function signature:
```rust
fn process_gkr_input_air<HS: GpuHashScheme>(
    w: &GkrInputWorkItem<HS>,
    buffers: &mut GkrThreadBuffers,
) -> Result<(), InteractionGpuError>
```

Replace per-AIR allocations:

1. **`intermediates`** (lines 152-156): Remove the `DeviceBuffer::with_capacity(...)` allocation. Use `&buffers.intermediates` directly. The pre-allocated buffer has `len >= TASK_SIZE * buffer_size` for all AIRs. For `is_global = false`, the kernel uses thread-local stack intermediates and ignores the global pointer, so the pre-allocated buffer is unused but harmless.

2. **`d_public_values`** (lines 132-136): Keep the `if air_ctx.public_values.is_empty()` conditional to produce `DeviceBuffer::new()` (null pointer) for AIRs with no public values — this preserves the null-pointer semantics the kernel may rely on. For non-empty public values, use `air_ctx.public_values.copy_to(&mut buffers.public_values)?` (the existing `MemCopyH2D::copy_to` at `copy.rs:68-87`) which does `cudaMemcpyAsync` into the pre-allocated buffer without reallocation. The check `self.len() > dst.len()` passes because the source length ≤ max_public_values_len = pre-allocated length. Then pass `&buffers.public_values` to the kernel.

3. **`d_partition_ptrs`** (lines 140-147): Use `buffers.partition_ptrs` instead of the old `d_partition_ptrs` parameter. Remove the resize check at line 144 since pre-allocated size is the max. Use `partition_ptrs.copy_to(&mut buffers.partition_ptrs)?` as before.

4. **`tmp`** (lines 164-172): Use `buffers.tmp` instead of the old `tmp` parameter. Remove the resize check at line 166. Note: `buffers.tmp.len()` equals `max_tmp_len` (the max across all AIRs), which may be larger than `height * num_interactions` for the current AIR. This is safe because:
   - `frac_vector_scalar_multiply_ext_fp` (line 198) uses `tmp.len()` as the element count — processing extra elements is harmless (multiplies stale data by scalar, result is never read).
   - `frac_matrix_vertically_repeat` (line 200) uses explicit `num_interactions`, `lifted_height`, `height` parameters to determine read/write bounds, not `tmp.len()`.
   - However, to avoid unnecessary GPU work on stale elements, the `frac_vector_scalar_multiply_ext_fp` call should pass `(height * num_interactions) as u32` instead of `tmp.len() as u32`. This is a one-line change at line 198.

### Change 5: Handle the single-threaded path consistently

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

For `num_threads <= 1` (lines 264-269): use the first (only) `GkrThreadBuffers` from `thread_buffers`:

```rust
if num_threads <= 1 {
    let mut bufs = thread_buffers.into_iter().next().unwrap();
    for w in &work_items {
        process_gkr_input_air(w, &mut bufs)?;
    }
}
```

## Invariants

1. **Correctness**: Every kernel call receives the same data as before. Buffers are overwritten before each kernel launch (`copy_to` for `partition_ptrs` and `public_values`, kernel writes for `intermediates`). Stale data from previous AIRs in pre-allocated buffers is always overwritten before use or ignored by the kernel. Note: `copy_to` performs a partial write — it copies `src.len()` elements into a buffer of `dst.len()` elements, leaving trailing elements stale. This is safe because the kernels never read buffer lengths; they use explicit size parameters from the proving key (e.g., `height`, `used_nodes_len`).

2. **Buffer sizing**: Pre-allocated buffers are sized to the maximum `len` across ALL AIRs in the segment. Every AIR's requirements fit. This is guaranteed by the pre-computation in Change 1.

3. **Stream ordering**: Each thread's kernels execute sequentially on its per-thread CUDA stream (`cudaStreamPerThread`). Buffer reuse across AIRs within the same thread is safe because CUDA stream ordering guarantees kernel N finishes before kernel N+1 reads the buffer.

4. **Cross-thread safety**: Each thread owns its own `GkrThreadBuffers` with its own GPU memory. No two threads share a buffer. The `SendPtr` pattern already used for `leaves_ptr` ensures output writes to non-overlapping regions.

5. **Cross-stream allocation/free**: Buffers are allocated on the main thread's stream and freed on worker threads' streams (via `Drop` when the worker's closure exits, after `current_stream_sync()`). This is safe: the worker stream is synchronized before the buffer drops, and `d_free` tracks allocations by pointer, not by stream.

6. **Public values null semantics**: AIRs with empty `public_values` still receive `DeviceBuffer::new()` (null pointer, len=0), preserving the current kernel's expectation.

7. **APC 0 path**: When `work_items.len() < 100` (APC 0 with ~20 AIRs per segment), the single-threaded path is used with 1 buffer set. Pre-allocation overhead is negligible.

8. **Memory budget**: If `total pre-allocation > 2GB`, the code reduces `num_threads` (halving) until the budget fits. In the worst case, falls back to 1 thread. Logged as a warning. The RTX 4090 has 24GB; typical max intermediates per thread is < 100MB, so 8 × 100MB = 800MB is well within budget.

## Measurement Plan

### Before changes
Run `run_pairing.sh` for APC {0, 100, 300}. Record:
- `prover.rap_constraints.logup_gkr.input_evals_time_ms` per segment (from metrics JSON)
- `prover.rap_constraints.logup_gkr_time_ms` per segment
- `stark_prove_excluding_trace_time_ms` per segment
- Full `spec.py` output for all three configs

### After changes
Run the same benchmarks. Compare:

**Primary metric**: `prover.rap_constraints.logup_gkr.input_evals_time_ms` (sum across segments)
- Target: < 400ms at APC 300 (vs current ~529ms, requiring >25% improvement)

**Secondary metric**: `stark_prove_excluding_trace_time_ms` (spec.py "STARK excl trace")
- Target: < 1600ms at APC 300 (vs current ~1720ms, requiring >7% improvement)

**No-regression**: STARK excl trace at APC 0 must not increase by more than 3% (< 2200ms).

### Nsight validation
Run nsight on APC 300. Compare kernel concurrency:
- Before: `evaluate_interactions_gkr_kernel` instances on different streams show minimal temporal overlap
- After: instances on different streams show concurrent execution (visible in nsight timeline)

### Correctness
All three APC configs must complete prove+verify successfully.

## Rollback Criteria

Revert if ANY of:
1. STARK excl trace APC 300 improvement < 60ms (< 3.5% improvement)
2. STARK excl trace APC 0 regresses by > 65ms
3. GPU OOM at any APC config
4. Prove+verify fails at any config
5. GKR input eval wall time does not improve by at least 15% at APC 300
