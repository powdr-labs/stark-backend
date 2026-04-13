# Plan: Multi-stream Round 0 Parallel AIR Processing

## Goal

Reduce Round 0 wall-clock time at APC 300 by processing AIRs across multiple OS threads, each with its own CUDA stream (`cudaStreamPerThread`). Small kernels from different streams execute concurrently on idle SMs of the RTX 4090, improving GPU utilization from <5% (height-8 AIRs on 1/128 SMs) to 10-30% (4 concurrent kernels on different SMs).

Target: Round 0 from 663ms to ~500-565ms (15-25% improvement). STARK excl trace from 2272ms to ~2100-2175ms (4-8% improvement).

## Current Code Path

### Entry point
`crates/cuda-backend/src/logup_zerocheck/mod.rs:598` — `sumcheck_uni_round0_polys()`

### Setup phase (lines 598-728, unchanged)
Pre-computes shared data: `eq_xis` hash map, lambda powers, logup combinations, eq_3b weights, selector bases. All this data is read-only during the per-AIR loop. Runs on the main thread.

### Per-AIR loop (lines 732-880, the target)
Iterates over `ctx.per_trace` (623 entries at APC 300). For each AIR:

1. **CPU metadata** (lines 741-772): Extract `SymbolicConstraints` from `single_pk.vk.symbolic_constraints` (line 744-745), compute `omega_root`, `local_constraint_deg`, collect `main_parts` raw pointers, H2D upload `d_main_parts` via `to_device()`.
2. **Constraint kernel** (lines 777-790): `evaluate_round0_constraints_gpu()` in `round0.rs:31-124`. Allocates 3 DeviceBuffers (`intermediates`, `temp_sums_buffer`, `sp_evals`), launches `zerocheck_ntt_eval_constraints` kernel. Returns `DeviceBuffer<EF>` of size `num_cosets_zc * skip_domain`.
3. **Constraint D2H + postprocess** (lines 791-827): `sum_buffer.to_host()` (pipeline drain via `COPY_EVENT` mutex), transpose to row-major, iDFT via `UnivariatePoly::from_geometric_cosets_evals_idft`, multiply by `(Z^{2^l_skip} - 1)`.
4. **Interaction rule building** (round0.rs:161-207, called from line 831): Builds a per-AIR interaction DAG using `SymbolicDagBuilder`, constructs `SymbolicRulesGpu`, computes numer/denom weights, does 2 H2D copies (`d_numer_weights`, `d_denom_weights`). This is non-trivial CPU work (~tens of microseconds per AIR).
5. **Interaction kernel** (lines 831-846): `evaluate_round0_interactions_gpu()` in `round0.rs:132-275`. Allocates ~6 DeviceBuffers (rules, intermediates, temp_sums, sp_evals, etc.), launches `logup_bary_eval_interactions_round0` kernel. Returns `DeviceBuffer<Frac<EF>>`.
6. **Interaction D2H + postprocess** (lines 847-879): `sum.to_host()`, extract numer/denom, transpose, iDFT. Writes to `batch_sp_poly[2*trace_idx]` and `[2*trace_idx+1]`.

### Why it's slow
- 623 AIRs on one stream: kernels execute sequentially. Small kernels (height 8-64) use 1-4 of 128 SMs.
- Nsight kernel distribution for Round 0 at APC 300:
  - <100us: 420 kernels, 20ms total
  - 100-500us: 783 kernels, 202ms total
  - 500us-1ms: 322 kernels, 221ms total
  - >1ms: 55 kernels, 101ms total
- Total GPU kernel time: 543ms, wall clock: 663ms. Overhead (CPU prep + D2H): 120ms.
- GPU throughput drops 10.7x per cell from APC 0 to APC 300 due to underutilized SMs.

### Key property enabling parallelism
Each AIR's processing is **fully independent**: no data flows between AIRs in Round 0. All shared data (`pk`, `eq_xis`, `selectors_base`, trace matrices, `beta_pows`, `d_lambda_pows`) is read-only. `DeviceBuffer<T>` is `Send + Sync` (`crates/cuda-common/src/d_buffer.rs:36-37`). Output slots `batch_sp_poly[trace_idx]` are disjoint per AIR.

### Known concurrency bottleneck
The global `MemoryManager` (`crates/cuda-common/src/memory_manager/mod.rs:30`) is behind a `Mutex`. Every `d_malloc` / `d_free` acquires this lock. Each AIR does ~10 allocations (3 in constraint eval + ~6 in interaction eval + 1 for `d_main_parts`), totaling ~6,230 lock acquisitions per Round 0. With 4 threads, this mutex is a serialization point. The lock is short-lived (no kernel wait inside), so contention adds ~microseconds per acquisition. Total estimated contention overhead: 10-20ms. This limits the achievable concurrency but does not negate the benefit of concurrent kernel execution.

## Changes

### Change 1: Add `to_host_on_current_stream()` to DeviceBuffer

**File:** `crates/cuda-common/src/copy.rs`

**What:** Add a new trait + impl that uses `cudaStreamSynchronize(cudaStreamPerThread)` instead of the global `COPY_EVENT` mutex:

```rust
use crate::stream::current_stream_sync;

pub trait MemCopyD2HStreamSync<T> {
    /// Like `to_host()`, but syncs on the calling thread's per-thread stream
    /// instead of the global COPY_EVENT. Safe to call from multiple threads
    /// without mutex contention.
    fn to_host_on_current_stream(&self) -> Result<Vec<T>, MemCopyError>;
}

impl<T> MemCopyD2HStreamSync<T> for DeviceBuffer<T> {
    fn to_host_on_current_stream(&self) -> Result<Vec<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();
        check(unsafe {
            cudaMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                cudaStreamPerThread,
            )
        })?;
        current_stream_sync().map_err(|e| MemCopyError::CopyD2H(e.into()))?;
        unsafe { host_vec.set_len(self.len()); }
        Ok(host_vec)
    }
}
```

**Why:** The existing `to_host()` uses a global `Mutex<CudaEvent>` (`COPY_EVENT`, line 12). With multi-threaded Round 0, the mutex serializes D2H calls across all threads — while one thread blocks waiting for its kernel (holding the mutex), other threads' kernels complete but the threads cannot copy results. `to_host_on_current_stream()` uses per-thread stream synchronization, so each thread blocks only on its own stream. For pageable host memory, `cudaMemcpyAsync` is already effectively synchronous (CUDA docs), so the subsequent `cudaStreamSynchronize` is redundant but harmless.

### Change 2: Extract per-AIR work into a standalone function

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**What:** Define a result struct and a work-item struct, then extract the loop body (lines 741-879) into a standalone function.

```rust
/// Result of processing one AIR in Round 0.
struct Round0AirResult {
    trace_idx: usize,
    zerocheck_poly: Option<UnivariatePoly<EF>>,
    logup_numer_poly: Option<UnivariatePoly<EF>>,
    logup_denom_poly: Option<UnivariatePoly<EF>>,
}

/// All read-only references needed to process one AIR in Round 0.
struct Round0AirWorkItem<'a, HS: GpuHashScheme> {
    trace_idx: usize,
    air_idx: usize,
    single_pk: &'a DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    n: isize,                    // from n_per_trace, the lift exponent
    selectors_cube: &'a DeviceMatrix<F>,
    public_values: &'a DeviceBuffer<F>,
    eq_3bs: &'a [EF],           // host slice, per-interaction weights
    cached_mains: &'a [CommittedTraceData<GenericGpuBackend<HS>>],
    common_main: &'a DeviceMatrix<F>,
    eq_xis: &'a FxHashMap<usize, EqEvalLayers<EF>>,
    d_lambda_pows: &'a DeviceBuffer<EF>,
    beta_pows: &'a [EF],        // host slice (Vec<EF>), NOT DeviceBuffer
    l_skip: usize,
    constraint_degree: usize,    // global max constraint degree
    xi: &'a [EF],
    max_temp_bytes: usize,       // per-thread memory limit (total / num_threads)
}
```

The function `process_air_round0` contains the exact logic from lines 741-879:
1. Constructs `SymbolicConstraints::from(&single_pk.vk.symbolic_constraints)` (CPU work, per-AIR)
2. Computes `local_constraint_deg`, `omega_root`, `num_cosets_zc`, `num_cosets_logup`
3. Builds `d_main_parts` from `cached_mains`/`common_main` raw pointers via `to_device()` (H2D on worker stream)
4. Calls `evaluate_round0_constraints_gpu()` — allocates intermediate buffers on worker stream, launches kernel on worker stream
5. Calls `to_host_on_current_stream()` — syncs worker stream, copies result
6. CPU postprocess: transpose + iDFT for constraints
7. Calls `evaluate_round0_interactions_gpu()` — builds interaction DAG (CPU), allocates buffers (worker stream), launches kernel (worker stream). Uses `beta_pows` as `&[EF]` host slice (round0.rs:139) and `eq_3bs` as `&[EF]` (round0.rs:140).
8. Calls `to_host_on_current_stream()` — syncs worker stream, copies result
9. CPU postprocess: extract numer/denom, transpose + iDFT
10. Returns `Round0AirResult`

**Why:** Encapsulates per-AIR logic into a `Send`-compatible function. All references in `Round0AirWorkItem` are to `Send + Sync` types (DeviceBuffer, DeviceMatrix, DeviceStarkProvingKey are all Send+Sync). The function can be called from any OS thread.

### Change 3: Restructure Round 0 into 3 phases with parallel execution

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**What:** Replace the sequential loop (lines 732-880) with:

```rust
// Phase 1: Build work items (main thread, pure CPU)
let per_thread_temp_bytes = self.memory_limit_bytes / NUM_ROUND0_STREAMS.max(1);
let work_items: Vec<Round0AirWorkItem<HS>> = izip!(
    &ctx.per_trace, &self.n_per_trace, &selectors_base,
    &self.public_values_per_trace, &self.eq_3b_per_trace,
).enumerate().map(|(trace_idx, ((air_idx, air_ctx), &n, sel, pv, eq3b))| {
    Round0AirWorkItem {
        trace_idx,
        air_idx: *air_idx,
        single_pk: &self.pk.per_air[*air_idx],
        n,
        selectors_cube: sel,
        public_values: pv,
        eq_3bs: eq3b,
        cached_mains: &air_ctx.cached_mains,
        common_main: &air_ctx.common_main,
        eq_xis: &self.eq_xis,
        d_lambda_pows,
        beta_pows: &self.beta_pows,
        l_skip,
        constraint_degree: self.constraint_degree,
        xi: &self.xi,
        max_temp_bytes: per_thread_temp_bytes,
    }
}).collect();

// Sort by descending height for better load balance across thread chunks.
// Without sorting, large AIRs may cluster in one chunk, creating a long tail.
work_items.sort_by(|a, b| b.common_main.height().cmp(&a.common_main.height()));

// Barrier: ensure all prior GPU work (trace uploads, pk data, selectors)
// is visible to worker thread streams. Required because cudaStreamPerThread
// in worker threads are distinct streams with no implicit ordering relative
// to the main thread's stream.
current_stream_sync()?;

// Phase 2: Process AIRs in parallel across N OS threads.
// Each thread gets its own CUDA stream via cudaStreamPerThread.
// Small kernels on different streams can execute concurrently on the GPU.
let num_threads = NUM_ROUND0_STREAMS.min(work_items.len());
let results: Vec<Round0AirResult> = if num_threads <= 1 {
    // Fallback: sequential (avoids thread overhead for few AIRs, e.g. APC 0)
    work_items.iter()
        .map(|w| process_air_round0(w))
        .collect::<Result<Vec<_>, _>>()?
} else {
    let chunk_size = (work_items.len() + num_threads - 1) / num_threads;
    std::thread::scope(|s| {
        let handles: Vec<_> = work_items
            .chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
                    chunk.iter()
                        .map(|w| process_air_round0(w))
                        .collect()
                })
            })
            .collect();
        let mut all_results = Vec::with_capacity(work_items.len());
        for handle in handles {
            all_results.extend(handle.join().unwrap()?);
        }
        Ok::<_, LogupZerocheckError>(all_results)
    })?
};

// Phase 3: Scatter results into batch_sp_poly (main thread, pure CPU)
for result in results {
    if let Some(poly) = result.zerocheck_poly {
        batch_sp_poly[2 * num_present_airs + result.trace_idx] = poly;
    }
    if let Some(poly) = result.logup_numer_poly {
        batch_sp_poly[2 * result.trace_idx] = poly;
    }
    if let Some(poly) = result.logup_denom_poly {
        batch_sp_poly[2 * result.trace_idx + 1] = poly;
    }
}
```

**Why:** Phase 1 is pure CPU. Phase 2 spawns N OS threads; each gets its own CUDA stream via `cudaStreamPerThread`. The RTX 4090 (compute capability 8.9) supports up to 128 concurrent kernels. With 4 threads, up to 4 kernels from different streams execute simultaneously. For small AIRs (1-4 SMs each), this uses 4-16 SMs out of 128, versus 1-4 SMs with the single-stream approach. The CPU work per AIR (SymbolicConstraints construction, interaction DAG building, weight computation) also runs in parallel across threads. Phase 3 is a trivial O(n) scatter.

### Change 4: Add configurable thread count constant

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**What:** Add a constant near the top of the file:

```rust
/// Number of OS threads (and thus CUDA streams) for parallel Round 0 processing.
/// More threads = more concurrent kernels but more peak GPU memory and more
/// MemoryManager mutex contention. 4 is a good balance for the RTX 4090.
const NUM_ROUND0_STREAMS: usize = 4;
```

**Why:** Controls parallelism level. 4 provides meaningful concurrency without excessive memory pressure. With `max_temp_bytes / 4` per thread, peak temporary GPU memory stays bounded at the same level as the single-threaded case.

## Invariants

1. **Correctness**: Each AIR's constraint and interaction evaluations are independent. The output polynomials in `batch_sp_poly` are written to disjoint indices (`trace_idx`). The final result is identical regardless of processing order.

2. **Memory ordering**: `current_stream_sync()` on the main thread before spawning ensures all prior GPU work (trace matrices, proving key data, selector uploads) is visible to worker thread streams. Within each worker thread, operations are ordered by its per-thread CUDA stream.

3. **DeviceBuffer lifetimes**: Intermediate buffers allocated in `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu` are allocated on the worker thread's stream (`cudaMallocAsync(cudaStreamPerThread)`) and freed when the DeviceBuffer drops (still on the same thread's stream via `cudaFreeAsync`). No cross-stream memory ownership transfer.

4. **Memory budget**: `max_temp_bytes` is divided by `NUM_ROUND0_STREAMS` before being passed to worker threads. With N=4 threads, each thread's temporary buffer budget is 1/4 of the original, ensuring total peak GPU memory for Round 0 temporaries remains within the original budget.

5. **D2H safety**: `to_host_on_current_stream()` syncs only the calling thread's stream. No global mutex is held during the sync, so other threads can proceed with their kernel launches and D2H concurrently.

6. **Fallback path**: When `num_threads <= 1` (very few AIRs, or `NUM_ROUND0_STREAMS = 1`), the code uses sequential processing through the same `process_air_round0` function. This ensures no regression for APC 0 (99 large AIRs) or testing with reduced thread count.

7. **No protocol change**: Only the execution order within the prover changes. The mathematical operations per AIR remain identical. Verifier is unaffected.

8. **MemoryManager contention**: The global MemoryManager mutex (`memory_manager/mod.rs:30`) serializes `d_malloc`/`d_free` across threads. Each AIR does ~10 alloc/free cycles. At ~1-3us per lock acquisition, 4 threads processing 623 AIRs adds ~10-20ms of contention overhead. This is acceptable for a prototype. A follow-on optimization could pre-allocate per-thread reusable buffers to eliminate most allocations.

## Measurement Plan

### Commands
Run the benchmark suite in the powdr repo:
```bash
cd /home/georg/powdr/results/pairing
PROVE_BIN="/home/georg/powdr/target/release/powdr_openvm_riscv"
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics after_apc000/metrics.json --recursion
$PROVE_BIN prove --artifact apc100.cbor --input 0 --metrics after_apc100/metrics.json --recursion
$PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics after_apc300/metrics.json --recursion
```

Analyze with `spec.py`:
```bash
python3 /home/georg/spec.py after_apc300/metrics.json after_apc300
```

### Expected outcomes
- **Round 0 at APC 300**: 663ms -> 500-565ms (15-25% improvement)
- **Round 0 at APC 0**: 177ms -> 170-180ms (no regression; large kernels already saturate GPU)
- **STARK excl trace at APC 300**: 2272ms -> 2100-2175ms (4-8% improvement)
- **STARK excl trace at APC 0**: ~2130ms (no regression)
- **Correctness**: prove+verify succeeds for all three APC configs

### Profiling
Run nsight on APC 300 to verify concurrent kernel execution:
```bash
nsys profile --output after_apc300/nsys_report --force-overwrite true \
    --trace cuda,nvtx,osrt --sample none --stats true \
    -- $PROVE_BIN prove --artifact apc300.cbor --input 0 --recursion
```
Check the timeline view: kernels from different streams should overlap temporally during Round 0.

## Rollback Criteria

- **Less than 10% improvement** in Round 0 at APC 300 (i.e., Round 0 stays above 597ms): ROLLBACK. The multi-stream overhead is not justified.
- **Any regression > 5%** in STARK excl trace at APC 0: ROLLBACK. The optimization must not hurt the non-APC case.
- **Correctness failure**: Any prove+verify failure: ROLLBACK immediately.
- **Memory OOM**: If the additional concurrent intermediate buffers cause GPU OOM, reduce `NUM_ROUND0_STREAMS` to 2 and re-test. If still OOM at 2 streams: ROLLBACK.
