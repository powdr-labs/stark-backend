# Plan: Multi-stream MLE Round Evaluation

## Goal

Overlap the logup and zerocheck evaluation paths within each MLE sumcheck round using OS threads with per-thread CUDA streams (cudaStreamPerThread). This follows the same multi-stream pattern used successfully for Round 0 (3.28x improvement) and GKR input eval (1.67x improvement).

The target is the `sumcheck_polys_batch_eval` function in `crates/cuda-backend/src/logup_zerocheck/mod.rs` (lines 1576-1974). Currently, within each MLE round, the function evaluates six sequential steps:
1. Late logup traces (lines 1846-1865) — writes `logup_tilde_evals`
2. Late zerocheck traces (lines 1867-1884) — writes `zerocheck_tilde_evals`
3. Early logup traces via `evaluate_logup_batched` (lines 1888-1900) — writes `logup_out`
4. Early zerocheck DAG traces (lines 1919-1930) — writes `zc_out`
5. Early zerocheck par-Y traces (lines 1933-1953) — writes `zc_out`
6. Early zerocheck monomial traces (lines 1957-1970) — writes `zc_out`

All six steps run sequentially on the calling thread's CUDA stream. Steps 1+3 (logup) and steps 2+4+5+6 (zerocheck) are independent: they read shared immutable data and write to disjoint output arrays.

## Current Code Path

### Entry: MLE round loop
`crates/cuda-backend/src/logup_zerocheck/mod.rs` lines 660-677:
```
for round in 1..=n_max:
    sp_round_evals = prover.sumcheck_polys_batch_eval(round, r[round - 1])
    batch_s = prover.compute_batch_s_poly(sp_round_evals, ...)
    // transcript observe/sample
    prover.fold_mle_evals(round, r_round)
```

### sumcheck_polys_batch_eval (lines 1576-1974)
**Phase 1** (lines 1591-1841): CPU metadata collection. Iterates all traces, splits into late_eval and early_eval. For early_eval, collects interpolation metadata and launches batched interpolation kernel (`batched_interpolate_columns_matrix_kernel`). Constructs `TraceCtx` structs pointing to interpolated device buffers.

**Phase 2** (lines 1843-1971): GPU evaluation. Six sequential steps (listed above), each creating batch objects, launching GPU kernels, copying results D2H, and processing on CPU.

### Key data accessed by both paths
**Read-only (shared)**: `self.pk`, `self.logup_combinations`, `self.lambda_combinations`, `self.lambda_pows`, `self.d_challenges`, `self.monomial_num_y_threshold`, `self.memory_limit_bytes`, `self.sm_count`

**Logup-only mutable**: `logup_out: Vec<[Vec<EF>; 2]>`, `self.logup_tilde_evals: Vec<[EF; 2]>`

**Zerocheck-only mutable**: `zc_out: Vec<Vec<EF>>`, `self.zerocheck_tilde_evals: Vec<EF>`

### CUDA synchronization primitives
- `to_host()` (`crates/cuda-common/src/copy.rs` line 102): Acquires global `COPY_EVENT` mutex, calls `cudaMemcpyAsync` + `record_and_wait(cudaStreamPerThread)`. Blocks until the per-thread stream completes all prior work.
- `to_host_on_current_stream()` (`crates/cuda-common/src/copy.rs` line 134): Mutex-free. Calls `cudaMemcpyAsync` + `current_stream_sync()`. Safe to call from multiple threads without contention.
- `MEMORY_MANAGER` (`crates/cuda-common/src/memory_manager/mod.rs` line 30): Global mutex protecting `d_malloc`/`d_free`. Each `DeviceBuffer::with_capacity()` and `Drop` acquires this. Hold time per acquisition: ~5-10µs (one `cudaMallocAsync` call + bookkeeping).

## Changes

### Change 1: Synchronize interpolation before thread spawn

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`
**Location**: After the interpolation kernel launch (line ~1773) and Phase 3 TraceCtx construction (line ~1841), immediately before Phase 2 evaluation begins.

Add `current_stream_sync()` call. This ensures the interpolated device buffer is fully written before the spawned evaluation threads read from it on their per-thread streams. Without this, the spawned threads' CUDA streams have no ordering guarantee relative to the calling thread's stream where the interpolation kernel was launched.

```rust
// After Phase 1 (interpolation + TraceCtx construction):
current_stream_sync().map_err(MemCopyError::from)?;
// Phase 2: multi-stream evaluation begins
```

### Change 2: Extract all self borrows before thread scope

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`
**Function**: `sumcheck_polys_batch_eval`, at the start of Phase 2

Rust's borrow checker requires explicit field-level borrow splitting when passing mutable references to different fields into separate closures. Extract ALL needed references before the `std::thread::scope`:

```rust
// Immutable refs (shared by both threads)
let pk = self.pk;
let logup_combinations = &self.logup_combinations;
let lambda_combinations = &self.lambda_combinations;
let lambda_pows = self.lambda_pows.as_ref();
let d_challenges_ptr = self.d_challenges.as_ptr();
let monomial_num_y_threshold = self.monomial_num_y_threshold;
let memory_limit_bytes = self.memory_limit_bytes;
let sm_count = self.sm_count;

// Mutable refs (disjoint, one per thread)
let logup_tilde_evals = &mut self.logup_tilde_evals;
let zerocheck_tilde_evals = &mut self.zerocheck_tilde_evals;
```

### Change 3: Use to_host_on_current_stream in spawned threads

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

All `to_host()` calls within the spawned evaluation closures must be replaced with `to_host_on_current_stream()`. This avoids the global `COPY_EVENT` mutex that would otherwise serialize D2H copies across threads.

Affected call sites:
- Logup thread: inside `LogupMonomialBatch::evaluate` result (line 1861 equivalent), inside `evaluate_logup_batched` (batch_mle.rs lines 528, 609)
- Zerocheck thread: after `ZerocheckMonomialBatch::evaluate` (line 1880 equivalent), after `ZerocheckMonomialParYBatch::evaluate` (line 1950), after `ZerocheckMonomialBatch::evaluate` for low traces (line 1967), inside `evaluate_zerocheck_batched`

Implementation approach: The batch evaluate functions (`LogupMonomialBatch::evaluate`, `ZerocheckMonomialBatch::evaluate`, etc.) return `DeviceBuffer` objects. The `to_host()` is called on these return values in the caller. Change these caller-side `to_host()` calls to `to_host_on_current_stream()`.

For `evaluate_logup_batched` and `evaluate_zerocheck_batched` (which internally call `to_host()` on intermediate results), add a `use_stream_sync: bool` parameter or refactor to return DeviceBuffers that the caller converts with `to_host_on_current_stream()`. The simplest approach: replace the internal `to_host()` calls in these functions with `to_host_on_current_stream()` unconditionally — `to_host_on_current_stream` works correctly in single-threaded mode too (it's just slightly different sync mechanism, but equally correct).

### Change 4: Thread-safe wrapper for TraceCtx

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs` or `batch_mle.rs`

`TraceCtx` contains raw CUDA device pointers (`*const EF`, etc.) which are not `Send`. The logup and zerocheck closures capture `Vec<&TraceCtx>` references.

Following the pattern from `gkr_input.rs` (GKR input multi-stream): wrap the closure data in a struct that asserts `Send`:

```rust
struct LogupEvalWork<'a> {
    late_traces: Vec<&'a TraceCtx>,
    early_traces: &'a [TraceCtx],
    // ... other needed refs
}
// SAFETY: CUDA device pointers in TraceCtx reference GPU global memory
// that is accessible from any CUDA stream on the same device.
unsafe impl<'a> Send for LogupEvalWork<'a> {}
```

### Change 5: Multi-stream dispatch with threshold

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Only use multi-stream when `early_eval.len() >= MIN_MLE_MULTISTREAM_TRACES` (set to 50). Below the threshold, run the existing sequential path. At APC 300 seg0 there are ~350 early traces per round (well above threshold). At APC 0, there are ~20 traces per round (below threshold, single-stream).

```rust
const MIN_MLE_MULTISTREAM_TRACES: usize = 50;

if early_eval.len() >= MIN_MLE_MULTISTREAM_TRACES {
    // Multi-stream path: spawn logup + zerocheck on separate threads
    std::thread::scope(|s| {
        let logup_handle = s.spawn(|| { /* logup closure */ });
        let zerocheck_handle = s.spawn(|| { /* zerocheck closure */ });
        let logup_result = logup_handle.join().unwrap()?;
        let zerocheck_result = zerocheck_handle.join().unwrap()?;
        Ok(())
    })?;
} else {
    // Single-stream path: existing sequential code
    // ... (steps 1-6 unchanged)
}
```

### Change 6: Replace to_host with to_host_on_current_stream globally in evaluation functions

**File**: `crates/cuda-backend/src/logup_zerocheck/batch_mle.rs`

Replace ALL five `to_host()` calls in this file with `to_host_on_current_stream()`:
1. Line 480: inside `evaluate_zerocheck_batched`, single-trace fallback path
2. Line 492: inside `evaluate_zerocheck_batched`, normal batch path
3. Line 528: inside `evaluate_logup_batched`, low-monomial batch path
4. Line 609: inside `evaluate_logup_batched`, high-batch normal path
5. Line 650: inside `evaluate_single_logup`, oversized-trace fallback path

This is safe in single-threaded mode — `to_host_on_current_stream()` calls `cudaStreamSynchronize(cudaStreamPerThread)` which has identical synchronization semantics to `to_host()`'s `COPY_EVENT.record_and_wait(cudaStreamPerThread)`. Both block the CPU until the per-thread stream drains. The only difference is `to_host_on_current_stream` avoids the global mutex.

The `to_host_on_current_stream` trait (`MemCopyD2HStreamSync`) needs to be imported wherever it replaces `to_host`.

### MEMORY_MANAGER mutex assessment

The MEMORY_MANAGER mutex (`crates/cuda-common/src/memory_manager/mod.rs` line 30) serializes all GPU buffer allocations/frees. Each evaluation thread performs ~10-20 allocations per round (5 `to_device()` uploads in batch constructors + 2 `with_capacity()` in evaluate + per-trace intermediates in batch_mle paths + sub-batching iterations when traces exceed memory_limit_bytes) and a similar number of frees (buffer Drops). Total per round: ~40-80 mutex acquisitions across both threads, at ~6µs hold time each = ~240-480µs contention per round. Over 14 rounds: ~3.4-6.7ms.

This is modest compared to the ~3ms per-round kernel time being overlapped. Critically, each batch's buffers are created, used, and dropped sequentially within each thread — the contention is only from interleaving across the two threads, not from concurrent live buffers. The failed `multistream-stacked-reduction-round0` task had 112 *concurrent live* buffers that disrupted the CUDA memory pool state. Our allocation pattern is sequential-within-thread and well within acceptable limits.

If measured contention exceeds expectations, pre-allocating tmp_sums and output buffers before the thread scope (passing pre-allocated DeviceBuffers to evaluate via new `evaluate_into` methods) would eliminate allocation-time contention entirely. This is a fallback, not required for the initial prototype.

## Invariants

1. **Correctness**: Logup and zerocheck evaluation paths compute independent parts of the sumcheck polynomial. The logup path evaluates interaction (bus) contributions; the zerocheck path evaluates constraint contributions. `compute_batch_s_poly` (line 1977) combines both. Multi-streaming changes execution order, not computation.

2. **No write aliasing**: Logup writes to `logup_out[trace_idx]` and `logup_tilde_evals[trace_idx]`. Zerocheck writes to `zc_out[trace_idx]` and `zerocheck_tilde_evals[trace_idx]`. These are separate `Vec` allocations with distinct base pointers. Even for the same trace_idx, the writes target different arrays.

3. **Interpolation completion**: `current_stream_sync()` after the interpolation kernel (Change 1) ensures the interpolated device buffer is globally visible before evaluation threads launch kernels reading from it on their per-thread streams.

4. **APC 0 non-regression**: Threshold guard (Change 5) ensures the sequential path is used when `early_eval.len() < 50`, matching current behavior.

5. **D2H correctness**: `to_host_on_current_stream()` syncs the calling thread's per-thread stream, which is the same stream the kernel was launched on. This correctly ensures the kernel output is ready before the D2H copy completes.

## Measurement Plan

```bash
cd /home/georg/powdr
AUTOPRECOMPILE_COUNT=300 openvm-riscv/scripts/run_pairing.sh
AUTOPRECOMPILE_COUNT=0 openvm-riscv/scripts/run_pairing.sh
python3 /home/georg/spec.py results/pairing/apc300/metrics.json after_apc300
python3 /home/georg/spec.py results/pairing/apc000/metrics.json after_apc000
```

Success criteria:
- MLE Rounds at APC 300 improves by at least 10ms (158ms → ≤148ms)
- STARK excl trace at APC 300 improves by at least 10ms (1297ms → ≤1287ms)
- No regression at APC 0 (STARK excl trace ≤ 2160ms)

## Rollback Criteria

Revert if:
- MLE Rounds improvement at APC 300 is less than 10ms
- OR any APC 0 metric regresses by more than 20ms
- OR proof verification fails at any APC configuration
