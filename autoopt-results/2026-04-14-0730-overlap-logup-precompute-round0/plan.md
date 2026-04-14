# Plan: Overlap Logup Combination Precompute with Round 0

## Goal

Hide ~22ms per segment of logup combination precomputation (d_eq_3b H2D uploads + precompute_logup_combinations kernel launches) behind the Round 0 multi-stream evaluation by running them concurrently on a background thread. The precompute output is only consumed during MLE rounds and has zero data dependencies with Round 0 evaluation. Per-segment Round 0 is 92-93ms; the precompute takes ~22ms per segment. Overlapping saves ~22ms × 2 segments = ~44ms total on STARK excl trace at APC 300.

## Current Code Path

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**Function**: `sumcheck_uni_round0_polys` (line 882)

**Per-segment timing (APC 300, from fresh measurement 2026-04-14)**: `ple_round0_time_ms` = 92ms (seg 0), 93ms (seg 1). Breakdown estimate: ~15ms setup (lambda pows, eq_3b CPU, eq_xis, sels, work items, sort, extract tables, batch array, barrier) + ~22ms precompute (d_eq_3b upload + logup combinations) + ~55ms multi-stream evaluation.

Current sequential order within this function:

1. **Lines 893-908**: Lambda powers upload + lambda combination precompute (needed for Round 0)
2. **Lines 912-944**: eq_3b host-side CPU computation (needed for Round 0 weight computation — each work item borrows `&self.eq_3b_per_trace[trace_idx]` at line 1082)
3. **Lines 945-955**: d_eq_3b device upload — ~395 `to_device()` calls per segment producing `self.d_eq_3b_per_trace`. NOT needed for Round 0. Only used by step 4.
4. **Lines 957-975**: logup_combinations precompute — iterates over all traces, calls `compute_logup_combinations()` per trace. Each call launches 2 GPU kernels (`precompute_logup_numer_combinations` + `precompute_logup_denom_combinations`). ~791 kernel launches per segment, ~18ms GPU time per segment. NOT needed for Round 0, only for MLE rounds.
5. **Lines 977-1002**: eq_xis + sels_per_trace_base (needed for Round 0 work items)
6. **Lines 1014-1137**: Work item construction, sorting, extract tables, batch array
7. **Line 1192-1194**: `current_stream_sync()` barrier
8. **Lines 1196+**: Round 0 multi-stream evaluation (~65-70ms, 8 threads at ~42% GPU utilization)

Steps 3-4 total ~22ms per segment and are independent of steps 5-8. There is ample GPU capacity for concurrent precompute kernels during Round 0.

**Data dependency analysis** (verified against source code):

| Data | Used by Round 0? | Used by precompute? | Conflict? |
|------|-------------------|---------------------|-----------|
| `self.pk` | Yes (shared &) | Yes (shared &) | No (both read-only) |
| `self.eq_3b_per_trace` (host) | Yes (work items borrow `&[EF]` at line 1082) | Yes (read for upload + CPU bus_term_sum) | No (both read-only) |
| `self.d_eq_3b_per_trace` (device) | No | Yes (written by upload) | No conflict |
| `self.d_beta_pows` | No (Round 0 uses host-side `beta_pows` via work item field at line 1087) | Yes (read by precompute kernels) | No |
| `self.logup_combinations` | No (first consumed at line 1739 in MLE rounds) | Yes (written by precompute) | No conflict |
| `self.beta_pows` (host) | Yes (work items borrow `&[EF]` at line 1087) | Yes (read for bus_term_sum CPU computation in `compute_logup_combinations` line 570) | No (both read-only) |

## Changes

### Change 1: Restructure `sumcheck_uni_round0_polys` to overlap precompute with Round 0

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**What**: Reorder the function body so that steps 3-4 (d_eq_3b upload + logup precompute) run on a background thread concurrent with step 8 (Round 0 multi-stream evaluation). Only applied in the multi-threaded path (num_threads > 1); the single-threaded path retains the current sequential behavior.

**New flow**:

```
1. Lambda powers + lambda combinations            (sequential, same as before)
2. eq_3b CPU computation → self.eq_3b_per_trace    (sequential, same as before)
3. eq_xis + sels_per_trace_base                    (sequential, MOVED BEFORE precompute)
4. Work item construction + sorting + extract tables + batch array  (sequential)
5. current_stream_sync() barrier
6. IF num_threads > 1:
     std::thread::scope:
       ├─ Background thread: d_eq_3b upload + logup_combinations precompute
       │   → returns (Vec<DeviceBuffer<EF>>, Vec<Option<LogupCombinations>>)
       │   → calls current_stream_sync() before returning
       └─ Round 0 worker threads (0..num_threads): process_air_round0 (same as before)
     Store background thread results into self
   ELSE (num_threads <= 1):
     d_eq_3b upload + logup_combinations precompute (sequential, same as current code)
     Single-threaded Round 0 evaluation
7. Continue to MLE rounds (which consume self.logup_combinations)
```

**Specific code changes**:

a) **Move lines 977-1002 (eq_xis + sels) to before lines 945** — these currently sit between the precompute and work item construction but are independent of the precompute. Moving them earlier doesn't change behavior.

b) **Remove lines 945-975 (d_eq_3b upload + logup precompute) from their current position.** They will be placed inside the `thread::scope` block for the multi-threaded path and kept sequential for the single-threaded path.

c) **In the multi-threaded path** (the existing `thread::scope` block that spawns Round 0 worker threads, around line 1292), spawn one additional background thread before the Round 0 workers:

```rust
// Capture shared references for the background thread
let pk_ref = self.pk;
let eq_3b_ref = &self.eq_3b_per_trace;
let d_beta_pows_ref = &self.d_beta_pows;
let beta_pows_ref = &self.beta_pows;
let per_trace_ref = &ctx.per_trace;

std::thread::scope(|s| {
    // Background thread: upload d_eq_3b + precompute logup combinations
    let precompute_handle = s.spawn(|| -> Result<_, LogupZerocheckError> {
        let d_eq_3b: Vec<DeviceBuffer<EF>> = eq_3b_ref
            .iter()
            .map(|eq_3bs| {
                if eq_3bs.is_empty() {
                    Ok(DeviceBuffer::new())
                } else {
                    eq_3bs.to_device().map_err(MemCopyError::from)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut logup_combs: Vec<Option<LogupCombinations>> =
            (0..num_present_airs).map(|_| None).collect();
        for (trace_idx, (air_idx, _)) in per_trace_ref.iter().enumerate() {
            let air_pk = &pk_ref.per_air[*air_idx];
            if air_pk.other_data.interaction_monomials.is_some()
                && !eq_3b_ref[trace_idx].is_empty()
            {
                logup_combs[trace_idx] = Some(
                    compute_logup_combinations(
                        pk_ref, *air_idx, d_beta_pows_ref,
                        &d_eq_3b[trace_idx], &eq_3b_ref[trace_idx], beta_pows_ref,
                    ).map_err(LogupZerocheckError::LogupCombinations)?
                );
            }
        }

        // Ensure all GPU work on this thread's stream is visible
        current_stream_sync().map_err(MemCopyError::from)?;
        Ok((d_eq_3b, logup_combs))
    });

    // Spawn Round 0 worker threads (same as current code)
    let round0_handles: Vec<_> = (0..num_threads).map(|tid| {
        s.spawn(move || { /* existing process_air_round0 logic */ })
    }).collect();

    // Join all
    // ... collect round0 results ...
    let precompute_result = precompute_handle.join().unwrap()?;
    (round0_results, precompute_result)
});

// Store precompute results
self.d_eq_3b_per_trace = precompute_result.0;
self.logup_combinations = precompute_result.1;
```

d) **In the single-threaded path** (num_threads <= 1, around line 1272): keep the d_eq_3b upload + logup precompute in sequential order, same as current code. The overlap provides no benefit here since the single-threaded Round 0 already saturates the GPU for the few large AIRs.

### Change 2: Handle borrow references for the background thread

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Before the `thread::scope` block, create local reference bindings that the scope can capture:
```rust
let pk_ref = self.pk;
let eq_3b_ref = &self.eq_3b_per_trace;
let d_beta_pows_ref = &self.d_beta_pows;
let beta_pows_ref = &self.beta_pows;
let per_trace_ref = &ctx.per_trace;
```

These are all shared borrows (`&`) that coexist with the shared borrows already taken by Round 0 work items. `thread::scope` ensures all threads complete before the scope exits, so all borrows are valid.

### Change 3: Ensure correct stream synchronization

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Three synchronization points are needed:

1. **Existing barrier (line 1192)**: `current_stream_sync()` — ensures main-thread GPU uploads (sels, lambda_pows, batch_array, extract tables) are visible to Round 0 worker threads. Stays as-is.

2. **Background thread exit**: `current_stream_sync()` inside the background thread ensures all d_eq_3b uploads and precompute kernel launches are complete before the thread returns results. This is critical because the `LogupCombinations` contain `DeviceBuffer` objects allocated on the background thread's `cudaStreamPerThread`. Without this sync, the buffers might not be visible to other streams.

3. **No additional barrier needed after thread::scope**: Rust's `thread::scope` guarantees the background thread has completed (including its `current_stream_sync()`) before the scope returns. The returned `DeviceBuffer` objects are ready for use on any stream.

## Invariants

1. **Correctness**: The logup_combinations values are identical — same inputs, same computation, same results. Only the timing changes.
2. **Stream safety**: Background thread's `cudaStreamPerThread` is independent from Round 0 worker threads' streams. The background thread's `current_stream_sync()` at exit ensures all GPU work is globally visible.
3. **No APC 0 regression**: At APC 0, num_threads = 1 (< 100 AIRs). The single-threaded path is unchanged — no background thread is spawned.
4. **Memory pool interaction**: The background thread allocates DeviceBuffers through `cudaMallocAsync` on its own stream. These are small allocations (~800 bytes per trace, ~600KB total). Round 0 workers use pre-allocated buffers for 95% of AIRs (from the threshold prealloc optimization), so mutex contention between the background thread and Round 0 workers is minimal (~30 fallback allocations from Round 0 vs ~791 precompute allocations — the mutex acquisition cost of ~2us × 800 = ~1.6ms is negligible).
5. **Borrow safety**: All references captured by the background thread are shared borrows (`&`) that coexist with the shared borrows held by Round 0 work items. No mutable aliases exist during the `thread::scope` lifetime. The background thread returns owned values, which are stored into `self` only after the scope exits.

## Measurement Plan

Run the benchmark in `/home/georg/powdr`:
```bash
cd results/pairing
RUST_LOG=info powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics current_apc300/metrics.json --recursion
RUST_LOG=info powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics current_apc000/metrics.json --recursion
```

Analyze with `spec.py`:
```bash
python3 ~/spec.py current_apc300/metrics.json after_apc300
python3 ~/spec.py current_apc000/metrics.json after_apc000
```

Check per-segment breakdown:
```python
# Verify per-segment ple_round0_time_ms decreased by ~20ms each
python3 -c "import json; [print(g) for g in json.load(open('metrics.json'))['gauge'] if 'ple_round0' in g['metric']]"
```

Expected results:
- ple_round0 per segment at APC 300: 92-93ms → ~70-73ms (~22ms reduction each)
- Round 0 total at APC 300: 202ms → ~158ms
- STARK excl trace at APC 300: 1327ms → ~1285ms
- APC 0: no regression (unchanged single-threaded path)

## Rollback Criteria

Revert if ANY of:
- Round 0 at APC 300 improves by less than 25ms total (below noise threshold for 2-segment sum)
- STARK excl trace at APC 300 regresses vs before
- Any metric at APC 0 regresses by more than 20ms
- Correctness failure (proof verification fails)
- LogUp GKR or other phases regress by more than 15ms at APC 300 (indicating memory pool disruption from concurrent allocations)
