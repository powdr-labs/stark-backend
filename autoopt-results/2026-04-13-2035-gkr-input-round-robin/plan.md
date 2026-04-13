# Plan: GKR Input Eval Round-Robin Work Assignment

## Goal

Replace contiguous work chunking in `log_gkr_input_evals()` with interleaved (round-robin) assignment to balance GPU kernel time and per-AIR fixed overhead across the 8 worker threads. This follows the exact pattern proven in the Round 0 optimization (`2026-04-13-1930-round0-interleaved-work-balance`), which reduced Round 0 thread spread from 26ms to 7ms and improved Round 0 by 36ms (1.13x) at APC 300.

## Current Code Path

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

**Function**: `log_gkr_input_evals()` (lines 212-354)

**Current flow**:
1. **Line 232-250**: Build `work_items: Vec<GkrInputWorkItem>` from `trace_interactions`
2. **Line 252**: Sort by descending height: `work_items.sort_by(|a, b| b.air_ctx.height().cmp(&a.air_ctx.height()))`
3. **Lines 254-279**: Pre-compute max buffer sizes across all work items
4. **Lines 285-303**: Determine `num_threads` (8 if ≥100 AIRs, else 1), apply memory budget check
5. **Lines 305-321**: Pre-allocate `GkrThreadBuffers` per thread
6. **Lines 323-327**: Single-thread path (APC 0, <100 AIRs)
7. **Lines 329-348**: Multi-thread path with **contiguous chunking**:
   ```rust
   let chunk_size = work_items.len().div_ceil(num_threads);
   std::thread::scope(|s| {
       let handles: Vec<_> = thread_buffers
           .into_iter()
           .zip(work_items.chunks(chunk_size))  // ← CONTIGUOUS
           .map(|(mut bufs, chunk)| {
               s.spawn(move || {
                   for w in chunk {
                       process_gkr_input_air(w, &mut bufs)?;
                   }
                   current_stream_sync()?;
                   Ok(())
               })
           })
           .collect();
   ```

**Why it's slow**: At APC 300, ~310 AIRs per segment sorted by descending height. With 8 threads and contiguous chunking (`chunks(39)`):
- Thread 0: items 0-38 (the 39 largest AIRs by height)
- Thread 7: items 273-309 (the 39 smallest AIRs by height)

Large AIRs have long kernel execution times (~1-13ms per AIR). Small AIRs have short kernel times (~10-100μs) but similar fixed per-AIR overhead (~1.5-2ms for kernel launch, allocation reuse, D2H copy, etc.). This creates thread imbalance: thread 0 finishes last because it processes all the largest AIRs, while thread 7 finishes early.

## Changes

### Change 1: Replace contiguous chunking with round-robin assignment

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`  
**Lines**: 329-348 (multi-thread path)

Replace the `work_items.chunks(chunk_size)` dispatch with round-robin index assignment, following the pattern from `mod.rs:950-985`.

**Before** (current code):
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
    for handle in handles {
        handle.join().unwrap()?;
    }
    Ok::<_, InteractionGpuError>(())
})?;
```

**After** (round-robin):
```rust
// Interleaved (round-robin) assignment on the height-sorted list.
// Item i goes to thread (i % num_threads). Each thread gets a balanced
// mix of large and small AIRs without needing a calibrated cost model.
let items_per_thread = work_items.len().div_ceil(num_threads);
let mut thread_indices: Vec<Vec<usize>> =
    (0..num_threads).map(|_| Vec::with_capacity(items_per_thread)).collect();
for (idx, _) in work_items.iter().enumerate() {
    thread_indices[idx % num_threads].push(idx);
}

std::thread::scope(|s| {
    let handles: Vec<_> = thread_buffers
        .into_iter()
        .zip(thread_indices.into_iter())
        .enumerate()
        .map(|(thread_id, (mut bufs, indices))| {
            let items = &work_items;
            s.spawn(move || -> Result<(), InteractionGpuError> {
                let t0 = std::time::Instant::now();
                for &idx in &indices {
                    process_gkr_input_air(&items[idx], &mut bufs)?;
                }
                current_stream_sync().map_err(InteractionGpuError::from)?;
                tracing::debug!(
                    thread_id,
                    elapsed_ms = t0.elapsed().as_millis(),
                    num_airs = indices.len(),
                    "gkr_input thread done"
                );
                Ok(())
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap()?;
    }
    Ok::<_, InteractionGpuError>(())
})?;
```

**Why this helps**: Round-robin distributes items so thread 0 gets items {0, 8, 16, 24, ...} (1st, 9th, 17th, 25th largest), thread 1 gets {1, 9, 17, 25, ...}, etc. Each thread receives approximately equal count (±1) and a balanced mix of heights. In the Round 0 case, this reduced the max-to-min thread spread from 26ms to 7ms.

### Change 2: Add diagnostic logging (debug level only)

The `tracing::debug!` per-thread logging (included in the code above) allows verifying the load balance improvement in debug builds without any runtime overhead at the default log level. This matches the pattern used in Round 0 (mod.rs:975).

## Invariants

1. **Correctness**: Each AIR is processed exactly once. The `leaves` buffer is pre-zeroed and each AIR writes to its own non-overlapping region via `leaves_ptr` offset. The order of AIR processing does not affect the result.
2. **No APC 0 regression**: The single-thread path (lines 323-327) is unchanged; the round-robin only applies when `num_threads > 1` (i.e., ≥100 AIRs). APC 0 has ~20 AIRs per segment and always uses the single-thread path.
3. **Thread buffer ownership**: Each thread still owns exactly one `GkrThreadBuffers`. The change from `.zip(work_items.chunks(chunk_size))` to `.zip(thread_indices.into_iter())` preserves the 1:1 pairing between thread buffers and threads.
4. **Memory usage**: Unchanged. Same number of pre-allocated buffers, same buffer sizes. No new allocations.
5. **Stream safety**: Each worker thread uses `cudaStreamPerThread` (its own stream). The work_items slice is read-only (`&work_items`). The `SendPtr(leaves_ptr)` writes to non-overlapping regions.

## Measurement Plan

Run the pairing benchmark for APC {0, 100, 300} using:
```bash
cd /home/georg/powdr/results/pairing
PROVE_BIN="/home/georg/powdr/target/release/powdr_openvm_riscv"
$PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics after_apc300.json --recursion
$PROVE_BIN prove --artifact apc100.cbor --input 0 --metrics after_apc100.json --recursion
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics after_apc000.json --recursion
```

Analyze with `spec.py`:
```bash
python3 autoopt/spec.py <metrics_path> <name>
```

**Key metrics to compare (before vs after)**:
- `prover.rap_constraints.logup_gkr.input_evals_time_ms` per segment — direct target
- `prover.rap_constraints.logup_gkr_time_ms` per segment — parent metric
- `stark_prove_excluding_trace_time_ms` per segment — overall STARK target
- LogUp GKR (spec.py) at APC {0, 100, 300}
- STARK excl trace (spec.py) at APC {0, 100, 300}

**Diagnostic-first step**: Before collecting full measurements, run a single APC 300 benchmark with `RUST_LOG=debug` to capture per-thread elapsed times and verify the load balance mechanism is working:
```
gkr_input thread done { thread_id: 0, elapsed_ms: X, num_airs: Y }
```
Compare the max-min thread spread before and after the change. If the spread reduction is negligible (< 5ms), investigate whether the cost model assumption (height correlates with per-AIR time) holds for GKR input eval before proceeding with full measurements.

**Stability**: Run APC 300 benchmark 3 times to verify improvement is consistent and above noise level.

## Expected Impact

At APC 300, current STARK excl trace is 1410ms (1.74x vs 2455ms baseline). GKR input eval is 318ms (seg0=256ms, seg1=62ms). The Round 0 round-robin saved 36ms on a 313ms Round 0 (11.5% relative). A similar relative improvement on GKR input eval would be ~37ms, though GKR has a different work profile (pre-allocated buffers reduce fixed overhead), so a conservative 10-25ms estimate is used. This translates to 0.7-1.8% of STARK excl trace — modest individually, but this optimization is low-risk and additive with future optimizations on other components.

## Rollback Criteria

Revert the optimization if ANY of:
1. **STARK excl trace APC 300 improvement < 8ms** (below measurement noise threshold)
2. **STARK excl trace APC 0 regression > 30ms** (APC 0 should be unchanged since it uses single-thread path)
3. **Any APC configuration fails prove+verify**
4. **GPU OOM at any APC configuration**
5. **LogUp GKR APC 100 regression > 50ms** (APC 100 is borderline for multi-threading threshold and could show regression like the initial Round 0 multi-stream attempt)
