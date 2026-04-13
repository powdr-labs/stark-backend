# Plan: Round 0 Interleaved Work Balance

## Goal

Reduce Round 0 wall time at APC 300 by replacing contiguous work chunking with greedy longest-processing-time-first (LPT) thread assignment. Currently, the sorted-then-chunked approach puts all large AIRs on thread 0, creating load imbalance. LPT distributes work so that each thread's total estimated cost is approximately equal, improving CPU-GPU overlap during post-processing phases.

The performance hypothesis: with contiguous chunking, thread 0 (holding all large AIRs) is the critical path because it accumulates both the largest GPU kernel waits AND the most CPU post-processing. Even though GPU SM saturation limits kernel concurrency (large kernels monopolize SMs regardless of which thread holds them), the CPU post-processing between kernels on the critical thread adds serial overhead that does not overlap with other threads' GPU work. By distributing large AIRs across threads, each thread's CPU post-processing interleaves with other threads' GPU kernel execution, reducing total wall time.

**Important caveat:** Prior Round 0 optimizations (pipeline-round0-kernel-launches, parallelize-round0-cpu-postprocess) showed that the gap between GPU kernel time and total Round 0 time is "primarily D2H copy latency and kernel launch overhead." The load balancing change addresses a different aspect — it reduces the critical thread's accumulated CPU time between GPU kernel launches, not the per-AIR overhead. The expected improvement is conservative: 15-30ms.

Target: Round 0 from ~295ms to ~265-280ms at APC 300 (15-30ms improvement). STARK excl trace from ~1433ms to ~1403-1418ms.

## Current Code Path

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

The Round 0 flow in `sumcheck_uni_round0_polys` (lines 899-966):

1. **Phase 1 (lines 899-926):** Build `work_items: Vec<Round0AirWorkItem>` from all present AIRs. Pure CPU metadata collection.

2. **Sort (line 929):** `work_items.sort_by(|a, b| b.common_main.height().cmp(&a.common_main.height()))` — sorts by descending trace height.

3. **Threading decision (lines 939-943):** If `work_items.len() >= 100`, use `NUM_ROUND0_STREAMS` (8) threads; else 1 thread.

4. **Chunk and dispatch (lines 950-966):**
   ```rust
   let chunk_size = work_items.len().div_ceil(num_threads);
   work_items.chunks(chunk_size).map(|chunk| {
       s.spawn(move || chunk.iter().map(|w| process_air_round0(w)).collect())
   })
   ```
   With 310 AIRs and 8 threads: chunk_size = 39. Thread 0 gets items 0-38 (largest), thread 7 gets items 273-309 (smallest).

5. **Per-AIR processing** in `process_air_round0` (lines 140-272): For each AIR:
   - H2D copy: main trace pointers (~16-48 bytes)
   - GPU kernel 1: zerocheck constraint evaluation
   - D2H sync + CPU post-process: transpose + iDFT → polynomial
   - GPU kernel 2: logup interaction evaluation (includes DAG construction on CPU)
   - D2H sync + CPU post-process: transpose + iDFT → polynomial pair

## Changes

### Change 0 (diagnostic): Add per-thread timing instrumentation

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, inside the thread spawn

Before implementing the LPT assignment, add `Instant::now()` / `elapsed()` measurements around each thread's work loop. Log per-thread completion times at debug level. This validates the load imbalance hypothesis — confirming whether thread 0 actually finishes much later than thread 7.

```rust
s.spawn(move || -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
    let t0 = std::time::Instant::now();
    let results: Vec<_> = chunk.iter().enumerate().map(|(i, w)| {
        let air_t0 = std::time::Instant::now();
        let result = process_air_round0(w)?;
        let air_ms = air_t0.elapsed().as_millis();
        if i < 5 { // Log top 5 AIRs per thread (largest, since sorted by height)
            tracing::debug!(thread_id, air_idx = i, height = w.common_main.height(), air_ms, "round0 per-air");
        }
        Ok(result)
    }).collect::<Result<_, _>>()?;
    tracing::debug!(thread_id, elapsed_ms = t0.elapsed().as_millis(), num_airs = chunk.len(), "round0 thread done");
    Ok(results)
})
```

This provides both per-thread totals AND per-AIR timing for the largest AIRs, validating whether `height` is a good cost proxy and whether GPU kernel wait or CPU post-processing dominates per-AIR cost.

If all threads finish within ~10ms of each other, abandon the LPT approach (the bottleneck is GPU SM saturation, not thread imbalance). If thread 0 is >30ms slower than the median, proceed with Change 1.

**Alternative to consider:** If the diagnostic shows modest imbalance (20-30ms), a simpler random shuffle of `work_items` before chunking could achieve similar balance with less code than full LPT. Decide after seeing the diagnostic data.

### Change 1: Replace contiguous chunking with greedy LPT assignment

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines 950-960

Use a greedy longest-processing-time-first (LPT) algorithm: iterate sorted work items (largest first) and assign each to the thread with the lowest current total estimated cost. Use `common_main.height()` as the cost proxy (height correlates with kernel runtime).

```rust
// Greedy LPT assignment: assign each work item to the least-loaded thread
let mut thread_loads = vec![0u64; num_threads];
let mut thread_items: Vec<Vec<&Round0AirWorkItem<HS>>> =
    (0..num_threads).map(|_| Vec::new()).collect();
for item in work_items.iter() {
    // Find thread with minimum load
    let min_thread = thread_loads
        .iter()
        .enumerate()
        .min_by_key(|(_, load)| *load)
        .unwrap()
        .0;
    thread_items[min_thread].push(item);
    thread_loads[min_thread] += item.common_main.height() as u64;
}

std::thread::scope(|s| {
    let handles: Vec<_> = thread_items
        .into_iter()
        .enumerate()
        .map(|(thread_id, items)| {
            s.spawn(move || -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
                let t0 = std::time::Instant::now();
                let results: Vec<_> = items.iter().map(|w| process_air_round0(w)).collect::<Result<_, _>>()?;
                tracing::debug!(thread_id, elapsed_ms = t0.elapsed().as_millis(), num_airs = items.len(), "round0 thread done");
                Ok(results)
            })
        })
        .collect();
```

**Why LPT over round-robin:** Round-robin on a sorted list is a reasonable heuristic but does not account for the wide variance in AIR heights. For example, if one AIR has height 2^20 (1M) and the next has 2^19 (512K), round-robin assigns them to threads 0 and 1. But the 2^20 AIR has roughly 2x the work. LPT accounts for this by assigning the 2^19 AIR to a different, less-loaded thread.

**Why `height` as cost proxy:** The constraint evaluation and interaction evaluation kernel runtime scales approximately linearly with trace height (for the coset-parallel kernel, the work is `height × num_cosets × constraint_complexity`). Height is the dominant factor and is available without additional computation.

### Change 2: Concurrent GPU memory analysis

**File:** Same as Change 1.

With LPT, each thread now holds some large AIRs. The per-thread temporary allocation budget is `memory_limit_bytes / NUM_ROUND0_STREAMS` (line 900), which is unchanged. The max allocation per AIR within a thread is bounded by this budget, enforced by `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu`. Since each AIR allocates and frees its temporaries before the next AIR starts (sequential within each thread), the per-thread peak memory does not increase with interleaving.

However, with contiguous chunking, thread 0 finishes large AIR allocations before thread 7 starts its small ones. With LPT, all threads allocate large-AIR temporaries concurrently. The worst case is `8 × (memory_limit_bytes / 8) = memory_limit_bytes`, which is the same total. No change needed, but monitor GPU memory usage in measurements.

## Invariants

1. **Correctness:** Every AIR is processed exactly once. The `trace_idx` field in `Round0AirResult` ensures results map to the correct output positions regardless of processing order.

2. **Per-thread memory budget:** Unchanged at `memory_limit_bytes / NUM_ROUND0_STREAMS`.

3. **APC 0 no-regression:** At APC 0, `work_items.len() < 100` so single-threaded path is taken. LPT change is in the multi-threaded branch only.

4. **Thread safety:** `process_air_round0` is already thread-safe (uses `cudaStreamPerThread`, no shared mutable state). LPT doesn't change the parallelism model.

## Measurement Plan

### Step 1: Diagnostic instrumentation run
Build with per-thread timing (Change 0). Run APC 300 once. Check log output for per-thread elapsed times. If thread imbalance is <20ms, stop and report — the approach cannot achieve meaningful improvement.

### Step 2: LPT implementation + benchmark
If instrumentation confirms >30ms thread imbalance, implement Change 1. Run the full benchmark suite for APC {0, 100, 300}:
```bash
cd /home/georg/powdr/results/pairing
/path/to/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics /tmp/after_apc300.json --recursion
/path/to/powdr_openvm_riscv prove --artifact apc100.cbor --input 0 --metrics /tmp/after_apc100.json --recursion
/path/to/powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics /tmp/after_apc000.json --recursion
```

Analyze with `spec.py`. Run APC 300 3x to confirm stability.

### Step 3: Compare per-thread timing before/after
Compare the per-thread elapsed times from Change 0's instrumentation with and without LPT. Verify that the slowest thread's time decreased.

## Rollback Criteria

- Diagnostic instrumentation shows thread imbalance <20ms → do not proceed with LPT
- Round 0 at APC 300 improves by less than 15ms → revert
- STARK excl trace at APC 300 regresses or improves by less than 15ms → revert
- APC 0 STARK excl trace regresses by more than 20ms → revert
- Any APC configuration fails to prove+verify → revert
