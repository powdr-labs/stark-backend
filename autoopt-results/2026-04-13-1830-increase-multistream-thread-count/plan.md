# Plan: Increase Multi-Stream Thread Count from 4 to 8

## Goal

Increase the number of OS threads (each with a per-thread CUDA stream via `cudaStreamPerThread`) from 4 to 8 for both Round 0 constraint evaluation and GKR input evaluation. At APC 300, ~310 AIRs per segment launch tiny kernels (1-4 SMs each) on the RTX 4090's 128 SMs. Nsys profiling shows 4 threads achieve only 1.7-2.1x effective parallelism (kernel time / wall time), leaving >90% of SMs idle. Doubling to 8 threads improves concurrent kernel occupancy, reduces chunk size for better load balance, and provides more overlap between GPU execution and per-thread CPU work.

## Current Code Path

### Round 0 (`sumcheck_uni_round0_polys`)

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

- **Line 93:** `const NUM_ROUND0_STREAMS: usize = 4;`
- **Line 898:** `per_thread_temp_bytes = self.memory_limit_bytes / NUM_ROUND0_STREAMS.max(1);` — divides memory budget across threads
- **Line 937-941:** Thread count selection: `if work_items.len() >= 100 { NUM_ROUND0_STREAMS.min(work_items.len()) } else { 1 }`
- **Line 948:** `chunk_size = work_items.len().div_ceil(num_threads);` — divides AIRs into chunks
- **Lines 949-963:** `std::thread::scope` spawns worker threads, each calling `process_air_round0()` per AIR

**Per-thread work** (`process_air_round0`, same file ~lines 1000-1100):
Each thread iterates its chunk of AIRs. Per AIR: launches constraint + interaction evaluation GPU kernels, performs D2H copy (`to_host_on_current_stream`), then CPU post-processing (transpose, iDFT, polynomial construction).

**Current timing (APC 300):** Seg0=174ms, Seg1=176ms, total=350ms

### GKR Input Eval (`log_gkr_input_evals`)

**File:** `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

- **Line 30:** `const NUM_GKR_INPUT_STREAMS: usize = 4;`
- **Line 258-262:** Thread count selection: same `>= 100` threshold
- **Lines 271-291:** `std::thread::scope` spawns workers, each calling `process_gkr_input_air()` per AIR

**Per-thread work** (`process_gkr_input_air`, lines 105-209):
Per AIR: launches `logup_gkr_input_eval()` CUDA kernel, handles height lifting, D2H copy via `to_host_on_current_stream`.

**Current timing (APC 300):** Seg0=266ms (includes 167ms stream-sync overhead from first-segment allocation), Seg1=72ms. Kernel wall spans: Seg0=99ms, Seg1=64ms.

### Why current parallelism is low

Nsys data shows effective parallelism with 4 threads:
- Round 0: ~1.7x (350ms wall, ~600ms total kernel time across both segments)
- GKR input eval Seg1: 3.4x (64ms wall, 214ms kernel time)
- GKR input eval Seg0: 2.1x (99ms wall, 209ms kernel time)

Root causes:
1. **Chunk imbalance:** With 4 threads processing ~78 AIRs each (sorted by descending height), the first chunk has all the largest AIRs. The thread processing the tallest AIRs takes longest, creating a long tail.
2. **Memory pool mutex:** Every `DeviceBuffer::with_capacity` call in `process_air_round0` and `process_gkr_input_air` acquires the global `MemoryManager` mutex. With 4 threads doing ~10 allocations per AIR, that's ~3,100 lock acquisitions per segment for Round 0.
3. **GPU underutilization:** At any instant, only 4 kernels are in-flight. Most kernels use 1-4 SMs, so 4-16 of 128 SMs are active (3-12%).

## Changes

### Change 1: Increase `NUM_ROUND0_STREAMS` from 4 to 8

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, line 93

**What:** Change `const NUM_ROUND0_STREAMS: usize = 4;` to `const NUM_ROUND0_STREAMS: usize = 8;`

**Why:** With 8 threads, each processes ~39 AIRs instead of ~78. This reduces chunk imbalance (the tallest-AIR thread finishes sooner relative to others) and doubles the number of concurrent GPU kernels (8-32 SMs active vs 4-16).

### Change 2: Increase `NUM_GKR_INPUT_STREAMS` from 4 to 8

**File:** `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`, line 30

**What:** Change `const NUM_GKR_INPUT_STREAMS: usize = 4;` to `const NUM_GKR_INPUT_STREAMS: usize = 8;`

**Why:** Same reasoning as Change 1. GKR input eval kernels average ~540μs each, small enough to benefit from more concurrent streams.

### Change 3: (No code change needed) Memory budget division

The existing code at line 898 already divides memory budget by `NUM_ROUND0_STREAMS`:
```rust
let per_thread_temp_bytes = self.memory_limit_bytes / NUM_ROUND0_STREAMS.max(1);
```

With 8 threads, each gets `memory_limit_bytes / 8` instead of `/4`. For app proofs (`log_blowup == 1`), `save_memory` is true (`crates/cuda-backend/src/device.rs`, line 32), so `memory_limit_bytes = gkr_mem_contribution` — the size of the GKR leaves buffer. At APC 300, `gkr_mem_contribution = total_leaves * sizeof(Frac<EF>)` ≈ 8-9 GiB (with total_leaves ≈ 2^28), giving each of 8 threads ~1 GiB. This is more than sufficient for the intermediate buffers needed per-AIR (typically <100 MiB for the largest AIRs at APC 300, where heights are ≤16K).

For GKR input eval, there is no explicit memory budget division — each thread allocates per-AIR buffers independently. The per-AIR allocation size is small (a few MB), so 8 threads won't approach GPU memory limits.

### Note on MemoryManager mutex contention

The global `MemoryManager` mutex (`crates/cuda-common/src/memory_manager/mod.rs`) is acquired for every `DeviceBuffer::with_capacity` and free operation. With 8 threads instead of 4, contention pressure increases. The total number of lock acquisitions remains ~3,100 per segment for Round 0 (same work, more threads), but serialization waiting time per acquisition increases with more contenders.

This is unlikely to be a blocker because mutex hold times (allocation lookup + cudaMallocAsync call ≈ 5-10μs) are small relative to per-AIR kernel execution times (100-2000μs). Even with doubled contention, total serialization adds at most 5-10ms per segment. If profiling shows contention is limiting, per-thread pre-allocated reusable buffer pools would be the follow-up optimization.

### No changes needed for threshold or fallback path

The existing `work_items.len() >= 100` threshold remains appropriate:
- APC 0: ~20 AIRs per segment → sequential path (unchanged)
- APC 100: ~120 AIRs per segment → 8 threads, 15 AIRs each (sufficient)
- APC 300: ~310 AIRs per segment → 8 threads, 39 AIRs each (optimal)

## Invariants

1. **Correctness:** Each thread writes to non-overlapping regions (Round 0: different `batch_sp_poly` entries via `trace_idx`; GKR input: different `leaves` offsets via `SendPtr`). Increasing thread count does not change the write pattern.
2. **APC 0 no regression:** The `>= 100` threshold means APC 0 (~20 AIRs/segment) always uses the sequential path, completely unaffected by the thread count constant.
3. **Memory safety:** Per-thread temp budget decreases from ~2 GiB to ~1 GiB (from `gkr_mem_contribution / 8`). The largest AIR at APC 300 (height ~16K, ~100 columns) needs ~50 MiB of intermediates. 1 GiB is sufficient with >20x headroom.
4. **Stream isolation:** Each OS thread gets its own `cudaStreamPerThread`. Kernels on different streams are independent. The `current_stream_sync()` barrier before thread spawning ensures all prior GPU work is visible.
5. **Result ordering:** Round 0 results are identified by `trace_idx` and scattered in Phase 3. GKR input results are accumulated atomically into the leaves buffer. Neither depends on thread ordering.

## Measurement Plan

Run the standard benchmark suite:

```bash
cd /home/georg/powdr && bash openvm-riscv/scripts/run_pairing.sh
```

Then analyze with:
```bash
python3 /home/georg/spec.py results/pairing/apc300/metrics.json after_apc300
python3 /home/georg/spec.py results/pairing/apc100/metrics.json after_apc100
python3 /home/georg/spec.py results/pairing/apc000/metrics.json after_apc000
```

### Success criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Round 0 (APC 300) | < 300ms (>14% reduction from 350ms) | `spec.py` Round 0 line |
| LogUp GKR (APC 300) | < 530ms (>5.7% reduction from 562ms) | `spec.py` LogUp GKR line |
| STARK excl trace (APC 300) | < 1500ms (>4.8% reduction from 1576ms) | `spec.py` STARK excl trace line |
| STARK excl trace (APC 0) | < 2250ms (no regression) | `spec.py` STARK excl trace line |
| APC 100 Round 0 | < 250ms (improvement from 273ms) | `spec.py` Round 0 line |

### Per-segment validation

Extract per-segment metrics from `metrics.json`:
```python
# Check per-segment Round 0 and input_eval timing
prover.rap_constraints.round0_time_ms (seg 0 and seg 1)
prover.rap_constraints.logup_gkr.input_evals_time_ms (seg 0 and seg 1)
```

Verify both segments show improvement (not just one).

### Nsys validation

Run nsys profiling for APC 300 and check:
1. Kernel concurrency: >8 concurrent kernel instances visible in the timeline during Round 0 and GKR input eval
2. Stream count: >8 distinct `streamId` values during the multi-threaded phases
3. Effective parallelism: total kernel time / wall span should be >3x (up from ~2x)

## Rollback Criteria

Revert if ANY of the following:
- STARK excl trace at APC 300 improves less than 4% (<60ms reduction)
- STARK excl trace at APC 0 regresses by more than 3% (>65ms increase)
- Any correctness failure (prove+verify fails)
- GPU OOM error at any APC configuration
- APC 100 shows more than 5% regression in STARK excl trace
