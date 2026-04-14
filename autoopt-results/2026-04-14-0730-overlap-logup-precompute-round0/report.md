# Report: Overlap Logup Combination Precompute with Round 0

## Description

The logup combination precomputation (d_eq_3b H2D uploads + `precompute_logup_combinations` kernel launches) ran sequentially on the default stream before Round 0 started, even though its output is only consumed later during MLE rounds. Round 0 uses only host-side eq_3b values for weight computation — the device-side precompute is entirely independent. By spawning the precompute on a background thread concurrent with Round 0's multi-stream evaluation (which runs at ~42% GPU utilization), the ~22ms per-segment precompute work was expected to be fully hidden behind Round 0's ~55ms GPU evaluation time.

## Implementation

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, function `sumcheck_uni_round0_polys`

**Changes**:

1. Removed the d_eq_3b upload (lines 945-955) and logup precompute loop (lines 957-975) from their original sequential position between eq_3b CPU computation and eq_xis/sels computation.

2. For the **single-threaded path** (num_threads <= 1, i.e., APC 0 with <100 AIRs): placed the d_eq_3b upload + logup precompute sequentially before the Round 0 loop, preserving the original behavior with no change.

3. For the **multi-threaded path** (num_threads > 1, i.e., APC 100/300 with >=100 AIRs): added a background thread inside the existing `std::thread::scope` block that runs concurrently with the Round 0 worker threads:
   - Background thread: uploads d_eq_3b per-trace, computes logup combinations, syncs its CUDA stream, and returns owned `(Vec<DeviceBuffer<EF>>, Vec<Option<LogupCombinations>>)`.
   - Round 0 workers: unchanged behavior.
   - After the scope exits, the returned precompute results are stored into `self.d_eq_3b_per_trace` and `self.logup_combinations`.

4. Captured shared references (`pk_ref`, `eq_3b_ref`, `d_beta_pows_ref`, `beta_pows_ref`, `per_trace_ref`) before the scope for the background thread to borrow. All are immutable borrows coexisting with Round 0 work item borrows.

**Deviations from plan**: None. The implementation followed the plan exactly.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1370ms | 1306ms | -1149ms, 1.88x lower | -64ms, 1.05x lower |
| Constraints | 1634ms | 946ms | 881ms | -753ms, 1.85x lower | -65ms, 1.07x lower |
| Round 0 | 662ms | 203ms | 182ms | -480ms, 3.64x lower | -21ms, 1.12x lower |
| LogUp GKR | 790ms | 570ms | 526ms | -264ms, 1.50x lower | -44ms, 1.08x lower |
| MLE Rounds | 180ms | 171ms | 171ms | -9ms, 1.05x lower | 0ms, unchanged |
| Openings | 413ms | 174ms | 177ms | -236ms, 2.33x lower | +3ms, noise |
| Trace Commit | 406ms | 247ms | 247ms | -159ms, 1.64x lower | 0ms, unchanged |

### APC 000

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2140ms | 2150ms | -3ms, unchanged | +10ms, noise |
| Round 0 | 178ms | 176ms | 176ms | -2ms, unchanged | 0ms, unchanged |
| LogUp GKR | 993ms | 1017ms | 1025ms | +32ms, noise | +8ms, noise |

### Per-segment Round 0 (APC 300, app_proof segments only)

| Segment | Before | After | Change |
|---------|--------|-------|--------|
| seg 0 | 93ms | 82ms | -11ms |
| seg 1 | 93ms | 83ms | -10ms |

Second confirmation run: seg 0 = 83ms, seg 1 = 81ms (consistent).

## Assessment

**Success**: The optimization achieved its goal of overlapping logup precompute with Round 0.

**Round 0 improvement**: 203ms -> 182ms at APC 300 (-21ms, 10.3%). The per-segment improvement was ~10-11ms rather than the planned ~22ms. The likely reason is that while GPU utilization during Round 0 was ~42%, the concurrent precompute allocations introduce some memory pool mutex contention with the Round 0 workers, reducing the effective overlap from ~22ms to ~10ms.

**STARK excl trace improvement**: 1370ms -> 1306ms at APC 300 (-64ms, 4.7%). This is larger than the Round 0 improvement alone because LogUp GKR also improved by 44ms (570ms -> 526ms). This secondary improvement is likely due to the precompute allocations now overlapping with Round 0 rather than being freed/reallocated sequentially before it, leaving the CUDA memory pool in a more favorable state for the subsequent GKR phase.

**No APC 0 regression**: STARK excl trace at APC 000 changed by +10ms (2140ms -> 2150ms), within measurement noise. The single-threaded path is unchanged.

**Cumulative progress**: STARK excl trace at APC 300 is now 1.88x lower vs baseline (2455ms -> 1306ms).

**Complexity**: Minimal. The change adds one background thread spawn inside an existing `thread::scope` block with clean ownership transfer. No new data structures, no new CUDA kernels, no FFI changes.

## Future Work

- **What worked well**: The background thread pattern with `thread::scope` provides a clean way to overlap independent GPU work with existing multi-stream evaluation. The ownership model (returning owned values, storing after scope) avoids borrow checker issues.

- **Further overlap opportunities**: The same pattern could be applied to other phases that do preparatory GPU work before a parallel evaluation phase. For example, any upload or precompute that's consumed by a later phase (not the immediately following parallel phase) could be overlapped.

- **Memory pool optimization**: The unexpected LogUp GKR improvement (+44ms) suggests that allocation ordering and pool state significantly affect subsequent phases. Investigating memory pool fragmentation patterns across phases could reveal further optimization opportunities.

- **Precompute on dedicated stream**: Instead of using `cudaStreamPerThread` (which is unique per OS thread), using an explicitly created CUDA stream for the precompute work could allow more fine-grained control over concurrency and prioritization.
