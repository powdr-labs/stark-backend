# Report: 2026-04-15-1600-fix-batch-round0-revert-and-reapply

## Description

The original plan aimed to fix the batch Round 0 descriptor-array optimization (commit 0d1bb2f9) that caused APC 300 STARK proving to hang for 12+ minutes. The plan had 5 steps:

1. Revert the broken batch commit (already done: commit 3ba30eee)
2. Re-apply PRUNE_SHMEM_KERNELS separately (already done: commit d491ff87)
3. Hoist logup rule construction from per-AIR Phase 2 to single-threaded Phase 1
4. Modify `evaluate_round0_interactions_gpu` to accept precomputed rules
5. Re-implement batch path using precomputed rules

Steps 1-2 were already committed before this task. Steps 3-4 were expected to provide 5-15ms improvement at APC 300 by eliminating redundant per-AIR DAG construction and H2D uploads.

## Implementation

Two approaches were attempted:

### Approach 1: Phase 1 Pre-computation (Plan Steps 3-4)

- Added `LogupRulesOnDevice` struct and `build_logup_rules_on_device()` to `round0.rs`
- Added `logup_rules: Option<LogupRulesOnDevice>` field to `Round0AirWorkItem`
- Pre-built logup rules in the Phase 1 work-item construction loop (single-threaded)
- Modified `evaluate_round0_interactions_gpu` to use precomputed rules when available

**Result**: Severe regression. APC 300 Round 0 went from 181ms to 277ms (+96ms). Root cause: Phase 1 is single-threaded, so all 623 AIRs' rule construction + H2D uploads were serialized. In the original code, this work was parallelized across 8 threads in Phase 2.

### Approach 2: Keygen-time Pre-computation (Alternative)

After the Phase 1 approach failed, I implemented a different strategy:

- Extended `AirDataGpu` in `pkey.rs` to store full `LogupRound0Rules` (encoded rules on GPU + interaction-to-rule-index mappings) instead of just `logup_round0_buffer_size`
- Pre-computed rules once at keygen time (during proving key construction)
- Modified `evaluate_round0_interactions_gpu` to use keygen-time rules: only compute per-challenge interaction weights at proving time, skip DAG construction entirely
- Removed `SymbolicConstraints::from()` call from `process_air_round0` (no longer needed for logup path)

Key files changed:
- `crates/cuda-backend/src/pkey.rs`: Added `LogupRound0Rules`, `InteractionRuleMapping` structs; extended `AirDataGpu::new()` to store full rules + mappings
- `crates/cuda-backend/src/logup_zerocheck/round0.rs`: Rewrote `evaluate_round0_interactions_gpu` to use keygen rules
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`: Updated `process_air_round0` and buffer size references

**Result**: Smaller but consistent regression. APC 300 Round 0 went from ~182ms to ~210ms (+28ms).

Step 5 (batch CUDA kernels) was not re-attempted because (a) the prerequisite Steps 3-4 caused regressions, and (b) the previous batch CUDA kernels caused a GPU hang.

## Results

### Approach 2 (Keygen-time rules) — Final State

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 0** | | | | | |
| STARK excl trace (ms) | 2153 | 1812 | 1789 | -364 (1.20x lower) | -23 (1.01x lower) |
| Round 0 (ms) | 178 | 178 | 179 | +1 (1.01x higher) | +1 (1.01x higher) |
| **APC 100** | | | | | |
| STARK excl trace (ms) | 2155 | 1301 | 1302 | -853 (1.66x lower) | +1 (1.00x) |
| Round 0 (ms) | 464 | 167 | 174 | -290 (2.67x lower) | +7 (1.04x higher) |
| **APC 300** | | | | | |
| STARK excl trace (ms) | 2455 | 1111 | 1154 | -1301 (2.13x lower) | +43 (1.04x higher) |
| Round 0 (ms) | 662 | 181 | 210 | -452 (3.15x lower) | +29 (1.16x higher) |

### Approach 1 (Phase 1 pre-computation) — Discarded

| Metric | Before | After (Phase 1) | Change |
|--------|--------|-----------------|--------|
| APC 0 Round 0 (ms) | 178 | 181 | +3 (noise) |
| APC 100 Round 0 (ms) | 167 | 219 | +52 (1.31x higher) |
| APC 300 Round 0 (ms) | 181 | 277 | +96 (1.53x higher) |

## Assessment

**The optimization did NOT achieve its goal.** Both approaches caused performance regressions at APC 100 and APC 300.

The fundamental issue is that the per-AIR CPU-side work (DAG construction, rule encoding) in the multi-threaded Phase 2 provides two benefits that are lost when the work is pre-computed:

1. **Parallelism**: With 8 worker threads, the CPU work runs concurrently. Pre-computing in Phase 1 (single-threaded) serializes it, causing a ~96ms regression at APC 300.

2. **Implicit GPU pacing**: Even with keygen-time pre-computation (which maintains Phase 2 parallelism), removing the CPU work between kernel launches changes the timing of how 8 threads issue GPU kernel launches. The ~0.1ms per-AIR CPU work acts as a natural pace limiter, spreading kernel launches over time. Without it, threads issue kernel launches in tighter bursts, potentially increasing GPU command queue contention. This explains the ~28ms regression at APC 300.

The batch path (Step 5) was not attempted because:
- Steps 3-4 are prerequisites (rules must be precomputed for the batch path)
- The previous batch CUDA kernels (commit 0d1bb2f9) caused a GPU hang at APC 300
- The plan's rollback criteria required stopping after Step 4 if the batch kernels hung

## Future Work

- **Debug the batch CUDA kernel hang**: The most promising path forward. `compute-sanitizer` should be used to check for out-of-bounds access or infinite loops in the batched zerocheck/logup kernels. The hang is likely in the kernel itself (not CPU setup), based on the plan's analysis.

- **Retain CPU pacing while pre-computing**: If pre-computing rules is needed for the batch path, a synthetic delay or throttle between per-AIR kernel launches could compensate for the lost CPU pacing. This is ugly but may be necessary.

- **Profile the GPU command queue**: Use nsight systems to compare kernel launch density (launches/ms) between the original and optimized code paths. If the regression is indeed due to GPU command queue contention, the nsight trace would show more overlapping kernel launches in the optimized version.

- **Parallelize Phase 1 rule construction**: If Phase 1 pre-computation is needed, it could be parallelized using rayon. However, this adds complexity and still doesn't address the GPU pacing issue.

- **Consider the SymbolicConstraints::from() cost**: The `SymbolicConstraints::from()` call per AIR is ~0.1ms. At APC 300 with 623 AIRs across 8 threads, this is ~8ms total. The DAG + rule construction adds another ~0.05ms per AIR. These are small compared to the GPU kernel time and serve as beneficial pacing rather than bottlenecks.
