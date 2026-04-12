# Report: Pre-compute Round 0 Logup Interaction Rules

## Description

The optimization aimed to eliminate redundant per-AIR CPU work in the Round 0 logup path by pre-computing static symbolic data at keygen time. At APC 300, ~370 AIRs with interactions per segment each rebuild a symbolic expression DAG, compile GPU evaluation rules, encode them, and H2D-transfer them — all of which depend only on the AIR's constraint structure (fixed at keygen time). The optimization moves this static work into `AirDataGpu::new()`, storing the compiled rules in the proving key for reuse.

A secondary optimization parallelizes Phase 2 (D2H + transpose + IDFT) processing of 623 AIR results using rayon, since each AIR writes to independent `batch_sp_poly` indices.

The hypothesis was that removing ~0.15-0.3ms of CPU setup per AIR (×370 AIRs ×2 segments) would reduce CPU-GPU pipeline bubbles and save 40-70ms on Round 0 at APC 300.

## Implementation

### Part 1: Pre-compute Round 0 logup rules (Changes 1-5)

**File: `crates/cuda-backend/src/pkey.rs`**
- Added `Round0InteractionRules` struct containing: pre-encoded rules on device (`d_rules`), `buffer_size`, `num_rules`, and `Round0WeightMap` (pre-computed mapping from interaction indices to rule indices for fast weight computation).
- Added `round0_interaction_rules: Option<Round0InteractionRules>` field to `AirDataGpu`.
- `Round0InteractionRules::new()` — builds the interaction-only DAG via `SymbolicDagBuilder`, compiles rules via `SymbolicRulesGpu::new(&dag, true)`, pre-computes the weight map (count_rule_idxs, message_offsets, message_rule_idxs, bus_indices), encodes rules, and H2D-transfers them. Runs once at keygen.
- `Round0InteractionRules::compute_weights()` — computes runtime-dependent `numer_weights`, `denom_weights`, `denom_sum_init` using direct array indexing instead of FxHashMap lookups.

**File: `crates/cuda-backend/src/logup_zerocheck/round0.rs`**
- Added `launch_round0_logup_kernel()` — accepts pre-computed rules + weights, performs only buffer allocation and kernel launch (the dynamic portion of the old `evaluate_round0_interactions_gpu`).

**File: `crates/cuda-backend/src/logup_zerocheck/mod.rs`**
- Removed `SymbolicConstraints::from(&single_pk.vk.symbolic_constraints)` call from Phase 1 loop (eliminated per-AIR full expression tree reconstruction).
- Replaced `evaluate_round0_interactions_gpu()` call with pre-computed rules path: `round0_rules.compute_weights()` → `launch_round0_logup_kernel()`.

### Part 2: Parallelize Phase 2 IDFT (Change 6)

**File: `crates/cuda-backend/src/logup_zerocheck/mod.rs`**
- Split Phase 2 into:
  - Phase 2a: Sequential D2H transfers (needs CUDA context)
  - Phase 2b: Parallel transpose + IDFT using `p3_maybe_rayon::prelude::par_iter()`
- Collected results into `Phase2Result` structs, then assigned back to `batch_sp_poly` sequentially.

### Deviations from plan
- No deviations. All 6 changes implemented as planned.

## Results

### Key Metrics

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **Round 0 APC 0** | 178ms | 173ms | 176ms | -2ms (1.01x lower) | +3ms (1.02x higher) |
| **Round 0 APC 100** | 463ms | 380ms | 380ms | -83ms (1.22x lower) | 0ms (1.00x) |
| **Round 0 APC 300** | 663ms | 570ms | 568ms | -95ms (1.17x lower) | -2ms (1.00x) |
| **STARK excl trace APC 0** | 2149ms | 2136ms | 2143ms | -6ms (1.00x) | +7ms (1.00x) |
| **STARK excl trace APC 100** | 2159ms | 2007ms | 2006ms | -153ms (1.08x lower) | -1ms (1.00x) |
| **STARK excl trace APC 300** | 2478ms | 2216ms | 2216ms | -262ms (1.12x lower) | 0ms (1.00x) |
| **APC300/APC0 scaling ratio** | 1.153 | 1.037 | 1.034 | — | — |

### Sub-component Breakdown (APC 300)

| Metric | Before Task | After Task | Change |
|--------|-------------|------------|--------|
| Constraints | 1545ms | 1553ms | +8ms |
| LogUp GKR | 789ms | 804ms | +15ms |
| Round 0 | 570ms | 568ms | -2ms |
| MLE Rounds | 183ms | 180ms | -3ms |
| Openings | 251ms | 249ms | -2ms |
| Trace Commit | 417ms | 411ms | -6ms |

All changes are within measurement noise (±10-15ms).

### Correctness

All 94 `openvm-cuda-backend` tests pass. Proof verification succeeded for all three APC configurations.

## Assessment

**The optimization did NOT achieve its goal.** Round 0 APC 300 decreased by only ~2ms (within noise), far below the ≥30ms minimum rollback threshold and the ≥80ms secondary success criterion. STARK excl trace APC 300 showed 0ms change, missing the ≥100ms primary criterion entirely.

**Root cause of failure:** The CPU work eliminated (DAG construction, rule compilation, rule encoding, H2D transfers) was already fully pipelined with GPU execution. In the Phase 1 loop, CUDA kernel launches are asynchronous — the host returns immediately after launch and can prepare the next AIR while the GPU executes the previous kernel. Since GPU kernel time per AIR (0.2-1.3ms for large domain AIRs) exceeds the CPU setup time after optimization (~0.05ms for weights), the GPU was never idle waiting for CPU work. The pipeline was GPU-bound, not CPU-bound.

The Phase 2 IDFT parallelization also showed no measurable improvement, suggesting Phase 2 CPU work was already fast (likely <10ms total for the small per-AIR IDFT matrices).

**The optimization was reverted** per rollback criteria (Round 0 APC 300 at 568ms > 539ms threshold).

## Future Work

### What was learned
1. **CPU-GPU pipelining is effective**: The existing Round 0 Phase 1 loop already achieves good GPU utilization because CUDA kernel launches are asynchronous. Reducing per-AIR CPU setup from ~0.3ms to ~0.05ms doesn't help when the GPU pipeline absorbs it.

2. **The 90ms "CPU overhead gap"** (total Round 0 time minus GPU kernel time) is not actually CPU compute overhead that can be eliminated. It likely includes:
   - CUDA runtime overhead for kernel dispatch and stream management
   - Memory allocation latency for device buffers (intermediates, temp_sums)
   - Pipeline drain at the end of Phase 1 (last kernel still executing when Phase 2 starts)

3. **Phase 2 IDFT is not a bottleneck**: The transpose + IDFT for small matrices (constraint_degree × skip_domain elements) is very fast, totaling <10ms even for 623 AIRs.

### More promising directions
1. **Reduce GPU kernel time directly**: The Round 0 bottleneck at APC 300 is ~480ms of GPU kernel time across ~370 AIRs per segment. Batch multiple small-AIR kernels into single launches (as the reverted batch-round0 commit attempted) would address the actual bottleneck — but needs debugging for the pairing benchmark hang.

2. **Investigate the batch-round0 regression**: Commit 74134b4e's approach of batching coset-parallel kernel launches was targeting the right bottleneck (many small kernel launches). The hang may be due to memory management issues or a buffer sizing bug.

3. **Profile GKR input evaluation**: At APC 300, LogUp GKR is ~800ms — the single largest sub-component. The 003-stacked-reduction-opt directory contains nsight data showing 5x GPU time reduction is possible through batching, but the CPU overhead offset is unexplained.

4. **Reduce per-AIR buffer allocation overhead**: Each Round 0 kernel launch allocates intermediates and temp_sums buffers via `DeviceBuffer::with_capacity`. Pre-allocating reusable buffers could reduce CUDA malloc overhead, which may be part of the 90ms gap.
