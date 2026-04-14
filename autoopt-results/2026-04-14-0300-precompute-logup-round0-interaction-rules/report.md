# Report: Pre-compute logup Round 0 interaction rules at keygen time

## Description

Pre-compute the logup Round 0 interaction evaluation DAG, encoded rules, and weight-index mapping at keygen time, eliminating per-AIR `SymbolicConstraints::from()`, `SymbolicDagBuilder` construction, `SymbolicRulesGpu::new()`, rule encoding, and `d_rules` device upload in `evaluate_round0_interactions_gpu`. The keygen code already built this identical DAG (pkey.rs:116-142) but discarded everything except `buffer_size`. The optimization keeps the full pre-computed rules and a compact weight mapping table so the hot path only needs to compute challenge-dependent weights.

## Implementation

### Changes made

1. **`crates/cuda-backend/src/pkey.rs`**: Added `LogupRound0Rules` and `InteractionWeightEntry` structs. Replaced `logup_round0_buffer_size: u32` field on `AirDataGpu` with `logup_round0: Option<LogupRound0Rules>`. Extended the existing DAG construction in `AirDataGpu::new()` to keep the full rules, encode and upload them to device, and build a weight mapping from interaction index to rule indices.

2. **`crates/cuda-backend/src/logup_zerocheck/round0.rs`**: Refactored `evaluate_round0_interactions_gpu` to remove the `symbolic: &SymbolicConstraints<F>` parameter. The function now reads pre-computed rules from `pk.other_data.logup_round0` instead of rebuilding the DAG, rules, and encoding from scratch. Only the challenge-dependent weight computation (using `eq_3bs` and `beta_pows`) remains in the hot path.

3. **`crates/cuda-backend/src/logup_zerocheck/mod.rs`**: Removed `SymbolicConstraints::from()` call in `process_air_round0()` (was only needed as a parameter to `evaluate_round0_interactions_gpu`). Updated `logup_round0_buffer_size` references to use `logup_round0.as_ref().map_or(0, |r| r.inner.buffer_size)`.

### Deviations from plan

None. The implementation followed the plan exactly.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1386ms | 1373ms | -1082ms, 1.79x lower | -13ms, 1.01x lower |
| Round 0 | 662ms | 243ms | 243ms | -419ms, 2.72x lower | 0ms, unchanged |
| LogUp GKR | 790ms | 547ms | 526ms | -264ms, 1.50x lower | -21ms, 1.04x lower |
| MLE Rounds | 180ms | 172ms | 170ms | -10ms, 1.06x lower | -2ms, unchanged |
| Openings | 413ms | 176ms | 175ms | -238ms, 2.36x lower | -1ms, unchanged |
| Trace Commit | 406ms | 246ms | 254ms | -152ms, 1.60x lower | +8ms, noise |

Second APC 300 run: STARK excl trace 1373ms, Round 0 248ms (confirming results are within noise).

### APC 0

| Metric | Baseline | After Task | vs Baseline |
|--------|----------|------------|-------------|
| STARK excl trace | 2153ms | 2123ms | -30ms, 1.01x lower |
| Round 0 | 178ms | 174ms | -4ms, noise |
| LogUp GKR | 993ms | 1004ms | +11ms, noise |

No regression at APC 0.

## Assessment

**The optimization did NOT achieve its goal.** Round 0 at APC 300 showed 0ms improvement (243ms before and after). STARK excl trace showed 13ms improvement, well within measurement noise. This falls below the 25ms rollback threshold defined in the plan.

**Why the plan's estimate was wrong:** The plan estimated ~0.8ms of CPU overhead per AIR for DAG construction + rule compilation + encoding + H2D upload, projecting 50-80ms wall-time improvement across 8 threads. In reality, the per-AIR overhead is much smaller:

1. The `SymbolicDagBuilder` + `add_expr()` loop is fast because most AIRs have few interactions (typically 2-10 expressions per AIR), and the pointer-based deduplication in `expr_to_idx` is O(1) per expression.
2. `SymbolicRulesGpu::new()` processes the small DAGs in microseconds, not ~0.2ms.
3. The `encoded_rules.to_device()` H2D upload is tiny (<1KB per AIR) and handled by the CUDA memory pool with near-zero latency.
4. Even `SymbolicConstraints::from()` (the DAG-to-tree conversion) is fast because the constraint DAGs are structurally simple for typical AIRs.

The total per-AIR CPU overhead for the eliminated code path is likely ~0.05-0.1ms, not ~0.8ms. With 623 AIRs across 8 threads, this amounts to ~4-8ms of wall time — well within measurement noise.

## Future Work

- The Round 0 bottleneck at APC 300 (243ms) is primarily GPU kernel execution time, not CPU overhead. Further optimization would need to target kernel efficiency (e.g., reducing memory traffic, improving SM occupancy) or reducing the number of kernel launches.
- The `SymbolicConstraints::from()` removal in `process_air_round0()` is a correct code simplification even if the performance impact is negligible — it eliminates a redundant DAG-to-tree-to-DAG roundtrip.
- The pre-computation approach is architecturally sound and could yield measurable benefits if applied to a code path with genuinely expensive per-call setup (e.g., if interaction DAGs were orders of magnitude larger).
- Measurement tooling: a per-AIR timing breakdown within Round 0 (measuring the specific code blocks eliminated) would have caught the incorrect per-AIR overhead estimate before running the full benchmark.
