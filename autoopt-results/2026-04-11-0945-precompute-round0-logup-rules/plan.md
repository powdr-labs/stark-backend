# Plan: Pre-compute Round 0 Logup Interaction Rules

## Goal

Eliminate redundant per-AIR DAG construction, rule compilation, and rule encoding/H2D transfers in the Round 0 logup path. At APC 300, ~370 AIRs with interactions per segment (2 segments) each rebuild the same static symbolic data on every `sumcheck_uni_round0_polys` call. This optimization moves all static computation to keygen time (`AirDataGpu::new`), leaving only runtime-dependent weight computation in the hot loop.

Secondary goal: parallelize Phase 2 (D2H + transpose + IDFT) using rayon to reduce sequential CPU processing of 623 AIR results.

Reference state (commit 6fccffce, post-d2h-sync and post-stacked-reduction):
- Round 0: APC 0 = 175ms, APC 300 = 569ms
- STARK excl trace: APC 0 = 2146ms, APC 300 = 2195ms
- Scaling ratio (APC 300 / APC 0): 1.023

Original baseline (pre-optimization): Round 0 APC 300 = 663ms, STARK excl trace APC 300 = 2478ms.

## Prerequisite: Revert Commit 74134b4e

Commit 74134b4e ("batch Round 0 coset-parallel kernel launches") causes the pairing benchmark to hang. Revert it before applying this optimization. The plan targets the code state at 6fccffce. If the batch commit is later fixed and re-applied, the pre-computed rules naturally integrate: the batch path's `LogupBatchMeta.d_rules` would point to the proving key's pre-computed device buffer instead of freshly encoded per-AIR buffers.

## Current Code Path

### Phase 1 per-AIR loop (mod.rs, `sumcheck_uni_round0_polys`, lines ~750-840 at 6fccffce)

For each of the ~623 present AIRs (per segment):

1. **`SymbolicConstraints::from(&single_pk.vk.symbolic_constraints)`** (line ~766): Reconstructs the full `Arc<SymbolicExpression>` tree from the flattened DAG. Allocates a `Vec<Arc<...>>` with one entry per DAG node, clones every constraint and interaction expression. This is marked "TEMPORARY conversion" in the source (dag.rs:384). Used solely to pass `single_air_constraints.interactions` to `evaluate_round0_interactions_gpu`.

2. **`evaluate_round0_constraints_gpu`** (line ~795): Already uses pre-compiled rules from the proving key (`pk.other_data.zerocheck_round0`). No redundant work here.

3. **`evaluate_round0_interactions_gpu`** (line ~812, defined in round0.rs:190-333): Rebuilds the interaction DAG and rules from scratch:
   - **DAG construction** (round0.rs:219-238): Creates `SymbolicDagBuilder`, calls `add_expr` for every interaction count and message field, builds `SymbolicExpressionDag`. This involves FxHashMap lookups, recursive expression traversal, algebraic simplifications, and deduplication.
   - **Rule compilation** (round0.rs:239): `SymbolicRulesGpu::new(&dag, true)` — builds `SymbolicRulesBuilder`, schedules buffer allocation via priority queue, generates three-address code rules, builds `dag_idx_to_rule_idx` map.
   - **Weight computation** (round0.rs:240-264): For each interaction, looks up `dag_builder.expr_to_idx` (pointer-keyed FxHashMap) to get dag_idx, then `rules.dag_idx_to_rule_idx` to get rule_idx. Accumulates into `numer_weights` and `denom_weights` vectors.
   - **Rule encoding + H2D** (round0.rs:267-268): Encodes each rule to `u128` via `Codec::encode()`, copies encoded rules to device.
   - **Buffer allocation + kernel launch** (round0.rs:270-332): Allocates intermediates and temp_sums buffers, launches `logup_bary_eval_interactions_round0` CUDA kernel.

Steps a-d are **static** — they depend only on the AIR's symbolic constraint structure, which is fixed at keygen time. Step e is **dynamic** — it depends on runtime trace data and domain parameters.

### Phase 2 sequential processing (mod.rs, lines ~840-900 at 6fccffce)

After `current_stream_sync()`, processes each AIR sequentially:
1. D2H transfer of `zc_result` and `logup_result` (`DeviceBuffer::to_host()`)
2. Transpose from coset-major to row-major layout
3. `UnivariatePoly::from_geometric_cosets_evals_idft` (small-matrix IDFT)
4. Construct `sp_0` polynomial
5. Store in `batch_sp_poly` at trace-idx-specific indices

Each AIR writes to independent indices: `batch_sp_poly[2*num_present_airs + trace_idx]` for zerocheck, `batch_sp_poly[2*trace_idx]` and `batch_sp_poly[2*trace_idx + 1]` for logup. No data dependencies between iterations.

### Why this is slow at APC 300

APC 300 has ~370 AIRs with interactions (vs ~99 at APC 0). Each `evaluate_round0_interactions_gpu` call includes CPU work (DAG construction, rule compilation, rule encoding, weight computation, H2D transfers) before launching the GPU kernel. At the reference state (6fccffce), Round 0 total is 569ms and nsight GPU kernel time is ~479ms, giving ~90ms of CPU overhead that is not overlapped with GPU execution.

The CPU overhead has three components:
1. **`SymbolicConstraints::from` + DAG construction + rule compilation + rule encoding** (~0.15-0.3ms per AIR): `SymbolicConstraints::from` rebuilds the full `Arc<SymbolicExpression>` tree from the flattened DAG (allocates `Vec<Arc<...>>`, clones every expression). Note: this call runs for ALL 623 AIRs per segment, not just the ~370 with interactions, because the resulting `single_air_constraints` is also used for a `debug_assert_eq!`. After it, `SymbolicDagBuilder::add_expr` (FxHashMap lookups, recursive traversal), `SymbolicRulesGpu::new` (buffer scheduling, three-address code gen), and `Codec::encode` (per-rule encoding) add further static overhead for the ~370 interaction-bearing AIRs. This is the dominant per-AIR CPU cost and is entirely static — depends only on the AIR's constraint structure.
2. **Weight computation** (~0.02-0.05ms per AIR): Simple array accumulation. Already fast, remains after optimization.
3. **Weight H2D transfer** (~0.01-0.02ms per AIR): Two small cudaMemcpy calls. Remains after optimization.

The optimization eliminates component 1, which is the largest. With the d2h-sync optimization (69fc59ca) already in place, the GPU pipeline partially overlaps CPU setup for AIR[i+1] with GPU execution for AIR[i]. However, when per-AIR CPU setup exceeds per-AIR GPU kernel time (which happens for the many small AIRs at APC 300), the GPU sits idle between kernels, creating pipeline bubbles. Eliminating the static CPU work reduces these bubbles and tightens the CPU-GPU pipeline.

**Realistic savings estimate for Part 1**: 40-70ms at APC 300 (reducing Round 0 from 569ms to ~499-529ms). This is bounded by the ~90ms total CPU overhead; the remaining overhead (weight computation + weight H2D + buffer allocation) cannot be eliminated.

**Realistic savings estimate for Part 2 (IDFT parallelization)**: 20-40ms at APC 300. With 623 AIRs doing sequential transpose + IDFT, parallelization on 8+ cores compresses this work.

**Combined estimate**: 60-110ms reduction in STARK excl trace at APC 300.

## Changes

### Change 1: Add `Round0InteractionRules` struct to `pkey.rs`

**File**: `crates/cuda-backend/src/pkey.rs`

Add a new struct and a field to `AirDataGpu`:

```rust
/// Pre-computed Round 0 logup interaction rules.
/// The DAG structure and compiled rules depend only on the AIR's symbolic constraints
/// (fixed at keygen time). Only the weights (which depend on runtime eq_3b and beta_pows)
/// need recomputation at proving time.
pub struct Round0InteractionRules {
    /// Encoded rules on device, compiled with buffer_vars=true for Round 0 kernel.
    pub(crate) d_rules: DeviceBuffer<u128>,
    /// Buffer size for GPU intermediate values.
    pub(crate) buffer_size: u32,
    /// Number of compiled rules. Used to allocate weight vectors at runtime.
    pub(crate) num_rules: usize,
    /// Per-interaction metadata for fast weight computation at runtime.
    pub(crate) weight_map: Round0WeightMap,
}

/// Maps interaction indices to rule indices for runtime weight computation.
/// Replaces the expensive DAG pointer lookups with direct array indexing.
pub struct Round0WeightMap {
    /// count_rule_idxs[i] = rule index for interaction i's count expression.
    pub count_rule_idxs: Vec<usize>,
    /// message_offsets[i]..message_offsets[i+1] indexes into message_rule_idxs.
    /// Length = num_interactions + 1.
    pub message_offsets: Vec<usize>,
    /// Flat array of rule indices for all message fields across all interactions.
    pub message_rule_idxs: Vec<usize>,
    /// bus_indices[i] = bus index for interaction i.
    pub bus_indices: Vec<u16>,
}
```

Add field to `AirDataGpu` (line 24):

```rust
pub struct AirDataGpu {
    pub interaction_rules: InteractionEvalRules,
    pub zerocheck_round0: ConstraintOnlyRules<true>,
    pub zerocheck_mle: ConstraintOnlyRules<false>,
    pub zerocheck_monomials: Option<ZerocheckMonomials>,
    pub interaction_monomials: Option<InteractionMonomials>,
    pub round0_interaction_rules: Option<Round0InteractionRules>,  // NEW
}
```

`Option` because AIRs with no interactions produce `None`.

### Change 2: Build `Round0InteractionRules` in `AirDataGpu::new()` 

**File**: `crates/cuda-backend/src/pkey.rs`, inside `AirDataGpu::new()` (lines 86-115)

Add construction after the existing rule building:

```rust
let round0_interaction_rules = if !symbolic_constraints.interactions.is_empty() {
    Some(Round0InteractionRules::new(&symbolic_constraints)?)
} else {
    None
};
```

The `Round0InteractionRules::new()` implementation follows the same pattern as the current `evaluate_round0_interactions_gpu` lines 219-268, but stores results persistently:

1. Build DAG via `SymbolicDagBuilder` with all interaction count/message expressions
2. Compile rules via `SymbolicRulesGpu::new(&dag, true)`
3. Build the weight map by walking `dag_builder.expr_to_idx` and `rules.dag_idx_to_rule_idx` for each interaction
4. Encode rules and H2D transfer to `d_rules`

This is the SAME code currently in round0.rs:219-268, extracted into a constructor that runs once at keygen rather than per-AIR per-segment.

### Change 3: Add `compute_round0_weights` helper

**File**: `crates/cuda-backend/src/pkey.rs` (or a new helper in `round0.rs`)

Add a method to `Round0InteractionRules` for runtime weight computation:

```rust
impl Round0InteractionRules {
    /// Compute runtime-dependent weights for the Round 0 logup kernel.
    /// Returns (d_numer_weights, d_denom_weights, denom_sum_init).
    pub fn compute_weights(
        &self,
        eq_3bs: &[EF],
        beta_pows: &[EF],
    ) -> Result<(DeviceBuffer<EF>, DeviceBuffer<EF>, EF), MemCopyError> {
        let map = &self.weight_map;
        let mut numer_weights = vec![EF::ZERO; self.num_rules];
        let mut denom_weights = vec![EF::ZERO; self.num_rules];
        let mut denom_sum_init = EF::ZERO;

        for i in 0..map.count_rule_idxs.len() {
            numer_weights[map.count_rule_idxs[i]] += eq_3bs[i];

            let msg_start = map.message_offsets[i];
            let msg_end = map.message_offsets[i + 1];
            let msg_len = msg_end - msg_start;

            denom_sum_init += eq_3bs[i]
                * beta_pows[msg_len]
                * F::from_u32(map.bus_indices[i] as u32 + 1);

            for (j, &rule_idx) in map.message_rule_idxs[msg_start..msg_end].iter().enumerate() {
                denom_weights[rule_idx] += eq_3bs[i] * beta_pows[j];
            }
        }

        let d_numer = numer_weights.to_device()?;
        let d_denom = denom_weights.to_device()?;
        Ok((d_numer, d_denom, denom_sum_init))
    }
}
```

This replaces the weight computation currently duplicated in round0.rs:240-264 and mod.rs:927-948. The key difference: it uses direct array indexing (`count_rule_idxs[i]`, `message_rule_idxs[start..end]`) instead of FxHashMap lookups through `dag_builder.expr_to_idx` and `rules.dag_idx_to_rule_idx`.

### Change 4: Add `launch_round0_logup_kernel` function

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Add a new function that accepts pre-computed rules and weights, performing only the buffer allocation and kernel launch (the dynamic part of the current `evaluate_round0_interactions_gpu`, lines 270-332):

```rust
/// Launch the Round 0 logup kernel with pre-computed rules and weights.
/// This is the runtime-only portion of evaluate_round0_interactions_gpu.
pub fn launch_round0_logup_kernel<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    round0_rules: &Round0InteractionRules,
    d_numer_weights: &DeviceBuffer<EF>,
    d_denom_weights: &DeviceBuffer<EF>,
    denom_sum_init: EF,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    // ... buffer sizing, intermediates allocation, kernel launch
    // Same as current round0.rs:270-332, but using:
    //   round0_rules.d_rules instead of freshly encoded d_rules
    //   round0_rules.buffer_size instead of rules.buffer_size
}
```

The existing `evaluate_round0_interactions_gpu` is kept for now (it still compiles and can serve as a fallback) but is no longer called from the hot path.

### Change 5: Simplify Phase 1 loop in `sumcheck_uni_round0_polys`

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Replace the logup portion of the Phase 1 loop. The key changes:

1. **Remove `SymbolicConstraints::from` call** (line ~766). The only remaining user was the logup path; the zerocheck path already uses `pk.other_data.zerocheck_round0`. The debug assert `single_air_constraints.max_constraint_degree() == local_constraint_deg` can use `single_pk.vk.max_constraint_degree` directly (which is what `local_constraint_deg` is already set from at line ~768).

2. **Replace `evaluate_round0_interactions_gpu` call** (lines ~812-828) with:
   ```rust
   let logup_result = if !eq_3bs.is_empty() {
       if let Some(round0_rules) = &single_pk.other_data.round0_interaction_rules {
           let (d_numer_w, d_denom_w, denom_sum_init) =
               round0_rules.compute_weights(eq_3bs, &self.beta_pows)?;
           launch_round0_logup_kernel(
               single_pk,
               round0_rules,
               &d_numer_w,
               &d_denom_w,
               denom_sum_init,
               selectors_cube.buffer(),
               &d_main_parts,
               public_values,
               eq_xi_tree.get_ptr(n_lift),
               1 << l_skip,
               1 << n_lift,
               height as u32,
               num_cosets_logup as u32,
               omega_root,
               max_temp_bytes,
           )?
       } else {
           DeviceBuffer::new()
       }
   } else {
       DeviceBuffer::new()
   };
   ```

3. **Remove imports** that are no longer needed: `SymbolicConstraints`, `evaluate_round0_interactions_gpu`.

### Change 6: Parallelize Phase 2 processing

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Split Phase 2 into two sub-phases:

**Phase 2a — Sequential D2H transfers:**
```rust
// Phase 2a: D2H transfers (must be sequential — needs CUDA context)
current_stream_sync().map_err(|e| LogupZerocheckError::from(MemCopyError::from(e)))?;
drop(d_main_parts_vec);

struct HostData {
    zc_host: Option<Vec<EF>>,
    logup_host: Option<Vec<Frac<EF>>>,
}
let host_data: Vec<HostData> = pending
    .iter()
    .map(|entry| {
        let zc_host = if !entry.zc_result.is_empty() {
            Some(entry.zc_result.to_host()?)
        } else {
            None
        };
        let logup_host = if !entry.logup_result.is_empty() {
            Some(entry.logup_result.to_host()?)
        } else {
            None
        };
        Ok(HostData { zc_host, logup_host })
    })
    .collect::<Result<Vec<_>, MemCopyError>>()?;
```

**Phase 2b — Parallel transpose + IDFT:**
```rust
// Phase 2b: Parallel transpose + IDFT (no CUDA dependency, independent per AIR)
use rayon::prelude::*;

struct Phase2Result {
    trace_idx: usize,
    zc_poly: Option<UnivariatePoly<EF>>,
    logup_numer: Option<UnivariatePoly<EF>>,
    logup_denom: Option<UnivariatePoly<EF>>,
}

let results: Vec<Phase2Result> = pending
    .into_par_iter()
    .zip(host_data.into_par_iter())
    .map(|(entry, data)| {
        // ... transpose + IDFT logic, identical to current sequential code
        // Returns Phase2Result with the computed polynomials
    })
    .collect();

// Assign results (sequential, fast — just pointer writes)
for result in results {
    if let Some(poly) = result.zc_poly {
        batch_sp_poly[2 * num_present_airs + result.trace_idx] = poly;
    }
    if let Some(poly) = result.logup_numer {
        batch_sp_poly[2 * result.trace_idx] = poly;
    }
    if let Some(poly) = result.logup_denom {
        batch_sp_poly[2 * result.trace_idx + 1] = poly;
    }
}
```

The existing IDFT logic inside the parallel closure is identical to the current sequential loop body (lines ~845-900 at 6fccffce). The only structural change is collecting results into `Phase2Result` structs instead of directly writing to `batch_sp_poly`.

## Invariants

1. **Rule equivalence**: The pre-computed `d_rules` and `buffer_size` must be byte-identical to what `evaluate_round0_interactions_gpu` would produce for the same AIR. Verified by: the constructor uses the same `SymbolicDagBuilder` + `SymbolicRulesGpu::new(&dag, true)` + `Codec::encode()` pipeline.

2. **Weight equivalence**: `compute_weights()` must produce the same `numer_weights`, `denom_weights`, and `denom_sum_init` values as the current code. Verified by: the same accumulation logic, same indexing into `eq_3bs` and `beta_pows`. The only difference is using pre-computed `count_rule_idxs[i]` instead of `dag_builder.expr_to_idx -> rules.dag_idx_to_rule_idx` lookups, which produce the same values.

3. **Kernel output equivalence**: The `logup_bary_eval_interactions_round0` CUDA kernel receives identical inputs (d_rules, buffer_size, d_numer_weights, d_denom_weights, denom_sum_init, trace data, domain params). Its output is unchanged.

4. **Phase 2 output equivalence**: The parallel Phase 2 produces the same `batch_sp_poly` values. Each AIR's transpose + IDFT is independent and deterministic. The `par_iter` order doesn't matter because results are assigned by `trace_idx`.

5. **Memory ownership**: The `Round0InteractionRules.d_rules` DeviceBuffer is owned by the proving key and lives for the entire proving session. No keepalive vectors needed. The runtime-allocated `d_numer_weights` and `d_denom_weights` are dropped after each kernel launch completes (they are consumed within the per-AIR scope, and the kernel is launched before the scope exits).

6. **No CUDA kernel changes**: The `logup_bary_eval_interactions_round0` kernel and its batched variant are unchanged.

## Measurement Plan

Run the pairing benchmark at APC {0, 100, 300} before and after:

```bash
# In the powdr repo:
openvm-riscv/scripts/run_pairing.sh
```

Collect `metrics.json` files, then analyze with `spec.py`:
```bash
python spec.py <path>/metrics.json <experiment_name>
```

**Metrics to compare:**
- Round 0 time (APC 0, 100, 300)
- STARK excl trace (APC 0, 100, 300)
- APC 300 / APC 0 ratio for STARK excl trace

**Before measurements** should match the reference state (commit 6fccffce, after reverting 74134b4e):
- Round 0: APC 0 ≈ 175ms, APC 300 ≈ 569ms
- STARK excl trace: APC 0 ≈ 2146ms, APC 300 ≈ 2195ms
- APC 300/APC 0 scaling ratio ≈ 1.023

**Success criteria** (from task.md):
- Primary: STARK excl trace APC 300 decreases by ≥100ms (2195ms → ≤2095ms)
- Secondary: Round 0 APC 300 decreases by ≥80ms (569ms → ≤489ms)

Note: the task's primary and secondary criteria are at the high end of the plan's estimated range (60-110ms combined, 40-70ms Round 0 Part 1). Meeting both requires Part 1 and Part 2 to land at or above midpoint. The rollback criteria below define the minimum bar for keeping the optimization.

**Correctness**: Run the full cuda-backend test suite:
```bash
cargo nextest run -p openvm-cuda-backend --test-threads=4
```

All 94 tests must pass. The pairing benchmark itself verifies proof correctness (verification is part of the benchmark flow).

## Rollback Criteria

- **Revert if** Round 0 APC 300 does not decrease by at least 30ms (remains above 539ms). This is the minimum expected from eliminating the static CPU work.
- **Revert if** STARK excl trace APC 0 increases by more than 20ms (above 2166ms)
- **Revert if** any cuda-backend test fails
- **Revert if** keygen time increases by more than 500ms (pre-computation should be <100ms per AIR, ~60ms total for 623 AIRs — but verify)
- **Scope out Phase 2 (Change 6) if** Part 1 alone meets the success criteria or if the D2H transfer overhead dominates Phase 2 time (leaving little for IDFT parallelization to compress)
