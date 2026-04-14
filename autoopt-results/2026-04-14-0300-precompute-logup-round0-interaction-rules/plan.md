# Plan: Pre-compute logup Round 0 interaction rules at keygen time

## Goal

Eliminate per-AIR CPU overhead in `evaluate_round0_interactions_gpu` by pre-computing the logup interaction DAG, encoded rules, and weight-index mapping at keygen time. This mirrors the existing pattern where zerocheck constraint rules are pre-computed as `pk.other_data.zerocheck_round0`.

Currently, each call to `evaluate_round0_interactions_gpu` (called per AIR per segment — 1246 times at APC 300) rebuilds:
1. `SymbolicConstraints::from()` — converts DAG back to expression trees (~0.3-0.5ms)
2. `SymbolicDagBuilder` + `SymbolicExpressionDag` construction (~0.2-0.3ms)
3. `SymbolicRulesGpu::new()` + rule encoding (~0.1-0.2ms)
4. `d_rules.to_device()` H2D upload (~0.05ms)

The keygen code at `pkey.rs:116-142` already builds this identical DAG but discards everything except `buffer_size`. We extend it to keep the full rules and a weight mapping.

## Current Code Path

### Keygen (pkey.rs:89-152)

`AirDataGpu::new()` calls:
1. `SymbolicConstraints::from(dag)` — converts `SymbolicConstraintsDag` to `SymbolicConstraints` (line 94)
2. `InteractionEvalRules::new()` — builds GKR input eval rules (line 95)
3. `ConstraintOnlyRules::<true>::new()` — builds zerocheck Round 0 rules (line 96)
4. `ConstraintOnlyRules::<false>::new()` — builds zerocheck MLE rules (line 97)
5. Lines 116-142: Builds the logup round0 interaction DAG (same DAG as `evaluate_round0_interactions_gpu`), extracts `rules.buffer_size`, discards the rest

### Round 0 hot path (logup_zerocheck/mod.rs:152-319)

`process_air_round0()` calls:
1. Line 157: `SymbolicConstraints::from(&single_pk.vk.symbolic_constraints)` — **redundant DAG→tree→DAG roundtrip**
2. Line 175: `main_parts.to_device()` — H2D of pointer array (~5 pointers)
3. Lines 197-212: `evaluate_round0_constraints_gpu()` — uses pre-computed `pk.other_data.zerocheck_round0` (already optimized)
4. Lines 262-279: `evaluate_round0_interactions_gpu()` — receives `&single_air_constraints`, rebuilds DAG from scratch

### evaluate_round0_interactions_gpu (round0.rs:162-320)

Lines 193-239 (per-AIR overhead, inside block scope):
1. Lines 194-211: `SymbolicDagBuilder::new()` + iterate interactions + `add_expr()` per count/message field + sort + dedup + build `SymbolicExpressionDag`
2. Line 213: `SymbolicRulesGpu::new(&dag, true)`
3. Lines 214-235: Compute `numer_weights[rule_idx]` and `denom_weights[rule_idx]` using `eq_3bs` and `beta_pows` — challenge-dependent, varies per proof
4. Lines 236-237: `d_numer_weights = numer_weights.to_device()`, `d_denom_weights = denom_weights.to_device()`
5. Lines 241-242: `encoded_rules.to_device()` → `d_rules`

Steps 1-2 and 5 are invariant (same result for any proof of the same AIR). Step 3-4 depend on prover challenges.

Lines 244-320: Buffer allocation, kernel launch, result copy — these remain unchanged.

## Changes

### 1. Add `LogupRound0Rules` struct to `pkey.rs`

```rust
/// Pre-computed logup Round 0 interaction evaluation rules.
/// Stores the encoded rules on device and a compact weight mapping
/// so the hot path only needs to compute challenge-dependent weights.
pub struct LogupRound0Rules {
    pub(crate) inner: EvalRules,
    /// For each interaction: (count_rule_idx, message_rule_indices).
    /// Used to compute numer_weights and denom_weights from eq_3bs/beta_pows
    /// without rebuilding the DAG.
    pub(crate) weight_map: Vec<InteractionWeightEntry>,
    /// Number of rules (length of numer_weights/denom_weights arrays)
    pub(crate) num_rules: usize,
}

/// Maps one interaction to its rule indices for weight computation.
pub struct InteractionWeightEntry {
    pub count_rule_idx: usize,
    pub message_rule_indices: Vec<usize>,
    /// Bus index for this interaction (copied from VK at keygen time).
    pub bus_index: u32,
}
```

Note: `bus_index` and `message_len` (= `message_rule_indices.len()`) are stored directly in `InteractionWeightEntry` rather than in a separate struct — this avoids a redundant type and avoids re-deriving them from the VK's `DagInteraction` at runtime (which would require a separate DAG→interaction lookup).

**File**: `crates/cuda-backend/src/pkey.rs`

### 2. Modify `AirDataGpu::new()` to build `LogupRound0Rules`

Replace lines 116-142 of `pkey.rs` (the current `logup_round0_buffer_size` computation) with full pre-computation:

```rust
let logup_round0 = if !symbolic_constraints.interactions.is_empty() {
    let mut dag_builder = SymbolicDagBuilder::new();
    let mut sorted_used_dag_idxs = Vec::new();
    for interaction in &symbolic_constraints.interactions {
        let count = dag_builder.add_expr(&interaction.count);
        sorted_used_dag_idxs.push(count);
        sorted_used_dag_idxs.extend(
            interaction.message.iter()
                .map(|field_expr| dag_builder.add_expr(field_expr)),
        );
    }
    sorted_used_dag_idxs.sort();
    sorted_used_dag_idxs.dedup();
    let dag = SymbolicExpressionDag {
        nodes: dag_builder.nodes,
        constraint_idx: sorted_used_dag_idxs,
    };
    let rules = SymbolicRulesGpu::new(&dag, true);

    // Build weight mapping: interaction_idx -> rule indices
    let weight_map: Vec<InteractionWeightEntry> = symbolic_constraints.interactions.iter()
        .map(|interaction| {
            let count_dag_idx = dag_builder.expr_to_idx
                [&(&interaction.count as *const SymbolicExpression<_>)];
            let count_rule_idx = rules.dag_idx_to_rule_idx[&count_dag_idx];
            let message_rule_indices: Vec<usize> = interaction.message.iter()
                .map(|msg| {
                    let msg_dag_idx = dag_builder.expr_to_idx
                        [&(msg as *const SymbolicExpression<_>)];
                    rules.dag_idx_to_rule_idx[&msg_dag_idx]
                })
                .collect();
            InteractionWeightEntry {
                count_rule_idx,
                message_rule_indices,
                bus_index: interaction.bus_index as u32,
            }
        })
        .collect();

    let num_rules = rules.rules.len();
    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;
    // logup round0 doesn't use used_nodes (see round0.rs:207 comment)
    let d_used_nodes = DeviceBuffer::new();

    Some(LogupRound0Rules {
        inner: EvalRules {
            d_rules,
            d_used_nodes,
            buffer_size: rules.buffer_size.try_into().expect("buffer_size exceeds u32"),
        },
        weight_map,
        num_rules,
    })
} else {
    None
};
```

Update `AirDataGpu` struct to replace `logup_round0_buffer_size: u32` with `logup_round0: Option<LogupRound0Rules>`.

**File**: `crates/cuda-backend/src/pkey.rs`

### 3. Update `logup_round0_buffer_size` references

All existing references to `pk.other_data.logup_round0_buffer_size` (used in Round 0 work item buffer size pre-computation, `logup_zerocheck/mod.rs`) must use `pk.other_data.logup_round0.as_ref().map_or(0, |r| r.inner.buffer_size)` instead.

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs` — search for `logup_round0_buffer_size` and update.

### 4. Refactor `evaluate_round0_interactions_gpu` to use pre-computed rules

Replace the function signature and body (round0.rs:162-320):

**New signature**:
```rust
pub fn evaluate_round0_interactions_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    beta_pows: &[EF],
    eq_3bs: &[EF],
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    prealloc_intermediates: Option<&mut DeviceBuffer<F>>,
    prealloc_temp_sums: Option<&mut DeviceBuffer<Frac<EF>>>,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError>
```

Remove the `symbolic: &SymbolicConstraints<F>` parameter. Instead use `pk.other_data.logup_round0`.

**New body** (lines 193-239 replaced with):
```rust
if eq_3bs.is_empty() {
    return Ok(DeviceBuffer::new());
}
let logup_r0 = pk.other_data.logup_round0.as_ref()
    .expect("logup_round0 rules missing for AIR with interactions");

let buffer_size = logup_r0.inner.buffer_size;
let d_rules = &logup_r0.inner.d_rules;

// Compute challenge-dependent weights using pre-computed mapping
let mut numer_weights = vec![EF::ZERO; logup_r0.num_rules];
let mut denom_weights = vec![EF::ZERO; logup_r0.num_rules];
let mut denom_sum_init = EF::ZERO;

for (interaction_idx, entry) in logup_r0.weight_map.iter().enumerate() {
    numer_weights[entry.count_rule_idx] += eq_3bs[interaction_idx];
    denom_sum_init += eq_3bs[interaction_idx]
        * beta_pows[entry.message_rule_indices.len()]
        * F::from_u32(entry.bus_index + 1);
    for (message_idx, &rule_idx) in entry.message_rule_indices.iter().enumerate() {
        denom_weights[rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
    }
}
let d_numer_weights = numer_weights.to_device()?;
let d_denom_weights = denom_weights.to_device()?;

// ... rest of function unchanged (buffer allocation, kernel launch, result copy)
// Uses d_rules from logup_r0 instead of locally constructed d_rules
```

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

### 5. Remove `SymbolicConstraints::from()` from `process_air_round0`

In `logup_zerocheck/mod.rs:152-319`, remove:
- Line 157: `let single_air_constraints = SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);`
- Line 264: Remove `&single_air_constraints` parameter from `evaluate_round0_interactions_gpu` call

The `single_air_constraints` local is ONLY used as a parameter to `evaluate_round0_interactions_gpu`. With the pre-computed rules, this is no longer needed.

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

### 6. Update the kernel launch in `evaluate_round0_interactions_gpu`

The kernel launch at round0.rs:295-320 currently uses the locally constructed `d_rules`. After the refactor, it uses `logup_r0.inner.d_rules` from the pre-computed data. The only change is the source of `d_rules` and `buffer_size` — all other arguments are unchanged. The existing call site at round0.rs:303 provides the correct argument order matching the FFI signature at `cuda/logup_zerocheck.rs:1015-1034`:

```rust
// Existing argument order (unchanged, only d_rules source changes):
logup_bary_eval_interactions_round0(
    temp_sums_buffer,        // &mut DeviceBuffer<Frac<EF>>
    &mut s_evals,            // &mut DeviceBuffer<Frac<EF>>
    selectors_cube,          // &DeviceBuffer<F>
    preprocessed_ptr,        // *const F
    main_parts,              // &DeviceBuffer<*const F>
    eq_cube,                 // *const EF
    public_values,           // &DeviceBuffer<F>
    &d_numer_weights,        // &DeviceBuffer<EF> (still per-proof)
    &d_denom_weights,        // &DeviceBuffer<EF> (still per-proof)
    denom_sum_init,          // EF (still per-proof)
    d_rules,                 // &DeviceBuffer<u128> — NOW from logup_r0.inner.d_rules
    buffer_size,             // u32 — NOW from logup_r0.inner.buffer_size
    intermediates,           // &mut DeviceBuffer<F>
    skip_domain,             // u32
    num_x,                   // u32
    height,                  // u32
    num_cosets,              // u32
    g_shift,                 // F
    max_temp_bytes,          // usize
)
```

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

## Invariants

1. **Correctness**: The pre-computed rules produce identical `d_rules` and weight mappings as the current per-call construction. The DAG builder is deterministic given the same symbolic constraints, and the keygen code already builds the exact same DAG (pkey.rs:117-135). The weight computation logic is unchanged — only the mapping from interaction→rule_idx is pre-computed.

2. **Existing zerocheck path unchanged**: `evaluate_round0_constraints_gpu` already uses `pk.other_data.zerocheck_round0` and is not modified.

3. **GKR input eval path unchanged**: `process_gkr_input_air` in gkr_input.rs uses `pk.other_data.interaction_rules` which is a different DAG construction (for the GKR input evaluation, not Round 0). These are distinct DAGs with different topological orderings and buffering strategies.

4. **Fallback for AIRs without interactions**: When `eq_3bs.is_empty()` or `logup_round0` is `None`, return empty buffer immediately (matches current behavior).

5. **Keygen determinism**: The pre-computation uses the same `SymbolicDagBuilder` → `SymbolicRulesGpu::new()` path as the current runtime code. The `expr_to_idx` pointer-based mapping in `SymbolicDagBuilder` is stable because the `symbolic_constraints` local is not moved between the DAG build loop and the weight-map build loop (both occur within the same block in `AirDataGpu::new()`). Note: if future refactoring moves `symbolic_constraints` ownership between these two phases, the `expr_to_idx` pointers would dangle. The current design keeps both phases in the same scope to prevent this.

6. **Memory**: The pre-computed `LogupRound0Rules` stores `d_rules` (on device, one buffer per AIR) and `weight_map` (host-side, compact). For 623 AIRs, the additional device memory is negligible (each AIR's rules are typically <1KB). Host memory for weight_map is ~10 entries × 623 AIRs = ~50KB.

## Measurement Plan

0. **Validate per-AIR overhead estimate** (before full benchmarking): Add a temporary `std::time::Instant` around the eliminated code block (round0.rs:193-239) for a single APC 300 run. Verify the measured per-AIR overhead is ≥0.3ms. If <0.3ms, the total improvement will be below the 25ms rollback threshold and the optimization should be reconsidered.

1. Build the binary in the powdr repo:
   ```bash
   cd /home/georg/powdr && cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"
   ```

2. Run APC 300 benchmark (3 runs for median):
   ```bash
   cd /home/georg/powdr/results/pairing
   RUST_LOG=info ./target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics after_apc300/metrics.json --recursion
   ```
   Analyze with `python3 /home/georg/spec.py after_apc300/metrics.json after_apc300`

3. Run APC 0 benchmark to verify no regression:
   ```bash
   RUST_LOG=info ./target/release/powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics after_apc000/metrics.json --recursion
   ```

4. Compare Round 0 time: expect 50-80ms improvement at APC 300, ≤25ms improvement at APC 0.
5. Compare STARK excl trace: expect 50-80ms improvement at APC 300.

## Rollback Criteria

- Round 0 improvement at APC 300 is less than 25ms (below noise floor).
- Any regression in STARK excl trace at APC 0 exceeding 20ms.
- Any regression in other phases (LogUp GKR, MLE Rounds, Openings) exceeding 15ms at any APC configuration.
- Proof verification fails.
