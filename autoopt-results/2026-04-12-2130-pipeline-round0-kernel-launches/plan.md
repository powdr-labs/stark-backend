# Plan: Pipeline Round 0 Kernel Launches

## Goal

Eliminate per-AIR GPU synchronization in the Round 0 constraint evaluation loop to allow the GPU to execute kernels back-to-back without CPU stalls. This targets the largest sub-component regression in "STARK excl. trace": Round 0 increases from 178ms (APC 0, 99 AIRs) to 662ms (APC 300, 623 AIRs) despite a >2x reduction in total cells and constraints.

## Current Code Path

The bottleneck is in `sumcheck_uni_round0_polys()` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:598-884`.

**Call chain (per AIR iteration, lines 732-880):**
1. CPU: Prepare per-AIR data (constraints DAG, trace pointers, etc.)
2. `evaluate_round0_constraints_gpu()` at `round0.rs:31-124`:
   - Allocates `intermediates` (`DeviceBuffer::with_capacity`) — calls `cudaMallocAsync`
   - Allocates `temp_sums_buffer` (`DeviceBuffer::with_capacity`) — calls `cudaMallocAsync`
   - Allocates `sp_evals` output buffer
   - Launches `zerocheck_ntt_eval_constraints` kernel via FFI
   - Returns `sp_evals` DeviceBuffer (intermediates and temp_sums dropped = `cudaFreeAsync`)
3. **SYNC**: `sum_buffer.to_host()` at `mod.rs:792` — `cudaMemcpyAsync` + `cudaEventSynchronize`
4. CPU: Transpose, iDFT, polynomial construction → `batch_sp_poly[2*N + trace_idx]`
5. `evaluate_round0_interactions_gpu()` at `round0.rs:132-275`:
   - **CPU**: Builds `SymbolicDagBuilder`, encodes rules, computes numer/denom weights (lines 162-206)
   - Uploads `d_rules`, `d_numer_weights`, `d_denom_weights` to GPU (3 H2D copies)
   - Allocates `intermediates`, `temp_sums_buffer`, `s_evals` output buffer
   - Launches `logup_bary_eval_interactions_round0` kernel via FFI
   - Returns `s_evals` DeviceBuffer
6. **SYNC**: `sum.to_host()` at `mod.rs:848` — `cudaMemcpyAsync` + `cudaEventSynchronize`
7. CPU: Unzip frac, transpose, iDFT → `batch_sp_poly[2*i]`, `batch_sp_poly[2*i+1]`

**Why this is slow with many AIRs:**
- 623 AIRs × 2 sync points = 1,246 GPU synchronizations via `cudaEventSynchronize`
- Each sync drains the GPU pipeline, preventing the next kernel from being submitted until the current one finishes AND the result is copied to host
- CPU work between syncs (DAG construction, iDFT, etc.) blocks kernel submission
- Per-AIR `cudaMallocAsync`/`cudaFreeAsync` for intermediates adds runtime API overhead
- All work runs on `cudaStreamPerThread` (per-thread default stream), so operations are serialized

**Profiling evidence (APC 300):**
- Total GPU kernel time in Round 0: ~490ms (measured via nsight kernel summary)
- Measured Round 0 wall time: 662ms
- Overhead (sync stalls, CPU work, alloc): ~172ms (26% of wall time)
- Key kernels: `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel` (220ms, 539 instances), `logup_r0_ntt_eval_interactions_coset_parallel_kernel` (195ms + 29ms, 735 instances)

## Changes

### Step 1: Extract interaction preparation from `evaluate_round0_interactions_gpu`

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Create a new struct and function to pre-compute per-AIR interaction data:

```rust
pub struct InteractionRound0Prep {
    pub d_rules: DeviceBuffer<u128>,
    pub d_numer_weights: DeviceBuffer<EF>,
    pub d_denom_weights: DeviceBuffer<EF>,
    pub denom_sum_init: EF,
    pub buffer_size: u32,
}
```

Note: The `SymbolicRulesGpu` is consumed during preparation (its `dag_idx_to_rule_idx` is used to map interaction expressions to rule indices for weight computation). Only the encoded device rules and computed weights survive into the struct. The `SymbolicDagBuilder` and its `expr_to_idx` map are also consumed during DAG construction and do not need to be stored.

Add function `prepare_round0_interactions(pk, symbolic, eq_3bs, beta_pows) -> Result<InteractionRound0Prep>` that extracts lines 161-212 from `evaluate_round0_interactions_gpu`. This performs:
- DAG construction via `SymbolicDagBuilder` (CPU)
- `SymbolicRulesGpu::new()` from the DAG (CPU)
- Rule encoding: `rules.rules.iter().map(|c| c.encode()).collect_vec()` (CPU)
- Weight computation: numer_weights, denom_weights, denom_sum_init using `rules.dag_idx_to_rule_idx` (CPU)
- H2D upload of encoded rules (`d_rules`) and weights (`d_numer_weights`, `d_denom_weights`)

**Why**: Separates CPU-heavy preparation from kernel launch, allowing all preparation to be done upfront before any kernel launches.

### Step 2: Create kernel-launch-only variants of the evaluation functions

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Add two new functions that take pre-allocated buffers and launch kernels without allocating or syncing:

```rust
pub fn launch_round0_constraints_kernel(
    pk: &DeviceStarkProvingKey<...>,
    // ... same params as current evaluate_round0_constraints_gpu ...
    intermediates: &mut DeviceBuffer<F>,
    temp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
) -> Result<(), Round0EvalError>
```

```rust
pub fn launch_round0_interactions_kernel(
    prep: &InteractionRound0Prep,
    // ... trace pointers, eq_cube, etc ...
    intermediates: &mut DeviceBuffer<F>,
    temp_sums: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
) -> Result<(), Round0EvalError>
```

These functions call the same FFI (`zerocheck_ntt_eval_constraints`, `logup_bary_eval_interactions_round0`) but use externally-owned buffers instead of allocating their own. They do NOT perform D2H copies.

**Why**: Decouples kernel launch from allocation and sync, enabling pipeline execution.

### Step 3: Restructure the main loop in `sumcheck_uni_round0_polys`

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs` (lines 720-884)

Replace the single per-AIR loop with three phases:

**Phase A — Prepare (CPU, before any kernel launches):**
```rust
// Pre-compute per-AIR metadata
let mut per_air_data: Vec<Round0AirData> = Vec::with_capacity(num_present_airs);
for (trace_idx, ...) in izip!(...).enumerate() {
    let single_pk = &self.pk.per_air[*air_idx];
    let symbolic = SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
    let local_deg = single_pk.vk.max_constraint_degree as usize;
    let num_cosets_zc = local_deg.saturating_sub(1);
    let num_cosets_logup = local_deg;

    // IMPORTANT: max_constraint_degree includes interaction expression degrees.
    // An AIR can have num_cosets_zc > 0 but no plain AIR constraints.
    // We must check constraint_idx.is_empty() to determine if the zerocheck kernel
    // should be launched, matching the guard at round0.rs:46.
    let has_constraints = !single_pk.vk.symbolic_constraints.constraints.constraint_idx.is_empty()
        && num_cosets_zc > 0;
    
    // Build interaction prep (currently done inside the loop at round0.rs:161-206)
    let has_interactions = !eq_3bs.is_empty();
    let interaction_prep = if has_interactions {
        Some(prepare_round0_interactions(single_pk, &symbolic, eq_3bs, &self.beta_pows)?)
    } else { None };
    
    // Compute main_parts, preprocessed_ptr (same as current lines 763-768)
    let d_main_parts = ...;
    
    per_air_data.push(Round0AirData {
        air_idx, n, height, local_deg, num_cosets_zc, num_cosets_logup,
        has_constraints, has_interactions,
        interaction_prep, d_main_parts, omega_root, eq_xi_ptr, ...
    });
}

// Pre-allocate shared intermediate buffers (max size across all AIRs).
// The sizing functions are thin wrappers over the existing unsafe FFI calls:
//   _zerocheck_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
//   _zerocheck_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
//   _logup_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
//   _logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
// For zerocheck, buffer_size comes from pk.other_data.zerocheck_round0.inner.buffer_size.
// For logup, buffer_size comes from InteractionRound0Prep.buffer_size.
let max_zc_intermed = per_air_data.iter().map(|d| zc_intermediates_size(d)).max().unwrap_or(0);
let max_zc_temp = per_air_data.iter().map(|d| zc_temp_size(d)).max().unwrap_or(0);
let max_logup_intermed = per_air_data.iter().map(|d| logup_intermediates_size(d)).max().unwrap_or(0);
let max_logup_temp = per_air_data.iter().map(|d| logup_temp_size(d)).max().unwrap_or(0);
let mut zc_intermediates = DeviceBuffer::with_capacity(max_zc_intermed.max(1));
let mut zc_temp_sums = DeviceBuffer::with_capacity(max_zc_temp.max(1));
let mut logup_intermediates = DeviceBuffer::with_capacity(max_logup_intermed.max(1));
let mut logup_temp_sums = DeviceBuffer::with_capacity(max_logup_temp.max(1));

// Pre-allocate per-AIR output buffers (only for AIRs that will actually launch kernels)
let mut zc_outputs: Vec<DeviceBuffer<EF>> = per_air_data.iter()
    .map(|d| if d.has_constraints { DeviceBuffer::with_capacity(d.num_cosets_zc * skip_domain) } else { DeviceBuffer::new() })
    .collect();
let mut logup_outputs: Vec<DeviceBuffer<Frac<EF>>> = per_air_data.iter()
    .map(|d| if d.has_interactions { DeviceBuffer::with_capacity(d.num_cosets_logup * skip_domain) } else { DeviceBuffer::new() })
    .collect();
```

**Phase B — Launch all kernels (no sync):**
```rust
for (trace_idx, data) in per_air_data.iter().enumerate() {
    // Launch zerocheck constraint kernel ONLY if this AIR has plain constraints
    // (has_constraints checks both constraint_idx.is_empty() and num_cosets_zc > 0)
    if data.has_constraints {
        launch_round0_constraints_kernel(
            &self.pk.per_air[data.air_idx], data.selectors, &data.d_main_parts,
            data.public_values, data.eq_xi_ptr, d_lambda_pows,
            skip_domain, data.num_x, data.height, data.num_cosets_zc,
            data.omega_root, max_temp_bytes,
            &mut zc_intermediates, &mut zc_temp_sums, &mut zc_outputs[trace_idx],
        )?;
    }
    // Launch logup interaction kernel if this AIR has interactions
    if let Some(prep) = &data.interaction_prep {
        launch_round0_interactions_kernel(
            prep, data.selectors, &data.d_main_parts,
            data.public_values, data.eq_xi_ptr,
            skip_domain, data.num_x, data.height, data.num_cosets_logup,
            data.omega_root, max_temp_bytes,
            &mut logup_intermediates, &mut logup_temp_sums, &mut logup_outputs[trace_idx],
        )?;
    }
}
// All kernels submitted. No to_host() calls above.
```

**Phase C — Sync once + batch process results:**
```rust
// Explicit stream sync before D2H copies using the existing safe wrapper
// (crates/cuda-common/src/stream.rs:81). This is cleaner than relying on the
// first to_host()'s event sync and avoids the COPY_EVENT mutex for the initial wait.
current_stream_sync()?;

for (trace_idx, data) in per_air_data.iter().enumerate() {
    // Process zerocheck output (same logic as current lines 791-826)
    if !zc_outputs[trace_idx].is_empty() {
        let q_evals = zc_outputs[trace_idx].to_host()?;
        // ... transpose, iDFT, compute sp_0 ...
        batch_sp_poly[2 * num_present_airs + trace_idx] = UnivariatePoly::new(coeffs);
    }
    // Process logup output (same logic as current lines 847-878)
    if !logup_outputs[trace_idx].is_empty() {
        let evals = logup_outputs[trace_idx].to_host()?;
        // ... unzip frac, transpose, iDFT ...
        batch_sp_poly[2 * trace_idx] = numer_poly;
        batch_sp_poly[2 * trace_idx + 1] = denom_poly;
    }
}
```

**Why**: After the explicit `cudaStreamSynchronize`, all kernels have completed and their output buffers are ready. The subsequent `to_host()` calls will perform near-instant D2H memcpy (no pending GPU work). The GPU pipeline was never stalled between kernel launches in Phase B.

### Step 4: Keep original functions as wrappers (backward compatibility)

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Keep `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu` as convenience wrappers that allocate, launch, and return. They remain available for any other callers but are no longer used by the hot path.

**Why**: Minimizes risk. Other code paths that use these functions are unaffected.

## Invariants

1. **Correctness**: Each AIR produces the same polynomial evaluations as before. The kernel inputs are identical; only the timing of D2H copies changes.
2. **Constraint kernel guard**: The zerocheck constraints kernel is launched only when `has_constraints == true`, which requires BOTH `!constraint_idx.is_empty()` AND `num_cosets_zc > 0`. This matches the existing guard at `round0.rs:46`. Note that `max_constraint_degree` (and thus `num_cosets_zc`) includes interaction expression degrees, so an AIR with interactions but no plain constraints can have `num_cosets_zc > 0` while having empty `constraint_idx`. Launching the zerocheck kernel on such an AIR would be incorrect.
3. **Intermediate buffer reuse**: On `cudaStreamPerThread`, kernels are serialized. The shared intermediate buffer is only accessed by one kernel at a time. Each kernel writes its output to a separate per-AIR DeviceBuffer.
4. **Memory safety**: Per-AIR output buffers are pre-allocated with the exact required capacity. Shared intermediate buffers are sized to the max across all AIRs. All buffers outlive the kernel launches (dropped after Phase C).
5. **Error handling**: If any kernel launch fails in Phase B, the error propagates immediately (before subsequent launches). CUDA runtime errors on the stream cause subsequent operations to fail, so partial failures are caught.
6. **Protocol unchanged**: The verifier is not modified. The same polynomials are produced; only the GPU execution order changes.

## Measurement Plan

Run the benchmark for APC 0, 100, 300 (using `run_pairing.sh` in the powdr repo):
```bash
cd /home/georg/powdr && openvm-riscv/scripts/run_pairing.sh
```

Analyze with `spec.py` and compare "STARK (excl. trace)" and its sub-components, particularly "Round 0":
```bash
python3 /home/georg/spec.py results/pairing/apc000/metrics.json after_apc000
python3 /home/georg/spec.py results/pairing/apc100/metrics.json after_apc100
python3 /home/georg/spec.py results/pairing/apc300/metrics.json after_apc300
```

**Expected outcomes (conservative estimate):**
- Round 0 at APC 300: from ~662ms to ~450-560ms (15-30% reduction). The saving comes from eliminating GPU idle time between kernel launches. The actual gain depends on how much of the 172ms overhead is GPU stall time vs. CPU post-processing (transpose/iDFT) that merely shifts from interleaved to post-batch.
- Round 0 at APC 0: roughly unchanged (only 99 AIRs, sync overhead is proportionally smaller)
- STARK excl trace at APC 300: from ~2455ms to ~2300-2400ms (~3-8% reduction)
- No regression in correctness (prove+verify passes)

**Optimistic estimate** (if most overhead is GPU stall time, not CPU work):
- Round 0 at APC 300: ~350-450ms (30-50% reduction)
- STARK excl trace at APC 300: ~2150-2250ms (~10-15% reduction)

Also run nsight profiling for APC 300 to verify kernel pipelining:
```bash
nsys profile --output results/pairing/apc300/nsys_report --trace cuda,nvtx --sample none \
  -- $PROVE_BIN prove --artifact apc300.cbor --input 0 --recursion
```

Look for reduced gaps between kernel launches in the GPU timeline.

## Rollback Criteria

- Less than 10% improvement in Round 0 at APC 300 after full implementation
- Any correctness regression (prove fails or verify rejects)
- Peak GPU memory increases by more than 20% (unlikely given small output buffer overhead)
- Regression in APC 0 performance by more than 5%
