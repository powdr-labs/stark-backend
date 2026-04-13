# Plan: Pre-allocate per-thread GPU buffers for Round 0

## Goal

Eliminate per-AIR `cudaMallocAsync`/`cudaFreeAsync` and `MemoryManager` mutex overhead in Round 0 constraint and interaction evaluation by pre-allocating reusable per-thread GPU buffers. This is the same optimization pattern applied to GKR input eval in task `2026-04-13-2100-prealloc-gkr-input-buffers`, which achieved 1.67x improvement at APC 300 (532ms → 318ms).

Round 0 currently takes 296ms at APC 300 (seg0=148ms, seg1=148ms). Each AIR allocates 4 large temporary `DeviceBuffer`s via the global `Mutex<MemoryManager>` + `cudaMallocAsync` (zerocheck `intermediates` and `temp_sums_buffer`, logup `intermediates` and `temp_sums_buffer`), plus 2 small output buffers (`sp_evals`, `s_evals`) and 4 small H2D uploads (`d_main_parts`, `d_numer_weights`, `d_denom_weights`, `d_rules`). With ~310 AIRs per segment across 8 threads, the 4 large temporaries alone cause ~2,480 alloc/free pairs per segment. The `cudaMallocAsync`/`cudaFreeAsync` calls cause implicit CUDA memory pool cross-stream synchronization, serializing the 8 worker threads.

**Expected improvement**: Round 0 296ms → ~190-220ms (~75-106ms reduction, 25-36%). STARK excl trace ~70-100ms improvement (5-7%). Based on GKR prealloc precedent where seg1 (many small AIRs, allocation-dominated) improved 4.15x while seg0 (few large AIRs, kernel-dominated) improved 1.07x.

## Current Code Path

### Entry point
`LogupZerocheckGpu::sumcheck_uni_round0_polys()` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:765`

### Phase 1: Build work items (lines 897-924)
- Creates `Vec<Round0AirWorkItem>` with read-only references per AIR
- Sets `max_temp_bytes = self.memory_limit_bytes / NUM_ROUND0_STREAMS` (line 898)
- Sorts by descending height for load balance (line 927)

### Phase 2: Process AIRs in parallel (lines 933-964)
- If ≥ 100 AIRs: spawns `NUM_ROUND0_STREAMS` (8) OS threads via `std::thread::scope`
- Each thread calls `process_air_round0()` per AIR in its chunk

### process_air_round0 (lines 138-269)
Per AIR, calls two functions that each allocate temporary GPU buffers:

**evaluate_round0_constraints_gpu** (`round0.rs:31-124`):
1. `intermediates: DeviceBuffer<F>::with_capacity(intermed_capacity)` — size from `_zerocheck_r0_intermediates_buffer_size()` (line 63-68)
2. `temp_sums_buffer: DeviceBuffer<EF>::with_capacity(temp_sums_capacity)` — size from `_zerocheck_r0_temp_sums_buffer_size()` (line 80)
3. `sp_evals: DeviceBuffer<EF>::with_capacity(num_cosets * skip_domain)` — output buffer (line 96)
4. Launches `zerocheck_ntt_eval_constraints()` kernel (line 101)
5. All three buffers freed on function return (Drop → cudaFreeAsync)

**evaluate_round0_interactions_gpu** (`round0.rs:132-275`):
1. `d_numer_weights`, `d_denom_weights`, `d_rules` — small H2D uploads per AIR (lines 204-210)
2. `intermediates: DeviceBuffer<F>::with_capacity(intermed_capacity)` — size from `_logup_r0_intermediates_buffer_size()` (line 222-227)
3. `temp_sums_buffer: DeviceBuffer<Frac<EF>>::with_capacity(temp_sums_capacity)` — size from `_logup_r0_temp_sums_buffer_size()` (line 233)
4. `s_evals: DeviceBuffer<Frac<EF>>::with_capacity(large_domain)` — output buffer (line 248)
5. Launches `logup_bary_eval_interactions_round0()` kernel (line 251)
6. All buffers freed on function return

### Per-AIR allocation count: 4 large temporary allocations + 2 small output allocations + 4 small H2D uploads = 10 GPU memory operations
### Per-segment at APC 300: ~310 AIRs × 10 = ~3,100 GPU memory operations (pre-allocation targets the 4 large temporaries = ~1,240 alloc/free pairs per segment)

## Changes

### Change 1: Add `Round0ThreadBuffers` struct and extend `Round0AirWorkItem`

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add a struct after `Round0AirWorkItem` (around line 137):

```rust
struct Round0ThreadBuffers {
    /// Pre-allocated buffer for zerocheck constraint intermediates
    zc_intermediates: DeviceBuffer<F>,
    /// Pre-allocated buffer for zerocheck constraint temp_sums
    zc_temp_sums: DeviceBuffer<EF>,
    /// Pre-allocated buffer for logup interaction intermediates
    logup_intermediates: DeviceBuffer<F>,
    /// Pre-allocated buffer for logup interaction temp_sums
    logup_temp_sums: DeviceBuffer<Frac<EF>>,
}
```

Add a field to `Round0AirWorkItem`:

```rust
/// Pre-computed logup round0 interaction rules buffer_size (from SymbolicRulesGpu).
/// Computed once during work item construction to avoid rebuilding the DAG per-AIR
/// both during max-size pre-computation and inside evaluate_round0_interactions_gpu.
logup_r0_buffer_size: u32,
```

Compute this field when building work items (inside the `.map()` at line 907):

```rust
let logup_r0_buffer_size = if !eq3b.is_empty() {
    let symbolic = SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
    compute_logup_round0_buffer_size(&symbolic).0
} else {
    0
};
```

This pre-computation allows the max-size loop (Change 2) to compute logup buffer sizes without needing its own `SymbolicConstraints::from()` call per AIR. Note: `process_air_round0` still does `SymbolicConstraints::from()` at line 142 because `evaluate_round0_interactions_gpu` needs the full symbolic constraints for weight computation and rule encoding — only the `buffer_size` lookup is cached.

Note: we do NOT pre-allocate `sp_evals` (output of zerocheck) or `s_evals` (output of logup) because these are small (num_cosets × skip_domain elements, typically < 100) and must be returned to the caller. The allocation overhead for small buffers is negligible.

### Change 2: Pre-compute max buffer sizes

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

After building `work_items` and before `current_stream_sync()` (between lines 924 and 929), add a max-size computation loop:

```rust
let mut max_zc_intermediates: usize = 0;
let mut max_zc_temp_sums: usize = 0;
let mut max_logup_intermediates: usize = 0;
let mut max_logup_temp_sums: usize = 0;

for w in &work_items {
    let single_pk = w.single_pk;
    let local_constraint_deg = single_pk.vk.max_constraint_degree as usize;
    let n_lift = w.n.max(0) as usize;
    let num_cosets_zc = local_constraint_deg.saturating_sub(1);
    let num_cosets_logup = local_constraint_deg;
    let skip_domain = 1u32 << w.l_skip;
    let num_x = 1u32 << n_lift;

    // Zerocheck constraint buffers
    let zc_rules = &single_pk.other_data.zerocheck_round0;
    if !single_pk.vk.symbolic_constraints.constraints.constraint_idx.is_empty()
        && num_cosets_zc > 0
    {
        let buffer_size: u32 = zc_rules.inner.buffer_size;
        let zc_inter = unsafe {
            _zerocheck_r0_intermediates_buffer_size(
                buffer_size, skip_domain, num_x, num_cosets_zc as u32, w.max_temp_bytes,
            )
        };
        let zc_temp = unsafe {
            _zerocheck_r0_temp_sums_buffer_size(
                buffer_size, skip_domain, num_x, num_cosets_zc as u32, w.max_temp_bytes,
            )
        };
        max_zc_intermediates = max_zc_intermediates.max(zc_inter);
        max_zc_temp_sums = max_zc_temp_sums.max(zc_temp);
    }

    // Logup interaction buffers — use the pre-computed logup_r0_buffer_size from Change 1
    if w.logup_r0_buffer_size > 0 {
        let logup_inter = unsafe {
            _logup_r0_intermediates_buffer_size(
                w.logup_r0_buffer_size, skip_domain, num_x, num_cosets_logup as u32, w.max_temp_bytes,
            )
        };
        let logup_temp = unsafe {
            _logup_r0_temp_sums_buffer_size(
                w.logup_r0_buffer_size, skip_domain, num_x, num_cosets_logup as u32, w.max_temp_bytes,
            )
        };
        max_logup_intermediates = max_logup_intermediates.max(logup_inter);
        max_logup_temp_sums = max_logup_temp_sums.max(logup_temp);
    }
}
```

Note: `logup_r0_buffer_size` is computed once per work item in Change 1 via `compute_logup_round0_buffer_size()`. This avoids a separate `SymbolicConstraints::from()` call in this loop.

### Change 3: Add memory budget check and allocate per-thread buffers

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

After computing max sizes and before `current_stream_sync()`:

```rust
let per_thread_bytes =
    max_zc_intermediates * size_of::<F>()
    + max_zc_temp_sums * size_of::<EF>()
    + max_logup_intermediates * size_of::<F>()
    + max_logup_temp_sums * size_of::<Frac<EF>>();

let mut num_threads = if work_items.len() >= 100 {
    NUM_ROUND0_STREAMS.min(work_items.len())
} else {
    1
};

// Safety valve: reduce threads if pre-allocation would exceed 2 GB
while num_threads > 1 && num_threads * per_thread_bytes > 2_000_000_000 {
    num_threads /= 2;
    tracing::warn!(
        "Reducing Round 0 thread count to {} due to memory budget ({}MB per thread)",
        num_threads,
        per_thread_bytes / (1024 * 1024)
    );
}

let thread_buffers: Vec<Round0ThreadBuffers> = (0..num_threads)
    .map(|_| Round0ThreadBuffers {
        zc_intermediates: if max_zc_intermediates > 0 {
            DeviceBuffer::with_capacity(max_zc_intermediates)
        } else {
            DeviceBuffer::new()
        },
        zc_temp_sums: if max_zc_temp_sums > 0 {
            DeviceBuffer::with_capacity(max_zc_temp_sums)
        } else {
            DeviceBuffer::new()
        },
        logup_intermediates: if max_logup_intermediates > 0 {
            DeviceBuffer::with_capacity(max_logup_intermediates)
        } else {
            DeviceBuffer::new()
        },
        logup_temp_sums: if max_logup_temp_sums > 0 {
            DeviceBuffer::with_capacity(max_logup_temp_sums)
        } else {
            DeviceBuffer::new()
        },
    })
    .collect();
```

### Change 4: Modify `evaluate_round0_constraints_gpu` to accept pre-allocated buffers

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Change the function signature to take optional pre-allocated buffers:

```rust
pub fn evaluate_round0_constraints_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    prealloc_intermediates: &mut DeviceBuffer<F>,       // NEW
    prealloc_temp_sums: &mut DeviceBuffer<EF>,          // NEW
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
```

Inside the function, replace the allocation code:
- Replace `let mut intermediates = ... DeviceBuffer::<F>::with_capacity(intermed_capacity)` with `let intermediates = prealloc_intermediates;` (using the pre-allocated buffer directly)
- Replace `let mut temp_sums_buffer = ... DeviceBuffer::<EF>::with_capacity(...)` with `let temp_sums_buffer = prealloc_temp_sums;`
- Keep `sp_evals` allocation as-is (small output buffer that must be returned)

The pre-allocated buffers have capacity ≥ the required size since we took the max across all AIRs. The kernel writes only the needed portion. The `sp_evals` output buffer is still allocated per-AIR since it's small and returned.

### Change 5: Modify `evaluate_round0_interactions_gpu` to accept pre-allocated buffers

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Add `prealloc_intermediates: &mut DeviceBuffer<F>` and `prealloc_temp_sums: &mut DeviceBuffer<Frac<EF>>` parameters. Do NOT pass `logup_r0_buffer_size` — the function still builds the full DAG and rules per-AIR (needed for `d_rules`, `d_numer_weights`, `d_denom_weights`, weight index mapping), so it obtains `buffer_size` from `rules.buffer_size` naturally. The pre-computed `logup_r0_buffer_size` is only needed in the max-size computation loop (Change 2).

Replace the internal `intermediates` and `temp_sums_buffer` allocations with the pre-allocated buffers.

Keep `d_numer_weights`, `d_denom_weights`, `d_rules`, and `s_evals` allocations as-is — these are small per-AIR data that can't be reused.

### Change 6: Update `process_air_round0` signature

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Change `process_air_round0` to accept `&mut Round0ThreadBuffers`:

```rust
fn process_air_round0<HS: GpuHashScheme>(
    w: &Round0AirWorkItem<HS>,
    buffers: &mut Round0ThreadBuffers,
) -> Result<Round0AirResult, LogupZerocheckError> {
```

Pass `&mut buffers.zc_intermediates` and `&mut buffers.zc_temp_sums` to `evaluate_round0_constraints_gpu`, and `&mut buffers.logup_intermediates` and `&mut buffers.logup_temp_sums` to `evaluate_round0_interactions_gpu`.

### Change 7: Update thread spawning to pass buffers

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Follow the exact GKR prealloc pattern (from `gkr_input.rs` lines 323-349):

```rust
if num_threads <= 1 {
    let mut bufs = thread_buffers.into_iter().next().unwrap();
    for w in &work_items {
        results.push(process_air_round0(w, &mut bufs)?);
    }
} else {
    let chunk_size = work_items.len().div_ceil(num_threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = thread_buffers
            .into_iter()
            .zip(work_items.chunks(chunk_size))
            .map(|(mut bufs, chunk)| {
                s.spawn(move || -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
                    chunk
                        .iter()
                        .map(|w| process_air_round0(w, &mut bufs))
                        .collect()
                })
            })
            .collect();
        // ... collect results as before
    })?;
}
```

### Change 8: Add helper function for logup buffer size computation

**File**: `crates/cuda-backend/src/logup_zerocheck/round0.rs`

Add a public helper that computes the logup round0 buffer_size without performing any GPU work:

```rust
/// Compute the buffer_size for logup round0 interaction rules.
/// Returns (buffer_size_as_u32, rules_count).
pub fn compute_logup_round0_buffer_size(
    symbolic: &SymbolicConstraints<F>,
) -> (u32, usize) {
    if symbolic.interactions.is_empty() {
        return (0, 0);
    }
    let mut dag_builder = SymbolicDagBuilder::new();
    let mut sorted_used_dag_idxs = Vec::new();
    for interaction in &symbolic.interactions {
        let count = dag_builder.add_expr(&interaction.count);
        sorted_used_dag_idxs.push(count);
        sorted_used_dag_idxs.extend(
            interaction.message.iter().map(|e| dag_builder.add_expr(e)),
        );
    }
    sorted_used_dag_idxs.sort();
    sorted_used_dag_idxs.dedup();
    let dag = SymbolicExpressionDag {
        nodes: dag_builder.nodes,
        constraint_idx: sorted_used_dag_idxs,
    };
    let rules = SymbolicRulesGpu::new(&dag, true);
    (rules.buffer_size.try_into().unwrap(), rules.rules.len())
}
```

This is called once per AIR on the main thread during work item construction (Change 1). The returned `buffer_size` is stored in `Round0AirWorkItem::logup_r0_buffer_size` and reused in the max-size pre-computation loop (Change 2), keeping that loop simple without needing its own `SymbolicConstraints::from()` call per AIR.

## Invariants

1. **Correctness**: Pre-allocated buffers must have capacity ≥ the required size for every AIR. The max-size computation ensures this. The kernels write only the needed portion and read using their own size parameters, so stale data in the buffer tail is never read.

2. **Memory safety**: The `DeviceBuffer` backing memory must remain valid for the kernel's lifetime. Since pre-allocated buffers live for the entire Round 0 phase (not just one kernel launch), this is satisfied.

3. **No APC 0 regression**: The `>= 100 AIR` threshold for multi-threading is preserved. APC 0 (~20 AIRs/segment) uses the sequential path with a single set of pre-allocated buffers.

4. **Memory budget**: The 2 GB safety valve prevents OOM from large pre-allocations. If per-thread memory exceeds budget, thread count is halved (same pattern as GKR prealloc).

5. **Output buffers unchanged**: `sp_evals` and `s_evals` are still allocated per-AIR and returned normally. Only internal temporary buffers are pre-allocated.

6. **Kernel semantics unchanged**: The CUDA kernels receive the same pointer arguments. They don't care whether the backing buffer was allocated per-AIR or pre-allocated at max size; they only use the explicit size parameters passed to them.

## Measurement Plan

Run the standard benchmark suite:
```bash
cd /home/georg/powdr/results/pairing
# APC 0, 100, 300
for apc in apc000 apc100 apc300; do
    /path/to/powdr_openvm_riscv prove --artifact ${apc}.cbor --input 0 \
        --metrics after_${apc}/metrics.json --recursion
done
```

Analyze with `spec.py` and compare:

**Primary success criteria** (all must pass):
1. Round 0 at APC 300: < 240ms (at least 19% improvement from 296ms)
2. STARK excl trace at APC 300: < 1380ms (at least 4% improvement from 1442ms)
3. STARK excl trace at APC 0: < 2200ms (no regression)

**Secondary targets**:
4. Round 0 seg1 at APC 300: < 100ms (expecting 50-70% reduction from 148ms due to allocation-dominated workload)
5. Round 0 at APC 100: < 210ms (improvement from 250ms)

## Rollback Criteria

Revert if ANY of these are true:
1. STARK excl trace at APC 300 improves by less than 40ms (< 2.8% of current 1442ms)
2. STARK excl trace at APC 0 regresses by more than 65ms
3. GPU OOM at any APC configuration
4. Prove+verify fails at any APC configuration
5. Round 0 at APC 300 improves by less than 30ms (< 10% of current 296ms)
