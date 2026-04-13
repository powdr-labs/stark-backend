# Plan: Pre-allocate per-thread Round 0 buffers with adaptive size threshold

## Goal

Eliminate per-AIR GPU memory allocation overhead in the Round 0 constraint and interaction evaluation by pre-allocating reusable per-thread buffer pools, following the pattern that saved 217ms in GKR input evaluation (`prealloc-gkr-input-buffers`). Use an adaptive size threshold to avoid the GPU memory pressure that caused the previous Round 0 pre-allocation attempt (`prealloc-round0-buffers`) to fail.

## Current Code Path

### Round 0 orchestration
**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs:767-1003`

`sumcheck_uni_round0_polys` processes 623 AIRs (at APC 300) across 8 OS threads with per-thread CUDA streams. Work items are sorted by descending height and distributed round-robin.

Each thread calls `process_air_round0` (mod.rs:140-272) per AIR, which calls two evaluation functions sequentially:

### Per-AIR evaluation: `evaluate_round0_constraints_gpu`
**File:** `crates/cuda-backend/src/logup_zerocheck/round0.rs:31-124`

Allocations per call (3 mutex acquisitions + 3 frees = 6 total):
1. `intermediates: DeviceBuffer<F>` — size from `_zerocheck_r0_intermediates_buffer_size()` (line 63-68)
2. `temp_sums_buffer: DeviceBuffer<EF>` — size from `_zerocheck_r0_temp_sums_buffer_size()` (line 80)
3. `sp_evals: DeviceBuffer<EF>` — size = `num_cosets * skip_domain` (line 96)

All three are freed on function return (Drop). Each alloc/free pair acquires the global `MEMORY_MANAGER` mutex (`crates/cuda-common/src/memory_manager/mod.rs:30`).

The zerocheck `buffer_size` comes from `pk.other_data.zerocheck_round0.inner.buffer_size` (round0.rs:51-53), which is pre-computed in the proving key and available during Phase 1 work item preparation.

### Per-AIR evaluation: `evaluate_round0_interactions_gpu`
**File:** `crates/cuda-backend/src/logup_zerocheck/round0.rs:132-274`

Allocations per call (6 mutex acquisitions + 6 frees = 12 total):
1. `intermediates: DeviceBuffer<F>` — size from `_logup_r0_intermediates_buffer_size()` (line 222-227)
2. `temp_sums_buffer: DeviceBuffer<Frac<EF>>` — size from `_logup_r0_temp_sums_buffer_size()` (line 233)
3. `s_evals: DeviceBuffer<Frac<EF>>` — size = `num_cosets * skip_domain` (line 248)
4. `d_rules` — from `encoded_rules.to_device()` (line 210)
5. `d_numer_weights` — from `numer_weights.to_device()` (line 204)
6. `d_denom_weights` — from `denom_weights.to_device()` (line 205)

Items 4-6 are data-dependent per AIR (weights depend on runtime `eq_3bs` and `beta_pows`; rules depend on a DAG built per-AIR). Items 1-3 can be pre-allocated.

**Critical note:** The logup `buffer_size` (round0.rs:212) is **not** available from `pk.other_data` — it comes from a `SymbolicRulesGpu` constructed dynamically inside the function (round0.rs:161-206) via `SymbolicDagBuilder` + `SymbolicRulesGpu::new(&dag, true)`. This DAG is deterministic per AIR but is NOT pre-computed in `AirDataGpu`. This must be addressed (see Step 1).

### Additional per-AIR allocations in `process_air_round0`
- `d_main_parts` — from `main_parts.to_device()` (mod.rs:162), ~8-16 elements

### Total allocation overhead
Per AIR: ~6 mutex acquisitions for the 3 pre-allocatable buffers in each of the 2 functions (intermediates + temp_sums + result = 3 alloc+free pairs × 2 functions = 12 mutex acquisitions). Plus ~8 more for items 4-6, d_main_parts, and their frees. Total: **~20 mutex acquisitions per AIR**.

With 623 AIRs × 2 segments = 1246 AIR evaluations: **~24,920 mutex acquisitions** across 8 concurrent threads.

### Why this is slow
The global `MEMORY_MANAGER` singleton (`Mutex<MemoryManager>`, mod.rs:30 of cuda-common) serializes all allocations across all threads. With 8 concurrent threads each doing ~20 alloc/free cycles per AIR, the mutex is heavily contended. The GKR pre-alloc optimization proved this pattern: eliminating ~1,240 mutex acquisitions per segment saved 214ms for GKR input eval.

### Why the previous Round 0 pre-alloc failed
The `prealloc-round0-buffers` attempt pre-allocated for the **maximum** AIR across all work items. The max intermediates buffer was ~1GB per thread × 8 threads = 8GB, which caused systemic GPU memory pressure (only 24GB total) that degraded all phases including GKR (+188ms) and Leaf Recursion (+100ms).

## Changes

### Step 1: Add `logup_round0_buffer_size` to `AirDataGpu`

**File:** `crates/cuda-backend/src/pkey.rs`

Add a new field to `AirDataGpu` (line 24-32):
```rust
pub struct AirDataGpu {
    pub interaction_rules: InteractionEvalRules,
    pub zerocheck_round0: ConstraintOnlyRules<true>,
    pub zerocheck_mle: ConstraintOnlyRules<false>,
    pub zerocheck_monomials: Option<ZerocheckMonomials>,
    pub interaction_monomials: Option<InteractionMonomials>,
    /// Buffer size for the logup round0 interaction evaluation DAG.
    /// Computed at keygen time to allow pre-computation of buffer sizes
    /// during Round 0 work item preparation.
    pub logup_round0_buffer_size: u32,
}
```

In `AirDataGpu::new` (pkey.rs:86-114), compute this using the same logic currently in `evaluate_round0_interactions_gpu` (round0.rs:161-211):
1. Create `SymbolicDagBuilder`, add interaction count/message expressions
2. Sort+dedup constraint indices
3. Build `SymbolicExpressionDag`, then `SymbolicRulesGpu::new(&dag, true)`
4. Store `rules.buffer_size` as `logup_round0_buffer_size`

This makes the logup `buffer_size` available from `pk.other_data.logup_round0_buffer_size` during Phase 1.

### Step 2: Add `Round0ThreadBuffers` struct

**File:** `crates/cuda-backend/src/logup_zerocheck/round0.rs`

```rust
pub(crate) struct Round0ThreadBuffers {
    /// Pre-allocated intermediates buffer for zerocheck (reused across AIRs)
    pub zc_intermediates: DeviceBuffer<F>,
    /// Pre-allocated temp sums buffer for zerocheck
    pub zc_temp_sums: DeviceBuffer<EF>,
    /// Pre-allocated result buffer for zerocheck (data copied to host before reuse)
    pub zc_sp_evals: DeviceBuffer<EF>,
    /// Pre-allocated intermediates buffer for logup
    pub logup_intermediates: DeviceBuffer<F>,
    /// Pre-allocated temp sums buffer for logup
    pub logup_temp_sums: DeviceBuffer<Frac<EF>>,
    /// Pre-allocated result buffer for logup
    pub logup_s_evals: DeviceBuffer<Frac<EF>>,
}
```

No separate capacity tracking fields — the pre-allocated `DeviceBuffer.len()` IS the capacity. The caller compares the AIR's required size against `buf.len()` to decide whether to use the pre-allocated buffer.

### Step 3: Add buffer size fields to `Round0AirWorkItem`

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Add pre-computed buffer sizes to the work item struct:

```rust
struct Round0AirWorkItem<'a, HS: GpuHashScheme> {
    // ... existing fields ...
    /// Pre-computed zerocheck intermediates capacity needed
    zc_intermed_cap: usize,
    /// Pre-computed zerocheck temp sums capacity needed
    zc_temp_sums_cap: usize,
    /// num_cosets_zc * skip_domain (zerocheck result size)
    zc_sp_evals_cap: usize,
    /// Pre-computed logup intermediates capacity needed (uses logup_round0_buffer_size from Step 1)
    logup_intermed_cap: usize,
    /// Pre-computed logup temp sums capacity needed
    logup_temp_sums_cap: usize,
    /// num_cosets_logup * skip_domain (logup result size)
    logup_s_evals_cap: usize,
}
```

Compute these during Phase 1 (work item preparation, mod.rs ~901-926):
- `local_constraint_deg = single_pk.vk.max_constraint_degree`
- `skip_domain = 1 << l_skip`
- `num_x = 1 << n_lift` where `n_lift = n.max(0) as usize`
- `num_cosets_zc = local_constraint_deg.saturating_sub(1)` (mod.rs:166)
- `num_cosets_logup = local_constraint_deg` (mod.rs:213)
- Zerocheck `buffer_size` from `single_pk.other_data.zerocheck_round0.inner.buffer_size`
- Logup `buffer_size` from `single_pk.other_data.logup_round0_buffer_size` (from Step 1)
- Call `_zerocheck_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets_zc, max_temp_bytes)` and the other 5 FFI size functions to get actual sizes.

### Step 4: Compute threshold and pre-allocate

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs` (in `sumcheck_uni_round0_polys`, between Phase 1 and Phase 2)

After building work items with their buffer sizes:

1. For each of the 6 buffer types, find the maximum across ALL work items.
2. If the maximum for any buffer type is 0 (e.g., all AIRs have no zerocheck constraints), use capacity 1 as a sentinel (DeviceBuffer panics on size 0). Track this so the fallback path is used for those buffers.
3. Compute total per-thread bytes from the 6 max values.
4. Memory budget: `per_thread_budget = min(memory_limit_bytes / num_threads, 256 * 1024 * 1024)`.
5. If total per-thread bytes > per_thread_budget: reduce `num_threads` until it fits, same pattern as GKR (gkr_input.rs:296-303).
6. If after reducing to `num_threads=1` it still exceeds 2GB: skip pre-allocation entirely and use the current dynamic path.
7. Pre-allocate `num_threads` sets of `Round0ThreadBuffers`.

### Step 5: Modify evaluation functions to accept pre-allocated buffers

**File:** `crates/cuda-backend/src/logup_zerocheck/round0.rs`

#### D2H copy handling (addresses review finding 1)

`DeviceBuffer` does not distinguish capacity from logical length — `len()` returns the allocation size. `to_host_on_current_stream()` copies `len()` elements. When a pre-allocated buffer has capacity larger than the actual output, the D2H copy would transfer excess data.

**Solution:** Create a correctly-sized `DeviceBuffer` view using `ManuallyDrop` for D2H copies:

```rust
use std::mem::ManuallyDrop;

/// Create a non-owning DeviceBuffer view with the correct logical length for D2H copies.
/// SAFETY: `buf` must have capacity >= `len`, and the caller must ensure `buf` outlives the view.
unsafe fn device_view<T>(buf: &DeviceBuffer<T>, len: usize) -> ManuallyDrop<DeviceBuffer<T>> {
    ManuallyDrop::new(DeviceBuffer::from_raw_parts(buf.as_mut_ptr(), len))
}
```

This creates a `DeviceBuffer` that points to the same GPU memory but has the correct `len` for D2H copies. `ManuallyDrop` prevents the Drop impl from freeing the pre-allocated memory.

#### `evaluate_round0_constraints_gpu` changes

Add parameter:
```rust
prealloc: Option<(&mut DeviceBuffer<F>, &mut DeviceBuffer<EF>, &mut DeviceBuffer<EF>)>,
// (intermediates, temp_sums, sp_evals)
```

Replace allocation of `intermediates` (lines 63-68):
```rust
let (mut intermediates, owns_intermediates) = if let Some((ref mut pre_inter, _, _)) = prealloc {
    if intermed_capacity > 0 && pre_inter.len() >= intermed_capacity {
        // Reuse pre-allocated buffer (no mutex, no alloc)
        // SAFETY: pre_inter has capacity >= intermed_capacity
        (unsafe { ManuallyDrop::new(DeviceBuffer::from_raw_parts(pre_inter.as_mut_ptr(), intermed_capacity)) }, false)
    } else {
        (ManuallyDrop::new(DeviceBuffer::<F>::with_capacity(intermed_capacity.max(1))), true)
    }
} else if intermed_capacity > 0 {
    (ManuallyDrop::new(DeviceBuffer::<F>::with_capacity(intermed_capacity)), true)
} else {
    (ManuallyDrop::new(DeviceBuffer::<F>::new()), false)
};
```

Apply the same pattern for `temp_sums_buffer` and `sp_evals`.

For the D2H copy of `sp_evals` in the caller (`process_air_round0`, mod.rs:184):
```rust
let actual_len = w.zc_sp_evals_cap;
let view = unsafe { device_view(&bufs.zc_sp_evals, actual_len) };
let q_evals = view.to_host_on_current_stream()?;
```

At function end: if `owns_intermediates`, explicitly drop (frees via mutex). If not, ManuallyDrop prevents freeing.

#### `evaluate_round0_interactions_gpu` changes

Same pattern. Add `prealloc: Option<(&mut DeviceBuffer<F>, &mut DeviceBuffer<Frac<EF>>, &mut DeviceBuffer<Frac<EF>>)>` for (intermediates, temp_sums, s_evals).

Items 4-6 (`d_rules`, `d_numer_weights`, `d_denom_weights`) remain dynamically allocated. These 3 alloc+free pairs per logup call still go through the mutex. This is unavoidable since their data is AIR-specific and runtime-dependent.

### Step 6: Modify `process_air_round0` to use pre-allocated buffers

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Change signature:
```rust
fn process_air_round0<HS: GpuHashScheme>(
    w: &Round0AirWorkItem<HS>,
    bufs: &mut Round0ThreadBuffers,
) -> Result<Round0AirResult, LogupZerocheckError>
```

Inside the function:
1. Check if zerocheck buffers fit:
   ```rust
   let zc_prealloc = if w.zc_intermed_cap <= bufs.zc_intermediates.len()
       && w.zc_temp_sums_cap <= bufs.zc_temp_sums.len()
       && w.zc_sp_evals_cap <= bufs.zc_sp_evals.len()
       && w.zc_sp_evals_cap > 0
   {
       Some((&mut bufs.zc_intermediates, &mut bufs.zc_temp_sums, &mut bufs.zc_sp_evals))
   } else {
       None
   };
   ```
2. Call `evaluate_round0_constraints_gpu(..., zc_prealloc)`.
3. For the D2H copy: if pre-allocated was used, create a `device_view` with the actual output length (`w.zc_sp_evals_cap`). If dynamic allocation was used, the buffer already has the correct length.
4. After `to_host_on_current_stream()`, the host data is in a `Vec<EF>`. The pre-allocated GPU buffer is safe to reuse for the next AIR.
5. Same pattern for logup buffers, using `bufs.logup_*`.

### Step 7: Wire up pre-allocated buffers in the thread spawn

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs` (Phase 2, lines 963-985)

Pass one `Round0ThreadBuffers` to each spawned thread:
```rust
let handles: Vec<_> = thread_buffers
    .into_iter()
    .zip(thread_items.into_iter())
    .enumerate()
    .map(|(thread_id, (mut bufs, indices))| {
        let items = &work_items;
        s.spawn(move || -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
            let t0 = std::time::Instant::now();
            let results: Vec<_> = indices
                .iter()
                .map(|&idx| process_air_round0(&items[idx], &mut bufs))
                .collect::<Result<_, _>>()?;
            tracing::debug!(thread_id, elapsed_ms = t0.elapsed().as_millis(), num_airs = indices.len(), "round0 thread done");
            Ok(results)
        })
    })
    .collect();
```

For the single-thread path (`num_threads <= 1`), pass the single buffer set similarly.

## Invariants

1. **Correctness**: Pre-allocated buffers are only used when they meet or exceed the required capacity. When they don't fit, the fallback path is identical to the current code.
2. **D2H copy correctness**: Result buffers (`sp_evals`, `s_evals`) use `ManuallyDrop<DeviceBuffer::from_raw_parts(ptr, actual_len)>` views for D2H copies, ensuring exactly the correct number of elements are transferred to host. No excess data is copied.
3. **Buffer reuse safety**: Result buffers are only reused after `to_host_on_current_stream()` completes. This function calls `current_stream_sync()` (copy.rs:147), ensuring all GPU operations on the buffer are complete before reuse.
4. **ManuallyDrop safety**: Views created via `ManuallyDrop` never free GPU memory. Only the owning `Round0ThreadBuffers` (dropped after all threads complete) frees the pre-allocated memory.
5. **No cross-thread sharing**: Each thread owns its buffer set exclusively.
6. **Memory budget**: Total pre-allocation is bounded by `min(256MB × num_threads, 2GB)`. If this exceeds budget, `num_threads` is reduced. If it still exceeds after `num_threads=1`, pre-allocation is skipped entirely.
7. **APC 0 behavior**: With < 100 AIRs, `num_threads = 1`. If the single AIR's buffers exceed pre-allocated capacity, the fallback path is used.
8. **Kernel buffer semantics**: Intermediates and temp_sums are scratch space — kernels write to them but the data is never read back to host. Using a pre-allocated buffer with capacity > required is harmless; the kernel indexes only up to `intermed_capacity`, not the buffer's allocated length.

## Measurement Plan

### Commands
```bash
# In powdr repo:
cd results/pairing
PROVE_BIN="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')/release/powdr_openvm_riscv"

# Build
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"

# Before measurements (3 runs for APC 300)
for i in 1 2 3; do
  $PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics before_apc300_$i/metrics.json --recursion
done
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics before_apc000/metrics.json --recursion

# After implementing the optimization, rebuild and re-measure:
for i in 1 2 3; do
  $PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics after_apc300_$i/metrics.json --recursion
done
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics after_apc000/metrics.json --recursion

python3 ~/spec.py after_apc300_1/metrics.json after_apc300
```

### Expected results
- **Round 0 at APC 300**: 276ms → 200-240ms (13-28% improvement)
- **STARK excl trace at APC 300**: 1421ms → 1345-1385ms (3-5% improvement)
- **Round 0 at APC 0**: 177ms → 177ms (no change — single-threaded, few AIRs)
- **No regression in other phases**: GKR, MLE Rounds, Trace Commit, Openings should be within noise

### Stability verification
Compare medians of 3 before-runs vs 3 after-runs for APC 300 to confirm improvement exceeds noise (~20ms).

### Nsight validation
Run nsight on APC 300 and verify:
- Fewer cudaMallocAsync/cudaFreeAsync API calls in the Round 0 NVTX span
- Reduced mutex contention visible in CPU thread timeline (fewer blocked intervals)

## Rollback Criteria

1. **Round 0 at APC 300 improves by less than 20ms** (median of 3 runs). Below the noise floor.
2. **STARK excl trace at APC 300 regresses by more than 20ms**. Any single-phase regression (GKR, Trace Commit, Openings) > 30ms.
3. **APC 0 STARK excl trace regresses by more than 30ms**. Fallback path must not add overhead.
4. **Any correctness failure** at any APC configuration (prove + verify must succeed).
