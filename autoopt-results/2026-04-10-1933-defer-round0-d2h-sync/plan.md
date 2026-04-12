# Plan: Defer Round 0 D2H Synchronization

## Goal

Restructure the Round 0 per-AIR loop in `sumcheck_uni_round0_polys` to eliminate blocking D2H synchronization points between GPU kernel launches. Currently, up to ~1,246 kernel results (2 per AIR instance, fewer if some AIRs lack constraints or interactions) trigger a `to_host()` call that blocks the CPU via `cudaEventSynchronize`, creating pipeline stalls. By deferring all D2H transfers to after all kernels are queued, the GPU executes kernels back-to-back without host-side stalls.

This targets Round 0, the single largest contributor to the GPU prover's failure to scale with APC count: Round 0 *increases* from 178ms to 663ms (+485ms, 3.7x) as APCs go from 0 to 300, despite total work (cells, constraints, bus interaction messages) all decreasing by >2x.

The time savings come from two distinct mechanisms:
1. **GPU idle time elimination**: In the current code, the GPU sits idle while the CPU processes each AIR's IDFT and polynomial construction. Deferring D2H lets all GPU kernels execute back-to-back without gaps.
2. **CPU-GPU overlap in Phase 1**: `evaluate_round0_interactions_gpu` performs CPU-side work on each call (DAG construction, rule encoding, weight computation, H2D transfers). In the deferred approach, this CPU work for AIR[i+1] overlaps with GPU kernel execution for AIR[i], since the CPU no longer blocks waiting for each kernel to finish.

## Current Code Path

### Call Chain

1. `logup_zerocheck_prove` (mod.rs:224) calls `prover.sumcheck_uni_round0_polys(ctx, lambda)`
2. `sumcheck_uni_round0_polys` (mod.rs:598-884):
   - Lines 600-728: Setup (lambda powers, eq precomputation, selectors, etc.)
   - Lines 730-880: **The per-AIR loop** (this is what we modify)
   - Lines 881-883: Emit metrics, return `batch_sp_poly`

### The Per-AIR Loop (lines 730-880)

For each AIR instance (`trace_idx` from 0 to `num_present_airs - 1`):

1. **Setup** (lines 742-771): Compute `single_pk`, `single_air_constraints`, `local_constraint_deg`, `omega_root`, build `d_main_parts` (device buffer of device pointers to trace matrices), look up `eq_xi_tree`.

2. **Zerocheck kernel launch + sync** (lines 777-827):
   - `evaluate_round0_constraints_gpu(...)` -> returns `DeviceBuffer<EF>` (result on GPU)
   - `sum_buffer.to_host()` at line 792 -- **BLOCKING SYNC** via `cudaEventSynchronize`
   - CPU IDFT: transpose to row-major, call `UnivariatePoly::from_geometric_cosets_evals_idft`
   - Polynomial construction: multiply by `(Z^{2^l_skip} - 1)`, debug assert sum = 0
   - Store result in `batch_sp_poly[2 * num_present_airs + trace_idx]`

3. **Logup kernel launch + sync** (lines 831-879):
   - `evaluate_round0_interactions_gpu(...)` -> returns `DeviceBuffer<Frac<EF>>` (result on GPU)
   - `sum.to_host()` at line 848 -- **BLOCKING SYNC** via `cudaEventSynchronize`
   - Unzip into numer/denom, optional normalization for negative `n`
   - Transpose to row-major, call `from_geometric_cosets_evals_idft` twice (numer + denom)
   - Store results in `batch_sp_poly[2 * trace_idx]` and `batch_sp_poly[2 * trace_idx + 1]`

### Why It's Slow

Each `to_host()` call (copy.rs:102-125) does:
1. `cudaMemcpyAsync(D2H, cudaStreamPerThread)` -- queue async copy
2. `COPY_EVENT.record_and_wait(cudaStreamPerThread)` -- record event then `cudaEventSynchronize` (CPU blocks until ALL prior stream work completes)

This means the CPU blocks after every single kernel, waiting for the GPU. Since kernels are tiny (each processes one AIR's constraints for a few cosets), the GPU is idle during CPU processing time, and the CPU is idle during GPU kernel execution. With 623 AIR instances at APC 300, there are up to 1,246 such blocking sync points (fewer if some AIRs lack constraints or interactions, since the `to_host()` calls are guarded by `is_empty()` checks).

### Key Properties

- **All loop iterations are independent**: zerocheck/logup for AIR[i] reads only from AIR[i]'s trace, selectors, and proving key. The shared read-only inputs (`d_lambda_pows`, `eq_xi_tree`, `beta_pows`) are immutable.
- **Result buffers are tiny**: `DeviceBuffer<EF>` holds `num_cosets_zc * skip_domain` elements (~4-16 EF values); `DeviceBuffer<Frac<EF>>` holds `num_cosets_logup * skip_domain` (~4-16 Frac values). Each is < 1KB.
- **`d_main_parts` is per-iteration**: Created at line 768 from device pointers into `air_ctx` trace matrices. The trace matrices themselves are owned by `ProvingContext` and live throughout. `d_main_parts` is just an index array of device pointers.
- **Temporary buffers inside kernel functions are stream-ordered freed**: `evaluate_round0_constraints_gpu` and `evaluate_round0_interactions_gpu` create large temporary buffers (`intermediates`, `temp_sums_buffer`) that are dropped when the function returns. This is safe because `DeviceBuffer::drop` calls `cudaFreeAsync` (stream-ordered) -- the GPU will finish reading the buffer before the free executes. This remains safe in the deferred approach because all operations are on the same `cudaStreamPerThread`.
- **Memory allocation is also stream-ordered**: `d_malloc` uses `cudaMallocAsync` for small buffers and VPMM for large ones. Both are ordered with respect to `cudaStreamPerThread`, so new allocations can reuse memory from prior stream-ordered frees.
- **MemoryManager accounting note**: During Phase 1, `MemoryManager.current_size` will undercount actual physical GPU memory in use because `d_free` decrements `current_size` immediately (mod.rs:108) while the actual `cudaFreeAsync` is deferred. This means `max_used_size` may not reflect true physical peak. This is not a correctness issue but should be understood during debugging. The `nvidia-smi` check in the rollback criteria validates actual physical memory usage.

## Changes

### Overview

Split the single per-AIR loop (lines 730-880) into two phases:
- **Phase 1** (GPU launch): Iterate all AIRs, launch kernels, collect GPU result buffers
- **Phase 2** (CPU processing): Explicit stream sync, then D2H transfers + IDFT + polynomial construction

### Step 1: Define a per-AIR metadata struct

Add a local struct inside `sumcheck_uni_round0_polys` to hold the data needed for Phase 2 processing:

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, inside `sumcheck_uni_round0_polys`

```rust
// Metadata stored per AIR during Phase 1, consumed in Phase 2
struct Round0Pending {
    trace_idx: usize,
    air_idx: usize,  // for debug assert only
    num_cosets_zc: usize,
    num_cosets_logup: usize,
    local_constraint_deg: usize,
    omega_root: F,
    n: isize,  // n_per_trace value, for normalization check
    zc_result: DeviceBuffer<EF>,       // may be empty
    logup_result: DeviceBuffer<Frac<EF>>,  // may be empty
}
```

Note: `n` is `isize` matching `self.n_per_trace: Vec<isize>` (mod.rs:423). The loop destructures `&n` yielding `isize`. Methods `is_negative()` and `unsigned_abs()` (returning `usize`) are on `isize`.

### Step 2: Phase 1 -- GPU launch loop

Replace lines 730-880 with a Phase 1 loop that launches kernels without D2H transfers.

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

The new Phase 1 loop:
```rust
let mut pending: Vec<Round0Pending> = Vec::with_capacity(num_present_airs);
// Collect d_main_parts to keep them alive until all kernels complete.
// Although cudaFreeAsync is stream-ordered (so dropping would be safe),
// we retain them defensively to avoid coupling correctness to VPMM behavior.
let mut d_main_parts_vec: Vec<DeviceBuffer<*const F>> = Vec::with_capacity(num_present_airs);

for (trace_idx, ((air_idx, air_ctx), &n, selectors_cube, public_values, eq_3bs)) in izip!(
    &ctx.per_trace,
    &self.n_per_trace,
    &selectors_base,
    &self.public_values_per_trace,
    &self.eq_3b_per_trace,
)
.enumerate()
{
    debug!("starting batch constraints for air_idx={air_idx} (trace_idx={trace_idx})");
    let single_pk = &self.pk.per_air[*air_idx];
    let single_air_constraints =
        SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
    let local_constraint_deg = single_pk.vk.max_constraint_degree as usize;
    debug_assert_eq!(
        single_air_constraints.max_constraint_degree(),
        local_constraint_deg
    );
    assert!(
        local_constraint_deg <= self.constraint_degree,
        "Max constraint degree ({local_constraint_deg}) of AIR {air_idx} exceeds the global maximum {}",
        self.constraint_degree
    );

    let log_large_domain = log2_ceil_usize(local_constraint_deg << l_skip);
    let omega_root = F::two_adic_generator(log_large_domain);

    assert!(!xi.is_empty(), "xi vector must not be empty");

    let height = air_ctx.common_main.height();
    let mut main_parts = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
    for committed in &air_ctx.cached_mains {
        main_parts.push(committed.trace.buffer().as_ptr());
    }
    main_parts.push(air_ctx.common_main.buffer().as_ptr());
    let d_main_parts = main_parts.to_device()?;

    let n_lift = n.max(0) as usize;
    let eq_xi_tree = &self.eq_xis[&n_lift];
    let max_temp_bytes = self.memory_limit_bytes;

    let num_cosets_zc = local_constraint_deg.saturating_sub(1);
    let zc_result = evaluate_round0_constraints_gpu(
        single_pk,
        selectors_cube.buffer(),
        &d_main_parts,
        public_values,
        eq_xi_tree.get_ptr(n_lift),
        d_lambda_pows,
        1 << l_skip,
        1 << n_lift,
        height as u32,
        num_cosets_zc as u32,
        omega_root,
        max_temp_bytes,
    )?;

    let num_cosets_logup = local_constraint_deg;
    let logup_result = evaluate_round0_interactions_gpu(
        single_pk,
        &single_air_constraints,
        selectors_cube.buffer(),
        &d_main_parts,
        public_values,
        eq_xi_tree.get_ptr(n_lift),
        &self.beta_pows,
        eq_3bs,
        1 << l_skip,
        1 << n_lift,
        height as u32,
        num_cosets_logup as u32,
        omega_root,
        max_temp_bytes,
    )?;

    d_main_parts_vec.push(d_main_parts);
    pending.push(Round0Pending {
        trace_idx,
        air_idx: *air_idx,
        num_cosets_zc,
        num_cosets_logup,
        local_constraint_deg,
        omega_root,
        n,
        zc_result,
        logup_result,
    });
}
```

Key differences from current code:
- No `to_host()` calls -- kernel results stay on GPU
- `d_main_parts` collected into `d_main_parts_vec` instead of being dropped each iteration
- Per-AIR metadata (coset counts, omega_root, n) stored in `pending` for Phase 2
- All setup code (assertions, variable computation) is identical to current code

### Step 3: Phase 2 -- Stream sync + D2H transfers + CPU processing

After Phase 1, explicitly synchronize the stream, then process all results. An explicit `current_stream_sync()` at the phase boundary makes the sync point visible in profiling and simplifies debugging, rather than relying on the first `to_host()` to implicitly sync.

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

```rust
// Phase 2: explicit stream sync, then D2H + CPU polynomial construction
// Note: current_stream_sync returns CudaError; LogupZerocheckError has no direct CudaError
// variant. Route through MemCopyError which has #[from] CudaError. This is semantically
// imprecise but functional. If preferred, add a StreamSync(CudaError) variant to
// LogupZerocheckError instead.
current_stream_sync().map_err(|e| LogupZerocheckError::from(MemCopyError::from(e)))?;

// Release d_main_parts now that all kernels have completed
drop(d_main_parts_vec);

for entry in pending {
    // Zerocheck processing (same logic as current lines 791-827)
    if !entry.zc_result.is_empty() {
        let q_evals = entry.zc_result.to_host()?;
        let q = {
            let mut values = EF::zero_vec(entry.num_cosets_zc << l_skip);
            for coset_idx in 0..entry.num_cosets_zc {
                for i in 0..1 << l_skip {
                    values[i * entry.num_cosets_zc + coset_idx] =
                        q_evals[(coset_idx << l_skip) + i];
                }
            }
            UnivariatePoly::from_geometric_cosets_evals_idft(
                RowMajorMatrix::new(values, entry.num_cosets_zc),
                entry.omega_root,
                entry.omega_root,
            )
        };
        let sp_0_deg = sumcheck_round0_deg(l_skip, entry.local_constraint_deg);
        let coeffs = (0..=sp_0_deg)
            .map(|i| {
                let mut c = -*q.coeffs().get(i).unwrap_or(&EF::ZERO);
                if i >= 1 << l_skip {
                    c += q.coeffs()[i - (1 << l_skip)];
                }
                c
            })
            .collect_vec();
        debug_assert_eq!(
            coeffs.iter().step_by(1 << l_skip).copied().sum::<EF>(),
            EF::ZERO,
            "Zerocheck sum is not zero for air_id: {}",
            entry.air_idx
        );
        batch_sp_poly[2 * num_present_airs + entry.trace_idx] = UnivariatePoly::new(coeffs);
    }

    // Logup processing (same logic as current lines 847-879)
    if !entry.logup_result.is_empty() {
        let evals = entry.logup_result.to_host()?;
        let (mut numer, denom): (Vec<EF>, Vec<EF>) =
            evals.into_iter().map(|frac| (frac.p, frac.q)).unzip();
        if entry.n.is_negative() {
            let norm_factor = F::from_u32(1 << entry.n.unsigned_abs()).inverse();
            for s in &mut numer {
                *s *= norm_factor;
            }
        }
        let mut numer_values = EF::zero_vec(entry.num_cosets_logup << l_skip);
        let mut denom_values = EF::zero_vec(entry.num_cosets_logup << l_skip);
        for coset_idx in 0..entry.num_cosets_logup {
            for i in 0..1 << l_skip {
                let src = (coset_idx << l_skip) + i;
                let dst = i * entry.num_cosets_logup + coset_idx;
                numer_values[dst] = numer[src];
                denom_values[dst] = denom[src];
            }
        }
        batch_sp_poly[2 * entry.trace_idx] = UnivariatePoly::from_geometric_cosets_evals_idft(
            RowMajorMatrix::new(numer_values, entry.num_cosets_logup),
            entry.omega_root,
            F::ONE,
        );
        batch_sp_poly[2 * entry.trace_idx + 1] =
            UnivariatePoly::from_geometric_cosets_evals_idft(
                RowMajorMatrix::new(denom_values, entry.num_cosets_logup),
                entry.omega_root,
                F::ONE,
            );
    }
}
```

Note: `current_stream_sync()` requires adding `use openvm_cuda_common::stream::current_stream_sync;` to the import block at the top of the file if not already imported. Check existing imports first.

After the explicit sync, each `to_host()` call still invokes `cudaMemcpyAsync` + `record_and_wait`, but since the stream is already synchronized, the event-wait returns immediately. The D2H copies are then essentially just synchronous memcpys of tiny buffers (< 1KB each).

### Step 4: Preserve debug logging

The current loop has `debug!("starting batch constraints for air_idx=...")` at line 741. This is kept in Phase 1 (see Step 2 code above). No additional logging changes needed.

### Summary of touched code

| File | What changes |
|---|---|
| `crates/cuda-backend/src/logup_zerocheck/mod.rs:730-880` | Split single per-AIR loop into Phase 1 (launch) + explicit sync + Phase 2 (D2H + CPU) |
| `crates/cuda-backend/src/logup_zerocheck/mod.rs` (imports) | Add `current_stream_sync` import if not present |

No other files are modified. No CUDA kernel changes. No FFI changes. No new crate dependencies.

### Follow-up: Rayon parallelization of Phase 2

The two-phase structure makes Phase 2 CPU processing a natural target for rayon parallelization. Since each AIR's IDFT writes to distinct indices in `batch_sp_poly` (`2 * trace_idx`, `2 * trace_idx + 1`, `2 * num_present_airs + trace_idx`), the CPU work is embarrassingly parallel. This can be added as an immediate follow-up by converting the `for entry in pending` loop to use `par_iter` with indexed writes. Expected additional savings: 50-100ms on APC 300. Not included in the initial prototype to isolate the deferred-sync improvement in measurements.

## Invariants

1. **Output equivalence**: `batch_sp_poly` must contain exactly the same polynomials as before. Each `batch_sp_poly[k]` is written at most once, using the same index formula. The IDFT input data and polynomial construction logic are unchanged.

2. **Stream ordering**: All kernels and D2H transfers use `cudaStreamPerThread`. Kernels execute in the same order as before (trace_idx 0, 1, 2, ...). D2H transfers now happen after all kernels instead of interleaved, but this doesn't affect correctness since each AIR's result buffer is independent.

3. **Memory safety**:
   - `d_main_parts_vec` keeps all per-AIR pointer arrays alive until the explicit `current_stream_sync()`, then dropped.
   - The trace matrices pointed to by values in `d_main_parts` are owned by `ProvingContext` and live throughout.
   - Temporary buffers inside `evaluate_round0_constraints_gpu` / `evaluate_round0_interactions_gpu` are freed via `cudaFreeAsync` (stream-ordered) and correctly reclaimed by VPMM.

4. **Debug assertions**: The zerocheck sum-is-zero assertion (line 820-824) still runs, using the same `air_idx` value stored in `Round0Pending`.

5. **Empty buffer handling**: Both `zc_result.is_empty()` and `logup_result.is_empty()` checks are preserved, matching the current `sum_buffer.is_empty()` / `sum.is_empty()` guards.

6. **Error conversion for `current_stream_sync`**: The `CudaError` from `current_stream_sync()` must be mapped to `LogupZerocheckError`. This can go through `MemCopyError::from(CudaError)` if that conversion exists, or directly via a new `From` impl. Check existing error conversion paths during implementation.

## Measurement Plan

### Before (baseline for this task)

Run the pairing benchmark for APC 0, 100, 300 on the current code:

```bash
cd /path/to/powdr
openvm-riscv/scripts/run_pairing.sh  # for each APC config
```

Collect `metrics.json` for each config. Analyze with:

```bash
python spec.py results/pairing/<config>/metrics.json before_<config>
```

Record the "Round 0" and "STARK (excl. trace)" values from the output.

### After

Apply the changes, then rerun the same benchmarks:

```bash
cd /path/to/powdr
openvm-riscv/scripts/run_pairing.sh  # for each APC config
```

Collect new `metrics.json` files. Analyze with:

```bash
python spec.py results/pairing/<config>/metrics.json after_<config>
```

### What to compare

| Metric | APC 0 | APC 100 | APC 300 |
|---|---|---|---|
| Round 0 (ms) | before vs after | before vs after | before vs after |
| STARK excl trace (ms) | before vs after | before vs after | before vs after |
| Ratio APC300/APC0 STARK excl trace | before vs after | | |

### Correctness verification

```bash
cargo nextest run -p openvm-cuda-backend --test-threads=4
```

All existing tests must pass. The pairing benchmark itself also verifies proof correctness.

### GPU memory check

Monitor physical GPU memory during the APC 300 run using `nvidia-smi` in a separate terminal to verify no significant peak memory increase vs baseline.

## Rollback Criteria

1. **Correctness failure**: Any test failure or proof verification failure -> immediate revert.
2. **Insufficient improvement**: If STARK excl trace for APC 300 improves by less than 100ms (< 4% of ~2,478ms), the optimization is not worth the code change -> revert.
3. **Regression on APC 0**: If APC 0 STARK excl trace regresses by more than 20ms -> investigate before committing.
4. **GPU memory regression**: If physical peak GPU memory (from `nvidia-smi`) increases by more than 50MB -> investigate. The `MemoryManager.max_used_size` may undercount during Phase 1 because `d_free` decrements `current_size` before the stream-ordered free physically executes, so rely on `nvidia-smi` for the ground truth.
