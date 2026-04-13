# Plan: Batch MLE main_ptrs Upload

## Goal

Eliminate per-trace `main_ptrs.to_device()` overhead in MLE sumcheck rounds by consolidating all per-round main_ptrs into a single flat vector and uploading it with one `to_device()` call. This removes ~3000-4000 `cudaMallocAsync` + `cudaMemcpyAsync` + `cudaFreeAsync` triplets per benchmark, saving an estimated 20-40ms from the 170ms MLE Rounds span at APC 300.

## Current Code Path

### Entry point

`LogupZerocheckGpu::sumcheck_polys_batch_eval()` at `crates/cuda-backend/src/logup_zerocheck/mod.rs:1111`

This function is called once per MLE round per segment. At APC 300 with 2 segments and ~8 rounds per segment, it executes ~16 times per benchmark.

### Per-trace main_ptrs upload (the bottleneck)

**Case A (late_eval, round > n_lift)** — lines 1188-1195:
```rust
let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
    .iter()
    .map(|m| MainMatrixPtrs { data: m.buffer().as_ptr(), air_width: ... })
    .collect_vec();
let main_ptrs_dev = main_ptrs.to_device()?;  // ← per-trace cudaMallocAsync + H2D
```

**Case B (early_eval, round <= n_lift)** — lines 1335-1347:
```rust
let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
    .iter()
    .map(|m| MainMatrixPtrs { data: base_ptr.wrapping_add(...), air_width: ... })
    .collect_vec();
let main_ptrs_dev = main_ptrs.to_device()?;  // ← per-trace cudaMallocAsync + H2D
```

Each call creates a `DeviceBuffer<MainMatrixPtrs<EF>>` via:
1. `d_malloc()` acquires `MEMORY_MANAGER` mutex + `cudaMallocAsync` (~5-10μs)
2. `copy_to()` → `cudaMemcpyAsync` H2D (~2-5μs)

When each `DeviceBuffer` is dropped (at `sumcheck_polys_batch_eval` exit), `d_free()` acquires the same mutex + calls `cudaFreeAsync`. So the full per-trace cost is: 2 mutex acquisitions + `cudaMallocAsync` + `cudaMemcpyAsync` + `cudaFreeAsync` + HashMap insert/remove, totaling ~15-20μs per trace.

**Count**: Case A fires once per trace across all rounds (~310 per segment × 2 segments = ~620 calls). Case B fires for each trace where `round <= n_lift`, decreasing as rounds progress. Total across all rounds and segments: ~3000-4000 calls. At ~15-20μs per call including deallocation, this is ~45-80ms of overhead, of which ~30-40ms is on the critical path (allocation + copy; deallocation is deferred to function exit).

### How main_ptrs_dev is consumed

**TraceCtx struct** — `batch_mle.rs:89-105`:
```rust
pub(crate) struct TraceCtx {
    // ...
    pub main_ptrs_dev: DeviceBuffer<MainMatrixPtrs<EF>>,  // owns the allocation
    // ...
}
```

All consumers only use the raw pointer:
1. **Batch builders** (`batch_mle.rs:188`, `batch_mle.rs:335`, `batch_mle_monomial.rs:184,407,672`): use `t.main_ptrs_dev.as_ptr()`
2. **Single-trace fallback** (`batch_mle.rs:473`): calls `evaluate_mle_constraints_gpu(&t.main_ptrs_dev)`, which internally uses `.as_ptr()` at `mle_round.rs:56`
3. **Single-trace fallback** (`batch_mle.rs:642`): calls `evaluate_mle_interactions_gpu(&t.main_ptrs_dev)`, which internally uses `.as_ptr()` at `mle_round.rs:111`

Neither fallback function uses `.len()` or any other `DeviceBuffer` method beyond `.as_ptr()`.

## Changes

### Change 1: Modify TraceCtx to use raw pointer

**File**: `crates/cuda-backend/src/logup_zerocheck/batch_mle.rs:89-105`

Replace:
```rust
pub main_ptrs_dev: DeviceBuffer<MainMatrixPtrs<EF>>,
```
With:
```rust
pub main_ptrs_ptr: *const MainMatrixPtrs<EF>,
```

**Why**: TraceCtx no longer owns individual allocations. It holds a pointer into a shared device buffer that is kept alive by the caller.

### Change 2: Update batch builder usage of main_ptrs

**Files**:
- `crates/cuda-backend/src/logup_zerocheck/batch_mle.rs:188` — `ZerocheckMleBatchBuilder::new()`
- `crates/cuda-backend/src/logup_zerocheck/batch_mle.rs:335` — `LogupMleBatchBuilder::new()`
- `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs:184` — `ZerocheckMonomialBatch::new()`
- `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs:407` — `ZerocheckMonomialParYBatch::new()`
- `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs:672` — `LogupMonomialBatch::new()`

Change `t.main_ptrs_dev.as_ptr()` → `t.main_ptrs_ptr` at each site.

Also update the doc comment at `batch_mle_monomial.rs:109-110` which references `main_ptrs_dev` by name.

**Why**: Trivial rename. The raw pointer is the same value that `.as_ptr()` returned.

### Change 3: Update fallback evaluation function signatures

**File**: `crates/cuda-backend/src/logup_zerocheck/mle_round.rs`

`evaluate_mle_constraints_gpu()` (line 24): Change parameter from `d_main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>` to `d_main_ptrs: *const MainMatrixPtrs<EF>`. Update the call site at line 56 from `d_main_ptrs.as_ptr()` to `d_main_ptrs`.

`evaluate_mle_interactions_gpu()` (line 78): Same change. Update call site at line 111.

**File**: `crates/cuda-backend/src/logup_zerocheck/batch_mle.rs`

Update call sites:
- Line 473: `&t.main_ptrs_dev` → `t.main_ptrs_ptr`
- Line 642: `&t.main_ptrs_dev` → `t.main_ptrs_ptr`

**Why**: These functions only need the raw pointer. Removing the `DeviceBuffer` reference requirement allows them to work with shared-buffer pointers. This trades a nominal compile-time type-safety guarantee for the performance gain, but both functions are `pub(crate)` internal helpers already operating in an unsafe context (raw device pointers throughout).

### Change 4: Batch main_ptrs upload in sumcheck_polys_batch_eval

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

#### 4a: Batch Case A (late_eval) main_ptrs

After the Case A loop (current lines 1173-1211), instead of calling `main_ptrs.to_device()` per trace:

1. Collect all Case A main_ptrs into a flat `Vec<MainMatrixPtrs<EF>>` with per-trace offsets.
2. After the loop, do a single `all_late_main_ptrs.to_device()`.
3. Each `late_eval` TraceCtx gets `main_ptrs_ptr` pointing into the shared buffer at its offset.
4. Store the shared buffer as `_keepalive_late_main_ptrs` to keep the device memory alive.

```rust
// Before Case A loop (pre-size based on trace count estimate):
let mut all_late_main_ptrs: Vec<MainMatrixPtrs<EF>> = Vec::with_capacity(self.n_per_trace.len() * 2);
let mut late_main_ptrs_offsets: Vec<usize> = Vec::new();

// Inside Case A (round == n_lift + 1):
late_main_ptrs_offsets.push(all_late_main_ptrs.len());
let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..].iter()
    .map(|m| MainMatrixPtrs { data: m.buffer().as_ptr(), air_width: ... })
    .collect_vec();
all_late_main_ptrs.extend(main_ptrs);
// Don't call to_device() yet — defer to after the loop

// After Case A loop:
let _keepalive_late_main_ptrs = if !all_late_main_ptrs.is_empty() {
    let buf = all_late_main_ptrs.to_device()?;
    // Backfill pointers into late_eval TraceCtx entries
    for (i, t) in late_eval.iter_mut().enumerate() {
        t.main_ptrs_ptr = unsafe { buf.as_ptr().add(late_main_ptrs_offsets[i]) };
    }
    Some(buf)
} else {
    None
};
```

**Implementation note**: The Case A loop currently pushes to `late_eval` inside the loop. We need to restructure slightly: defer `late_eval.push()` until after the device upload, or push with a placeholder pointer and backfill. The backfill approach is simpler — push with `std::ptr::null()` initially, then fill in after upload.

#### 4b: Batch Case B (early_eval) main_ptrs

After Phase 3 (current lines 1303-1364), instead of calling `main_ptrs.to_device()` per trace:

1. Extend the Phase 3 loop to collect all Case B main_ptrs into a flat `Vec<MainMatrixPtrs<EF>>` with per-trace offsets.
2. After the loop, do a single `all_early_main_ptrs.to_device()`.
3. Each `early_eval` TraceCtx gets `main_ptrs_ptr` pointing into the shared buffer at its offset.
4. Store the shared buffer as `_keepalive_early_main_ptrs`.

```rust
// Before Phase 3 loop (pre-size based on Case B trace count):
let mut all_early_main_ptrs: Vec<MainMatrixPtrs<EF>> = Vec::with_capacity(case_b_traces.len() * 2);
let mut early_main_ptrs_offsets: Vec<usize> = Vec::with_capacity(case_b_traces.len());

// Inside Phase 3 per-trace loop:
early_main_ptrs_offsets.push(all_early_main_ptrs.len());
let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..].iter()
    .map(|m| MainMatrixPtrs { data: base_ptr.wrapping_add(...), air_width: ... })
    .collect_vec();
all_early_main_ptrs.extend(main_ptrs);

// After Phase 3 loop:
let _keepalive_early_main_ptrs = if !all_early_main_ptrs.is_empty() {
    let buf = all_early_main_ptrs.to_device()?;
    for (i, t) in early_eval.iter_mut().enumerate() {
        t.main_ptrs_ptr = unsafe { buf.as_ptr().add(early_main_ptrs_offsets[i]) };
    }
    Some(buf)
} else {
    None
};
```

**Why**: This is the main optimization. ~310 Case B traces per round → 1 upload. The total data is ~310 × 2 entries × 16 bytes = ~10KB per round — a trivial single upload.

### Change 5: Update test file

**File**: `crates/cuda-backend/src/tests.rs:707-726`

Change:
```rust
let main_ptrs_dev = main_ptrs.to_device().unwrap();
// ...
main_ptrs_dev,
```
To:
```rust
let main_ptrs_dev = main_ptrs.to_device().unwrap();
// ...
main_ptrs_ptr: main_ptrs_dev.as_ptr(),
```

Add `_keepalive` binding to keep `main_ptrs_dev` alive for the test scope.

**Why**: Test constructs TraceCtx directly and needs to match the new field name.

## Invariants

1. **Pointer validity**: Every `main_ptrs_ptr` in a TraceCtx must point into a device buffer that outlives all uses of that TraceCtx. The `_keepalive_*` variables in `sumcheck_polys_batch_eval` ensure this — they are declared before the evaluation calls and dropped at function exit.

2. **Pointer alignment**: `MainMatrixPtrs<EF>` is `#[repr(C)]` with natural alignment. Pointers into a flat `DeviceBuffer<MainMatrixPtrs<EF>>` are correctly aligned since the buffer is a contiguous array.

3. **Correctness**: The raw pointer values are identical to what `.as_ptr()` returned from the individual `DeviceBuffer`s. The data content (matrix pointers + widths) is unchanged. GPU kernels receive the same pointer values.

4. **No behavioral change for APC 0**: At APC 0, the multi-threading threshold is not met and there are few traces (~20 per segment). The batching still works but produces a small flat vector. No regression expected.

5. **Fallback path correctness**: The fallback evaluation functions (`evaluate_mle_constraints_gpu`, `evaluate_mle_interactions_gpu`) receive the same pointer value they received before. The only difference is the function signature changes from `&DeviceBuffer<T>` to `*const T`.

## Measurement Plan

Run the standard benchmark:
```bash
cd /home/georg/powdr && openvm-riscv/scripts/run_pairing.sh
```

Analyze with spec.py:
```bash
python /home/georg/spec.py results/pairing/apc300/metrics.json apc300
python /home/georg/spec.py results/pairing/apc100/metrics.json apc100
python /home/georg/spec.py results/pairing/apc000/metrics.json apc000
```

### Success criteria

| Metric | APC 300 Target | Rationale |
|--------|---------------|-----------|
| MLE Rounds | < 155ms | ≥10% improvement from 170ms (conservative, given ~3000-4000 eliminated calls) |
| STARK excl trace | < 1395ms | ≥12ms net improvement |
| STARK excl trace APC 0 | < 2205ms | No regression (< 3% increase from 2140ms) |

### Verification

1. Prove + verify passes for all 3 APC configs (0, 100, 300)
2. Run APC 300 three times to confirm consistency (MLE Rounds std < 10ms)

## Rollback Criteria

Revert if ANY of:
1. MLE Rounds at APC 300 improves by < 10ms (below measurement noise)
2. STARK excl trace at APC 0 regresses by > 65ms
3. Prove + verify fails for any APC config
4. GPU OOM at any APC config
