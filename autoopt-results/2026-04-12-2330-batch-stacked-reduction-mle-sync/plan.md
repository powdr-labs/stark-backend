# Plan: Batch Stacked Reduction MLE Sync

## Goal

Eliminate per-window GPU synchronization in the stacked reduction MLE rounds. Currently `batch_sumcheck_poly_eval()` performs a synchronous D2H copy (`to_host()` with full GPU pipeline drain) per trace-matrix window per round. Both the regular and degenerate CUDA kernels already atomically accumulate into `d_accum`, and the final result is a sum across all windows — so a single `fill_zero` + `to_host` per round suffices. This reduces thousands of pipeline-draining syncs per segment to ~17 (one per MLE round), targeting the 287ms MLE rounds at APC 300 (91% of the 317ms Stacked Reduction total).

Note on terminology: `ht_diff_idxs` creates one window per trace matrix (not per unique height). Two traces with the same height produce separate windows. At APC 300 with 623 AIR instances and multiple commits, the window count is several hundred.

## Current Code Path

### Entry point

`prove_stacked_opening_reduction_gpu()` at `crates/cuda-backend/src/stacked_reduction.rs:185-270`:
- Calls `batch_sumcheck_uni_round0_poly()` (round 0) — not affected by this change
- Calls `fold_ple_evals()` (round 0 PLE folding) — not affected
- MLE loop `for round in 1..=n_stack` (lines 243-256):
  - `batch_sumcheck_poly_eval(round, ...)` — **THE BOTTLENECK**
  - `transcript.observe_ext(...)` — Fiat-Shamir
  - `fold_mle_evals(round, u_round)` — GPU fold + CPU eq_ub update

### The bottleneck: `batch_sumcheck_poly_eval()`

`crates/cuda-backend/src/stacked_reduction.rs:794-912`

```
Lines 801-802: H2D copy of q_eval_ptrs (small, once per round)
Lines 804-831: D2H of 2 single EF elements for eq_stable/k_rot_stable (when n_max >= round-1)

Lines 832-907: THE CRITICAL LOOP
  for window in self.ht_diff_idxs.windows(2):     // one iteration per height group
      Line 844: fill_zero(d_accum)                  // cudaMemsetAsync (async, submits to stream)
      Lines 848-878: DEGENERATE path (log_height < l_skip + round):
          Line 861: eq_ub_slice.copy_to(&mut self.d_eq_ub)  // H2D per window
          Line 864: stacked_reduction_sumcheck_mle_round_degenerate(...)  // kernel launch
      Lines 879-900: REGULAR path:
          Line 887: stacked_reduction_sumcheck_mle_round(...)  // kernel launch
      Line 904: d_accum.to_host()                   // SYNCHRONOUS D2H (pipeline drain!)
      Line 905: reduce_raw_u64_to_ef(&h_accum)      // CPU reduction

Lines 909-911: Sum all per-window results:
  Ok(from_fn(|i| s_evals_batch.iter().map(|evals| evals[i]).sum::<EF>()))
```

### Why it's slow

Each `to_host()` at line 904 calls `cudaMemcpyAsync` followed by `record_and_wait()` (`crates/cuda-common/src/copy.rs:106-119`), which fully drains the GPU pipeline. With ~H height groups and 17 MLE rounds (n_stack=17 in production config), this creates ~17H pipeline-draining syncs per segment.

Per-segment measured metrics at APC 300: MLE rounds = 137-150ms. GPU kernel time for stacked reduction MLE kernels is ~30ms per segment. The remaining ~110-120ms is synchronization overhead.

### CUDA kernels (no changes needed)

Both kernels already use `sumcheck::atomic_add_fpext_to_u64(output, reduced)` to accumulate:
- `stacked_reduction_sumcheck_mle_round_kernel` at `cuda/src/stacked_reduction.cu:228-302`
- `stacked_reduction_sumcheck_mle_round_degenerate_kernel` at `cuda/src/stacked_reduction.cu:306-368`

### Degenerate kernel Rust wrapper

`crates/cuda-backend/src/cuda/stacked_reduction.rs:245-273`:
Currently takes `eq_ub_ptr: &DeviceBuffer<EF>` and calls `.as_ptr()`. Must be changed to accept a raw `*const EF` to allow pointer offsets into a pre-uploaded buffer.

### eq_ub_per_trace lifecycle

- Initialized to `vec![EF::ONE; unstacked_cols.len()]` at construction (line 415)
- Updated on CPU in `fold_mle_evals()` lines 986-994 after each round:
  ```rust
  for (s, eq_ub) in zip(&self.unstacked_cols, &mut self.eq_ub_per_trace) {
      if round + l_skip > s.log_height as usize {
          *eq_ub *= eval_eq_mle(&[u_round], &[F::from_bool(b == 1)]);
      }
  }
  ```
- Currently re-uploaded per degenerate window: `eq_ub_slice.copy_to(&mut self.d_eq_ub)`
- After our change: re-uploaded once per round as a full buffer

## Changes

### Change 1: Modify degenerate kernel wrapper to accept raw pointer

**File:** `crates/cuda-backend/src/cuda/stacked_reduction.rs`
**Function:** `stacked_reduction_sumcheck_mle_round_degenerate` (lines 245-273)

Change parameter from `eq_ub_ptr: &DeviceBuffer<EF>` to `eq_ub_ptr: *const EF`. The underlying FFI function already expects `*const EF`, and the current wrapper just calls `.as_ptr()`. This allows the caller to pass a pointer offset into a larger device buffer.

Update the call from `eq_ub_ptr.as_ptr()` to `eq_ub_ptr` directly.

### Change 2: Replace `d_eq_ub` with `d_eq_ub_all` in struct

**File:** `crates/cuda-backend/src/stacked_reduction.rs`
**Struct:** `StackedReductionGpu` (line 52)

Replace field `d_eq_ub: DeviceBuffer<EF>` (line 102) with `d_eq_ub_all: DeviceBuffer<EF>`.

In `new()` (around line 432-436), allocate with capacity `unstacked_cols.len()` (full length) instead of `max_window_len`. Preserve the empty guard to avoid `DeviceBuffer::with_capacity(0)` panic:
```rust
let d_eq_ub_all = if unstacked_cols.is_empty() {
    DeviceBuffer::new()
} else {
    DeviceBuffer::with_capacity(unstacked_cols.len())
};
```

### Change 3: Restructure `batch_sumcheck_poly_eval` inner loop

**File:** `crates/cuda-backend/src/stacked_reduction.rs`
**Function:** `batch_sumcheck_poly_eval` (lines 794-912)

Replace lines 832-911 with the batched approach:

```rust
// Upload full eq_ub_per_trace to device (once per round)
self.eq_ub_per_trace.copy_to(&mut self.d_eq_ub_all)?;

// Single fill_zero for the entire round
self.d_accum.fill_zero().map_err(StackedReductionError::FillZero)?;

for window in self.ht_diff_idxs.windows(2) {
    let window_len = window[1] - window[0];
    let unstacked_cols_ptr = unsafe { self.d_unstacked_cols.as_ptr().add(window[0]) };
    let lambda_pows_ptr = unsafe { self.d_lambda_pows.as_ptr().add(2 * window[0]) };
    let log_height = self.unstacked_cols[window[0]].log_height as usize;

    // NO fill_zero per window — accumulate across all windows

    if log_height < l_skip + round {
        let eq_r = self.eq_stable[log_height];
        let k_rot_r = self.k_rot_stable[log_height];
        let stacked_height = self.stacked_height(round);
        unsafe {
            // Pointer offset into pre-uploaded buffer (replaces per-window H2D copy)
            let eq_ub_ptr = self.d_eq_ub_all.as_ptr().add(window[0]);
            stacked_reduction_sumcheck_mle_round_degenerate(
                &self.d_q_eval_ptrs,
                eq_ub_ptr,       // raw pointer with offset
                eq_r,
                k_rot_r,
                unstacked_cols_ptr,
                lambda_pows_ptr,
                &mut self.d_accum,
                stacked_height,
                window_len,
                l_skip,
                round,
            )
            .map_err(StackedReductionError::SumcheckMleRoundDegenerate)?;
        }
    } else {
        let hypercube_dim = log_height - l_skip - round;
        let num_y = 1 << hypercube_dim;
        let stacked_height = self.stacked_height(round);
        unsafe {
            stacked_reduction_sumcheck_mle_round(
                &self.d_q_eval_ptrs,
                &self.eq_r_ns,
                &self.k_rot_ns,
                unstacked_cols_ptr,
                lambda_pows_ptr,
                &mut self.d_accum,
                stacked_height,
                window_len,
                num_y,
                self.sm_count,
            )
            .map_err(StackedReductionError::SumcheckMleRound)?;
        };
    }
    // NO to_host per window
}

// Single D2H copy for the entire round
let h_accum = self.d_accum.to_host()?;
let evals = reduce_raw_u64_to_ef(&h_accum);
// reduce_raw_u64_to_ef returns Vec<EF>; convert to fixed-size array
Ok(from_fn(|i| evals[i]))
```

Key differences from current code:
1. `self.eq_ub_per_trace.copy_to(&mut self.d_eq_ub_all)` once per round (replaces per-window `eq_ub_slice.copy_to`)
2. `self.d_accum.fill_zero()` once per round (replaces per-window)
3. Degenerate kernel gets `self.d_eq_ub_all.as_ptr().add(window[0])` (pointer offset)
4. `self.d_accum.to_host()` once per round (replaces per-window)
5. No more `s_evals_batch` collection or final `sum()` — the atomic accumulator already holds the total

### Change 4: Remove d_eq_ub reallocation logic

**File:** `crates/cuda-backend/src/stacked_reduction.rs`

Remove the per-window dynamic reallocation logic (lines 858-859):
```rust
if eq_ub_slice.len() > self.d_eq_ub.len() {
    self.d_eq_ub = DeviceBuffer::with_capacity(eq_ub_slice.len());
}
```
This is no longer needed since `d_eq_ub_all` has fixed capacity = `unstacked_cols.len()`.

## Invariants

1. **Correctness of atomic accumulation**: Both kernels use `atomic_add_fpext_to_u64` to accumulate into `d_accum`. The final result is a sum across all windows (lines 909-911). Letting all windows accumulate into the same zero-initialized buffer produces the same sum.

   **u64 overflow safety**: Each `block_reduce_sum` produces an FpExt value whose Fp components have raw values < P ≈ 2^31. Each block's atomic contribution per u64 slot is at most P-1 < 2^31. The total across all kernel launches in a round is bounded by (total number of blocks across all launches) × 2^31. The per-launch overflow guard at `stacked_reduction.cu:527` (`assert(num_y * grid.y < 1u << 32)`) only protects individual launches. For cross-launch safety: even with a conservative 20,000 total blocks across all windows in a round, the cumulative value is 20,000 × 2^31 ≈ 2^45, well within u64 range (2^64). Note that the existing per-launch assertion does NOT guarantee cross-launch safety — the cumulative bound is ensured by the arithmetic above.

2. **eq_ub_per_trace freshness**: The upload `self.eq_ub_per_trace.copy_to(&mut self.d_eq_ub_all)` at the start of each `batch_sumcheck_poly_eval` call happens AFTER the previous round's `fold_mle_evals` has updated eq_ub_per_trace on the CPU (lines 986-994). So the device buffer always contains the current values.

3. **Pointer validity**: `d_eq_ub_all` is a field of `StackedReductionGpu`, so it persists for the duration of all kernel launches within a round. The `as_ptr().add(window[0])` produces valid pointers because `window[0] + window_len <= unstacked_cols.len() == d_eq_ub_all.len()`.

4. **eq_stable/k_rot_stable D2H**: The 2 synchronous D2H copies at lines 814-828 (for eq_r_ns[0] and k_rot_ns[0]) happen BEFORE the window loop. These are still needed and are not affected by this change. They synchronize the stream, ensuring any prior async work is complete before the window loop begins.

5. **APC 0 behavior**: At APC 0 with 99 AIR instances, there are fewer height groups, so fewer syncs are eliminated. The optimization should not regress APC 0 because: (a) the bulk upload of eq_ub_per_trace is tiny for few columns, (b) the single fill_zero + to_host per round is the same cost as the first window's operations in the current code.

6. **Verifier unchanged**: No changes to the verifier or protocol. The optimization only changes the order/batching of GPU operations; the computed polynomial evaluations are identical.

## Measurement Plan

Run the benchmark for APC 0, 100, 300:
```bash
cd /home/georg/powdr/results/pairing
PROVE_BIN=/home/georg/powdr/target/release/powdr_openvm_riscv
for apc in apc000 apc100 apc300; do
    mkdir -p after_${apc}
    $PROVE_BIN prove --artifact ${apc}.cbor --input 0 --metrics after_${apc}/metrics.json --recursion
done
```

Analyze with spec.py:
```bash
python3 /home/georg/spec.py after_apc000/metrics.json after_apc000
python3 /home/georg/spec.py after_apc300/metrics.json after_apc300
```

**Primary metric**: `prover.openings.stacked_reduction.mle_rounds_time_ms` for app_proof segments.

**Expected results at APC 300**:
- Stacked Reduction MLE rounds: 287ms → ~80-120ms (50-70% reduction)
- Stacked Reduction total: 317ms → ~110-150ms (45-65% reduction)
- STARK excl trace: 2488ms → ~2300-2400ms (~4-8% reduction)

**Secondary check**: Verify APC 0 does not regress (Stacked Reduction should stay ~113ms ± noise).

## Rollback Criteria

Revert if:
- Less than 15% improvement in Stacked Reduction MLE rounds at APC 300 (i.e., less than 43ms improvement from baseline 287ms)
- Any correctness regression (prove+verify failure)
- APC 0 STARK excl trace regresses by more than 5%
