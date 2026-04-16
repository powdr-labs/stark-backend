# Plan: Batch Stacked Reduction Round 0 Descriptor Arrays

*Note: Directory name is a misnomer from the first plan iteration. The actual optimization targets stacked reduction Round 0 batching, matching task.md.*

## Goal

Reduce kernel launch overhead in the stacked reduction Round 0 by batching per-trace kernel launches into per-height-group batched launches using descriptor arrays. At APC 300, the stacked reduction Round 0 calls `_stacked_reduction_sumcheck_round0` (which launches 2 CUDA kernels: `stacked_reduction_round0_block_sum_kernel` + `final_reduce_block_sums<true>`) and `_stacked_reduction_fold_ple` (1 kernel) once per backing trace matrix (~400 per segment). The current total is 2391 kernel launches across both segments (797 × 3 kernel types). By batching same-height traces, we eliminate ~1400 launches.

## Current Code Path

### Round 0 sumcheck loop

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, lines 504-549

```rust
for ((trace_ptr, trace_height, trace_width), window) in zip(
    mem::take(&mut self.trace_ptrs),
    self.ht_diff_idxs.windows(2),
) {
    let log_height = trace_height.ilog2();
    let n = log_height as isize - l_skip as isize;
    let d_g_output = if n >= 0 { &mut d_g_pos } else { &mut d_g_neg[(-n-1) as usize] };

    let block_sums_len = _stacked_reduction_r0_required_temp_buffer_size(
        trace_height as u32, trace_width as u32, l_skip as u32) as usize;
    if block_sums_len > self.d_block_sums.len() {
        self.d_block_sums = DeviceBuffer::with_capacity(block_sums_len);
    }

    let lambda_pows_ptr = self.d_lambda_pows.as_ptr().add(2 * window[0]);
    stacked_reduction_sumcheck_round0(
        &self.eq_r_ns, trace_ptr, lambda_pows_ptr,
        &mut self.d_block_sums, d_g_output,
        trace_height, trace_width, l_skip,
    );
}
```

Each `stacked_reduction_sumcheck_round0` call launches TWO CUDA kernels:
1. `stacked_reduction_round0_block_sum_kernel` — computes G0, G1, G2 partial block sums
2. `final_reduce_block_sums<true>` — reduces block sums and ADDs into the G output buffer

### PLE fold loop

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, lines 703-745

```rust
for stacked in &self.stacked_per_commit {                // <-- per-commit loop
    let folded_evals = DeviceBuffer::with_capacity(...);
    folded_evals.fill_zero()?;
    let mut dst_offset = 0;
    for trace in &stacked.traces {                        // <-- per-trace loop
        stacked_reduction_fold_ple(
            trace.buffer().as_ptr(),
            folded_evals.as_mut_ptr().add(dst_offset),
            &self.d_omega_skip_pows, &d_inv_lagrange_denoms,
            trace.height(), trace.width(), l_skip,
        );
        dst_offset += new_height * trace.width();
    }
    self.q_evals.push(folded_evals);                      // per-commit output
}
```

The fold_ple loop is nested: outer loop per commit (each commit has its own `folded_evals` buffer), inner loop per trace within that commit.

### CUDA kernel signatures

**File**: `crates/cuda-backend/src/cuda/stacked_reduction.rs`, lines 44-63

```c
fn _stacked_reduction_sumcheck_round0(
    eq_r_ns, trace_ptr, lambda_pows, block_sums, output,
    trace_height, trace_width, l_skip, num_x) -> i32;

fn _stacked_reduction_fold_ple(
    src, dst, omega_skip_pows, inv_lagrange_denoms,
    trace_height, trace_width, l_skip) -> i32;
```

### nsight kernel data (APC 300, full benchmark including recursion)

- `stacked_reduction_round0_block_sum_kernel`: 22ms total, 797 instances, avg 28µs
- `final_reduce_block_sums<true>`: 7.8ms total, 797 instances, avg 10µs
- `stacked_reduction_fold_ple_kernel`: 9ms total, 797 instances, avg 11µs
- **Total: 38.8ms GPU, 2391 kernel launches**

Per-segment wall clock:
- Stacked reduction round 0: 14ms (seg0) + 12ms (seg1) = 26ms total
- Of this, the GPU kernel time is ~15ms per segment (from 38.8ms / ~2.5 segments+recursion)
- The remaining ~11ms per segment is kernel launch overhead, block_sums resize, lambda_pows pointer arithmetic

### Grouping analysis

Traces within a segment are grouped by height (same `n = log_height - l_skip`). At APC 300:
- ~20 distinct heights per segment
- Some heights have 200+ traces (e.g., APC traces with height 2^4)
- Others have 1-5 traces (large AIRs with height 2^18+)

Since `n = log_height - l_skip`, all traces of the same height map to the same G output bucket. Grouping by `trace_height` is sufficient.

## Changes

### Change 1: Add batched Round 0 block_sum CUDA kernel

**File**: New file `crates/cuda-backend/cuda/src/stacked_reduction_batched.cu`

```c
struct StackedR0Desc {
    const Fp *trace_ptr;
    const FpExt *lambda_pows;  // pre-offset: self.d_lambda_pows.as_ptr() + 2 * window[0]
    uint32_t trace_width;
    uint32_t blocks_for_this_desc;  // number of grid blocks assigned to this descriptor
};

// Maps a global blockIdx.x to a (descriptor_index, local_block_index) pair
// using binary search on the prefix-sum of blocks_for_this_desc.
// Follows the BlockCtx pattern from batch_mle.cu.
```

The batched kernel reuses the existing per-trace kernel body. Each block is assigned to one descriptor via the `BlockCtx` binary search. Block sums are written to `block_sums[global_block_offset * NUM_G * skip_domain]`.

**Final reduce strategy**: Call `final_reduce_block_sums<true>` per-descriptor after the batched block_sum kernel. This keeps the final reduce unchanged and simple. With ~20 height groups per segment and 1 final_reduce per descriptor within each group, the total final_reduce calls stay at ~400 per segment — but each reduce processes the blocks from just one descriptor. This is identical to the current behavior for each descriptor.

Rationale: The final_reduce kernel is tiny (10µs avg, 7.8ms total). Keeping it per-descriptor avoids implementing a batched reduce, adds zero risk, and the 400 launches × 5µs = 2ms overhead is acceptable relative to the savings from batching the larger block_sum kernel.

### Change 2: Add batched PLE fold CUDA kernel

**File**: Same file `crates/cuda-backend/cuda/src/stacked_reduction_batched.cu`

```c
struct FoldPleDesc {
    const Fp *src;        // trace buffer pointer
    FpExt *dst;           // output buffer pointer (pre-offset within per-commit folded_evals)
    uint32_t trace_width;
};
```

Same BlockCtx binary search pattern. Each block processes elements from one descriptor.

### Change 3: Add FFI bindings

**File**: `crates/cuda-backend/src/cuda/stacked_reduction.rs`

Add `repr(C)` structs for `StackedR0Desc` and `FoldPleDesc`, extern "C" declarations for the batched launchers, and safe Rust wrapper functions.

### Change 4: Orchestration in Rust — Round 0

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, modify `batch_sumcheck_uni_round0_poly`

Replace the per-trace loop (lines 504-549) with:

1. **Pre-sort by height** (already sorted via `ht_diff_idxs`): Iterate `trace_ptrs` and group consecutive traces with the same `trace_height` into batches.

2. **Pre-compute max block_sums size**: `max_block_sums_len = max over all traces of _stacked_reduction_r0_required_temp_buffer_size(h, w, l_skip)`. Allocate `self.d_block_sums` once at this size.

3. **For each height group** (traces with same `trace_height`):
   a. If group has < 10 traces: process sequentially (existing code path)
   b. If group has ≥ 10 traces:
      - Build `Vec<StackedR0Desc>` on host. Each descriptor's `lambda_pows` field = `self.d_lambda_pows.as_ptr() + 2 * window_start_for_this_trace`. The `window_start_for_this_trace` is computed from the corresponding `ht_diff_idxs[i]`.
      - Upload descriptor array to device: `descs.to_device()`
      - Compute total blocks needed: sum of `blocks_for_this_desc` across all descriptors in group
      - Allocate shared block_sums buffer for all descriptors: `total_blocks * NUM_G * skip_domain`
      - Launch batched `block_sum` kernel with `grid = total_blocks`
      - For each descriptor in the group: call `final_reduce_block_sums<true>` to accumulate that descriptor's block sums into the appropriate `d_g_output` buffer
   
   Note: All traces of the same height have the same `n`, so they all accumulate into the same G bucket (`d_g_pos` or `d_g_neg[k]`).

### Change 5: Orchestration in Rust — PLE fold

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, modify `fold_ple_evals`

Batching is scoped **within** the per-commit outer loop (lines 703-745). Within each commit:

1. Group traces by `trace.height()`.
2. For groups with ≥ 10 traces: build `FoldPleDesc` array, upload, launch batched kernel. Each descriptor's `dst` = `folded_evals.as_mut_ptr() + dst_offset_for_this_trace` (pre-computed from the sequential scan).
3. For groups with < 10 traces: use existing per-trace path.

The per-commit `q_evals.push(folded_evals)` logic is unchanged.

## Invariants

1. **Correctness**: Each trace produces the same G0/G1/G2 partial block sums via the same kernel body. The block sums are reduced and added to G output identically. The fold_ple writes to the same per-trace non-overlapping output regions. Output is bit-identical to the sequential path.

2. **APC 0 no-regression**: At APC 0, ~20 traces per segment. Most height groups have < 10 traces, falling below the batching threshold. The sequential path runs unchanged.

3. **No VPMM pool impact**: Descriptor arrays are tiny (~400 × 24 bytes = ~10 KB). The block_sums pre-allocation replaces per-trace resizing with one upfront allocation. No per-thread accumulation buffers (the failure mode of `multistream-stacked-reduction-round0` which allocated 112 DeviceBuffers). Net memory impact is negligible.

4. **fold_ple per-commit boundary**: Batching is scoped within each commit's trace loop. The per-commit `folded_evals` allocation and `q_evals.push` are unchanged.

5. **lambda_pows offsets**: Each descriptor's `lambda_pows` is set to `d_lambda_pows.as_ptr() + 2 * ht_diff_idxs[i]` where `i` is the trace's index in `trace_ptrs`. This matches the current sequential computation at line 535.

## Measurement Plan

1. Run benchmark for APC 0, 100, 300 before and after.
2. Compare `prover.openings.stacked_reduction.round0_time_ms` per segment.
3. Compare `STARK (excl. trace)` overall via `spec.py`.
4. Run nsight profile on APC 300 to verify:
   - `stacked_reduction_round0_block_sum_kernel` instance count decreased (797 → ~80-120)
   - `stacked_reduction_fold_ple_kernel` instance count decreased similarly
   - `final_reduce_block_sums<true>` instance count unchanged (~797)
   - Total kernel GPU time unchanged or slightly improved
5. APC 0 must show no regression (within ±10ms noise on STARK excl trace).

### Expected results

- Launch overhead reduction: ~1400 eliminated launches × ~5µs = ~7ms across full benchmark
- Per-segment stacked reduction round 0: 14ms → 10-12ms (seg0), 12ms → 9-11ms (seg1)
- STARK excl trace APC 300: 1101ms → ~1087-1095ms (~6-14ms improvement)
- APC 0: no change (below batching threshold)

## Rollback Criteria

- Less than 5ms improvement on stacked reduction round 0 total at APC 300.
- Any regression > 10ms on STARK excl trace at APC 0.
- Any correctness failure (proof verification fails at any APC configuration).
