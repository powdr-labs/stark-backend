# Plan: Multi-stream Stacked Reduction Round 0 and PLE Fold

## Goal

Reduce the Stacked Reduction phase from ~74ms to ~37ms at APC 300 by multi-streaming the per-trace sequential kernel launches in `batch_sumcheck_uni_round0_poly` and `fold_ple_evals`. These two functions launch ~797 small per-trace CUDA kernels sequentially on the default stream. Distributing across 8 OS threads with per-thread `cudaStreamPerThread` — the proven pattern from Round 0 (task 4, 1.89x) and GKR input eval (task 5, 1.40x) — should yield ~37ms savings.

## Current Code Path

### `batch_sumcheck_uni_round0_poly` (`stacked_reduction.rs:481-556`)

Computes the Round 0 univariate sumcheck polynomial. The per-trace loop (lines 505-549) is the bottleneck:

```
for each trace (797 iterations at APC 300):
    1. Check if d_block_sums needs realloc (line 529-531)
    2. Compute lambda_pows offset for this trace (line 535)
    3. Launch stacked_reduction_sumcheck_round0() kernel (line 537-548)
       → stacked_reduction_round0_block_sum_kernel: avg 28μs, 797 instances, 22ms total
       → final_reduce_block_sums<true>: avg 10μs, 797 instances, 8ms total
    4. Results accumulate into d_g_pos or d_g_neg[k] (non-atomic +=)
```

**Accumulation is NON-atomic**: `final_reduce_block_sums<true>` at `sumcheck.cuh:251` uses `output[out_idx] += sum` — a plain read-modify-write, not `atomicAdd`. This relies on sequential same-stream execution for correctness. Per-thread accumulation buffers are therefore **mandatory** for multi-streaming, with a post-join reduction step.

**Buffer sharing**: `self.d_block_sums` (line 106) is a shared scratch buffer resized on demand (line 529-531). Multi-threading requires per-thread scratch buffers.

**Trace data ownership**: `mem::take(&mut self.trace_ptrs)` at line 506 destructively moves `trace_ptrs` out of `self`. Since `trace_ptrs` is only used in this function (confirmed by grep), the work item builder should consume it via `mem::take` as now, then distribute the items to threads.

### `fold_ple_evals` (`stacked_reduction.rs:689-787`)

Folds prisma-linear evaluations after Round 0. The per-trace loop (lines 714-743):

```
for each stacked_per_commit group (2-3 groups):
    allocate folded_evals buffer, fill_zero
    for each trace in group (total ~797 across all groups):
        1. Compute dst_offset (running sum of new_height * trace.width())
        2. Launch stacked_reduction_fold_ple() kernel (line 730-739)
           → stacked_reduction_fold_ple_kernel: avg 11μs, 797 instances, 9ms total
```

**Data independence**: Each trace writes to a non-overlapping region of `folded_evals` (at `dst_offset`). No race conditions. The only dependency is computing the correct `dst_offset` per trace — this must be pre-computed.

### Nsys Profile Data (APC 300)

| Kernel | Time | Instances | Avg | Phase |
|--------|------|-----------|-----|-------|
| `stacked_reduction_round0_block_sum_kernel` | 22ms | 797 | 28μs | SR Round 0 |
| `final_reduce_block_sums<true>` | 8ms | 797 | 10μs | SR Round 0 |
| `stacked_reduction_fold_ple_kernel` | 9ms | 797 | 11μs | SR PLE fold |

Note: `fold_ple_from_evals_kernel` (9ms, 797 instances in nsys) is in `logup_zerocheck/utils.cu` — part of the LogUp zerocheck code path, NOT stacked reduction. It is NOT a target of this optimization.

Total sequential GPU time for targets: ~39ms. With ~6ms launch overhead (797 launches × ~8μs): ~45ms wall time. Multi-streamed (8 threads): ~6ms GPU + ~2ms overhead = ~8ms.

## Changes

### Change 1: Multi-stream `batch_sumcheck_uni_round0_poly`

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, function `batch_sumcheck_uni_round0_poly` (lines 481-556)

**What to change**:

1. **Pre-compute per-trace work items** (before the loop): Use `mem::take(&mut self.trace_ptrs)` as currently, then build a vector of work items containing `(trace_ptr: *const F, trace_height: usize, trace_width: usize, lambda_offset: usize, n_value: isize)`. Also compute `max_block_sums_len` across all traces by calling `_stacked_reduction_r0_required_temp_buffer_size` for each.

2. **Sort by descending height** for load balance (same pattern as Round 0 multi-stream).

3. **Determine multi-threading threshold**: Use `work_items.len() >= 100` (consistent with existing thresholds in Round 0 and GKR input eval).

4. **Pre-allocate per-thread resources**:
   - Per-thread `d_block_sums`: sized to `max_block_sums_len` (computed in step 1).
   - Per-thread `d_g_pos`: `NUM_G * skip_domain` EF elements, zero-filled before dispatch.
   - Per-thread `d_g_neg`: `l_skip` buffers each of `NUM_G * skip_domain` EF elements, zero-filled.

5. **Sync barrier**: `current_stream_sync()` before spawning threads (ensure prior GPU work — buffer zero-fills, eq_r_ns uploads — is visible to worker streams).

6. **Multi-threaded dispatch**: Use `std::thread::scope` with round-robin work assignment. Each work item is a struct containing raw `*const F` pointers (which are `Copy + Send`), sizes, and the lambda offset. Shared read-only device buffers (`eq_r_ns`, `d_lambda_pows`) are passed as raw pointers extracted before the spawn. Each thread:
   - Iterates its assigned work items
   - For each: launch `stacked_reduction_sumcheck_round0` into its per-thread `d_g_pos`/`d_g_neg[(-n-1)]` depending on `n_value`
   - After all traces: `current_stream_sync()`

7. **Reduce per-thread accumulators**: After threads join, for each of the `1 + l_skip` accumulation buffers (`d_g_pos` and each `d_g_neg[k]`): launch a simple elementwise-add kernel on the default stream that sums 8 per-thread buffers into the final buffer. This kernel has grid size `(NUM_G * skip_domain)` and each thread computes `out[i] = sum_t(thread_buf_t[i])`. Alternatively, use `N-1` pairwise `vector_add` calls (each adding one thread's buffer to the accumulator). The pairwise approach reuses existing infrastructure; the fused approach is faster but requires a new kernel. Start with pairwise for simplicity.

8. **Fallback**: If `work_items.len() < 100`, use the existing single-thread loop (modified to consume the pre-built work items instead of iterating `trace_ptrs` directly).

**Why**: Eliminates sequential execution of 797 kernel launches. The `stacked_reduction_sumcheck_round0` function internally handles the block sum + reduction for one trace. By running traces on different streams concurrently, small kernels (28μs average) can overlap on the GPU's 128 SMs.

### Change 2: Multi-stream `fold_ple_evals`

**File**: `crates/cuda-backend/src/stacked_reduction.rs`, function `fold_ple_evals` (lines 689-787)

**What to change**:

1. **Pre-compute trace offsets per group**: For each `stacked_per_commit` group, iterate traces to compute `(trace_ref, dst_offset, trace_height, trace_width)` work items. The `dst_offset` for each trace = running sum of `max(trace.height(), skip_domain) / skip_domain * trace.width()` from previous traces. This is pre-computed sequentially on the CPU (cheap — just arithmetic).

2. **Sync barrier**: After `folded_evals.fill_zero()` for each group, call `current_stream_sync()` to ensure the zero-fill is visible to worker streams.

3. **Multi-threaded dispatch per group**: For each `stacked_per_commit` group, if that group's trace count >= 100:
   - Extract `folded_evals.as_mut_ptr()` as a raw `*mut EF` (Send-safe)
   - Extract `d_omega_skip_pows` and `d_inv_lagrange_denoms` as raw `*const` pointers
   - Distribute work items across `NUM_STACKED_REDUCTION_STREAMS` threads via round-robin
   - Each thread launches `stacked_reduction_fold_ple` for its traces, writing to `folded_evals_ptr.add(dst_offset)`
   - No per-thread accumulation needed (non-overlapping writes)
   - Each thread calls `current_stream_sync()` at completion
   - If a group has < 100 traces, use the existing sequential loop for that group

4. **Keep post-processing unchanged**: The eq/k_rot computations after the per-commit loop (lines 747-786) are a few scalar operations on the default stream.

**Why**: Each trace's PLE fold writes to a non-overlapping region of the output buffer, making this embarrassingly parallel. The 797 sequential kernel launches (9ms GPU) become ~100 per-thread, completing in ~2ms.

### Change 3: Add thread count constant

**File**: `crates/cuda-backend/src/stacked_reduction.rs`

Add at module level:
```rust
/// Number of OS threads for parallel Stacked Reduction processing.
const NUM_STACKED_REDUCTION_STREAMS: usize = 8;
```

## Invariants

1. **Accumulation correctness (Round 0)**: `final_reduce_block_sums<true>` uses non-atomic `output[out_idx] += sum` (`sumcheck.cuh:251`). Per-thread accumulation buffers are mandatory. The sum of per-thread `d_g_pos` buffers must equal the sequential single-buffer result. This requires:
   - Each per-thread buffer is zero-initialized before dispatch
   - The reduction after threads join sums ALL per-thread buffers
   - The reduction is exact (BabyBear field arithmetic is associative and commutative)

2. **PLE fold offset correctness**: Pre-computed `dst_offset` values must match the sequential computation. Verified by: both use the same formula `max(trace.height(), skip_domain) / skip_domain * trace.width()`.

3. **Stream synchronization**: Worker stream kernels must see all prior default-stream GPU work (fill_zero, buffer uploads). Enforced by `current_stream_sync()` before spawn.

4. **No APC 0 regression**: At APC 0 (~99 AIR instances total), the per-group trace count will be below the 100 threshold, so both functions fall back to the existing sequential code path.

5. **Send safety**: Raw pointers (`*const F`, `*mut EF`, `*const EF`) are `Copy + Send`. Work items contain only raw pointers, sizes, and indices — no references to `&self`. Shared read-only device buffers are accessed via raw pointers extracted before `thread::scope`.

6. **Memory budget**: Per-thread buffers for Round 0:
   - d_g_pos: `NUM_G * skip_domain * sizeof(EF)` = `3 * 4096 * 8` = 96KB per thread
   - d_g_neg: same, × l_skip buckets (typically 12) = ~1.2MB per thread
   - d_block_sums: sized to max per trace, typically < 1MB per thread
   - Total: ~2.5MB × 8 threads = ~20MB. Negligible vs 24GB GPU memory.

## Measurement Plan

Run the prove command for APC {0, 100, 300}. Compare:
- Stacked Reduction time (from spec.py breakdown)
- STARK excl trace total
- Verify no regression at APC 0

### Expected results at APC 300:
- Stacked Reduction: 74ms → ~37ms (2x improvement)
- STARK excl trace: 1349ms → ~1312ms (-37ms, 1.03x improvement)
- Cumulative vs baseline: 1.87x (up from 1.82x)

### Expected results at APC 0:
- Stacked Reduction: 76ms → 76ms (unchanged, below threshold)
- STARK excl trace: 2138ms → 2138ms (unchanged)

### Verification commands:
```bash
cd /home/georg/powdr/results/pairing
# Run benchmark for each APC config
/home/georg/powdr/target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics current_apc300/metrics.json --recursion
python3 /home/georg/spec.py current_apc300/metrics.json current_apc300
```

## Rollback Criteria

- Stacked Reduction at APC 300 improves by **less than 15ms** (below measurement noise threshold for ~74ms phase)
- **OR** STARK excl trace at APC 300 improves by **less than 10ms**
- **OR** any regression at APC 0 exceeding **10ms** in STARK excl trace
- **OR** any correctness failure (proof verification fails)
