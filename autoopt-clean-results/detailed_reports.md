
# Detailed per-change reports

Each section gives the commit ref, a short description, the measured impact on this clean branch, and upstreaming notes. Deltas are at APC 300 against the immediately preceding step.

## 1. Multi-stream Round 0 (8 streams)

- **Commit:** `df618f95` (originally `06837bbe`, with the `4 -> 8` thread-count bump from original `04df94b0` folded in)
- **Files:** `crates/cuda-backend/src/logup_zerocheck/{mod.rs,round0.rs}`, `crates/cuda-common/src/copy.rs`
- **Diffstat:** +265/-140 (405 lines)
- **Measured:** APC 300 `2,307 -> 1,915 ms` (-392 ms, 1.20x — biggest single step). APC 100 `1,982 -> 1,745 ms` (-237 ms). APC 0 +17 ms (noise).
- **Dependencies introduced:** prerequisite for steps 7 and 8.

**What it does.** Round 0 processes 600+ AIRs, each launching kernels that use 1-4 of the RTX 4090's 128 SMs — >95% idle. This change restructures the loop into three phases and distributes Phase 2 across 8 OS threads, each using its own `cudaStreamPerThread`. Adds `Round0AirResult` / `Round0AirWorkItem` structs, a new `MemCopyD2HStreamSync` trait for mutex-free per-stream D2H copies, and a `>=100 AIRs` threshold to keep APC 0 on the sequential path (multi-threaded allocations on APC 0 polluted the VPMM pool for subsequent phases).

**Upstreaming considerations.**
- Second-largest diff in the set.
- Introduces a persistent pattern used by later changes (multi-stream + threshold + per-thread work items). Reviewers should focus on: thread safety of `SendPtr`-like share patterns, correctness of `to_host_on_current_stream`, and the `>=100` threshold's rationale (APC 0 regression without it).
- Memory budget is divided across worker threads (`memory_limit_bytes / NUM_ROUND0_STREAMS`) to keep peak usage within the original envelope.

## 2. Multi-stream GKR input eval (8 streams)

- **Commit:** `df0c54ed` (originally `0a811ac7`, with the `4 -> 8` thread-count bump folded in)
- **File:** `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
- **Diffstat:** +184/-108 (292 lines)
- **Measured:** APC 300 `1,915 -> 1,642 ms` (-273 ms, 1.17x). APC 100 `1,745 -> 1,739 ms` (-6 ms, noise). APC 0 +6 ms (noise). LogUp GKR phase `652 -> 386 ms` at APC 300.
- **Dependencies introduced:** prerequisite for step 6.

**What it does.** Applies the same multi-stream pattern to `log_gkr_input_evals()`. Adds `GkrInputWorkItem`, extracts the per-AIR inner loop into `process_gkr_input_air`, then uses `std::thread::scope` with 8 OS threads when AIR count `>= 100`. Requires a `SendPtr<T>` newtype for the stacked write-regions (different AIRs write non-overlapping offsets of the same DeviceBuffer — safe but invisible to the borrow checker).

**Upstreaming considerations.**
- Parallel structure to step 1; mostly review-once-review-twice.
- `SendPtr` safety rests on the stacked-layout invariant (non-overlapping write regions by unique `trace_idx`). A short comment at the struct would help reviewers.
- APC 100's -6 ms here is a measurement artifact; the underlying GKR-phase-level improvement at APC 100 is larger but partly offset by Round 0 drifting back up. The full effect is visible at APC 300.

## 3. Batch stacked-reduction MLE sync

- **Commit:** `d0c52334` (originally `d478fb18`)
- **Files:** `crates/cuda-backend/src/stacked_reduction.rs`, `cuda/stacked_reduction.rs`
- **Diffstat:** +22/-40 (62 lines; net -18 lines)
- **Measured:** APC 300 `1,642 -> 1,458 ms` (-184 ms, 1.13x). APC 100 `1,739 -> 1,489 ms` (-250 ms, 1.17x). APC 0 `1,818 -> 1,789 ms` (-29 ms). Stacked Reduction phase itself drops from 312 -> 125 ms at APC 300.

**What it does.** The per-window `to_host()` in `batch_sumcheck_poly_eval` forced a full GPU pipeline drain for every trace-matrix window of every MLE round (~7,000 drains/segment at APC 300). Since both kernels already accumulate atomically into a shared `d_accum`, the per-window syncs are unnecessary. Replaces them with one `fill_zero` + one `to_host` per MLE round (~17 drains/segment) and a one-shot upload of `eq_ub_per_trace` per round. The `d_eq_ub` buffer is resized from per-window to covering all columns up front.

**Upstreaming considerations.**
- Local change, no API or protocol impact.
- Actually reduces line count (-18 net).
- Best per-line payoff in the run: 62 lines for ~184 ms at APC 300.
- Independent of Round 0 / GKR multi-stream work (different files), so it could land before or after steps 1-2 without conflict. Placed here in the chain because its -184 ms slots between step 2's -273 and step 4's -142 cleanly.

## 4. Batch stacking scatter kernel

- **Commit:** `e50c8fb5` (originally `a8d81ec5`)
- **Files:** `crates/cuda-backend/cuda/src/matrix.cu`, `cuda/matrix.rs`, `stacked_pcs.rs`, `error.rs`
- **Diffstat:** +83/-32 (115 lines)
- **Measured:** APC 300 `1,458 -> 1,316 ms` (-142 ms, 1.11x). APC 100 `1,489 -> 1,407 ms` (-82 ms). APC 0 flat.

**What it does.** `stack_traces_into_expanded()` issued one `cudaMemcpyAsync` or `batch_expand_pad_wide` launch per trace column (~53K per segment). The rewrite is a two-phase approach: build a `Vec<StackColDesc>` on the CPU (pure pointer arithmetic, no CUDA calls), then do one H2D upload + one `stack_columns_kernel` launch. `nsys` confirms D2D memcpy count drops from 101K to 325 per run.

**Upstreaming considerations.**
- Adds a new CUDA kernel + tiny FFI wrapper; the Rust-side call site is simpler than what it replaces.
- Zero impact on APC 0 (only ~4K columns — the CPU-side overhead was already negligible there).
- Most of the gain is Trace Commit, which was the phase this change targeted.
- Independent of every other step; could be moved anywhere in the chain.

## 5. Batch stacked-reduction MLE round kernels

- **Commit:** `dfbd71b5` (originally `13bd3e96`, iteration 9)
- **Files:** `crates/cuda-backend/cuda/src/stacked_reduction.cu`, `cuda/stacked_reduction.rs`, `src/stacked_reduction.rs`
- **Diffstat:** +456/-49 (505 lines; 239 CUDA + 139 FFI + 127 caller)
- **Measured:** APC 300 `1,316 -> 1,250 ms` (-66 ms, 1.05x). APC 100 `1,407 -> 1,396 ms` (-11 ms). APC 0 -12 ms. Stacked Reduction phase itself `125 -> 76 ms` (-49 ms, 1.64x).
- **Depends on:** step 3 (same files).

**What it does.** `batch_sumcheck_poly_eval` previously launched ~15K per-AIR kernels per proof (~10K degenerate + ~5K non-degenerate) with only ~62 ms GPU compute vs ~115 ms wall time — launch overhead was the majority. This replaces the per-AIR launches with two batched descriptor-array kernels per MLE round:

- `batched_degenerate_mle_round_kernel`: one CUDA block per descriptor, fixed 256-thread block.
- `batched_nondegen_mle_round_kernel`: uses binary search on a prefix-sum array so each block can find its owning descriptor in O(log N). Preserves the original 2D grid layout via local block-index decomposition.

A `compute_mle_launch_params` Rust helper replicates the CUDA-side auto-tuning heuristic for the stride computation, so the batched Rust caller builds descriptors consistent with the original kernels.

**Notes.** Complements step 3 rather than superseding it. Step 3 eliminated per-window syncs *between* rounds; step 5 eliminates per-AIR launches *within* each round. Both are needed to get from ~312 -> ~76 ms on Stacked Reduction.

**Upstreaming considerations.**
- Largest diff among steps 1–7 (505 lines, mostly CUDA), but mechanically straightforward: the batched kernels are line-by-line equivalents of the per-AIR kernels, just parameterized by descriptor index.
- Descriptor arrays are small (~24 bytes × ~620 AIRs ≈ 15 KB per round) — upload cost negligible.
- Adds an assumption that kernel behavior is independent across descriptors, which the original already satisfied (atomic accumulation into shared `d_accum`). Reviewers should double-check the binary-search block mapping in the non-degenerate kernel — that's the one novel piece.

## 6. Pre-allocate per-thread GKR input buffers

- **Commit:** `55f1a588` (originally `0cdd5aea`)
- **File:** `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`
- **Diffstat:** +93/-36 (129 lines)
- **Measured:** APC 300 `1,250 -> 1,223 ms` (-27 ms, 1.02x). APC 100 `1,396 -> 1,373 ms` (-23 ms). APC 0 +19 ms (noise, APC 0 stays on single-threaded path).
- **Depends on:** step 2.

**What it does.** Pre-computes the max `intermediates`, `public_values`, `partition_ptrs`, and `tmp` buffer sizes across AIRs, then allocates one set of max-sized buffers per worker thread before spawning. Workers reuse via `copy_to` (H2D into existing memory) instead of alloc/free per AIR. Adds a 2 GB memory budget safety valve that halves `num_threads` if pre-allocation would exceed it. Eliminates ~1,240 `MemoryManager` mutex acquisitions / segment and the implicit CUDA pool synchronization across streams.

**Note on the measured delta.** The original run reported this change as -217 ms at APC 300 because it landed before the thread-count bump (then at 4 streams). Here the bump is folded into step 2, so by step 6 most of the mutex-contention headroom is already gone — the 27 ms here is the residual win.

**Upstreaming considerations.**
- Moderate diff, localized to one file.
- Memory budget safety valve avoids surprise OOMs on GPUs where max buffer size is large.
- Arguably mergeable with step 2 (same file, same pattern) if a reviewer prefers a single "multi-stream GKR input eval with pre-allocation" change.

## 7. Pre-allocate per-thread Round 0 buffers + round-robin work balance

- **Commit:** `8edd72e2` (bundles originals `6b940845` and `e6f614f2`, iterations 15 and 17)
- **Files:** `crates/cuda-backend/src/logup_zerocheck/{mod.rs,round0.rs}`, `crates/cuda-backend/src/pkey.rs`
- **Diffstat:** +343/-42 (385 lines total)
- **Measured:** APC 300 `1,223 -> 1,213 ms` (-10 ms, 1.01x). APC 100 `1,373 -> 1,347 ms` (-26 ms, 1.02x). APC 0 flat.
- **Depends on:** step 1. Prerequisite for step 8.

**What it does.** Two tightly coupled Round 0 dispatcher improvements:

1. **Round-robin dispatch** (iteration 15). Replaces `work_items.chunks(chunk_size)` with `work_items[idx]` where `idx % num_threads` chooses the thread. On the height-sorted list this gives each thread a balanced mix of large and small AIRs. The original plan's "LPT using height as cost" was abandoned after instrumentation showed height is a poor cost proxy (per-AIR fixed overhead ~1.8 ms dominates for small AIRs).

2. **Pre-allocated per-thread buffers with p95 threshold** (iteration 17). Adds `Round0ThreadBuffers` and pre-allocates `zc_intermediates`, `zc_temp_sums`, `logup_intermediates`, `logup_temp_sums` sized to the 95th percentile of per-AIR requirements (~3 MB/thread) rather than the max (~193 MB/thread). ~95% of AIRs reuse the pre-allocated buffers; the remainder falls back to dynamic allocation. A new `logup_round0_buffer_size` field on `AirDataGpu` lets Phase 1 compute buffer sizes without reconstructing the interaction DAG per AIR.

**Note on the measured delta.** The APC-300 delta (-10 ms) is smaller than the APC-100 delta (-26 ms) because by this point in the chain the remaining APC-300 Round 0 time is already low and the big wins come from step 8's GPU extraction. This step still earns its place because it is a hard prerequisite for step 8 and delivers a solid improvement at APC 100.

**Upstreaming considerations.**
- Prerequisite for step 8: iteration 21 is written against the `Round0ThreadBuffers` worker signature introduced here.
- The `p95` decision is the non-obvious bit. The original run tried `max`-sized prealloc first and saw +350 ms STARK at APC 300 from GPU memory pressure. The keygen pre-computation of `logup_round0_buffer_size` is also important — recomputing the DAG per AIR at runtime would cost ~5-10 ms at APC 300.
- Bundling iterations 15 and 17 is optional if a reviewer prefers them split. Splitting is mechanical: apply `6b940845` first, then the rest. This commit keeps them together because `e6f614f2` does not merge without `6b940845`.

## 8. GPU-side Round 0 polynomial extraction + overlap logup precompute

- **Commit:** `1ca18279` (bundles originals `423044cb` and `21c878af`, iterations 21 and 22)
- **Files:** `crates/cuda-backend/cuda/src/logup_zerocheck/round0_extract.cu` (new), `src/cuda/logup_zerocheck.rs`, `src/logup_zerocheck/mod.rs`
- **Diffstat:** +466/-134 (600 lines; 99 CUDA + 19 FFI + 482 caller)
- **Measured:** APC 300 `1,213 -> 1,148 ms` (-65 ms, 1.06x). APC 100 `1,347 -> 1,331 ms` (-16 ms). APC 0 `1,804 -> 1,792 ms` (-12 ms — recovers APC 0 drift from earlier steps). Round 0 phase `238 -> 180 ms` at APC 300.
- **Depends on:** steps 1 and 7.

**What it does.** Two changes landed together because they cannot be measured cleanly apart:

1. **GPU-side Round 0 polynomial extraction** (iteration 21). For every Round 0 AIR the original code did `cudaStreamSynchronize` to read the evaluation results, then ran CPU post-processing (transpose + iDFT + unshift + Lagrange interpolation + coefficient adjustment) before moving to the next AIR — ~1,246 pipeline drains per proof at APC 300. This change moves the entire post-processing onto the GPU via a pre-computed transformation matrix:
   - `compute_round0_extract_tables(d, l_skip)` runs `UnivariatePoly::from_geometric_cosets_evals_idft` on unit basis vectors at setup to capture the full CPU pipeline as a small matrix per unique degree.
   - `round0_extract_zerocheck_kernel` and `round0_extract_logup_kernel` apply the matrix to the per-AIR evaluation output, writing polynomial coefficients directly into a shared `d_batch_array` on device.
   - `process_air_round0` now launches the extract kernel instead of syncing + post-processing. A single D2H of `d_batch_array` at the end of the loop replaces the ~1,246 per-AIR drains. `Round0AirResult` goes away.

2. **Overlap logup combination precompute with Round 0** (iteration 22). The `d_eq_3b` upload + `compute_logup_combinations` loop used to run sequentially on the default stream before Round 0, but its output is only consumed during MLE rounds, not by Round 0. Moves it to a background thread inside the existing `thread::scope` in the multi-threaded path, running concurrently with Round 0 worker threads on its own `cudaStreamPerThread`. Round 0 workers receive `batch_ptr_val` (a `usize` copy of `d_batch_array.as_mut_ptr()`) so they can write evaluation results straight into the shared batch buffer while the precompute runs in parallel. The single-threaded path (APC 0) runs the precompute sequentially as before.

**Upstreaming considerations.**
- Largest single commit in the set (600 lines). The new CUDA file is 99 lines and does a dense matrix-vector multiply; easy to review.
- The pre-computed transform matrix is small (a few hundred KB total) and depends only on the constraint degree `d` and `l_skip`, which are fixed per keygen — no per-proof setup cost.
- Correctness note for the overlap-logup half: the scope joins Round 0 workers first, then the precompute thread, so there's no UAF on the returned `DeviceBuffer`s. The background thread calls `current_stream_sync()` before returning so the buffers are globally visible when assigned into `self` after the scope exits.
- Reviewers should check: batch-array offset accounting in `Round0AirWorkItem` (three offsets for zerocheck / numer / denom), the reconstruction step in `batch_sp_poly` that reads those offsets after the D2H, and the handoff of the precompute thread's returned `d_eq_3b_per_trace` / `logup_combinations` into `self`.
