# Report: Batched GLOBAL-Mode GKR Input Evaluation

## Description

Replace per-AIR GLOBAL-mode GKR input evaluation kernel launches with a single batched descriptor-array kernel for small AIRs (height <= TASK_SIZE = 65536). At APC 300, ~211-224 GLOBAL AIRs per segment each launch a 256-block kernel with TASK_SIZE threads regardless of actual height, wasting 90-99% of threads for small AIRs and serializing execution across 8 streams. The batched kernel was expected to use GPU-native block scheduling instead of coarse-grained stream scheduling, eliminating ~300 kernel launch overheads per segment and saving 20-50ms from LogUp GKR.

## Implementation

### CUDA Kernel (`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`)

Added `BatchGkrInputDesc` struct containing per-AIR pointers (d_fracs, d_preprocessed, d_main, d_public_values, d_rules, d_used_nodes, d_pair_idxs) and metadata (permutation_height, buffer_size, intermediates_offset, used_nodes_len). Added `BatchBlockCtx` struct (local_block_idx, air_idx). Implemented `batched_gkr_input_eval_kernel` using the same DAG evaluation logic as the existing per-AIR kernel, with per-AIR intermediates regions in a shared buffer.

### FFI Bindings (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)

Added `BatchGkrInputDesc` and `BatchGkrInputBlockCtx` repr(C) structs. Added `_batched_gkr_input_eval` extern C function and safe `batched_gkr_input_eval` wrapper.

### Rust Orchestration (`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`)

Modified `log_gkr_input_evals()` to:
1. Partition work items into batch-eligible (GLOBAL, height <= TASK_SIZE) and non-eligible
2. Build flat device buffers for partition_ptrs and public_values (single H2D per buffer)
3. Build descriptor and block-context arrays, allocate shared intermediates buffer
4. Memory budget check: cap batch intermediates at 64 MB (L2 cache size) — exceeded = fallback
5. Launch batched kernel on main thread's stream, concurrent with multi-stream workers
6. Post-kernel height normalization for batch-eligible AIRs needing it

### Key deviation from plan

The plan specified stride=1 intermediates layout for "better L1 cache locality." Testing revealed this was **incorrect** — stride=1 gives poor warp coalescing (adjacent threads access memory `buffer_size * 16` bytes apart). Changed to stride=`permutation_height` (same interleaved layout as the original per-AIR kernel) which gives perfect coalescing. However, even with corrected coalescing, the optimization still failed due to the total intermediates size exceeding L2 cache.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1292ms | 1291ms | -1164ms, 1.90x lower | -1ms, 1.00x |
| LogUp GKR (APC 300) | 790ms | 528ms | 525ms | -265ms, 1.50x lower | -3ms, 1.01x |
| Round 0 (APC 300) | 662ms | 181ms | 180ms | -482ms, 3.68x lower | -1ms, 1.01x |
| MLE Rounds (APC 300) | 180ms | 159ms | 160ms | -20ms, 1.13x lower | +1ms, 1.01x |
| STARK excl trace (APC 0) | 2153ms | 2147ms | 2160ms | -7ms, 1.00x | +13ms, 1.01x |
| LogUp GKR (APC 0) | 993ms | 1023ms | 1034ms | +41ms, 1.04x | +11ms, 1.01x |

All differences are within measurement noise. No improvement, no regression.

### Intermediate results (before 64 MB cap)

Without the L2 budget cap, the batched kernel caused a **massive regression**:

| Intermediates layout | LogUp GKR APC 300 | STARK excl trace APC 300 |
|---------------------|-------------------|-------------------------|
| Before (per-AIR) | 528ms | 1292ms |
| Stride=1 (bad coalescing) | 944ms (+79%) | 1718ms (+33%) |
| Stride=height (good coalescing) | 694ms (+31%) | 1466ms (+13%) |
| With 64 MB cap (fallback) | 525ms (no change) | 1291ms (no change) |

## Assessment

**Failure.** The optimization does not improve performance. The 64 MB L2 cache budget cap causes the batched path to be disabled for all significant workloads (APC 300 needs 443-576 MB of intermediates across 211-224 batch-eligible AIRs).

### Root cause analysis

The plan identified ~300 kernel launches per segment as overhead. However, the intermediates memory requirement is the binding constraint:

1. **Per-AIR intermediates volume**: Each batch-eligible GLOBAL AIR needs `height * buffer_size` EF elements (16 bytes each) in a unique region. With ~211 AIRs, the total is 576 MB for segment 0 — **8x the 72 MB L2 cache** on RTX 4090.

2. **Per-stream intermediates reuse**: The existing 8-stream approach allocates `TASK_SIZE * max_buffer_size` per thread (~40 MB). Each stream processes AIRs sequentially, reusing the same buffer. This means the intermediates are **hot in L2 cache** between AIR evaluations, because only 1 AIR's worth of intermediates is live at a time per stream.

3. **Batching destroys reuse**: The batched kernel needs ALL AIRs' intermediates live simultaneously (one GPU kernel = one launch = all blocks run). This causes massive L2 cache thrashing, overwhelming any savings from reduced kernel launch overhead.

4. **Coalescing matters but isn't sufficient**: The initial stride=1 layout caused a 79% regression due to poor warp coalescing. Fixing to stride=height reduced this to 31% regression. But even with perfect coalescing, the L2 overflow alone makes batching slower.

5. **Kernel launch overhead is small**: The ~300 launches per segment at ~5us each = ~1.5ms. Even eliminating all of it doesn't move the needle vs the 528ms total LogUp GKR.

## Future Work

- **The per-AIR multi-stream approach is near-optimal for GLOBAL GKR input eval.** The intermediates buffer reuse pattern (sequential per-AIR on each stream) is fundamentally better than any batched approach when `sum(height * buffer_size)` exceeds L2 cache.
- **Further GKR improvement needs algorithmic changes**, not kernel launch optimization. The dominant cost is memory bandwidth for trace reads (main + preprocessed), which is proportional to total rows and can't be reduced by batching.
- **Batching only works when the aggregate working set fits in L2.** The SCATTER batching (prior task) also showed minimal improvement — SCATTER AIRs are hidden behind GLOBAL timing anyway.
- **The descriptor-array CUDA infrastructure is correct and tested.** If a future optimization reduces per-AIR intermediates requirements (e.g., via shared DAG evaluation or reduced buffer_size), the batched kernel could be re-enabled.
