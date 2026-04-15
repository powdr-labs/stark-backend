# Report: Warp-Tiled GKR Intermediates Buffer Layout

## Description

Reorganize the GKR input evaluation GLOBAL-mode intermediates buffer from a thread-interleaved layout (stride = 65536) to a warp-tiled layout (stride = 32) to improve GPU cache locality. The hypothesis was that the `evaluate_interactions_gkr_kernel<true>` kernel (the single largest GPU kernel at ~458ms, 14.6% of total GPU time at APC 300) suffers from systematic TLB misses and defeats hardware prefetching because consecutive DAG node accesses within a thread jump 65536 × 16 = 1MB apart. With warp-tiled layout, per-warp working set drops from ~100MB scattered to ~50KB contiguous, enabling hardware prefetch (stride 512B vs 1MB), reducing TLB pressure by ~8x, and improving L2 sector reuse.

## Implementation

Four files were modified, following the plan exactly with no deviations:

1. **`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`** (kernel + launcher):
   - Added `buffer_size` parameter to `evaluate_interactions_gkr_kernel` template and `_logup_gkr_input_eval` launcher
   - Changed GLOBAL intermediates pointer setup from `base + task_offset` with `stride = task_stride (65536)` to `base + warp_id * buffer_size * WARP_SIZE + lane_id` with `stride = WARP_SIZE (32)`
   - Used existing `WARP_SIZE` constant from `launcher.cuh`

2. **`crates/cuda-backend/src/cuda/logup_zerocheck.rs`** (FFI binding):
   - Added `buffer_size: u32` to `_logup_gkr_input_eval` extern declaration and `logup_gkr_input_eval` safe wrapper

3. **`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`** (call site):
   - Passed `buffer_size as u32` (from `rules.inner.buffer_size`) to the `logup_gkr_input_eval` call

The total allocation size is unchanged (TASK_SIZE * buffer_size elements). Only the mapping from (warp, node, lane) to memory address changed.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1314ms | 1317ms | -1138ms, 1.86x lower | +3ms, within noise |
| LogUp GKR | 790ms | 529ms | 534ms | -256ms, 1.48x lower | +5ms, within noise |
| Round 0 | 662ms | 180ms | 182ms | -480ms, 3.64x lower | +2ms, within noise |
| MLE Rounds | 180ms | 174ms | 172ms | -8ms, 1.05x lower | -2ms, within noise |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2128ms | 2158ms | +5ms, within noise | +30ms, within noise |
| LogUp GKR | 993ms | 1002ms | 1022ms | +29ms, within noise | +20ms, within noise |

### nsight Kernel-Level (APC 300)

| Kernel | Before (plan baseline) | After |
|--------|----------------------|-------|
| `evaluate_interactions_gkr_kernel<true>` total GPU time | ~464ms | 458ms |
| Instances | 486 | 486 |
| Median per-instance | ~742μs | ~742μs |

Three APC 300 runs with the optimization: 1302ms, 1311ms, 1317ms (median 1311ms). Before: 1314ms. The difference is within measurement noise.

## Assessment

**Result: failure** — The warp-tiled layout produced no measurable improvement at APC 300 or APC 0. The nsight profile confirms the kernel total GPU time is unchanged (458ms vs ~464ms baseline).

The hypothesis that TLB misses and defeated hardware prefetch were the dominant bottleneck appears to be incorrect. Possible explanations:

1. **Memory bandwidth, not latency, is the bottleneck.** The kernel's main bottleneck may be total memory bandwidth consumption (reading main/preprocessed trace data for each row), not the intermediates access pattern. Even with perfect intermediates caching, the kernel still needs to read column data from global memory for every DAG source operand.

2. **Intermediates accesses are a small fraction of total memory traffic.** With ~100 DAG nodes per interaction set but each node potentially reading 1-2 main trace values (column-major, height-stride), the intermediates reads/writes may be only 10-20% of total memory traffic. Improving their locality by 2-8x means only 1-2% total traffic reduction.

3. **GPU L2 already handles the old layout adequately.** The 72MB L2 cache on the RTX 4090 may absorb enough of the intermediates traffic even at stride-65536, especially since all 32 warp lanes coalesce into a single cache line per node access. The old layout was already coalesced for warp-level access; only per-thread temporal reuse was poor.

4. **Hardware prefetcher may not trigger on either pattern.** The GPU prefetcher may not activate for this access pattern regardless of stride, since the kernel alternates between intermediates reads and main/preprocessed column lookups with complex control flow (DAG evaluation with branches).

## Future Work

- The GKR input eval kernel is now memory-bandwidth-bound on main/preprocessed trace reads, not intermediates locality. Future improvements should target:
  - **Reducing redundant trace reads** — if the same column cell is read by multiple DAG nodes in one interaction, caching it in registers
  - **Fusing multiple interaction sets per kernel launch** — batching AIRs to amortize kernel launch overhead and improve occupancy
- The warp-tiled layout is not harmful (no regression) and is arguably a cleaner memory layout. It could be kept for code clarity, but since it provides no measurable benefit, reverting avoids unnecessary complexity.
- The kernel's 458ms total GPU time at APC 300 across 486 instances suggests ~943μs average per instance. Given the multi-stream parallelism (8 threads), wall-clock GKR input eval time is dominated by the critical stream, not total GPU time. Further reducing per-instance time requires algorithmic changes, not memory layout tweaks.
