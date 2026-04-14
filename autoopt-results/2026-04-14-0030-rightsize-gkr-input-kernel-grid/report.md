# Report: Right-size GKR input eval kernel grid for small AIRs

## Description

The GKR input evaluation CUDA launcher for `evaluate_interactions_gkr_kernel<GLOBAL=true>` always uses `TASK_SIZE=65536` as the thread count, launching 256 blocks regardless of the AIR's actual `permutation_height`. For the majority of AIRs at APC 300 (height << 65536), over 98% of threads exit at the boundary check, yet each 256-block kernel saturates the RTX 4090's 128 SMs, preventing concurrent kernel execution from the 8 multi-stream worker threads.

The hypothesis was that changing the launcher to use `min(TASK_SIZE, permutation_height)` would allow small-AIR kernels to use proportionally fewer blocks (e.g., 4 blocks for height=1024 instead of 256), freeing SMs for concurrent kernel execution from other streams.

## Implementation

**Single code change** in `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`, line 231:

Before:
```cpp
auto count = is_global ? TASK_SIZE : permutation_height;
```

After:
```cpp
auto count = is_global ? min((uint32_t)TASK_SIZE, permutation_height) : permutation_height;
```

No deviations from the plan. The Rust-side pre-allocation was intentionally left unchanged (at least one AIR has height >= TASK_SIZE, so the pre-allocated maximum stays the same).

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace APC 300 | 2455 ms | 1364 ms | 1358 ms | -1097 ms, 1.81x lower | -6 ms, 1.00x (noise) |
| STARK excl trace APC 100 | 2155 ms | 1746 ms | 1745 ms | -410 ms, 1.23x lower | -1 ms, 1.00x (noise) |
| STARK excl trace APC 0 | 2153 ms | 2138 ms | 2140 ms | -13 ms, 1.01x lower | +2 ms, 1.00x (noise) |
| LogUp GKR APC 300 | 790 ms | 522 ms | 517 ms | -273 ms, 1.53x lower | -5 ms, 1.01x (noise) |
| GKR input evals seg0 APC 300 | ~241 ms* | 242 ms | 238-254 ms | ~0 ms | -4 to +12 ms (noise) |
| GKR input evals seg1 APC 300 | ~62 ms* | 62 ms | 61-62 ms | ~0 ms | -1 to 0 ms (noise) |

*Baseline per-segment values not available; using pre-optimization task values as reference.

Second APC 300 run: STARK excl trace = 1383 ms, confirming the ~1358 ms first run was slightly lucky rather than a real improvement.

## Assessment

**The optimization did not improve performance.** The STARK excl trace improvement at APC 300 was -6 ms in the first run and +19 ms in the second, both well within measurement noise. This is far below the plan's rollback threshold of 30 ms.

The hypothesis — that SM saturation from oversized GLOBAL-mode kernel grids was preventing multi-stream concurrency — appears to be wrong, or at least not the primary bottleneck. Possible explanations:

1. **Large AIRs dominate wall time**: A few AIRs with height >= TASK_SIZE have unchanged grid sizes. If these large AIRs account for most of the wall time on each thread, reducing the grid for small AIRs doesn't help — the thread still blocks on the large kernel.

2. **Memory bandwidth, not SM count, is the bottleneck**: Even with fewer blocks, the GPU may be memory-bandwidth-limited rather than compute-limited. The GKR input eval kernel has low arithmetic intensity (reading partition data, evaluating DAG nodes), so freeing SMs doesn't help if memory bandwidth is already saturated.

3. **CUDA runtime serialization**: The CUDA runtime's kernel launch and scheduling overhead may serialize submissions from 8 OS threads regardless of grid sizes. The `cudaStreamPerThread` approach may not achieve true hardware-level concurrency.

4. **Already achieving reasonable concurrency**: The previous pre-allocation optimization eliminated mutex contention and allocation serialization, possibly already achieving near-optimal concurrency. The remaining ~240 ms seg0 time may be close to the kernel-execution floor.

## Future Work

- **Profile with nsys** to verify whether inter-stream kernel overlap actually changed. Compare total `evaluate_interactions_gkr_kernel<true>` block counts and overlap ratios before vs after.
- **Focus on the large-AIR tail**: The few AIRs with height >= TASK_SIZE likely dominate per-thread wall time. Splitting these across threads (sub-tiling) or using more efficient kernel designs for large AIRs could be more impactful.
- **Memory bandwidth analysis**: Use nsys memory throughput metrics to determine whether GKR input eval is compute-bound or memory-bound. If memory-bound, SM-level concurrency improvements won't help.
- **Kernel fusion**: Combining multiple small-AIR kernels into a single batched kernel (descriptor-array approach) could reduce kernel launch overhead and improve occupancy efficiency more than grid-sizing alone.
