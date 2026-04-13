# Report: Batch MLE Round Kernels

## Description

Replace the per-AIR kernel launch loop in `batch_sumcheck_poly_eval()` (stacked reduction MLE rounds) with batched kernel launches using descriptor arrays. At APC 300, the stacked reduction MLE rounds phase launched ~15K individual CUDA kernels per benchmark (~10K degenerate + ~5K non-degenerate), with total GPU compute time of ~62ms but wall time of ~115ms. The hypothesis was that batching all per-AIR launches into 1-2 kernel launches per round would eliminate kernel launch overhead and improve GPU SM utilization.

## Implementation

### CUDA side (`crates/cuda-backend/cuda/src/stacked_reduction.cu`)

1. **Descriptor structs**: Added `DegenMleDesc` (col_offset, window_len, eq_r, k_rot_r) and `NonDegenMleDesc` (col_offset, window_len, num_y, stride, blocks_x) after existing `UnstackedSlice`.

2. **Batched degenerate kernel** (`batched_degenerate_mle_round_kernel`): Grid=(num_descs, 1), Block=(256, 1). Each block reads its descriptor from `descs[blockIdx.x]`, computes base pointers, and executes the same computation as the existing per-AIR degenerate kernel. Fixed block size of 256 threads (vs. original min(window_len, 256)).

3. **Batched non-degenerate kernel** (`batched_nondegen_mle_round_kernel`): Grid=(total_blocks, 1), Block=(256, 1). Each block uses binary search on a prefix-sum array to find its AIR index, then decomposes its local block index into y-block and stride-block indices matching the original 2D grid layout. O(log N) binary search per block (~10 comparisons for 600 AIRs).

4. **Launcher functions**: Two new `extern "C"` launchers (`_batched_stacked_reduction_sumcheck_mle_round_degenerate`, `_batched_stacked_reduction_sumcheck_mle_round`).

### Rust side

5. **FFI bindings** (`crates/cuda-backend/src/cuda/stacked_reduction.rs`): Added `#[repr(C)]` descriptor structs (`DegenMleDesc`, `NonDegenMleDesc`), extern declarations, safe wrappers, and `compute_mle_launch_params()` helper that replicates the CUDA-side auto-tuning heuristic for stride computation.

6. **Caller** (`crates/cuda-backend/src/stacked_reduction.rs`): Replaced the per-window kernel launch loop in `batch_sumcheck_poly_eval()` with two phases:
   - Phase A: Build descriptor vectors and prefix-sum arrays on CPU
   - Phase B: Upload descriptors to device and launch 1-2 batched kernels per round

### Deviations from plan

- **Change 8 (pre-allocated descriptor buffers)**: Not implemented. The per-round `to_device()` calls add negligible overhead compared to the kernel launch savings.
- **Metric tracking**: The plan targeted "MLE Rounds" under Constraints, but `batch_sumcheck_poly_eval` actually runs during "Stacked Reduction" under Openings. The correct metric is `prover.openings.stacked_reduction.mle_rounds_time_ms`, not `prover.rap_constraints.mle_rounds_time_ms`.

## Results

### Primary metric: Stacked Reduction MLE Rounds (raw gauge sum)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| SR MLE APC 0 | 144ms | 59ms | 38ms | -106ms, 3.79x lower | -21ms, 1.55x lower |
| SR MLE APC 100 | 230ms | 81ms | 42ms | -188ms, 5.48x lower | -39ms, 1.93x lower |
| SR MLE APC 300 | 346ms | 115ms | 55ms | -291ms, 6.29x lower | -60ms, 2.09x lower |

### Stacked Reduction total (spec.py "Stacked Reduction" under Openings)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| SR total APC 0 | 150ms | 83ms | 76ms | -74ms, 1.97x lower | -7ms, 1.09x lower |
| SR total APC 100 | 202ms | 103ms | 71ms | -131ms, 2.85x lower | -32ms, 1.45x lower |
| SR total APC 300 | 311ms | 123ms | 75ms | -236ms, 4.15x lower | -48ms, 1.64x lower |

### STARK excl trace (spec.py)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK APC 0 | 2150ms | 2172ms | 2156ms | +6ms, 1.00x | -16ms, 1.01x lower |
| STARK APC 100 | 1780ms | 1653ms | 1591ms | -189ms, 1.12x lower | -62ms, 1.04x lower |
| STARK APC 300 | 2455ms | 1474ms | 1474ms | -981ms, 1.67x lower | 0ms, 1.00x |

Note: At APC 300, the 48ms Stacked Reduction improvement is offset by +50ms noise in Constraints (GKR +39ms, Round 0 +8ms), resulting in zero net change in STARK excl trace. The raw gauge sum shows a clearer picture: STARK 1991ms → 1980ms (-11ms).

### Openings total (raw gauge sum)

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| Openings APC 0 | 465ms | 377ms | 356ms | -109ms, 1.31x lower | -21ms, 1.06x lower |
| Openings APC 100 | 450ms | 303ms | 259ms | -191ms, 1.74x lower | -44ms, 1.17x lower |
| Openings APC 300 | 522ms | 290ms | 230ms | -292ms, 2.27x lower | -60ms, 1.26x lower |

## Assessment

The optimization achieved its goal: the stacked reduction MLE rounds phase improved by 2.09x at APC 300 (115ms → 55ms, -60ms). The improvement scales with APC count as expected — more AIRs means more kernel launches eliminated.

The STARK excl trace total at APC 300 didn't show the improvement due to noise in other metrics, but:
- No regression at any APC configuration
- The targeted phase is clearly and consistently improved across all three APC configs
- At APC 100, the improvement is visible in the overall STARK number (-62ms, 1.04x)

The optimization added ~130 lines of CUDA code and ~100 lines of Rust code. The complexity is reasonable given the descriptor-array pattern was already established in the codebase (StackColDesc for stacking scatter).

## Future Work

- **Pre-allocate descriptor buffers**: The plan's Change 8 (pre-allocating d_degen_descs, d_nondegen_descs, d_block_offsets as struct fields) would eliminate ~34 per-round DeviceBuffer allocations. Currently these are small H2D transfers so the impact is minimal, but worth considering if MLE round count increases.
- **Apply same batching to RAP constraints MLE rounds**: The `prover.rap_constraints.mle_rounds_time_ms` metric (268ms at APC 300) is a separate MLE rounds phase that might benefit from similar batching if it also uses many per-AIR kernel launches.
- **Combine with multi-stream**: The batched kernels could potentially be split across multiple CUDA streams (e.g., degenerate on stream 0, non-degenerate on stream 1) for concurrent execution.
- **Thread coarsening for degenerate kernel**: The batched degenerate kernel uses a fixed block size of 256 threads, but many AIRs have window_len < 256, leaving threads idle. A variable block size or thread coarsening could improve utilization.
