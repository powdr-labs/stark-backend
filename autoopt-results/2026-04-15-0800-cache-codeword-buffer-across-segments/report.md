# Report: cache-codeword-buffer-across-segments

## Description

The optimization aimed to eliminate the cold-start allocation overhead in the first segment's `rs_code_matrix` by pre-warming the GPU memory pool (VPMM) during `GpuDevice::new()`. Per-segment metrics show `rs_code_matrix_time_ms` = 58ms for seg 0 vs 0ms for seg 1 at APC 300 — the entire gap is the first-time VPMM page creation (128 pages × 2MB via `cuMemCreate + cuMemMap`) for the 192MB codeword buffer. By allocating and immediately freeing a 256MB `DeviceBuffer` at device initialization, the VPMM pool gets pre-warmed pages ready for reuse when the first segment's `rs_code_matrix` needs them.

## Implementation

**Changed file**: `crates/cuda-backend/src/device.rs`

Added a 256MB `DeviceBuffer<u32>` allocation and immediate free in `GpuDevice::new()`, after `ensure_device_ntt_twiddles_initialized()` and before prover config construction. The buffer is scoped so it drops immediately, returning the VPMM pages to the pool's free list.

Also added `POOL_WARMUP_BYTES` constant (256 × 1024 × 1024) and imported `DeviceBuffer` from `openvm_cuda_common`.

**Alternative attempted**: Modified `VpmmConfig` to default to 128 `initial_pages` (pre-allocating at pool construction time via `#[ctor::ctor]`). This barely helped (`rs_code_matrix` seg 0: 58ms → 51ms vs 14ms with DeviceBuffer approach) because the pre-allocated pages get consumed by intermediate allocations between program start and proving time. Reverted.

**No deviations from plan** for the primary approach.

## Results

### APC 300

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2455ms | 1292ms | 1250ms | -1205ms, 1.96x lower | -42ms, 1.03x lower |
| Trace Commit | 406ms | 255ms | 211ms | -195ms, 1.92x lower | -44ms, 1.21x lower |
| rs_code_matrix seg 0 | — | 58ms | 14ms | — | -44ms, 4.14x lower |
| stacked_commit seg 0 | — | 164ms | 120ms | — | -44ms, 1.37x lower |
| LogUp GKR | 790ms | 519ms | 520ms | -270ms, 1.52x lower | +1ms, unchanged |
| Round 0 | 662ms | 180ms | 179ms | -483ms, 3.70x lower | -1ms, unchanged |
| MLE Rounds | 180ms | 159ms | 159ms | -21ms, 1.13x lower | 0ms, unchanged |
| Stacked Reduction | 311ms | 75ms | 75ms | -236ms, 4.15x lower | 0ms, unchanged |

### APC 0

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace | 2153ms | 2134ms | 2271ms | +118ms, 1.05x higher | +137ms, 1.06x higher |
| Trace Commit | 518ms | 526ms | 485ms | -33ms, 1.07x lower | -41ms, 1.08x lower |
| LogUp GKR | 993ms | 1013ms | 1190ms | +197ms, 1.20x higher | +177ms, 1.17x higher |
| Round 0 | 178ms | 176ms | 176ms | -2ms, unchanged | 0ms, unchanged |

APC 0 was verified with 2 runs without warmup (2134ms, 2169ms) and 2 runs with warmup (2271ms, 2285ms). The GKR regression is consistent and not noise.

## Assessment

**The optimization did NOT achieve its goal.** While it successfully eliminated the cold-start codeword allocation overhead at APC 300 (rs_code_matrix seg 0: 58ms → 14ms, saving ~44ms on STARK excl trace), it caused a severe regression at APC 0 (+137ms STARK excl trace, driven by +177ms in LogUp GKR).

The APC 0 regression exceeds the rollback threshold of 20ms by 7x. The optimization is **reverted**.

**Root cause of regression**: The warmup allocation creates a 256MB free region in the VPMM pool with specific stream/event state. This disrupts the pool state for subsequent allocations — particularly the GKR input evaluation pre-allocated buffers and their memory-bandwidth-bound kernel execution. This is the same pattern observed in the `multistream-stacked-reduction-round0` task, where per-thread buffer allocations through the CUDA memory pool caused a ~200ms GKR regression. The VPMM pool is extremely sensitive to allocation order and free list state because different virtual address layouts produce different TLB and memory access patterns for bandwidth-bound kernels.

## Future Work

- **Root cause investigation**: The VPMM pool state sensitivity to allocation patterns needs to be understood systematically. Three separate optimizations have now triggered ~150-200ms GKR regressions at APC 0 by changing pool state. The underlying mechanism — likely TLB locality or virtual address fragmentation affecting memory bandwidth — needs to be characterized with nsight memory profiling.
- **Pool-transparent warmup**: A warmup mechanism that pre-creates physical pages without going through the allocation/free cycle (i.e., directly populating `active_pages` without adding to `free_regions`) might avoid the pool state disruption. This would require a new VPMM API.
- **Lazy page creation**: Instead of pre-warming, the VPMM could be modified to create physical pages in larger batches (e.g., 128 pages at once) instead of one at a time, reducing the per-page overhead from ~0.4ms to amortized ~0.05ms.
- **The 58ms cold-start is genuine and worth fixing**: The improvement at APC 300 (-44ms) was clear and targeted. The challenge is doing it without side effects on the bandwidth-sensitive GKR phase.
