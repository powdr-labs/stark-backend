# Report: Decouple VPMM Pool Threshold from Page Size

## Description

The previous optimization (vpmm-bulk-page-creation) increased the VPMM page size from 2 MiB to 16 MiB, which gave a surprising -186ms improvement on APC 300. Approximately 136ms of that came from medium-sized allocations (2-16 MiB) bypassing VPMM and going through `cudaMallocAsync` instead. This task attempted to extend that benefit by introducing a separate `pool_threshold` parameter (default 4x page_size = 64 MiB) so allocations in the 16-64 MiB range (including per-thread GKR intermediates at ~20-40 MiB each) would also bypass VPMM.

## Implementation

### Files changed:
- `crates/cuda-common/src/memory_manager/vm_pool.rs`: Added `pool_threshold: Option<usize>` to `VpmmConfig`, `pool_threshold: usize` to `VirtualMemoryPool`. Resolution logic with `saturating_mul` for overflow guard when VPMM is unsupported. Environment variable `VPMM_POOL_THRESHOLD` support.
- `crates/cuda-common/src/memory_manager/mod.rs`: Changed allocation routing from `size < self.pool.page_size` to `size < self.pool.pool_threshold`.
- `crates/cuda-common/src/memory_manager/tests.rs`: Added `pool_threshold: None` to all 4 `VpmmConfig` struct literals.

### Deviations from plan: None. All 5 changes implemented as specified.

## Results

### STARK excl trace (ms)

| Config | Baseline | Before Task | After (4x) | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| APC 0  | 2153     | 1779        | 1830       | -323ms, 1.18x lower | +51ms, 1.03x higher |
| APC 100| 2155     | 1302        | 1330       | -825ms, 1.62x lower | +28ms, 1.02x higher |
| APC 300| 2455     | 1110        | 1130       | -1325ms, 2.17x lower | +20ms, 1.02x higher |

### Alternative thresholds tested at APC 0

| Threshold | STARK excl trace (ms) | vs Before |
|-----------|----------------------|-----------|
| 1x (control) | 1773 | -6ms (noise) |
| 2x (32 MiB)  | 1811 | +32ms |
| 4x (64 MiB)  | 1830, 1833 | +51-54ms |

### Component breakdown at APC 0 (4x threshold)

| Component | Before | After (4x) | Delta |
|-----------|--------|------------|-------|
| LogUp GKR | 700 | 713 | +13ms |
| Round 0 | 180 | 179 | -1ms |
| MLE Rounds | 116 | 134 | +18ms |
| Openings | 298 | 320 | +22ms |
| WHIR | 218 | 227 | +9ms |
| Stacked Reduction | 78 | 91 | +13ms |
| Trace Commit | 476 | 476 | 0ms |

## Assessment

**Failure.** The optimization caused consistent regressions across all APC configurations:
- APC 0: +51ms (exceeds 20ms rollback threshold)
- APC 100: +28ms
- APC 300: +20ms

The regression at APC 0 scaled monotonically with the threshold multiplier (1x: 0ms, 2x: +32ms, 4x: +51ms), confirming it is caused by routing more allocations through `cudaMallocAsync` instead of VPMM.

The regression is distributed across LogUp GKR (+13ms), MLE Rounds (+18ms), and Openings (+22ms) at APC 0. These are memory-bandwidth-bound phases. The pattern is consistent with the VPMM pool state sensitivity observed in `cache-codeword-buffer-across-segments` and `multistream-stacked-reduction-round0`.

The hypothesis that medium-sized allocations (16-64 MiB) would benefit from bypassing VPMM was incorrect. While the previous task's 2-16 MiB bypass was beneficial (a side effect of increasing page_size), the mechanism was not simply "avoid VPMM overhead" but rather the specific memory layout that resulted from that page size change. Extending the bypass to 16-64 MiB allocations likely changes the CUDA memory pool's internal allocation patterns, degrading memory access locality for subsequent bandwidth-bound kernels.

## Future Work

- The VPMM pool state sensitivity pattern is now well-documented across 3 tasks (cache-codeword-buffer-across-segments, multistream-stacked-reduction-round0, this task). Any optimization that changes allocation routing or pool state must be tested for APC 0 regression as a first step.
- The 16 MiB page size (previous task) appears to be a local optimum: it routes 2-16 MiB allocations through cudaMallocAsync (beneficial) while keeping 16+ MiB through VPMM (beneficial for memory layout). Moving the threshold in either direction regresses.
- Future VPMM optimizations should focus on reducing overhead *within* VPMM (e.g., faster cuMemCreate, smarter page reuse) rather than bypassing it for more allocation sizes.
- nsight profiling of memory access patterns could reveal why VPMM allocations produce better bandwidth for subsequent kernels, potentially informing a custom allocator that preserves this property.
