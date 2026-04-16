# Report: Tune VPMM Page Size for WHIR

## Description

The VPMM page size was increased from 2 MiB to 16 MiB in a prior optimization (vpmm-bulk-page-creation), yielding a net -186ms improvement at APC 300. However, WHIR regressed by ~25ms because its 2-4 MiB buffers shifted from VPMM (virtual memory pool) to cudaMallocAsync. This task tested intermediate page sizes (4, 8 MiB) to find a sweet spot that preserves the large-allocation benefits while routing WHIR buffers back through VPMM.

## Implementation

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs:176`

One-line change: default page size multiplier from `8 * granularity` (16 MiB) to `4 * granularity` (8 MiB).

The implementation is trivial — the core work was empirical parameter sweeping:

1. **Before measurement**: Benchmarked current 16 MiB at APC 0, 100, 300 (2 runs for noise confirmation)
2. **4 MiB sweep** (via `VPMM_PAGE_SIZE=4194304`): Tested at APC 300
3. **8 MiB sweep** (via `VPMM_PAGE_SIZE=8388608`): Tested at APC 0, 300 (2 runs at APC 300 for noise confirmation)
4. **12 MiB attempt**: Failed — not a power of 2, so `12582912 % DEFAULT_VA_SIZE != 0` assertion fires
5. **After measurement**: Built with 8 MiB default, measured all 3 APC configs

**Deviation from plan**: The plan suggested testing 10 and 12 MiB page sizes. These are not powers of 2, so they fail the `va_size.is_multiple_of(page_size)` assertion with the default 8 TiB VA size. Only power-of-2 multiples of the 2 MiB granularity (2, 4, 8, 16 MiB) are viable without also changing VA_SIZE.

## Results

### APC 300 (primary metric)

| Metric | Baseline | Before (16 MiB) | After (8 MiB) | vs Baseline | vs Before |
|--------|----------|-----------------|---------------|-------------|-----------|
| STARK excl trace | 2455ms | 1110ms | 1120ms | -1335ms, 2.19x lower | +10ms, 1.01x higher |
| WHIR | 100ms | 125ms | 115ms | +15ms, 1.15x higher | -10ms, 1.09x lower |
| LogUp GKR | 790ms | 363ms | 387ms | -403ms, 2.04x lower | +24ms, 1.07x higher |
| Trace Commit | 406ms | 197ms | 198ms | -208ms, 2.05x lower | +1ms, noise |
| Round 0 | 662ms | 181ms | 180ms | -482ms, 3.68x lower | -1ms, noise |
| Stacked Reduction | 311ms | 74ms | 75ms | -236ms, 4.15x lower | +1ms, noise |
| MLE Rounds | 180ms | 164ms | 160ms | -20ms, 1.13x lower | -4ms, noise |

### APC 0 (regression check)

| Metric | Baseline | Before (16 MiB) | After (8 MiB) | vs Baseline | vs Before |
|--------|----------|-----------------|---------------|-------------|-----------|
| STARK excl trace | 2153ms | 1804ms | 1907ms | -246ms, 1.13x lower | +103ms, 1.06x higher |
| WHIR | 220ms | 219ms | 219ms | 0ms, noise | 0ms, noise |
| LogUp GKR | 993ms | 718ms | 829ms | -164ms, 1.20x lower | +111ms, 1.15x higher |
| Trace Commit | 518ms | 476ms | 481ms | -37ms, 1.08x lower | +5ms, noise |

### APC 100

| Metric | Before (16 MiB) | After (8 MiB) | vs Before |
|--------|-----------------|---------------|-----------|
| STARK excl trace | 1308ms | 1320ms | +12ms, 1.01x higher |
| WHIR | 154ms | 149ms | -5ms, 1.03x lower |
| LogUp GKR | 475ms | 505ms | +30ms, 1.06x higher |

### Page Size Sweep Summary (APC 300)

| Page Size | STARK excl trace | WHIR | LogUp GKR | Trace Commit | Net vs 16 MiB |
|-----------|------------------|------|-----------|--------------|---------------|
| 16 MiB (current) | 1113ms avg | 125ms | 365ms avg | 197ms | baseline |
| 8 MiB | 1126ms avg | 117ms avg | 388ms avg | 198ms avg | +13ms worse |
| 4 MiB | 1177ms | 103ms | 438ms | 222ms | +64ms worse |

## Assessment

**Result: Failure.** No intermediate page size improves STARK excl trace at APC 300. The rollback criteria from the plan are triggered on all counts:

1. **No page size improves STARK excl trace by >= 10ms** — all tested sizes were net-worse
2. **APC 0 regresses by > 20ms** — 8 MiB causes +103ms regression (GKR +111ms)
3. **WHIR recovery insufficient** — the -10ms WHIR improvement at 8 MiB is offset 2.4x by the +24ms GKR regression

The root cause is that smaller VPMM page sizes increase the number of cuMemCreate calls for medium-to-large allocations (16-256 MiB range), which directly impacts LogUp GKR — a memory-bandwidth-bound phase that is sensitive to VPMM pool state and allocation layout. This is the same pool state sensitivity documented in tasks `cache-codeword-buffer-across-segments` and `multistream-stacked-reduction-round0`.

The 25ms WHIR regression from baseline (100ms → 125ms at APC 300) is a tolerable cost of the 16 MiB page size setting. At WHIR's 5% share of STARK excl trace, this ~25ms penalty is dwarfed by the benefits the larger page size provides to other phases.

## Future Work

- **WHIR-specific allocation**: Instead of a global page size change, WHIR could pre-allocate a dedicated VPMM scratch buffer (similar to the prealloc-whir-fold-buffers attempt) that is large enough to hold all round buffers. This was tried and only saved 4ms, suggesting the WHIR regression is not purely allocation overhead.
- **Dual-pool architecture**: A two-pool VPMM design — small pages (2-4 MiB) for WHIR-class allocations, large pages (16 MiB) for general use — would avoid the global tradeoff. This adds significant complexity for a ~10ms target.
- **WHIR kernel optimization**: The more productive path is to optimize WHIR's actual kernel performance rather than tuning allocation routing. The 125ms WHIR time at APC 300 includes ~65ms of Merkle tree hashing — kernel-level optimization (e.g., batched Poseidon2 across rounds) may yield larger gains.
- **Non-power-of-2 page sizes**: The current VA_SIZE alignment constraint limits viable page sizes to powers of 2. Relaxing this (e.g., setting VA_SIZE to `page_size * large_N`) would enable testing 6 MiB, 10 MiB, 12 MiB, but the GKR regression trend suggests no sweet spot exists.
