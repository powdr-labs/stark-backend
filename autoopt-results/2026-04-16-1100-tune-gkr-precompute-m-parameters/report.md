# Report: Tune GKR PrecomputeM Parameters

## Description

The optimization aimed to reduce GKR fractional sumcheck inner round cost at APC 300 by tuning the PrecomputeM strategy parameters. The primary vector was the `MIN_N` threshold (default 22) that controls which GKR layers use PrecomputeM vs FoldEval. The hypothesis was that at APC 300 (many small traces), smaller GKR layers would be better served by FoldEval's simpler per-round cost than PrecomputeM's M-build + reduction + multifold overhead (~6.6ms per layer). Secondary sweeps targeted `TARGET_BLOCKS` (GPU parallelism of M-build kernel) and `TAIL_TILE` (direct block granularity control).

## Implementation

### Diagnostic (Step 1.5)
Added temporary `eprintln!` in `choose_round_strategy()` to log per-layer strategy decisions. Found that at APC 300:
- **App proof segments** (2 segments, 27 outer rounds each): PrecomputeM at rem_n = 22,23,24,25,26 (5 layers per segment, 10 total)
- **Leaf recursion**: 5 PrecomputeM layers (rem_n 22-26)
- **Internal recursion**: 4 PrecomputeM layers (rem_n 22-25)
- **Total**: 14 PrecomputeM layers, matching nsight's 14 `precompute_m_build_partial` instances

### Buffer sizing bug fix
When testing MIN_N=24, discovered a CUDA `cudaErrorIllegalAddress` crash caused by the work buffer being undersized. The buffer size computation at `fractional.rs:628` used the compile-time constant `GKR_WINDOW_DEFAULT_MIN_N` (22) instead of the runtime `precompute_m_min_n` value. When the env var sets MIN_N higher, FoldEval fallback rounds at rem_n > 22 need larger work buffers than `1 << 22` elements.

**Fix**: Moved `precompute_m_min_n()` env var read before the buffer sizing computation and replaced `1 << GKR_WINDOW_DEFAULT_MIN_N` with `1 << precompute_m_min_n`.

File: `crates/cuda-backend/src/logup_zerocheck/fractional.rs` (lines 619-649)

### Parameter sweep
Used existing `SWIRL_CUDA_GKR_PRECOMPUTE_M_*` environment variables to test 10 configurations at APC 300.

## Results

### MIN_N sweep (primary)

| MIN_N | LogUp GKR (ms) | STARK excl trace (ms) | vs default |
|-------|----------------|----------------------|------------|
| disabled | 446 | 1200 | +87ms |
| 28 | 415 | 1181 | +68ms |
| 26 | 385 | 1137 | +24ms |
| 24 | 375 | 1119 | +6ms |
| **22 (default)** | **367** | **1113** | **0** |
| 20 | 365 | 1111 | -2ms |
| 18 | 368 | 1109 | -4ms |
| 16 | 383 | 1132 | +19ms |

### TARGET_BLOCKS sweep (secondary)

| TARGET_BLOCKS | LogUp GKR (ms) | STARK excl trace (ms) | vs default |
|---------------|----------------|----------------------|------------|
| 512 | 370 | 1115 | +2ms |
| **1024 (default)** | **367** | **1113** | **0** |
| 2048 | 377 | 1122 | +9ms |

### TAIL_TILE sweep

| TAIL_TILE | LogUp GKR (ms) | STARK excl trace (ms) | vs default |
|-----------|----------------|----------------------|------------|
| 256 | 375 | 1127 | +14ms |
| **4096 (default)** | **367** | **1113** | **0** |

### Full comparison table

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace (APC 300) | 2455ms | 1113ms | 1113ms | 1342ms lower (2.21x) | 0ms (1.00x) |
| LogUp GKR (APC 300) | 790ms | 367ms | 363ms | 427ms lower (2.18x) | -4ms (1.01x, noise) |
| Round 0 (APC 300) | 662ms | 182ms | 186ms | 476ms lower (3.56x) | +4ms (noise) |
| MLE Rounds (APC 300) | 180ms | 161ms | 161ms | 19ms lower (1.12x) | 0ms (1.00x) |
| STARK excl trace (APC 0) | 2153ms | 1796ms | 1808ms | 345ms lower (1.19x) | +12ms (noise) |
| LogUp GKR (APC 0) | 993ms | 715ms | 724ms | 269ms lower (1.37x) | +9ms (noise) |

## Assessment

The optimization **did not achieve its goal**. No parameter combination improved GKR inner rounds at APC 300 by ≥10ms (the rollback threshold). The current defaults are near-optimal.

The plan's core hypothesis was wrong: raising MIN_N to shift layers from PrecomputeM to FoldEval makes GKR *worse*, not better. The MIN_N sweep shows a clear monotonic trend — every increase in MIN_N (fewer PrecomputeM layers) increases GKR time. Disabling PrecomputeM entirely adds 79ms to GKR (+87ms STARK excl trace).

**Why PrecomputeM is faster than FoldEval for these layers**: The windowed approach (w=3) processes 3 inner rounds in a single GPU pass. Each PrecomputeM window costs ~6.6ms total (3.2ms M-build + 3.4ms multifold). The equivalent 3 FoldEval rounds would cost ~1.5ms in compute_round kernels but also require 3 separate D2H syncs + CPU transcript observe/sample cycles + tree revert operations. The per-round overhead of FoldEval (~2-3ms including D2H + CPU) exceeds PrecomputeM's amortized cost.

**Why lowering MIN_N doesn't help either**: At rem_n=18-21, the additional PrecomputeM layers process smaller data (2^15-2^18 groups), where the M-build kernel underutilizes the GPU (fewer tail blocks) while adding reduction overhead. The min_blocks threshold (64) already correctly filters out the smallest layers.

The secondary sweeps (TARGET_BLOCKS and TAIL_TILE) also showed no improvement, confirming that the M-build kernel's GPU utilization is already well-balanced at the current defaults.

The only code change — fixing the work buffer sizing to use the runtime MIN_N value — is a correctness fix for the env var override path. It doesn't affect default behavior.

## Future Work

- **PrecomputeM is confirmed valuable**: Any future optimization should preserve or enhance PrecomputeM, not replace it. The 79ms benefit of PrecomputeM at APC 300 suggests the windowed approach successfully amortizes per-round overhead.
- **Per-round D2H elimination**: The main remaining GKR overhead is the per-round D2H sync + CPU transcript processing in FoldEval rounds (rem_n < 22). A GPU-side transcript approach (attempted in `2026-04-14-1800-gpu-gkr-transcript-processing`, which failed because per-round CPU work was already fast at ~15-25μs) might help if combined with batching multiple transcript operations.
- **Adaptive window size**: The current fixed w=3 window could be made adaptive — larger windows (w=4 or w=5) for the largest layers could reduce the number of M-build + multifold invocations further, though this requires more M-table memory (2^w entries).
- **FoldEval kernel fusion**: The `compute_round_and_revert` + `compute_round_and_fold_inplace` kernels in FoldEval rounds could potentially be fused into multi-round kernels (analogous to how PrecomputeM processes multiple rounds).
