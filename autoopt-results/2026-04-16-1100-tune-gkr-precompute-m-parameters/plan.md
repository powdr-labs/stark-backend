# Plan: Tune GKR PrecomputeM Parameters

## Goal

Reduce GKR fractional sumcheck inner round cost at APC 300 by tuning the PrecomputeM strategy parameters, primarily the `MIN_N` threshold that controls which GKR layers use PrecomputeM vs FoldEval. The GKR inner rounds are the largest sequential component (213ms = 107ms per segment, 19% of STARK excl trace). The directly tunable kernel is `precompute_m_build_partial_kernel` (45ms total). The `multifold_kernel` (47ms) has fixed launch geometry unaffected by these parameters, but can be eliminated entirely for layers shifted to FoldEval via `MIN_N`. The MIN_N sweep is the primary optimization vector: raising it from 22 to 24-26 could shift 2-4 layers from PrecomputeM to FoldEval, eliminating their M-build + reduction + multifold overhead (~6.6ms per layer). The TARGET_BLOCKS and TAIL_TILE sweeps are secondary, targeting the M-build kernel's GPU utilization for layers that remain on PrecomputeM.

## Current Code Path

### PrecomputeM strategy selection (`crates/cuda-backend/src/logup_zerocheck/fractional.rs`)

1. **Line 225-245**: `choose_round_strategy()` decides FoldEval vs PrecomputeM per GKR layer. Uses `precompute_m_min_n` (default 22 from line 36) as the minimum log-size for PrecomputeM.

2. **Line 203-223**: `choose_precompute_m_window_w()` computes the window size (always w=3 from `GKR_WINDOW_SIZE`, line 35) and tail_tile. The tail_tile determines how many thread blocks the M-build kernel launches, computed by `precompute_m_build_tail_tile()` (lines 186-201).

3. **Lines 186-201**: `precompute_m_build_tail_tile()` clamps `k / target_blocks` between `PRECOMPUTE_M_MIN_TAIL_TILE` (256) and `PRECOMPUTE_M_TAIL_TILE` (4096), where k = 2^(rem_n - w) is the number of groups.

### Environment variable overrides (lines 144-184)

| Variable | Default | Description |
|---|---|---|
| `SWIRL_CUDA_GKR_PRECOMPUTE_M` | enabled | Enable/disable PrecomputeM strategy |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N` | 22 | Min log-size for PrecomputeM (layers with rem_n < min_n use FoldEval) |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS` | 64 | Minimum blocks for M-build kernel (if fewer blocks would result, fall back to FoldEval) |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS` | 1024 | Target blocks for M-build kernel (controls tail_tile) |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_TAIL_TILE` | override | Direct tail_tile override (clamped to [256, 4096]) |

### Kernel behavior

- **precompute_m_build_partial**: launches `num_tail_blocks` thread blocks, each processing `tail_tile` groups. More blocks = better GPU utilization but more reduction overhead.
- **multifold**: launches grid proportional to pq_size / 2^(w+pending_fold). Fixed per GKR layer.
- **frac_precompute_m_eval_round**: lightweight kernel per window sumcheck round (~0.5ms each).

### Current profile (APC 300 nsight)

| Kernel | GPU time | Instances | Avg |
|---|---|---|---|
| precompute_m_build_partial<true,3> | 45ms | 14 | 3.2ms |
| multifold<4> | 47ms | 14 | 3.4ms |
| compute_round_and_revert | 70ms | 141 | 0.5ms |
| bit_rev_frac_build_k2 | 50ms | 6 | 8.3ms |
| compute_round_and_fold_inplace | 9ms | 1370 | 6.6μs |
| frac_build_tree_two_layers | 12ms | 66 | 0.18ms |
| **Total GKR inner** | **~233ms** | | |

## Changes

### Step 1: Baseline measurement (no code changes)

Measure current GKR inner round time at APC 300 with default parameters. Record per-segment `logup_gkr_time_ms` and per-kernel GPU times from nsight.

### Step 1.5: Diagnostic — log per-layer strategy decisions

Add temporary `tracing::info!` in `choose_round_strategy()` (line 225) to log `(round, rem_n, strategy)` for each GKR outer round. Run once at APC 300 to determine:
- How many layers use PrecomputeM vs FoldEval
- What `rem_n` values the PrecomputeM layers have
- The improvement ceiling (number of layers × ~6.6ms per eliminated PrecomputeM layer)

This diagnostic directly informs which sweep ranges are productive.

### Step 2: Parameter sweep

Using the existing environment variables, test configurations at APC 300 (2 runs each for noise). **Front-load Sweep 2 (MIN_N)** since it has the highest probability of impact.

**Sweep 2 (primary): MIN_N (FoldEval vs PrecomputeM threshold)**
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N=24` (fewer layers use PrecomputeM)
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N=26` (only largest layers use PrecomputeM)
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N=28` (almost all layers use FoldEval)

Hypothesis: At APC 300, smaller GKR layers (from many small traces) may be better served by FoldEval's simpler per-round cost than PrecomputeM's M-build + reduction + multifold overhead (~6.6ms per layer). Raising MIN_N shifts these layers to FoldEval, eliminating both `precompute_m_build_partial` and `multifold` invocations for those layers.

**Sweep 1 (secondary, only if Sweep 2 shows promise): TARGET_BLOCKS**
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS=512` (fewer, larger blocks)
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS=2048` (more, smaller blocks)

Hypothesis: Higher TARGET_BLOCKS increases M-build parallelism but the `precompute_m_reduce_partials_kernel` iterates over `num_blocks` in a serial loop per thread, so more blocks increases reduction cost linearly. Net effect is unclear; only test if Sweep 2 leaves PrecomputeM layers that could benefit.

**Sweep 3 (secondary): TAIL_TILE (direct block granularity)**
- `SWIRL_CUDA_GKR_PRECOMPUTE_M_TAIL_TILE=256` (minimum tile, maximum blocks)

Note: TAIL_TILE override bypasses TARGET_BLOCKS computation (line 193-194). Only test independently from Sweep 1.

**Sweep 4: Disable PrecomputeM entirely**
- `SWIRL_CUDA_GKR_PRECOMPUTE_M=0` (all layers use FoldEval)

FoldEval-only baseline for comparison. Establishes the maximum possible savings from eliminating all PrecomputeM overhead.

**Sweep 5 (combined): Best MIN_N + best TARGET_BLOCKS**
Test the best MIN_N value from Sweep 2 combined with the best TARGET_BLOCKS from Sweep 1 (if applicable).

### Step 3: Identify best configuration

From sweep results, identify the parameter combination that minimizes GKR inner round time at APC 300 without regressing APC 0.

### Step 4: Verify at APC 0 and APC 100

Run the best configuration at APC 0 and APC 100. Verify no regression beyond noise (±20ms for STARK excl trace).

### Step 5: Apply the change

If a beneficial configuration is found, update the default constants in `fractional.rs`:
- `PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS` (line 71)
- `PRECOMPUTE_M_DEFAULT_MIN_BLOCKS` (line 70)
- `GKR_WINDOW_DEFAULT_MIN_N` (line 36)
- `PRECOMPUTE_M_TAIL_TILE` (line 68) / `PRECOMPUTE_M_MIN_TAIL_TILE` (line 69)

## Invariants

1. **Correctness**: Changing PrecomputeM parameters does not affect the mathematical result — both FoldEval and PrecomputeM produce identical sumcheck proofs (verified by existing test `test_precompute_m_multi_window_matches_fused` at line 1256). All 94 tests must pass.

2. **No APC 0 regression**: The parameter change must not regress APC 0 STARK excl trace by > 20ms. If the optimal APC 300 config hurts APC 0, use adaptive logic (e.g., derive parameter from num_traces or total_leaves).

3. **Parameter bounds**: TARGET_BLOCKS ≥ 1, MIN_BLOCKS ≥ 1, tail_tile ∈ [256, 4096], MIN_N ≥ GKR_WINDOW_SIZE.

## Measurement Plan

For each sweep configuration:
1. Set the relevant environment variable(s)
2. Run APC 300 benchmark (from powdr repo):
   ```
   SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS=<value> \
   powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics <output>.json --recursion
   ```
3. Analyze with `spec.py` — record LogUp GKR time, STARK excl trace, and per-segment logup_gkr_time_ms.
4. For the best config: also run APC 0 and APC 100 with nsight profiling.

## Rollback Criteria

- Revert if no parameter combination improves GKR inner rounds at APC 300 by ≥ 10ms
- Revert if the best APC 300 config regresses APC 0 STARK excl trace by > 20ms
- Revert if any test fails
