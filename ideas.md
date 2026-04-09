# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1558ms** (baseline 2491ms, **-37.5%**)
APC000 baseline: 2176ms. Target: APC300 STARK < APC000/2 = 1084ms. Gap: ~474ms.

## Priority Order

### 1. Direct allocation for GKR leaves buffer
Bypass VPMM pool for the 512MB GKR leaves allocation using cudaMallocAsync directly. The VPMM pool's contiguous block search adds ~300ms on segment 0 (vs 2ms on segment 1 which reuses freed blocks). Implementation started on branch `003-stacked-reduction-opt` - `d_malloc_direct` and `DeviceBuffer::with_capacity_direct` already added, needs testing.

**Expected savings:** ~200-300ms for APC300 segment 0.

### 2. Batch non-degenerate stacked reduction MLE kernel
5076 launches of stacked_reduction_sumcheck_mle_round_kernel at 4.3us each (21.7ms). Batch similarly to degenerate kernel.

**Expected savings:** ~15ms.

### 3. SP1-inspired slope-based interpolation at fixed evaluation points
SP1 uses {0, 2, 4} evaluation points with precomputed slopes in sumcheck. Could reduce per-round overhead in GKR fractional sumcheck.

**Expected savings:** ~10-20ms.

### 4. Profile-guided kernel occupancy tuning with ncu
Use NVIDIA Nsight Compute to check register pressure and occupancy for the hottest kernels.

**Expected savings:** Unknown.

### 5. Logup Round 0 kernel batching (CUDA-level)
Batch the 735 logup_r0_ntt_eval_interactions kernel launches into grouped dispatches. Requires new CUDA kernel for batched barycentric/NTT evaluation.

**Expected savings:** ~30-50ms on top of parallel streams.

## Done

### 001-round0-parallel-streams
**Idea:** Dispatch per-AIR Round 0 kernels across 8 OS threads with per-thread CUDA streams. Deferred D2H to avoid COPY_EVENT mutex serialization. Interleaved thread assignment for load balancing.
**Result:** APC300 Round 0: 670ms -> 183ms (-73%). STARK excl. trace: 2491ms -> ~2098ms (-16%).
**Task dir:** 001-round0-parallel-streams/ (has plan.md, report.md, before/after metrics)

### 003-stacked-reduction-deferred-sync
**Idea:** Cherry-picked commit c868e4b. Eliminated ~32,500 per-window GPU-CPU sync barriers in stacked reduction MLE rounds by batching all window kernel launches before a single D2H transfer.
**Result:** APC300 Stacked Reduction: 311ms -> 90ms (-71%).

### 003-gkr-input-batching
**Idea:** Cherry-picked commit 514d9d0. Batched GKR input evaluation across AIRs using BlockCtx/GkrInputCtx pattern. Reduced ~791 kernel launches to 2-12 per segment.
**Result:** APC300 LogUp GKR: 800ms -> ~600ms (-25%).

### 003-batched-degenerate-stacked
**Idea:** Batched stacked_reduction_sumcheck_mle_round_degenerate_kernel: collected all degenerate windows into DegenerateWindowCtx array and launched a single kernel with one block per window. Reduced 10,348 launches to 1.
**Result:** APC300 Stacked Reduction: 128ms -> 90ms (-30%).

### 003-batched-round0-zerocheck
**Idea:** Batched Round 0 zerocheck evaluation using BlockCtx/Round0ZerocheckCtx. Groups small AIRs by matching dimensions. For APC300: batches 279/306 traces (91%) into 12 groups. Minimal additional gain on top of parallel streams.
**Result:** ~5ms improvement (parallel streams already capture most benefit).

### 003-logup-rules-caching
**Idea:** Cache Round 0 logup DAG (SymbolicRulesGpu + interaction mappings) in proving key at keygen. Avoids per-AIR DAG rebuilding during proving.
**Result:** ~2ms (DAG building hidden by parallel streams).

### 003-parallel-logup-precompute
**Idea:** Parallelize precompute_logup_combinations calls across 8 threads. Each call launches 2 GPU kernels per AIR; overlapping on separate streams reduces 37ms sequential precompute.
**Result:** Round 0: 216ms -> 183ms (-15%). STARK: ~1612ms -> ~1572ms.

### 003-selector-caching
**Idea:** Cache selector DeviceBuffers by trace height. Traces with same height share same selector (is_first, is_transition, is_last). Reduces ~600 VPMM allocations to ~20.
**Result:** ~3ms.

### 003-parallel-eq3b
**Idea:** Parallelize eval_eq_mle computation for eq_3b weights across 8 threads (CPU-bound).
**Result:** ~2ms.

### 003-vpmm-precommit-pages
**Idea:** Pre-commit 8192 VPMM pages (~16GB) at startup to avoid cuMemSetAccess overhead during first large allocations. Configurable via VPMM_PAGES env var.
**Result:** Moves ~50ms cuMemSetAccess from STARK span to startup. No APC000 regression.

## Tested and rejected

- **GKR threshold increase (10->32):** Hurts RTX 4090 occupancy due to register pressure.
- **l_skip tuning:** Previous Report 1 found l_skip=4 is optimal.
- **calculate_zero_hash caching:** Hidden behind pipeline overlap (Report 1).
- **16 Round 0 threads:** No improvement over 8 (memory contention).
- **VPMM pre-warming in commit:** Helps APC300 but fragments pool for APC000.
- **RS matrix cache disable:** Huge regression in WHIR (+269ms) due to recomputation.
- **VPMM bypass for large allocations (>256MB):** cudaMallocAsync doesn't provide pool reuse, causing +274ms Round 0 regression.
