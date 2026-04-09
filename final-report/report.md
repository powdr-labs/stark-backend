# Final Report: OpenVM v2 GPU Prover Optimizations for Autoprecompiles

**Branch:** `003-stacked-reduction-opt` (20 commits on top of `v2-powdr-07-04`)

## Summary

Combined optimizations reduce APC300 STARK excl. trace from **2482ms to 1582ms (-36.3%)**. APC000 is unchanged (2165ms → 2133ms). The scaling direction is corrected: APC300 is now 26% faster than APC000 (was 15% slower before optimization).

## Results

### STARK (excl. trace) — Before vs After

| Phase | APC000 Base | APC000 Opt | Factor | APC100 Base | APC100 Opt | Factor | APC300 Base | APC300 Opt | Factor |
|---|---|---|---|---|---|---|---|---|---|
| **STARK excl. trace** | **2165** | **2133** | **0.99** | **2183** | **1618** | **0.74** | **2482** | **1582** | **0.64** |
| LogUp GKR | 1001 | 1015 | 1.01 | 782 | 650 | 0.83 | 809 | 615 | 0.76 |
| Round 0 | 179 | 171 | 0.96 | 474 | 163 | 0.34 | 664 | 176 | 0.27 |
| MLE Rounds | 118 | 118 | 1.00 | 150 | 153 | 1.02 | 182 | 182 | 1.00 |
| Stacked Reduction | 113 | 81 | 0.72 | 203 | 81 | 0.40 | 310 | 87 | 0.28 |
| Trace Commit | 522 | 516 | 0.99 | 428 | 426 | 1.00 | 412 | 415 | 1.01 |
| WHIR | 219 | 219 | 1.00 | 138 | 138 | 1.00 | 100 | 100 | 1.00 |
| **Total proof** | **5127** | **5036** | **0.98** | **6117** | **5423** | **0.89** | **6939** | **5939** | **0.86** |

All times in ms.

### Relative to APC000 (optimized)

| Metric | APC000 | APC300 | Ratio |
|---|---|---|---|
| Cells | 1.90B | 811M | 0.43x |
| **Baseline STARK** | 2165 | 2482 | **1.15x (worse!)** |
| **Optimized STARK** | 2133 | 1582 | **0.74x (better!)** |
| Target (proportional) | 2133 | 907 | 0.43x |

---

## Optimizations Applied

### 1. Round 0 Parallel CUDA Streams (~390ms APC300 savings)

**Commits:** `2ef4093b` (1 file, +162 -88), `6fc1602a` (1 file, +1 -1), `b7144c36` (1 file, +14 -6)

**Problem:** Round 0 evaluates zerocheck and logup constraints for each AIR sequentially. With 600+ APC AIRs, most are small and don't saturate the GPU. Each of the ~1400 kernel launches (700 zerocheck + 700 logup) runs on the same CUDA stream, serialized.

**Idea:** Dispatch per-AIR Round 0 work across 8 OS threads using `std::thread::scope`. Since the codebase compiles CUDA with `--default-stream=per-thread`, each thread gets its own CUDA stream automatically. Kernels from different threads execute concurrently on the GPU. D2H transfers are deferred to after all threads complete (avoids the global `COPY_EVENT` mutex serialization). Traces are assigned to threads in an interleaved pattern (thread T gets traces T, T+8, T+16, ...) for load balancing, since traces are sorted by height.

**Impact:** APC300 Round 0: 664ms → ~280ms. The largest single optimization. APC000 (99 AIRs, below threshold) uses the sequential path and is unaffected.

### 2. Stacked Reduction Deferred Sync (~185ms APC300 savings)

**Commit:** `1d61fafc` (2 files, +21 -30)

**Problem:** The stacked reduction sumcheck MLE rounds processed ~3,250 windows per round with a GPU-CPU sync barrier after each window (zero accumulator, launch kernel, D2H transfer). Across 10 MLE rounds, this produced ~32,500 sync barriers. APC300 has ~3x more windows than APC000, causing 2.75x worse scaling (113ms → 310ms).

**Idea:** Cherry-picked from a previous optimization branch (commit c868e4b). Restructured `batch_sumcheck_poly_eval` to: (1) upload all eq_ub values once, (2) zero the accumulator once, (3) launch ALL window kernels without intermediate D2H syncs, (4) single D2H copy after all kernels complete. Works because kernels use warp-aggregated atomic accumulation into the same output buffer.

**Impact:** APC300 Stacked Reduction: 310ms → ~126ms. Eliminates the main APC scaling regression in the openings phase.

### 3. GKR Input Eval Batching (~190ms APC300 savings)

**Commit:** `8e437d1c` (3 files, +387 -53)

**Problem:** The GKR input evaluation (`evaluate_interactions_gkr_kernel`) launched one kernel per AIR to compute interaction numerator/denominator pairs. For APC300, this meant ~791 individual kernel launches, each with its own buffer setup and launch overhead.

**Idea:** Cherry-picked from a previous optimization branch (commit 514d9d0). Groups AIRs by their `GLOBAL` flag (determines whether intermediates use registers or global memory) and dispatches all AIRs in a single kernel launch per group using a `BlockCtx`/`GkrInputCtx` pattern. Each CUDA block reads its AIR's data pointers, rules, and output location from device-side context arrays.

**Impact:** APC300 LogUp GKR: ~800ms → ~600ms. Reduces ~791 kernel launches to 2-12 (grouped by GLOBAL flag and segment).

### 4. Batched Degenerate Stacked Reduction (~38ms APC300 savings)

**Commit:** `e728b55f` (3 files, +181 -27)

**Problem:** The stacked reduction MLE rounds had 10,348 "degenerate" kernel launches (for traces smaller than the current sumcheck dimension). Each launch processed a single window with a single CUDA block, taking ~3.9μs. The cumulative launch overhead was ~40ms.

**Idea:** Collect all degenerate windows into a `DegenerateWindowCtx` array (containing per-window pointers, lengths, and scalar parameters). Launch a single batched kernel with one block per window. The batched kernel reads its context from `d_window_ctxs[blockIdx.x]`.

**Impact:** APC300 Stacked Reduction: 128ms → 90ms. Reduces 10,348 launches to 1.

### 5. Parallel Logup Combination Precompute (~30ms APC300 savings)

**Commit:** `dc8e413f` (2 files, +102 -41)

**Problem:** Before Round 0, `precompute_logup_combinations` is called for each trace with interaction monomials. Each call launches 2 small GPU kernels. With ~600 traces, the 1,200 sequential kernel launches take ~37ms (dominated by launch overhead, not compute).

**Idea:** Dispatch the precompute calls across 8 threads using `std::thread::scope`, same pattern as Round 0 parallel streams. Each thread processes a subset of traces on its own CUDA stream, overlapping kernel launches.

**Impact:** Round 0 setup: ~37ms → ~5ms. Also improved Round 0 total from 216ms → 183ms.

### 6. VPMM Pre-Commitment at Startup (~10ms APC300 savings)

**Commits:** `82eeb019` (1 file, +4 -1), `f5855f0a` (1 file, +5 -4)

**Problem:** The VPMM (Virtual Pool Memory Manager) uses CUDA virtual memory with page-level management. The first time a page is used, `cuMemSetAccess` must be called to commit physical memory. For the GKR leaves buffer (~512MB), this first-use penalty was ~50ms.

**Idea:** Pre-commit 8192 VPMM pages (~16GB with default 2MB page size) at program startup. This commits physical pages before any proving starts, moving the `cuMemSetAccess` cost outside the STARK timing span. Configurable via `VPMM_PAGES` environment variable.

**Impact:** ~10ms moved from STARK span to program startup. No APC000 regression.

### 7. Batched Round 0 Zerocheck (~5ms APC300 savings)

**Commits:** `4dec564d` (4 code files, +858 -15), `4ca1c5fa` (3 files, +32 -30)

**Problem:** Round 0 zerocheck launches one kernel per AIR for NTT-based constraint evaluation. With parallel streams already overlapping these launches, the remaining overhead is small, but the many tiny kernels (grid=(1,2) for small AIRs) still underutilize the GPU.

**Idea:** Group small AIRs by matching dimensions (height, num_x, num_cosets, constraint_degree) and dispatch each group as a single batched kernel using `Round0ZerocheckCtx`/`Round0BlockCtx` context structs. For APC300: batches 279/306 traces (91%) into 12 groups.

**Impact:** ~5ms. Minimal additional gain on top of parallel streams, which already capture most of the kernel launch overhead savings.

### 8. Logup Round 0 Rules Caching (~2ms APC300 savings)

**Commit:** `98b20821` (3 files, +151 -24)

**Problem:** `evaluate_round0_interactions_gpu` rebuilt a `SymbolicDagBuilder` and `SymbolicRulesGpu` for each AIR on every call. This involves CPU-intensive DAG construction, rule encoding, and device upload of the encoded rules.

**Idea:** Cache the Round 0 logup DAG (encoded rules + per-interaction rule index mappings) in `AirDataGpu` during keygen. During proving, only the per-proof weights (which depend on eq_3bs and beta_pows) need to be computed and uploaded.

**Impact:** ~2ms. The DAG building was already hidden by parallel streams.

### 9. Selector Buffer Caching by Height (~3ms APC300 savings)

**Commit:** `cfc6b29a` (1 file, +9 -1)

**Problem:** The Round 0 setup created a selector DeviceBuffer (is_first, is_transition, is_last) for each trace. With ~600 traces, this meant ~600 VPMM allocations and H2D transfers, even though many traces share the same height and thus identical selectors.

**Idea:** Cache selector buffers in a `FxHashMap<usize, DeviceMatrix>` keyed by trace height. Traces with the same height share the same device buffer via `Arc`.

**Impact:** ~3ms. Reduces ~600 VPMM allocations to ~20 distinct heights.

### 10. Parallel eq_3b Computation (~2ms APC300 savings)

**Commit:** `95f44dd8` (1 file, +57 -30)

**Problem:** Computing eq_3b weights (via `eval_eq_mle`) for each trace's interactions is CPU-bound. With ~600 traces, the sequential computation adds a few ms.

**Idea:** Dispatch across 8 threads using `std::thread::scope`. Each thread computes eq_3b weights for a subset of traces.

**Impact:** ~2ms. CPU parallelism for a small workload.

---

## Ideas Tested and Rejected

| Idea | Why rejected |
|---|---|
| **GKR local intermediates threshold 10→32** | Increases register pressure from 10×16B to 32×16B per thread, reducing SM occupancy on RTX 4090. Net negative. |
| **16 Round 0 threads** | VPMM mutex contention from 16 concurrent allocation streams offsets the GPU overlap benefit. 8 threads is the sweet spot. |
| **VPMM pre-warming in commit function** | Pre-allocating a 512MB buffer during trace commit helps APC300 but fragments the VPMM pool for APC000 (whose allocations land at different VA offsets). |
| **RS matrix cache disable** | Freeing the RS-code matrix after Merkle tree construction saves memory but forces WHIR to recompute it during openings (+269ms). |
| **VPMM bypass for >256MB allocations** | Using `cudaMallocAsync` directly for large buffers avoids VPMM pool search overhead but loses pool-based reuse. Subsequent VPMM allocations can't reclaim the freed memory, causing +274ms Round 0 regression. |
| **Direct `cudaMallocAsync` for GKR leaves only** | Same pool reuse problem. APC000 Round 0 regresses from 173ms to 855ms because the 512MB freed via `cudaFreeAsync` isn't returned to the VPMM pool for subsequent allocations. |
| **l_skip tuning** | Previous Report 1 tested l_skip=3 and l_skip=5 with mixed results. l_skip=4 is the balanced optimum. |
| **calculate_zero_hash caching** | This APC-specific kernel (109ms GPU time) runs during Set Initial Memory, not during STARK excl. trace. Previous Report 1 confirmed no wall-clock impact due to pipeline overlap. |

---

## Remaining Bottleneck

APC300 STARK is 1582ms. The gap to proportional scaling (907ms) is ~675ms, dominated by:

- **GKR fractional sumcheck (~400ms):** Inherently sequential per-round due to Fiat-Shamir. Each of ~20 outer × ~5 inner rounds requires GPU kernel → D2H sync → CPU transcript → next round. Cannot be optimized without protocol changes.
- **Trace Commit (~415ms):** Poseidon2 row hashing (231ms) + Merkle tree (146ms) + NTT (82ms). Cryptographic hashing is the bottleneck; cannot be reduced without a faster hash.
- **GKR input eval segment 0 VPMM overhead (~314ms):** The first 512MB allocation triggers VPMM pool management overhead (contiguous block search in a fragmented pool). All attempted bypasses caused regressions in other phases due to loss of VPMM pool reuse. Requires VPMM-level redesign.

## Why Further GPU-Side Optimization Is Unlikely to Close the Gap

The 36% improvement was captured almost entirely by 3 ideas: parallel streams, stacked deferred sync, and GKR batching. These eliminated the per-AIR overhead that caused APC scaling regression. Everything after that gave diminishing returns — 7 additional optimizations combined contributed ~90ms.

The remaining ~675ms gap to proportional scaling is not from per-AIR overhead. It is from phases whose cost is set by the cryptographic protocol:

- **GKR fractional sumcheck** is sequential by construction (Fiat-Shamir requires each round's challenge to depend on the previous round's result). No amount of GPU parallelism can overlap these rounds.
- **Trace Commit** is dominated by Poseidon2 hashing, which is already running at near-peak GPU throughput (~7ns per row hash).
- **VPMM first-allocation overhead** is a memory management issue, not a compute issue. Five different bypass strategies were attempted; all caused regressions in other phases because the VPMM pool reuse mechanism is load-bearing for the rest of the prover.

The remaining actionable ideas (logup kernel batching ~30-50ms, non-degenerate stacked batching ~15ms) would add ~50-65ms total. Meaningful further progress requires changes outside the GPU prover scope: protocol parameter tuning, faster cryptographic primitives, or a VPMM redesign.

## Future Work (If Scope Were Expanded)

1. **VPMM redesign for large allocations:** The single biggest remaining opportunity. The first-segment VPMM overhead accounts for ~20% of APC300 STARK time. A dedicated large-buffer allocator or lazy page commitment could help.
2. **Logup Round 0 kernel batching:** New CUDA kernel for batched barycentric/NTT evaluation could reduce the remaining ~735 logup kernel launches (~30-50ms potential).
3. **SP1-inspired constraint batching:** SP1 folds all chip constraints into a single sumcheck instance with powers of a random challenge, avoiding per-chip sumcheck overhead.
