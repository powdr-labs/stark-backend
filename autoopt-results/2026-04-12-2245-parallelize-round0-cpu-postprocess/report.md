# Report: Parallelize Round 0 CPU Post-Processing

## Description

The optimization aimed to parallelize CPU-bound post-processing (transpose, iDFT, polynomial coefficient assembly) in the Round 0 constraint evaluation loop across AIRs using rayon. The hypothesis was that ~200ms of Round 0's 662ms total at APC 300 was spent on serial CPU post-processing (based on the ~462ms GPU kernel time measured by nsight in the previous task, leaving ~200ms of CPU-only work). With 623 AIRs processed independently, this work is embarrassingly parallel and should compress to ~7-13ms with 32 cores.

## Implementation

**File changed:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

**Changes:**
1. Added `p3_maybe_rayon::prelude::*` import to the existing `openvm_stark_backend` use block
2. Defined `Round0AirRaw` struct to hold per-AIR raw D2H data and metadata
3. Split the single per-AIR loop (lines 732-880) into two phases:
   - **Phase 1 (sequential):** GPU kernel launches + D2H copies, collecting raw results into `Vec<Round0AirRaw>`
   - **Phase 2 (parallel via rayon):** CPU post-processing (transpose, iDFT, polynomial construction) using `into_par_iter()`
4. Added a scatter step to write parallel results back into `batch_sp_poly`
5. Preserved the `debug_assert` for zerocheck sum validation inside the parallel closure

**Deviations from plan:** None. Implementation followed the plan exactly.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| **APC 0** | | | | | |
| Round 0 | 178 ms | 179 ms | 178 ms | 0ms (1.00x) | -1ms (1.01x lower) |
| STARK excl trace | 2153 ms | 2165 ms | 2178 ms | +25ms (1.01x higher) | +13ms (1.01x higher) |
| **APC 100** | | | | | |
| Round 0 | 464 ms | 465 ms | 466 ms | +2ms (1.00x) | +1ms (1.00x) |
| STARK excl trace | 2155 ms | 2163 ms | 2160 ms | +5ms (1.00x) | -3ms (1.00x) |
| **APC 300** | | | | | |
| Round 0 | 662 ms | 667 ms | 657 ms | -5ms (1.01x lower) | -10ms (1.02x lower) |
| STARK excl trace | 2455 ms | 2468 ms | 2455 ms | 0ms (1.00x) | -13ms (1.01x lower) |

## Assessment

**Result: FAILURE** — The optimization did not meet the 10% improvement threshold for Round 0 at APC 300. The measured improvement was only 10ms (1.5%), far below the required 66ms (10%).

**Root cause analysis:**

The 200ms CPU overhead estimate was incorrect. The gap between total Round 0 time (~662ms) and GPU kernel time (~462ms from nsight) includes not just CPU post-processing compute, but also:
- D2H memory copy latency (`to_host()` calls, still sequential in Phase 1)
- Kernel launch overhead and driver-side bookkeeping
- SymbolicConstraints construction from proving key (CPU, per AIR)
- Various small allocations and metadata assembly

The actual CPU compute (transpose + iDFT) per AIR is likely very small — with `l_skip` typically being small and `num_cosets_zc` being low (constraint degree - 1), each transpose+iDFT operation takes microseconds, not the ~0.32ms/AIR estimated. The total CPU post-processing across 623 AIRs may be only ~10-20ms, making parallelization yield marginal gains.

The dominant overhead in the 200ms gap is likely D2H synchronization and kernel launch overhead, which this optimization does not address (D2H copies must remain sequential on the CUDA stream).

## Future Work

- **What worked well:** The two-phase pattern (GPU collection then CPU processing) is a clean refactor with no correctness risk. The code is arguably more readable.
- **What didn't work:** The CPU post-processing is not the bottleneck. The time gap between GPU kernel execution and total Round 0 is dominated by D2H copies and kernel launch overhead, not transpose/iDFT compute.
- **Better approaches for Round 0:**
  - Multi-stream GPU parallelism: launch kernels for independent AIRs on separate CUDA streams to overlap kernel execution with D2H copies
  - Kernel fusion: combine constraint and interaction evaluation into a single kernel launch per AIR to reduce launch overhead
  - Batched kernel launches: group small AIRs into a single kernel invocation
  - GPU-side post-processing: move the transpose and iDFT to GPU kernels to avoid D2H copies entirely for intermediate results
- **Key learning:** When nsight shows a gap between GPU kernel time and wall clock time, the gap is not necessarily CPU compute — it includes synchronization, D2H transfers, and driver overhead. Profile CPU-side work independently (e.g., with tracing spans around the transpose/iDFT code) before assuming it's the bottleneck.
