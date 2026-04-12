# Report: Defer Round 0 D2H Synchronization

## Description

Round 0 of the constraint evaluation phase in the GPU prover processes each AIR instance sequentially, launching a zerocheck kernel and a logup kernel per AIR, each followed by a **blocking D2H transfer** (`to_host()` -> `cudaEventSynchronize`). At APC 300 with 623 AIR instances, this creates up to ~1,246 blocking synchronization points, causing the GPU to sit idle during CPU-side IDFT processing and the CPU to block waiting for each tiny kernel result (<1KB each).

The optimization restructures the per-AIR loop in `sumcheck_uni_round0_polys` into two phases:
- **Phase 1 (GPU launch)**: Iterate all AIRs, launch both kernels per AIR, collect GPU result buffers without any D2H transfers.
- **Phase 2 (CPU processing)**: Explicit `current_stream_sync()`, then D2H transfers + IDFT + polynomial construction.

This was expected to eliminate GPU pipeline stalls between kernel launches and allow CPU setup work for AIR[i+1] to overlap with GPU kernel execution for AIR[i].

## Implementation

**File modified**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Changes:
1. Added `use openvm_cuda_common::stream::current_stream_sync;` import.
2. Defined a local `Round0Pending<F, EF>` struct to hold per-AIR metadata (trace_idx, air_idx, coset counts, omega_root, n, zc_result, logup_result) needed for Phase 2 processing.
3. Replaced the single per-AIR loop (original lines 730-880) with:
   - **Phase 1 loop**: Iterates all AIR instances, launches zerocheck and logup kernels, stores `DeviceBuffer` results and metadata in `pending: Vec<Round0Pending>`. Also collects `d_main_parts` into a separate vec to keep them alive.
   - **Explicit stream sync**: `current_stream_sync().map_err(...)` after all kernels are queued.
   - **Phase 2 loop**: Processes all results — `to_host()` calls (now fast since stream is already synced), IDFT, polynomial construction. Logic is identical to the original code.
4. Error conversion: `CudaError` from `current_stream_sync()` is mapped through `MemCopyError::from(CudaError)` -> `LogupZerocheckError::from(MemCopyError)`, using existing `#[from]` derives.

**Deviations from plan**: None. The implementation follows the plan exactly.

## Results

All measurements are averages of 2 runs (confirmed consistent to within ~3ms).

### Round 0

| Config | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| APC 0 | 178ms | 178ms | 175ms | -3ms (1.02x lower) | -3ms (1.02x lower) |
| APC 100 | 463ms | 466ms | 380ms | -83ms (1.22x lower) | -86ms (1.23x lower) |
| APC 300 | 663ms | 662ms | 569ms | -94ms (1.17x lower) | -93ms (1.16x lower) |

### STARK (excl. trace)

| Config | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| APC 0 | 2149ms | 2165ms | 2146ms | -3ms (1.00x lower) | -19ms (1.01x lower) |
| APC 100 | 2159ms | 2156ms | 2077ms | -82ms (1.04x lower) | -79ms (1.04x lower) |
| APC 300 | 2478ms | 2458ms | 2374ms | -104ms (1.04x lower) | -84ms (1.04x lower) |

### Constraints (total)

| Config | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| APC 0 | 1290ms | 1301ms | 1290ms | 0ms | -11ms |
| APC 100 | 1398ms | 1394ms | 1310ms | -88ms (1.07x lower) | -84ms (1.06x lower) |
| APC 300 | 1663ms | 1641ms | 1542ms | -121ms (1.08x lower) | -99ms (1.06x lower) |

### Scaling Ratio (APC 300 / APC 0 STARK excl trace)

| | Baseline | Before | After |
|---|---|---|---|
| Ratio | 1.153 | 1.135 | 1.106 |

### Correctness

All 94 `openvm-cuda-backend` tests pass. Proof verification succeeds for all APC configurations (the benchmark verifies proofs as part of the recursion step).

### GPU Memory

Peak GPU memory (from prove output) unchanged at ~1.6 GiB for APC 300, same as baseline.

## Assessment

The optimization **achieved a measurable improvement** but fell short of the primary success criteria:

- **Primary target (STARK excl trace APC 300 -150ms)**: Achieved -84ms vs before task, -104ms vs baseline. **Not fully met**, though directionally positive.
- **Secondary target (Round 0 APC 300 -30%)**: Achieved -14% (662->569ms). **Not met** — the target was 30%.
- **Scaling ratio target (<1.10)**: Achieved 1.106, very close to the 1.10 target. **Mostly met**.

The optimization clearly works — Round 0 time for APC 300 dropped by 93ms, scaling proportionally with instance count (APC 100 saw -86ms, APC 0 saw negligible change as expected). The gap from the predicted ~200-300ms improvement is likely because:

1. **CPU-side work in `evaluate_round0_interactions_gpu` already provided partial overlap**: The logup kernel launch function does significant CPU work (DAG construction, rule encoding, weight computation, H2D transfers for interaction data) before launching its kernel. Some of this CPU work was already overlapping with GPU execution in the original code, since the blocking sync comes *after* the kernel launch, not before the setup of the next iteration.
2. **The `cudaEventSynchronize` overhead itself is smaller than estimated**: The plan estimated ~200-300ms from eliminating 1,246 sync points. The actual overhead per sync point is likely <100us (hardware dependent), putting the total at ~60-120ms rather than 200-300ms — consistent with the observed 93ms improvement.

Despite not meeting the ambitious targets, the optimization is clearly worth keeping:
- 84-104ms improvement on APC 300 STARK excl trace (3-4% reduction)
- Zero correctness risk (pure loop restructuring, no CUDA kernel changes)
- Minimal code complexity increase
- Improved scaling ratio (1.15 -> 1.11)

## Future Work

- **What worked well**: The two-phase approach cleanly separates GPU launch from CPU processing. The `Round0Pending` struct is a natural abstraction. The explicit `current_stream_sync()` makes the synchronization point visible and debuggable.

- **Rayon parallelization of Phase 2**: Now that Phase 2 is isolated, the CPU-side IDFT processing is embarrassingly parallel (each AIR writes to distinct `batch_sp_poly` indices). Converting to `par_iter` could save an additional 30-50ms on APC 300 by parallelizing the 623 independent IDFTs. This is a natural follow-up.

- **Multi-stream pipelining**: Use multiple CUDA streams to overlap Phase 1 GPU execution with Phase 2 CPU processing. While the deferred approach eliminated CPU-GPU interleaving overhead, it now serializes all GPU work before all CPU work. With 2-4 streams and careful synchronization, GPU kernels for later AIRs could execute while the CPU processes earlier results. Expected additional savings: 50-100ms.

- **Apply the same pattern to other sumcheck rounds**: MLE rounds and other constraint evaluation phases may have similar per-AIR sequential D2H patterns. The two-phase approach could be applied there, though the savings would be smaller (fewer instances per round after folding).

- **Batch stacked reduction degenerate kernels**: The next-highest-impact optimization target. Stacked Reduction at APC 300 is 309ms with ~10,348 tiny single-block kernel launches. Batching these into ~17 launches per round could save ~100-150ms.
