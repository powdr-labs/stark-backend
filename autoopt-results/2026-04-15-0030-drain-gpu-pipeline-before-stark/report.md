# Report: Drain GPU Pipeline Before STARK Timing Span

## Description

The hypothesis was that async trace generation GPU kernels (especially `apc_apply_bus_kernel` at APC 300) were still executing when the STARK prover began, causing ~130-189ms of pipeline stall that inflated the `stark_prove_excluding_trace` metric. The fix was to add an explicit `current_stream_sync()` before the STARK timing span, draining any pending GPU operations so the metric accurately reflects only STARK work.

The expected mechanism: trace gen launches GPU kernels asynchronously, the CPU returns quickly, but those kernels are still running on `cudaStreamPerThread` when `prove()` is called. The first GPU operation inside STARK (the `stacked_commit` in Trace Commit) would then block on `to_host()` → `cudaEventSynchronize` until all prior stream work completes, inflating the Trace Commit sub-metric.

## Implementation

Three changes were made:

1. **`crates/stark-backend/src/prover/hal.rs`**: Added `drain_pending_device_ops()` method to the `ProverDevice` trait with a default no-op implementation (for CPU backends). Returns `Result<(), <Self as ProverDevice<PB, TS>>::Error>` to match the error-handling pattern.

2. **`crates/cuda-backend/src/gpu_backend.rs`**: Implemented `drain_pending_device_ops()` for `GpuDevice`, calling `openvm_cuda_common::stream::current_stream_sync()` with `ProverError::CurrentStreamSync` error mapping.

3. **`crates/stark-backend/src/prover/mod.rs`**: Restructured `Coordinator::prove` into two parts:
   - `prove()` (trait method): calls `drain_pending_device_ops()` under a `prover.drain_pipeline` tracing span, then delegates to `prove_stark()`.
   - `prove_stark()` (private method on a separate impl block): carries the `#[instrument(name = "stark_prove_excluding_trace")]` attribute and contains the original prove body unchanged.

No deviations from the plan.

## Results

| Metric | Baseline | Before Task | After Task | vs Baseline | vs Before |
|--------|----------|-------------|------------|-------------|-----------|
| STARK excl trace APC 300 | 2455ms | 1300ms | 1312ms | -1143ms, 1.87x lower | +12ms, 1.01x higher |
| STARK excl trace APC 0 | 2153ms | 2155ms | 2169ms | +16ms, 1.01x higher | +14ms, 1.01x higher |
| Trace Commit APC 300 | 406ms | 255ms | 250ms | -156ms, 1.62x lower | -5ms, 1.02x lower |
| Trace Commit APC 0 | 518ms | 531ms | 530ms | +12ms, 1.02x higher | -1ms, unchanged |
| Total APC 300 | 6868ms | 5702ms | 5711ms | -1157ms, 1.20x lower | +9ms, unchanged |
| Total APC 0 | 5077ms | 5059ms | 5051ms | -26ms, unchanged | -8ms, unchanged |
| APC 0/APC 300 STARK ratio | 0.88x | 1.66x | 1.65x | — | unchanged |
| Drain pipeline seg0 APC 300 | — | — | 0ms | — | — |
| Drain pipeline seg1 APC 300 | — | — | 0ms | — | — |

Per-segment detail (APC 300):

| Segment | Before main_trace_commit | After main_trace_commit | After drain_pipeline |
|---------|-------------------------|------------------------|---------------------|
| seg0 | 166ms | 161ms | 0ms |
| seg1 | 89ms | 89ms | 0ms |

## Assessment

**The optimization did NOT achieve its goal.** The drain pipeline sync completes in 0ms for all segments at all APC configurations, conclusively disproving the pipeline stall hypothesis.

The drain time of 0ms means that by the time `Coordinator::prove()` is called, all trace gen GPU kernels have already completed on `cudaStreamPerThread`. There is no async GPU work pending from prior phases.

This means the ~160ms Trace Commit time in segment 0 at APC 300 is actual commit work (stacking, Merkle tree construction, etc.), not pipeline stall from trace gen kernels. The previous analysis that attributed the difference between 164ms Trace Commit and ~35ms "actual commit work" to pipeline stall was incorrect — the ~35ms figure from the "debug run" was likely measured differently or in a different context.

The STARK excl trace metric was already accurately measuring STARK work before this change. The +12ms variation at APC 300 is within normal measurement noise.

## Future Work

- The 160ms Trace Commit at APC 300 seg0 is genuine commit work. To reduce it further, the stacking scatter kernel (already optimized in a prior task) or the Merkle tree construction could be investigated.
- The hypothesis that trace gen and STARK compete for GPU resources was based on nsight profiling showing kernel overlap. The actual code path may have implicit synchronization points (e.g., between trace gen and prove invocation in the caller) that drain the pipeline before `prove()` is called. Investigating the caller's code path could clarify where this sync happens.
- The 0ms drain time suggests the caller already performs an equivalent sync (possibly through memory copies or event waits during trace transport/commit preparation). Understanding this implicit sync point would prevent similar incorrect hypotheses in future.
