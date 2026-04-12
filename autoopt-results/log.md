# AutoOpt Log

## 2026-04-10-1933-defer-round0-d2h-sync

**Idea**: Defer D2H synchronization in Round 0 by splitting the per-AIR loop into a GPU launch phase and a CPU processing phase, eliminating ~1,246 blocking sync points.

**Result**: success — STARK excl trace APC 300: -104ms vs baseline (2478ms -> 2374ms, 1.04x lower); Round 0 APC 300: -94ms vs baseline (663ms -> 569ms, 1.17x lower)

**Summary**: Restructured `sumcheck_uni_round0_polys` to launch all GPU kernels before processing any results. The improvement scales with APC count as expected (APC 0: negligible, APC 100: -86ms Round 0, APC 300: -93ms Round 0). The absolute savings were lower than the plan's estimate of 200-300ms, likely because CPU-side work in the logup kernel setup already provided partial GPU overlap in the original code. The optimization establishes a clean two-phase structure that enables future rayon parallelization of Phase 2 CPU work.

## 2026-04-10-2145-batch-degenerate-stacked-reduction

**Idea**: Batch all degenerate-window kernel launches in Stacked Reduction into a single GPU kernel launch per round, eliminating ~10,348 tiny kernel launches at APC 300.

**Result**: success — STARK excl trace APC 300: -283ms vs baseline (2478ms -> 2195ms, 1.13x lower); Stacked Reduction APC 300: -161ms vs baseline (308ms -> 147ms, 2.10x lower) (vs previous: -165ms, 312ms -> 147ms, 2.12x lower)

**Summary**: Added a batched CUDA kernel where each block handles one degenerate window, all atomically accumulating into a shared output buffer. This replaced ~10,348 individual kernel launches (each with fill_zero + H2D + kernel + D2H sync) with a single launch cycle. The STARK excl trace APC 300/APC 0 scaling ratio improved from 1.153 (baseline) to 1.022, achieving near-parity. Combined with the previous Round 0 optimization, cumulative STARK excl trace improvement at APC 300 is 283ms (11.4%).

## 2026-04-11-0945-precompute-round0-logup-rules

**Idea**: Pre-compute Round 0 logup interaction rules (DAG, compiled GPU rules, weight map) at keygen time instead of rebuilding per-AIR at proving time. Also parallelize Phase 2 IDFT processing with rayon.

**Result**: failure — Round 0 APC 300: -2ms vs before (570ms -> 568ms), STARK excl trace APC 300: 0ms change (2216ms -> 2216ms). Reverted.

**Summary**: Moved static DAG construction, rule compilation, rule encoding, and H2D transfers from per-AIR proving-time to keygen-time pre-computation. The CPU work eliminated (~0.15-0.3ms per AIR) was already fully pipelined with asynchronous GPU kernel execution, so removing it produced no measurable improvement. The GPU pipeline was GPU-bound, not CPU-bound — the host launches kernels faster than the GPU completes them, so reducing CPU setup time has no effect. Phase 2 IDFT parallelization also showed no impact since the sequential IDFT was already fast (<10ms total). Key learning: the ~90ms "CPU overhead gap" between total Round 0 time and GPU kernel time is not eliminable CPU compute — it includes CUDA runtime dispatch overhead, buffer allocation latency, and pipeline drain.
