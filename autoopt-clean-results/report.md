# AutoOpt 2026-04-12 (Clean Run)

This is a clean re-implementation of the `stark-backend` improvements found by [autoopt](https://github.com/georgwiese/autoopt) using [this context markdown files](https://gist.github.com/georgwiese/d68a67c583ecdfb3ffd0e448c657b68c).
Each improvement is in a separate commit, roughly sorted by their impact.

## Benchmark & problem statement

For this experiment, we looked into the [pairing guest program](https://github.com/powdr-labs/powdr/tree/openvm-v2-integration-07-04/openvm-riscv/guest-pairing) ([APC analyzer](https://powdr-labs.github.io/powdr/autoprecompile-analyzer/?data=https%3A%2F%2Fgist.githubusercontent.com%2Fleonardoalt%2F73b455213ef4f987bf330559b714cb65%2Fraw%2F521c6644ae3efa35d39b5e742520f9dddf2a7480%2Fapc_candidates_pairing_v2.json)).
In this guest program, applying autoprecompiles achieves significant reductions in the number of trace cells, constraint instances and bus interaction messages.

| Experiment             | Trace cells      | Constraint instances | Bus interaction messages |
|------------------------|------------------|----------------------|--------------------------|
| APC=0 (vanilla OpenVM) | 1.90B            | 1.13B                | 965.5M                   |
| APC=100                | 1.19B (1.60x ↓)  | 40.4M (1.52x ↓)      | 621.8M (1.55x ↓)         |
| APC=300                | 811.2M (2.35x ↓) | 505.2M (2.23x ↓)     | 448.5M (2.15x ↓)         |

However, this does not translate to a reduction in STARK proving time (excluding trace generation):
- For APC=100, the proving time is [up 1.10x](https://powdr-labs.github.io/powdr/openvm/metrics-viewer/?data=https%3A%2F%2Fgithub.com%2Fpowdr-labs%2Fstark-backend%2Fblob%2Fautoopt-2026-04-12-clean%2Fautoopt-clean-results%2Fstep-08-gpu-round0-overlap-logup%2Fcombined_metrics.json&baseline=baseline_apc000&run=baseline_apc100).
- For APC=300, the proving time is [up 1.29x](https://powdr-labs.github.io/powdr/openvm/metrics-viewer/?data=https%3A%2F%2Fgithub.com%2Fpowdr-labs%2Fstark-backend%2Fblob%2Fautoopt-2026-04-12-clean%2Fautoopt-clean-results%2Fstep-08-gpu-round0-overlap-logup%2Fcombined_metrics.json&baseline=baseline_apc000&run=baseline_apc300).

## Results

With the proposed changes, the trend reverses:

![STARK proving time excluding trace by iteration](stark_excl_trace_by_iteration.svg)

With all 8 optimizations applied, the STARK proving time excluding trace generation is:
- For APC=100, the proving time is [down 1.35x](https://powdr-labs.github.io/powdr/openvm/metrics-viewer/?data=https%3A%2F%2Fgithub.com%2Fpowdr-labs%2Fstark-backend%2Fblob%2Fautoopt-2026-04-12-clean%2Fautoopt-clean-results%2Fstep-08-gpu-round0-overlap-logup%2Fcombined_metrics.json&baseline=step-08-gpu-round0-overlap-logup_apc000&run=step-08-gpu-round0-overlap-logup_apc100).
- For APC=300, the proving time is [down 1.56x](https://powdr-labs.github.io/powdr/openvm/metrics-viewer/?data=https%3A%2F%2Fgithub.com%2Fpowdr-labs%2Fstark-backend%2Fblob%2Fautoopt-2026-04-12-clean%2Fautoopt-clean-results%2Fstep-08-gpu-round0-overlap-logup%2Fcombined_metrics.json&baseline=step-08-gpu-round0-overlap-logup_apc000&run=step-08-gpu-round0-overlap-logup_apc300).

While still not scaling linearly with the statistics above, the proposed prover changes lead to a significant reduction in proving time as we increase the number of autoprecompiles.

## Overview of the changes

The following table summarizes the effect of the proposed changes, on STARK proving time (excluding trace generation):

| # | Change | Diffstat | APC 0 | APC 100 | APC 300 |
|---|--------|----------|-------|---------|---------|
| 0 | Baseline (`VPMM_PAGE_SIZE=16777216`) | — | 1,795ms | 1,982ms | 2,307ms |
| 1 | Multi-stream Round 0 (8 streams) | +265/-140 | 1,812ms<br>vs base: +17ms (1.01x ↑)<br>vs prev: +17ms (1.01x ↑) | 1,745ms<br>vs base: -237ms (1.14x ↓)<br>vs prev: -237ms (1.14x ↓) | 1,915ms<br>vs base: -392ms (1.20x ↓)<br>vs prev: -392ms (1.20x ↓) |
| 2 | Multi-stream GKR input eval (8 streams) | +184/-108 | 1,818ms<br>vs base: +23ms (1.01x ↑)<br>vs prev: +6ms (1.00x ↑) | 1,739ms<br>vs base: -243ms (1.14x ↓)<br>vs prev: -6ms (1.00x ↓) | 1,642ms<br>vs base: -665ms (1.41x ↓)<br>vs prev: -273ms (1.17x ↓) |
| 3 | Batch stacked-reduction MLE sync | +22/-40 | 1,789ms<br>vs base: -6ms (1.00x ↓)<br>vs prev: -29ms (1.02x ↓) | 1,489ms<br>vs base: -493ms (1.33x ↓)<br>vs prev: -250ms (1.17x ↓) | 1,458ms<br>vs base: -849ms (1.58x ↓)<br>vs prev: -184ms (1.13x ↓) |
| 4 | Batch stacking scatter kernel | +83/-32 | 1,797ms<br>vs base: +2ms (1.00x ↑)<br>vs prev: +8ms (1.00x ↑) | 1,407ms<br>vs base: -575ms (1.41x ↓)<br>vs prev: -82ms (1.06x ↓) | 1,316ms<br>vs base: -991ms (1.75x ↓)<br>vs prev: -142ms (1.11x ↓) |
| 5 | Batch stacked-reduction MLE round kernels | +456/-49 | 1,785ms<br>vs base: -10ms (1.01x ↓)<br>vs prev: -12ms (1.01x ↓) | 1,396ms<br>vs base: -586ms (1.42x ↓)<br>vs prev: -11ms (1.01x ↓) | 1,250ms<br>vs base: -1,057ms (1.85x ↓)<br>vs prev: -66ms (1.05x ↓) |
| 6 | Pre-allocate GKR input buffers | +93/-36 | 1,804ms<br>vs base: +9ms (1.01x ↑)<br>vs prev: +19ms (1.01x ↑) | 1,373ms<br>vs base: -609ms (1.44x ↓)<br>vs prev: -23ms (1.02x ↓) | 1,223ms<br>vs base: -1,084ms (1.89x ↓)<br>vs prev: -27ms (1.02x ↓) |
| 7 | Pre-allocate Round 0 buffers + round-robin balance | +343/-42 | 1,804ms<br>vs base: +9ms (1.01x ↑)<br>vs prev: 0ms (flat) | 1,347ms<br>vs base: -635ms (1.47x ↓)<br>vs prev: -26ms (1.02x ↓) | 1,213ms<br>vs base: -1,094ms (1.90x ↓)<br>vs prev: -10ms (1.01x ↓) |
| 8 | GPU Round 0 poly extract + overlap logup precompute | +466/-134 | 1,792ms<br>vs base: -3ms (1.00x ↓)<br>vs prev: -12ms (1.01x ↓) | 1,331ms<br>vs base: -651ms (1.49x ↓)<br>vs prev: -16ms (1.01x ↓) | 1,148ms<br>vs base: -1,159ms (2.01x ↓)<br>vs prev: -65ms (1.06x ↓) |

Each  change has a significant effect on STARK proving time, with each change being a reasonably-sized diff. The [full diff](https://github.com/powdr-labs/stark-backend/compare/v2-powdr-07-04...powdr-labs:stark-backend:1ca18279fa6f12d907d2c4bc7265eeaeda2025d7) is +1,732/-402 lines of code across 14 files.

See the [detailed reports](./detailed-reports.md) for more information on each change.

### Experimental setup

All experiments were run on an NVidea GeForce RTX 4090.
Also, we set `VPMM_PAGE_SIZE=16777216` for every run, including the baseline. This optimization was one of the items found by `autoopt`. It benefits all runs, including the baseline (1.19x faster).

### Noise

Each step is a single benchmark run. The original autoopt run characterized noise at roughly ±20 ms for APC 300 and ±15 ms for APC 0/100. APC 0 drifts in the ±30 ms band throughout steps 2-7; APC 0 uses the single-threaded path for Round 0 / GKR input eval (`<100` AIRs), so most of these changes are effectively no-ops for APC 0 and the small positive drifts vs baseline read as noise rather than a real regression. Step 8 pulls APC 0 back to -3 ms vs baseline.

## Notes on the original run

See the [original `autoopt-2026-03-12` summary report](https://github.com/powdr-labs/stark-backend/blob/autoopt-2026-04-12/autoopt-results/summary-report.md) for all the changes that were tried. Note that the "Increase default VPMM page size 2 MiB -> 16 MiB" idea can also be implemented by setting `VPMM_PAGE_SIZE=16777216`, which is now part of the baseline.
