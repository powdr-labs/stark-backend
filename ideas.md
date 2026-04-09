# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1558ms** (baseline 2491ms, **-37.5%**)
Target (<1084ms): gap ~474ms.

## Key bottleneck: GKR input eval seg0 overhead (316ms vs 2ms seg1)
The 314ms difference comes from VPMM pool management for the first 512MB
allocation (finding/splitting contiguous free regions). This is structural
to the VPMM design and requires allocator-level changes to fix.

## Remaining ideas

### 1. Custom allocator for GKR leaves buffer
Bypass VPMM entirely for the leaves buffer: allocate a persistent buffer
once and reuse across segments. Requires changing log_gkr_input_evals to
accept a pre-allocated buffer.

### 2. Batch non-degenerate stacked reduction MLE kernel (5076 launches, 21ms)

### 3. SP1-inspired slope-based interpolation in sumcheck
Fixed evaluation points with precomputed slopes could reduce per-round overhead.

### 4. Profile-guided kernel occupancy tuning with ncu
