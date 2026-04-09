# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1573ms** (baseline 2491ms, **-36.9%**)
Target (<1084ms): gap ~489ms, dominated by GKR (~593ms) + Trace Commit (~417ms).

## Still to try

### 1. Batch d_eq_3b uploads into single transfer
600 per-trace H2D transfers for eq_3b → single concatenated upload with offset tracking.
Expected: ~3-5ms.

### 2. Overlap eq_xis construction with d_eq_3b upload
eq_xis and d_eq_3b upload are independent. Run on separate threads/streams.
Expected: ~2-3ms overlap.

### 3. Batch non-degenerate stacked reduction MLE kernel (5076 launches, 21.7ms)
Similar to degenerate batching. Expected: ~15ms.

### 4. Pre-allocate GKR leaves buffer before STARK span
Move the 512MB VPMM allocation cost (~170ms for seg0) outside of STARK excl trace.
Challenge: fragmentation for APC000. Needs size-aware approach.

### 5. Investigate GPU occupancy for hot kernels
Use ncu to check if any kernel has low occupancy that could be improved with launch bounds.
