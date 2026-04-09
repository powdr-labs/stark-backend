# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1643ms** (baseline 2491ms, **-34%**)
Target (cell-proportional to APC000): ~913ms. Remaining gap: ~730ms.

APC300 breakdown: GKR 614ms | Trace Commit 419ms | Round 0 233ms | MLE 182ms | Stacked Red 89ms | WHIR 101ms

## Priority Order

### 1. Round 0 kernel batching (cross-AIR, CUDA-level)
735 zerocheck + 735 logup coset-parallel kernel launches, many with grid=(1,2) = 2 blocks. Batching into single launches (one block per AIR) would improve GPU SM utilization from ~2% to ~50% for small AIRs. The pacheco branch has code (d2c8980..a172f59) but needs NttEvalContext API fixes and conflict resolution with parallel streams.

**Expected savings:** ~50-100ms for APC300 (improved utilization, reduced launch overhead).

### 2. MLE memory allocation reduction
fold_mle_evals allocates ~5000 DeviceBuffers across 10 rounds × 500 traces, each going through the VPMM global mutex. Pre-allocate a buffer pool and reuse across rounds (ping-pong pattern).

**Expected savings:** ~10-20ms for APC300.

### 3. Trace Commit stacking optimization
419ms for APC300. The stacked matrix has 106K columns but short heights. Investigate if the commitment can be structured to avoid per-column overhead.

**Expected savings:** Unknown, needs investigation.

### 4. GKR fractional sumcheck pipelining
614ms, inherently sequential per-round (Fiat-Shamir). But ~370ms is CPU overhead between rounds (VPMM mutex, buffer allocation, eq construction). Could pre-allocate round buffers and reduce per-round allocation overhead.

**Expected savings:** ~20-50ms for APC300.

### 5. Combine: batched Round 0 + parallel streams
Implement batched kernels for small AIRs, fall back to per-AIR for large AIRs, dispatch both on parallel streams. This compounds the batching win with the stream overlap win.

**Expected savings:** ~80-150ms for APC300.

### 6. SP1 GPU prover analysis
Research what SP1 does differently for GPU proving. May reveal architectural ideas.

## Tested but rejected

- GKR threshold 10->32: Hurts RTX 4090 occupancy
- l_skip tuning: Report 1 found l_skip=4 is optimal
- calculate_zero_hash caching: Hidden behind pipeline overlap (Report 1)
- 16 Round 0 threads: No improvement over 8 (memory contention)
- GKR eq_buffer caching: Only ~1ms total, not the bottleneck
