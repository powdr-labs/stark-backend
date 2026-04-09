# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1612ms** (baseline 2491ms, **-35.3%**)
Target (<1084ms): gap ~528ms. This gap is structurally limited:
- GKR fractional sumcheck: ~400ms (sequential Fiat-Shamir, cannot pipeline)
- Trace Commit: ~419ms (Poseidon2 cryptographic hashing)
- These alone: ~819ms, leaving only 265ms for everything else

Reaching the 2x target requires either protocol-level changes or faster
cryptographic primitives, both out of scope.

APC300 breakdown: GKR 602ms | Trace Commit 417ms | Round 0 216ms | MLE 179ms | Stacked 88ms | WHIR 100ms

## Remaining practical ideas (diminishing returns)

### 1. Reduce MLE fold allocation count
Pre-allocate output buffers for fold_mle_evals (ping-pong pattern).
~5000 allocations through VPMM mutex per proof. Expected: ~10-20ms.

### 2. Logup Round 0 kernel batching (CUDA kernel)
Need new batched CUDA kernel for barycentric/NTT logup evaluation.
Expected: ~30-50ms.

### 3. VPMM page pre-commitment
GKR input eval segment 0 pays 170ms cuMemSetAccess penalty for first
large allocation. Pre-warming during commit (outside STARK span)
helps APC300 but hurts APC000 (fragmentation). Needs size-aware pre-warming.
Expected: ~100-170ms for APC300 STARK, negative impact on APC000.

### 4. Investigate Poseidon2 kernel optimization
231ms for row hashing. Check if there's a faster compression approach
for narrow stacked matrices (27 columns per row for APC300).

### 5. Multi-GPU distribution
Use 2+ GPUs to parallelize segment processing. Each GPU handles one
segment independently. Expected: ~2x speedup for 2 segments.
