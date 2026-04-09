# GPU Prover Optimization Ideas

Current best APC300 STARK excl. trace: **~1620ms** (baseline 2491ms, **-35%**)
Target (<1084ms = APC000/2): gap ~536ms.

APC300 breakdown: GKR 612ms | Trace Commit 419ms | Round 0 216ms | MLE 179ms | Stacked Red 88ms | WHIR 101ms

Remaining gap dominated by GKR (612ms) + Trace Commit (419ms) = 1031ms.
These are algorithmically hard to optimize (sequential Fiat-Shamir, cryptographic hashing).

## Priority Order

### 1. Reduce Trace Commit per-column overhead
With 106K columns (27x more than APC000), per-column Merkle tree overhead doesn't scale proportionally. Investigate whether leaf hashing can be more efficient for narrow rows. The poseidon2_compressing_row_hashes_kernel takes 231ms.

**Expected savings:** Unknown, needs kernel-level investigation.

### 2. GKR D2H sync reduction
GKR fractional sumcheck has ~100 D2H sync points across ~20 outer × ~5 inner rounds. Each sync waits for GPU kernel completion (~3ms). Can D2H be pipelined by launching next round's eq_buffer computation while current round's sum_evals are being transferred?

**Expected savings:** ~20-40ms (reduce sync wait time through overlap).

### 3. Reduce VPMM mutex contention
65K allocation + 65K deallocation calls total 92ms through global mutex. Consider: per-thread sub-pools, lock-free allocation for small buffers, or pre-allocating round scratch buffers.

**Expected savings:** ~20-40ms.

### 4. Logup Round 0 kernel batching
Batch the 735 logup_r0_ntt_eval_interactions kernel launches similar to zerocheck batching. Requires new batched CUDA kernel for barycentric/NTT evaluation.

**Expected savings:** ~30-50ms (on top of parallel streams).

### 5. Combined: all small wins compounding
Combine ideas #2, #3, #4 together for potential ~70-130ms total improvement.
