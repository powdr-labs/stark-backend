# Plan: Warp-Tiled GKR Intermediates Buffer Layout

## Goal

Improve GPU cache locality in the GKR input evaluation GLOBAL-mode kernel by reorganizing the intermediates buffer from a thread-interleaved layout to a warp-tiled layout. This targets the single largest GPU kernel (`evaluate_interactions_gkr_kernel<true>`, 464ms total GPU time, 486 instances at APC 300) which currently suffers from systematic L1/L2 cache misses due to a stride-65536 access pattern in its DAG node cache.

## Current Code Path

### Call chain

1. `log_gkr_input_evals()` at `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:215` — orchestrates multi-stream dispatch
2. `process_gkr_input_air()` at `gkr_input.rs:100` — processes one AIR on one OS thread
3. `logup_gkr_input_eval()` FFI wrapper at `crates/cuda-backend/src/cuda/logup_zerocheck.rs` — safe Rust wrapper
4. `_logup_gkr_input_eval()` C launcher at `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu:216` — launches CUDA kernel
5. `evaluate_interactions_gkr_kernel<true>` at `gkr_input.cu:65` — the kernel itself

### Current intermediates layout

The GLOBAL-mode intermediates buffer is allocated per-thread at `gkr_input.rs:308`:
```rust
intermediates: DeviceBuffer::with_capacity(max_intermediates_len),
// where max_intermediates_len = TASK_SIZE * max_buffer_size = 65536 * B
```

In the kernel (`gkr_input.cu:80-87`), each CUDA thread computes its pointer into the shared buffer:
```cpp
uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;  // thread's global ID
uint32_t task_stride = gridDim.x * blockDim.x;                 // = 65536 (TASK_SIZE)
intermediates_ptr = (FpExt *)d_intermediates + task_offset;
intermediate_stride = task_stride;  // = 65536
```

DAG nodes are accessed as `intermediates_ptr[node_idx * intermediate_stride]`, meaning:
- Thread `i` reads node `j` at address: `base + i + j * 65536`
- Within a warp (32 consecutive threads), all access the same node `j` at consecutive addresses → **coalesced** for a single node
- But within a single thread, consecutive nodes `j` and `j+1` are **65536 × 16 = 1MB apart** → every node-to-node access misses L1 and likely L2

### Why this is slow

- **Per-warp working set**: `buffer_size × 65536 × 16 bytes` ≈ 100 × 65536 × 16 ≈ **100MB** — far exceeds L1 (128KB) and L2 (72MB)
- **L1 hit rate**: ~0% (working set >> L1 capacity)
- **L2 hit rate**: ~72% (100MB vs 72MB L2) — 28% of accesses go to DRAM
- Each DAG evaluation reads/writes intermediates ~2× per node (read source + write result if buffered)
- With ~100 nodes per row and 65536 concurrent threads, this generates enormous memory traffic

### Grid configuration

- TASK_SIZE = 65536 = 2^16 (`gkr_input.cu:214`, `gkr_input.rs:27`)
- Block size = 256 threads
- Grid size = 256 blocks
- Total threads = 65536
- `num_rows_per_tile = ceil(height / 65536)` — each thread processes 1+ rows

## Changes

### 1. Add `buffer_size` parameter to CUDA kernel and launcher

**File:** `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`

Add `uint32_t buffer_size` parameter to `evaluate_interactions_gkr_kernel`:

```cpp
template <bool GLOBAL>
__global__ void evaluate_interactions_gkr_kernel(
    FracExt *__restrict__ d_fracs,
    // ... existing params ...
    const uint32_t num_rows_per_tile,
    const uint32_t buffer_size  // NEW
)
```

Add `uint32_t buffer_size` parameter to `_logup_gkr_input_eval` launcher (`gkr_input.cu:216`), pass through to kernel call (`gkr_input.cu:234`, `gkr_input.cu:249`). For the `GLOBAL=false` template instantiation, `buffer_size` is unused but still passed (consistent signature).

### 2. Change GLOBAL intermediates layout in kernel

**File:** `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`, lines 80-92

Replace the current GLOBAL pointer setup:

```cpp
// BEFORE:
uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t task_stride = gridDim.x * blockDim.x;
if constexpr (GLOBAL) {
    intermediates_ptr = (FpExt *)d_intermediates + task_offset;
    intermediate_stride = task_stride;
}
```

With warp-tiled layout (note: `WARP_SIZE` is available from `launcher.cuh`):

```cpp
// AFTER:
uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t task_stride = gridDim.x * blockDim.x;
if constexpr (GLOBAL) {
    uint32_t warp_id = task_offset / WARP_SIZE;
    uint32_t lane_id = task_offset % WARP_SIZE;
    intermediates_ptr = (FpExt *)d_intermediates + warp_id * buffer_size * WARP_SIZE + lane_id;
    intermediate_stride = WARP_SIZE;
}
```

Note: `WARP_SIZE` (= 32) is already defined in `launcher.cuh` as `static const size_t WARP_SIZE = 32;`. Use it directly rather than introducing a new constant.

The `task_offset` and `task_stride` variables are unchanged — they're still used for the row-processing loop:
```cpp
uint32_t row = task_offset + j * task_stride;  // unchanged
```

The DAG evaluation helper (`evaluate_dag_entry_gkr`) and the inner loops access intermediates via `intermediates_ptr[node * intermediate_stride]`, which now reads:
- `base + warp_id * B * 32 + lane_id + node * 32`
- For a warp: 32 consecutive threads access `warp_id * B * 32 + [0..31] + node * 32` → coalesced (same as before)
- For a thread: consecutive nodes are `32 * 16 = 512 bytes` apart → **fits in L1 cache lines**

### 3. Update Rust FFI binding

**File:** `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `buffer_size: u32` to the `_logup_gkr_input_eval` extern declaration (around line 278). Update the safe wrapper `logup_gkr_input_eval()` to accept and pass `buffer_size`.

### 4. Update Rust call site

**File:** `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`, line 174

Pass `buffer_size as u32` to the `logup_gkr_input_eval()` call. The value is already available as `rules.inner.buffer_size` (line 158).

## Memory Layout Comparison

### Current (thread-interleaved, stride = 65536)

```
Memory: [t0_n0, t1_n0, ..., t65535_n0, t0_n1, t1_n1, ..., t65535_n1, ...]
         |<---- 65536 elements ---->|  |<---- 65536 elements ---->|
```

Thread 0 accesses nodes at offsets: 0, 65536, 131072, ... (1MB apart)

### Proposed (warp-tiled, stride = 32)

```
Memory: [w0: t0_n0..t31_n0, t0_n1..t31_n1, ..., t0_nB..t31_nB] [w1: ...] [w2: ...] ...
         |<---- 32 * buffer_size elements per warp ---->|
```

Thread 0 accesses nodes at offsets: 0, 32, 64, ... (512 bytes apart)
Per-warp contiguous block: `buffer_size × 32 × 16` bytes ≈ **50KB** (for buffer_size=100)

### Why the new layout is faster

The primary performance mechanisms are:

1. **TLB pressure reduction.** With stride 1MB, every intermediates access touches a new 4KB page, causing a TLB miss. With stride 512B, ~8 consecutive node accesses hit the same page, reducing TLB misses by ~8x.

2. **Hardware prefetch enablement.** The GPU hardware prefetcher can detect and prefetch accesses with strides up to ~2KB. The current 1MB stride is far outside this range; the proposed 512B stride is within it, allowing the prefetcher to hide memory latency.

3. **L2 sector-level efficiency.** L2 cache lines are 128 bytes. With 1MB stride, each cache line loaded is used once and then evicted. With 512B stride, a thread's consecutive node accesses are close enough (~4 cache lines apart) that lines loaded for node `j` may still be resident when node `j+1` is accessed, especially for the write-then-read pattern of buffered intermediates.

4. **Temporal reuse across rows.** When `num_rows_per_tile > 1`, the same intermediates addresses are overwritten for each row iteration. With warp-tiled layout, the per-warp 50KB working set has a chance of surviving in L2 between row iterations. With the current 100MB footprint, it's guaranteed to be evicted.

| Metric | Current | Proposed |
|--------|---------|----------|
| Inter-node stride per thread | 1MB (65536 × 16B) | 512B (32 × 16B) |
| Per-warp contiguous working set | ~100MB (scattered) | ~50KB (contiguous) |
| TLB pages per 100-node DAG eval | ~100 new pages | ~13 new pages |
| Hardware prefetch | Disabled (stride > 2KB) | Enabled (stride = 512B) |

## Invariants

1. **Buffer capacity unchanged.** Total allocation remains `TASK_SIZE * buffer_size` FpExt elements. Only the mapping from (warp, node, lane) to memory address changes.
2. **No cross-thread intermediates access.** Each thread reads/writes only its own scratch region. The layout change doesn't affect data sharing.
3. **SCATTER mode (`GLOBAL=false`) unaffected.** Uses local stack array with stride 1. The `buffer_size` parameter is passed but unused.
4. **Row-processing loop unchanged.** `task_offset` and `task_stride` still govern which rows each thread processes.
5. **DAG evaluation semantics unchanged.** Same nodes, same operations, same results. Only the physical memory addresses change.
6. **Coalescing for same-node warp access preserved.** Threads 0..31 still access consecutive addresses for a given node (offset `[0..31]` within the warp tile).
7. **Output buffer unchanged.** Writes to `d_fracs` are independent of intermediates layout.
8. **Proof correctness preserved.** The kernel produces identical (p, q) fractional pairs for all AIRs.

## Measurement Plan

### Commands

```bash
# In the powdr repo:
cd /home/georg/powdr/results/pairing

# Build
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"

# APC 300 (primary target — run 3x for median)
RUST_LOG=info ./target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics apc300/metrics.json --recursion

# APC 0 (regression check)
RUST_LOG=info ./target/release/powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics apc000/metrics.json --recursion

# nsight profiling (APC 300 — verify kernel time reduction)
nsys profile --output nsys_apc300 --force-overwrite true --trace cuda,nvtx,osrt --sample none --stats true -- ./target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --recursion
```

### Expected results

**Kernel-level (nsight):**
- `evaluate_interactions_gkr_kernel<true>` total GPU time: 464ms → 310-380ms (1.2-1.5x speedup)

**Wall-clock critical-path estimate:**
At APC 300, GKR input eval runs on 8 streams per segment. Seg0 dominates (~248 GLOBAL instances, ~246ms wall time). If per-instance kernel time improves by 1.3x, the critical stream's wall time drops from ~246ms to ~189ms, saving ~57ms on seg0. Seg1 contributes ~14ms more. Total GKR input eval savings: ~57ms.

The GKR protocol rounds (~232ms, sequential) are unaffected.

- **LogUp GKR at APC 300**: 532ms → 470-500ms (30-60ms improvement)
- **STARK excl trace at APC 300**: 1317ms → 1260-1290ms (30-60ms improvement)
- **APC 0 regression**: no change (single-threaded path with <100 AIRs, and at APC 0 fewer large AIRs already have good per-AIR cache locality)

### Verification

1. Proof verifies at all APC configs (0, 100, 300)
2. `spec.py` shows LogUp GKR improvement ≥ 20ms at APC 300
3. `nsys stats --report cuda_gpu_kern_sum` shows `evaluate_interactions_gkr_kernel<true>` speedup ≥ 1.2x
4. APC 0 STARK excl trace within ±20ms of before (no regression)

## Rollback Criteria

- Revert if STARK excl trace improvement at APC 300 < 20ms (below measurement noise threshold)
- Revert if APC 0 regresses by > 20ms
- Revert if any APC configuration fails proof verification
