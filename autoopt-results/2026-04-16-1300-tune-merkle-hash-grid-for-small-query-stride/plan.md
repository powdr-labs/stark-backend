# Plan: Tune Merkle Hash Grid for Small Query Stride

## Goal

Increase GPU SM utilization for the `poseidon2_compressing_row_hashes_kernel` when the Merkle tree has a small `query_stride` (≤ 4096). At APC 300, the stacked codeword matrix is wide (~53K columns per segment) but short (query_stride ≈ 1024), producing only 32 thread blocks — occupying 32 of 128 SMs (25%). Reducing `blockDim.x` increases the block count to 128+, achieving near-full SM utilization and an estimated 2-3x speedup on this compute-bound kernel. This specifically improves APC 300 Trace Commit without affecting APC 0 (which already has 2048+ blocks).

## Current Code Path

### Launcher

**File**: `crates/cuda-backend/cuda/src/merkle_tree.cu`, lines 214-231

```cpp
extern "C" int _poseidon2_compressing_row_hashes(
    digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    auto [grid, block] = kernel_launch_params(query_stride, 512 >> log_rows_per_query);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;
    poseidon2_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(...);
    return CHECK_KERNEL();
}
```

**`kernel_launch_params`** (`crates/cuda-common/include/launcher.cuh`, line 14):
```cpp
inline std::pair<dim3, dim3> kernel_launch_params(size_t count, size_t threads_per_block) {
    size_t block = std::min(count, threads_per_block);
    size_t grid = div_ceil(count, block);
    return std::make_pair(dim3(grid, 1, 1), dim3(block, 1, 1));
}
```

### Current launch geometry at APC 300

- `log_rows_per_query` = `k_whir` = 4 (from `SystemParams::k_whir()`)
- `threads_per_block` = 512 >> 4 = 32
- `query_stride` ≈ 1024 (codeword_height / 2^k_whir, where codeword_height ≈ 2^14)
- `blockDim` = (32, 16) = 512 threads/block
- `gridDim` = (ceil(1024/32), 1) = (32, 1) = **32 blocks**
- SM utilization: 32 / 128 = **25%**

### Current launch geometry at APC 0

- Same `k_whir` = 4
- `query_stride` ≈ 65536 (codeword_height / 16, where codeword_height ≈ 2^20)
- `blockDim` = (32, 16) = 512 threads/block
- `gridDim` = (ceil(65536/32), 1) = (2048, 1) = 2048 blocks
- SM utilization: **100%** (2048 >> 128)

### Kernel structure

The kernel (`merkle_tree.cu`, line 16) has two phases:

1. **Row hashing** (lines 39-51): Each thread (indexed by `stride_idx = blockDim.x * blockIdx.x + threadIdx.x`) iterates over all `width` columns, absorbing elements into a Poseidon2 sponge (CELLS=16, CELLS_RATE=8). At APC 300 with width ≈ 53K, this is ≈ 6625 `poseidon2_mix` calls per thread.

2. **Intra-query Merkle reduction** (lines 53-71): `log_rows_per_query` levels of tree compression in shared memory across the y-dimension. `blockDim.y = 2^log_rows_per_query = 16`. Each level halves active y-threads, combining adjacent digests via `poseidon2_mix`.

**Key observation**: `blockDim.x` controls how many stride positions are processed per block. Reducing it increases the block count without changing the total thread count or work assignment — each thread still processes one stride position.

### Callers

- `stacked_commit` → `MerkleTreeGpu::new_with_hash` → `compress_rows` → `_poseidon2_compressing_row_hashes` — the main trace commitment Merkle tree, called once per segment.
- `prove_whir_opening_gpu` → `MerkleTreeGpu::new_with_hash` — WHIR round commits, which have width=4 (tiny rows, not affected by this optimization).
- There is an extension-field variant `_poseidon2_compressing_row_hashes_ext` with the same launcher pattern (line 233-250), used for extension-field Merkle trees. The same fix should be applied there for consistency.

## Changes

### Change 1: Add adaptive blockDim.x selection in the hash launcher

**File**: `crates/cuda-backend/cuda/src/merkle_tree.cu`, lines 214-231

Replace the fixed `512 >> log_rows_per_query` with an adaptive choice:

```cpp
extern "C" int _poseidon2_compressing_row_hashes(
    digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    // Target: enough blocks to fill all SMs (≥128 blocks on RTX 4090).
    // Each block has blockDim.y = 2^log_rows_per_query threads for Merkle
    // levels, and blockDim.x threads for stride parallelism.
    // Default blockDim.x = 512 >> log_rows_per_query (e.g. 32 for k=4).
    // When query_stride is small, this produces too few blocks.
    // We reduce blockDim.x until grid has ≥ MIN_BLOCKS blocks.
    constexpr size_t MIN_BLOCKS = 128;
    size_t default_bx = 512 >> log_rows_per_query;
    size_t bx = default_bx;
    while (bx > 1 && div_ceil(query_stride, bx) < MIN_BLOCKS) {
        bx >>= 1;
    }
    auto [grid, block] = kernel_launch_params(query_stride, bx);
    block.y = 1 << log_rows_per_query;
    size_t shared_stride = block.x * div_ceil(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    poseidon2_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query
    );
    return CHECK_KERNEL();
}
```

**Why this helps**: At APC 300 with query_stride=1024, the loop reduces bx from 32 → 16 → 8, giving grid.x = ceil(1024/8) = 128 blocks. This uses all 128 SMs instead of 32. The shared memory per block shrinks proportionally (from 8192 to 2048 bytes), well within limits.

**APC 0 is unaffected**: query_stride=65536, div_ceil(65536,32) = 2048 ≥ 128, so bx stays at 32 (default path).

### Change 2: Apply the same fix to the ext variant

**File**: `crates/cuda-backend/cuda/src/merkle_tree.cu`, lines 233-250

Apply the identical `MIN_BLOCKS` adaptive logic to `_poseidon2_compressing_row_hashes_ext`. Same loop, same constant.

### Change 3 (optional, if needed): Adjust __syncthreads usage

The kernel uses `__syncthreads()` in the Merkle reduction phase (lines 61, 71). With blockDim = (8, 16) = 128 threads = 4 warps, `__syncthreads` is slightly more expensive than within a single warp. However, for correctness, `__syncthreads` is required when any pair of threads in different warps share data via shared memory. At blockDim.x = 8, two x-threads can be in different warps (if y-threads of the same warp span different x-indices). So `__syncthreads` is still required. No change needed to the kernel itself.

## Invariants

1. **Correctness**: The kernel output must be identical regardless of `blockDim.x`. Each thread processes exactly one stride position (`stride_idx = blockDim.x * blockIdx.x + threadIdx.x`). The Merkle reduction in shared memory operates within a block's y-dimension. Reducing blockDim.x only changes how stride positions are distributed across blocks — the per-thread computation is unchanged.

2. **APC 0 no-regression**: When query_stride is large (≥ default_bx × MIN_BLOCKS), the adaptive loop doesn't fire and the launch geometry is identical to today.

3. **Shared memory correctness**: `shared_stride = block.x * div_ceil(block.y, 2)`. With smaller block.x, shared_stride shrinks, but each y-layer exchange accesses `shared_offset = ((leaf_idx >> (layer+1)) << layer) * block.x + threadIdx.x`, which stays within bounds.

4. **Thread count**: Total threads = grid.x × block.x × block.y = query_stride × (1 << log_rows_per_query) = height. This is preserved regardless of the block.x choice.

## Measurement Plan

1. Run `run_pairing.sh` for APC 0, 100, 300 before and after the change.
2. Compare `STARK (excl. trace)` and `Trace Commit` metrics.
3. Run nsight profile on APC 300 to verify:
   - `poseidon2_compressing_row_hashes_kernel` instance count is unchanged (25)
   - Average/total kernel time decreased for the large instances (main trace commits)
   - Grid dimensions increased from 32 to 128 for main trace commits
4. APC 0 must show no regression (within ±20ms noise).

## Rollback Criteria

- Less than 15ms improvement on STARK excl trace at APC 300.
- Any regression > 20ms on STARK excl trace at APC 0.
- Any correctness failure (proof verification fails).
