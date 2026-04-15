# Plan: Batched Descriptor-Array GLOBAL-Mode GKR Input Evaluation

## Goal

Replace per-AIR GLOBAL-mode GKR input evaluation kernel launches with a single batched descriptor-array kernel for small AIRs (height ≤ TASK_SIZE). This targets the largest STARK excl trace component: LogUp GKR at 529ms (41% of 1301ms total). The GKR input eval accounts for ~310ms of wall-clock time with 509ms of GPU kernel time compressed to 310ms via 8-stream multi-threading (1.64x compression). By using GPU-native block scheduling instead of coarse-grained stream scheduling and eliminating ~300 full-grid (256-block) kernel launches per segment, we expect to save 20-50ms from LogUp GKR.

**Calibration against prior work**: Three prior optimizations targeted this same phase — right-sizing grid (0ms), intermediates locality change (0ms), and SCATTER batching (-8ms). The key difference here is that GLOBAL AIRs each launch 256 blocks (vs 4-16 for SCATTER), and there are ~300 such launches per segment, each briefly monopolizing all 128 SMs for 2 waves. This is a fundamentally different overhead profile than the SCATTER case.

## Current Code Path

### Entry point
`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:215` — `log_gkr_input_evals()`

### Flow
1. **Phase 1** (lines 232-252): Build work items sorted by descending height
2. **Phase 2** (lines 254-279): Pre-compute max buffer sizes for pre-allocation
3. **Barrier** (line 282): `current_stream_sync()` to ensure `fill_zero` is visible to workers
4. **Phase 3** (lines 285-289): Decide thread count (≥100 AIRs → 8 threads, else 1)
5. **Worker loop**: Each thread processes its assigned work items sequentially via `process_gkr_input_air()` (line 112)
6. **Per-AIR processing** (line 112-209): For each AIR:
   - Collect trace pointers, public values, interaction rules
   - Determine GLOBAL vs SCATTER mode (`buffer_size > 10`, line 160)
   - Launch `_logup_gkr_input_eval` CUDA kernel with `count = TASK_SIZE = 65536` for GLOBAL mode

### CUDA kernel
`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu:66` — `evaluate_interactions_gkr_kernel<GLOBAL>`

For GLOBAL mode:
- **Grid**: 256 blocks × 256 threads = 65536 threads (always, regardless of AIR height)
- **Intermediates**: global memory, `intermediates_ptr = d_intermediates + task_offset`, `intermediate_stride = task_stride = gridDim.x * blockDim.x`
- **Tile loop**: each thread processes `num_rows_per_tile` rows: `row = task_offset + j * task_stride`
- **Output**: writes to `d_fracs[(interaction_idx * permutation_height + row) * 2 + is_denom]` — per-row, no cross-thread reduction

### Why it's slow
At APC 300 (2 segments, 623 AIR instances):
- ~300 GLOBAL-mode AIRs per segment (those with `buffer_size > 10`)
- Each launches a 256-block kernel (65536 threads)
- Small AIRs (height 256-4096) use <6% of launched threads; the rest check `row < permutation_height` and exit
- 8-stream multi-threading achieves only 1.64x compression (509ms GPU → 310ms wall) because:
  - Each 256-block launch briefly occupies all 128 SMs (2 waves), blocking other streams' kernels
  - Per-launch overhead (~5μs × 300 launches = ~1.5ms per segment) and CPU-side per-AIR setup (H2D for partition_ptrs, public_values) add up
  - CPU iteration to collect per-AIR metadata before each launch

### Key property enabling batching
**Output independence**: Each row's output goes to a unique position in `d_fracs`. There is NO cross-row or cross-thread reduction. Intermediates are per-thread scratch (rewritten for each row in the tile loop, since `num_rows_per_tile == 1` for batch-eligible AIRs). This means different AIRs' computations are fully independent and can share a kernel launch.

## Changes

### 1. CUDA batched kernel (`crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`)

Add a descriptor struct:

```c
struct BatchGkrInputDesc {
    FracExt *d_fracs;           // output base for this AIR (pre-offset by stacked layout)
    const Fp *d_preprocessed;   // preprocessed trace buffer
    const uint64_t *d_main;     // partition pointers (device)
    const Fp *d_public_values;  // public values (device, may be null)
    const Rule *d_rules;        // interaction rules
    const size_t *d_used_nodes; // used nodes array
    const uint32_t *d_pair_idxs;// pair index array
    size_t used_nodes_len;      // length of used_nodes
    uint32_t permutation_height;// this AIR's height
    uint32_t buffer_size;       // intermediates scratch size per thread for this AIR
    uint32_t intermediates_offset; // offset into shared intermediates buffer (in EF elements)
    uint32_t block_start;       // first block index for this AIR (for BlockCtx lookup)
};
```

Reuse the existing `BlockCtx` struct from `crates/cuda-backend/cuda/include/eval_ctx.cuh:18` (fields: `local_block_idx_x`, `air_idx`). Build a `BlockCtx` array (one entry per block, pre-built on Rust side) following the pattern in `crates/cuda-backend/cuda/src/logup_zerocheck/batch_mle.cu:119`.

New kernel: `batched_gkr_input_eval_kernel(const BatchGkrInputDesc *descs, const BlockCtx *block_ctxs, FpExt *intermediates_base, const FpExt *d_challenges)`

Per-block behavior:
- Load `ctx = block_ctxs[blockIdx.x]`
- Load `desc = descs[ctx.air_idx]`
- Compute `row = ctx.local_block_idx * blockDim.x + threadIdx.x`
- Guard: `if (row >= desc.permutation_height) return`
- Compute intermediates pointer: `FpExt *my_intermediates = intermediates_base + desc.intermediates_offset + row * desc.buffer_size`
- Set `intermediate_stride = 1` (contiguous per-thread scratch)
- Execute the same DAG evaluation loop as `evaluate_interactions_gkr_kernel` (lines 98-206 of current kernel)
- Write output to `desc.d_fracs` using `desc.permutation_height` for indexing

**Intermediates layout**: Each batch-eligible AIR allocates `height * buffer_size` entries in the shared intermediates buffer. Thread at row `r` uses entries `[intermediates_offset + r * buffer_size, intermediates_offset + (r+1) * buffer_size)`. Since different AIRs have different `buffer_size` values, the per-AIR allocation varies. The Rust side computes `intermediates_offset` via prefix sum over `height * buffer_size` for each eligible AIR.

**Why stride=1 is correct for batch-eligible AIRs**: These AIRs have `num_rows_per_tile == 1`, meaning each thread processes exactly ONE row and exits. The intermediates are used only within that single row evaluation — no reuse across tile iterations. With stride=1, consecutive DAG node results for the same thread are contiguous, improving L1 cache locality vs the current stride=65536 layout.

Add a C launcher: `_batched_gkr_input_eval(const BatchGkrInputDesc *descs, const BlockCtx *block_ctxs, uint32_t total_blocks, FpExt *intermediates, const FpExt *d_challenges)`.

**Height normalization temporaries**: For batch-eligible AIRs where `height != lifted_height`, the kernel must write to a temporary buffer (not directly to `leaves_ptr`), matching the current code's `trace_output` logic (gkr_input.rs:168-172). Allocate a shared tmp buffer sized by prefix sum of `height * num_interactions` (in Frac<EF> elements) for affected AIRs. Each such AIR's `d_fracs` descriptor field points into its section of the tmp buffer. After the batched kernel completes, normalization runs per-AIR from the tmp buffer to the final leaves position.

### 2. Rust FFI bindings (`crates/cuda-backend/src/cuda/logup_zerocheck.rs`)

Add:
- `#[repr(C)] struct BatchGkrInputDesc` matching the CUDA struct (with raw pointer fields)
- Reuse existing `BlockCtx` from `crates/cuda-backend/src/cuda/logup_zerocheck.rs:17`
- `unsafe impl Send for BatchGkrInputDesc` + `Sync`
- `extern "C" fn _batched_gkr_input_eval(descs: *const BatchGkrInputDesc, block_ctxs: *const BlockCtx, total_blocks: u32, intermediates: *mut EF, d_challenges: *const EF) -> i32`
- Safe wrapper `batched_gkr_input_eval_gpu(d_descs: &DeviceBuffer<BatchGkrInputDesc>, d_block_ctxs: &DeviceBuffer<BlockCtx>, intermediates: &mut DeviceBuffer<EF>, d_challenges: &DeviceBuffer<EF>)`

### 3. Rust orchestration (`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`)

Modify `log_gkr_input_evals()`:

**Phase 1 (extended)**: When building work items, also collect per-AIR metadata needed for the batched path:
- `is_global` flag (from `buffer_size > 10`)
- `buffer_size` value
- `height`
- For GLOBAL AIRs with `height ≤ TASK_SIZE` (i.e., `num_rows_per_tile == 1`): mark as `batch_eligible`

**New Phase 1.5 — Build batched descriptors for eligible AIRs**:
1. Partition work items into `batch_eligible` and `non_eligible` lists
2. For each batch-eligible AIR:
   - Compute `blocks_per_air = height.div_ceil(256)` (matches kernel block size of 256)
   - Compute `intermediates_per_air = height * buffer_size` (in EF elements)
3. Prefix-sum `block_start` values: `block_start[i] = sum(blocks_per_air[0..i])`
4. Prefix-sum `intermediates_offset` values: `offset[i] = sum(intermediates_per_air[0..i])`
5. Build `Vec<BatchGkrInputDesc>` with all per-AIR pointers and metadata
6. Build `Vec<GkrInputBlockCtx>` by iterating: for each AIR `a`, for `b` in `0..blocks_per_air[a]`, push `GkrInputBlockCtx { local_block_idx: b, air_idx: a }`
7. Collect per-AIR `partition_ptrs` into a flat `Vec<u64>`, upload once via `to_device()`. Each descriptor's `d_main` points into this buffer.
8. Collect per-AIR `public_values` into a flat `Vec<F>`, upload once. Each descriptor's `d_public_values` points into this buffer (or null if empty).
9. Upload `descs.to_device()`, `block_ctxs.to_device()`
10. Allocate shared intermediates: `DeviceBuffer::<EF>::with_capacity(total_intermediates)`

**Phase 3 (modified)**: Run batched and multi-stream paths concurrently:
- Launch the batched kernel on the **main thread's per-thread default stream** immediately before entering `thread::scope`. Since `--default-stream=per-thread` is set in `crates/cuda-builder/src/lib.rs:41`, each spawned worker thread gets its own CUDA stream. The batched kernel on the main thread's stream runs concurrently with all worker kernels on their respective streams — no explicit `CudaStream::new()` needed.
- Inside `thread::scope`: spawn worker threads for non-eligible AIRs (large GLOBAL + all SCATTER) using existing `process_gkr_input_air()` logic.
- After `thread::scope` returns (all workers joined and synced): call `current_stream_sync()` to ensure the batched kernel on the main thread's stream has completed.
- Then run height-normalization (`frac_vector_scalar_multiply_ext_fp` + `frac_matrix_vertically_repeat`) for batch-eligible AIRs that need it (where `height != lifted_height`), reading from the shared tmp buffer and writing to final leaves positions.

**APC 0 behavior**: At APC 0, `num_threads = 1` (< 100 AIRs). The batched kernel still runs for eligible GLOBAL AIRs, replacing sequential per-AIR launches. Non-eligible AIRs run on the main thread sequentially. This should be at least as fast.

### 4. Handle per-AIR data on device

- **Rules, used_nodes, pair_idxs**: Already on device as `pk_air.other_data.interaction_rules.inner.d_rules` etc. (uploaded at keygen time). Descriptors store raw pointers to these existing device buffers.
- **partition_ptrs**: Currently uploaded per-AIR via `partition_ptrs.copy_to(&mut buffers.partition_ptrs)`. For the batched path, collect all partition pointer lists into one flat device buffer, with each descriptor pointing to its section.
- **public_values**: Same pattern — flat buffer with per-AIR offsets.

## Invariants

1. **Output correctness**: Each row in `d_fracs` must produce the same value as the per-AIR kernel. The DAG evaluation logic is identical; only the launch geometry and intermediates layout change.
2. **No cross-AIR interference**: Each AIR's blocks write to disjoint regions of `d_fracs` (guaranteed by different `leaves_ptr` offsets from the stacked layout).
3. **Intermediates isolation**: Each thread's scratch is at a unique offset: `intermediates_offset + row * buffer_size` for `row < permutation_height`. No two threads share scratch space. Different AIRs have different `intermediates_offset` values (prefix sum ensures non-overlap). Different rows within an AIR have different `row` values.
4. **Varying buffer_size**: The `buffer_size` field in each descriptor tells the kernel how many EF elements to stride per row. The intermediates offset computation on the Rust side uses each AIR's own `buffer_size` to compute the prefix sum correctly.
5. **Fallback for large AIRs**: AIRs with `height > TASK_SIZE` (`num_rows_per_tile > 1`) continue using the existing per-AIR kernel on worker threads. The tile loop in those kernels reuses intermediates across tiles (same thread processes multiple rows), requiring the current `task_stride` layout. The batched kernel's stride=1 layout only works when `num_rows_per_tile == 1`.
6. **APC 0 safety**: At APC 0 (99 AIRs, below multi-stream threshold), the batched kernel replaces sequential per-AIR launches. Non-eligible AIRs run sequentially. No regression expected.
7. **Height normalization**: Eligible AIRs where `height != lifted_height` need post-processing after the batched kernel. Sync the main thread's stream first, then run normalization kernels sequentially.
8. **Memory budget**: The shared intermediates buffer replaces per-thread pre-allocated buffers for batch-eligible AIRs. Add a total size check: if `total_intermediates * sizeof(EF) > 2 GB`, fall back to the existing per-AIR multi-stream path for all AIRs (skip batching). This mirrors the existing budget check at `gkr_input.rs:292-303`.

## Measurement Plan

1. Build: `cargo check -p openvm-cuda-backend`
2. Test: `cargo nextest run -p openvm-cuda-backend --test-threads=4`
3. Benchmark APC 0: `cd /home/georg/powdr && openvm-riscv/scripts/run_pairing.sh` (APC 0 only)
4. Benchmark APC 300: same script, APC 300
5. Compare:
   - LogUp GKR at APC 300: expect 529ms → ~490-510ms
   - STARK excl trace at APC 300: expect 1301ms → ~1260-1280ms
   - No regression at APC 0

## Rollback Criteria

- Less than 20ms improvement on STARK excl trace at APC 300
- Any regression at APC 0 (STARK excl trace increase > 15ms)
- Test failures
- cudaErrorIllegalAddress or other CUDA errors
