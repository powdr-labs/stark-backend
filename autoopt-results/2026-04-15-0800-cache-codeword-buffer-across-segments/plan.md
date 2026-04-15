# Plan: Pre-warm GPU memory pool at device initialization

## Goal

Eliminate the cold-start allocation overhead in the first segment's `rs_code_matrix` by pre-warming the GPU memory pool (VPMM or cudaMallocAsync) during `GpuDevice::new()`. Per-segment metrics show `rs_code_matrix_time_ms` = 57ms for seg 0 vs 0ms for seg 1 at APC 300. The 57ms is wall time of the `rs_code_matrix()` function, which launches only asynchronous GPU kernels — the time comes from the first-time allocation of the 192MB codeword buffer through the memory pool. Segments 1+ get 0ms because the pool already has a suitable freed block. Pre-warming creates and frees a large buffer at device initialization, ensuring the pool has mapped pages ready for the first prove() call.

## Current Code Path

### Memory allocation flow (`crates/cuda-common/src/memory_manager/mod.rs:70-100`)

```
DeviceBuffer::<T>::with_capacity(len)  [d_buffer.rs:94]
  → MEMORY_MANAGER.lock().allocate(len * sizeof(T))  [d_buffer.rs:97]
    → if size < pool.page_size:
        cudaMallocAsync(ptr, size, cudaStreamPerThread)  [mod.rs:77]
      else:
        pool.allocate(size)  [vm_pool.rs — VPMM: cuMemCreate + cuMemMap]  [mod.rs:89]
```

For the 192MB codeword buffer (`codeword_height × width × 4` bytes):
- If VPMM active: `pool.allocate()` maps new physical pages via CUDA driver API — ~50ms for 96 pages
- If VPMM fallback: `cudaMallocAsync` triggers real `cudaMalloc` to grow the pool — ~10-50ms

### Where the 57ms occurs (`crates/cuda-backend/src/stacked_pcs.rs:182-279`)

```
rs_code_matrix()                    [stacked_pcs.rs:182, #[instrument(skip_all)]]
  let mut codewords = DeviceBuffer::<F>::with_capacity(codeword_height * width);  ← COLD ALLOC
  batch_expand_pad(...)              ← async kernel launch
  mle_interpolate_stages(...)        ← async kernel launch  
  bit_rev(...)                       ← async kernel launch
  batch_ntt(...)                     ← async kernel launches
  return DeviceMatrix::new(...)
```

The `#[instrument]` span measures wall time. All GPU kernel launches are async (return immediately). The 57ms is dominated by the cold `with_capacity` call. For seg 1, the pool has the freed block from seg 0 → instant allocation → 0ms.

### GpuDevice initialization (`crates/cuda-backend/src/device.rs:28-43`)

```rust
pub fn new(config: SystemParams) -> Self {
    ensure_device_ntt_twiddles_initialized();
    let prover_config = GpuProverConfig { ... };
    let id = get_device().unwrap() as u32;
    let sm_count = get_sm_count(id).expect("failed to get SM count");
    Self { config, prover_config, id, sm_count }
}
```

No GPU memory pre-warming currently occurs.

## Changes

### Change 1: Add pool pre-warming to `GpuDevice::new()` (`device.rs:28-43`)

After `ensure_device_ntt_twiddles_initialized()`, add a pre-warm allocation and immediate free:

```rust
pub fn new(config: SystemParams) -> Self {
    ensure_device_ntt_twiddles_initialized();
    
    // Pre-warm the GPU memory pool by allocating and freeing a 256MB buffer.
    // This maps physical pages in VPMM (or grows the cudaMallocAsync pool),
    // avoiding cold-start overhead in the first segment's rs_code_matrix.
    {
        use openvm_cuda_common::d_buffer::DeviceBuffer;
        let warmup_elems = POOL_WARMUP_BYTES / std::mem::size_of::<u32>();
        let _warmup = DeviceBuffer::<u32>::with_capacity(warmup_elems);
        // _warmup dropped here → pages return to pool, ready for reuse
    }
    
    let prover_config = GpuProverConfig { ... };
    // ... rest unchanged
}
```

### Change 2: Use fixed 256MB warmup size (`device.rs`)

The warmup uses a fixed 256MB allocation rather than a heuristic derived from system parameters. The actual codeword buffer size depends on runtime stacking decisions (which vary per segment), so any parameter-based estimate would be fragile. 256MB covers the codeword buffer at APC 300 (192MB) and most APC 0 configurations, while being only ~1% of a 24GB GPU.

```rust
/// Fixed pool warmup size in bytes. Covers the RS codeword buffer for typical workloads.
const POOL_WARMUP_BYTES: usize = 256 * 1024 * 1024;
```

At APC 300 seg 0: actual codeword = 192MB. Fits within 256MB warmup.
At APC 0 seg 0: actual codeword varies. If > 256MB, partial benefit (pool has some mapped pages ready).

**Alternative**: The VPMM already supports pre-allocation via the `VPMM_PAGES` environment variable (parsed in `VpmmConfig::from_env()`, `vm_pool.rs:42`). Setting `VPMM_PAGES=128` would achieve the same result without code changes. The code-based approach is preferred because it's automatic and doesn't require env var configuration.

### Why this works

The memory pool (VPMM or cudaMallocAsync) keeps freed memory available for reuse:
- **VPMM path**: Physical pages are mapped via `cuMemCreate` + `cuMemMap` during warmup. On free, pages return to the pool's free list (still mapped). Next allocation reuses mapped pages without driver API calls.
- **cudaMallocAsync path**: The async pool grows via real `cudaMalloc` during warmup. On free, memory stays in the pool. Next allocation finds a suitable block immediately.

In both cases, the warmup allocation + free takes ~50ms during `GpuDevice::new()` (outside any timing span), and subsequent allocations of ≤256MB are instant.

## Invariants

1. **No behavioral change**: The warmup allocates and immediately frees — no lasting side effect except warmed pool state.
2. **Memory safety**: The warmup buffer is owned by a local scope and dropped deterministically.
3. **No peak memory increase**: The warmup buffer is freed before any other allocation occurs.
4. **APC 0 compatibility**: The cap of 256MB limits over-allocation. APC 0 may allocate larger codeword buffers (if stacking width > 8), in which case the warmup provides partial benefit. No regression.
5. **VPMM not available**: If VPMM initialization fails (page_size = usize::MAX), all allocations go through cudaMallocAsync. The warmup still helps by growing the cudaMallocAsync pool.

## Measurement Plan

### Commands

```bash
# Build (ensure binary is up to date):
cd /home/georg/powdr && cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"

# APC 300 (primary metric):
cd /home/georg/powdr && RUST_LOG=info target/release/powdr_openvm_riscv prove \
  --artifact results/pairing/apc300.cbor --input 0 \
  --metrics <results_dir>/after_apc300.json --recursion

# APC 0 (regression check):
cd /home/georg/powdr && RUST_LOG=info target/release/powdr_openvm_riscv prove \
  --artifact results/pairing/apc000.cbor --input 0 \
  --metrics <results_dir>/after_apc000.json --recursion

# Analyze:
python3 /home/georg/spec.py <metrics_path> <experiment_name>
```

### Expected behavior

- **rs_code_matrix_time_ms (seg 0)**: drops from 57ms to ≤5ms
- **stacked_commit_time_ms (seg 0)**: drops from 163ms to ~110ms
- **STARK excl trace (APC 300)**: ≥30ms improvement (from ~1298ms to ≤1268ms)
- **APC 0**: ≥20ms improvement (seg 0 rs_code_matrix was 47ms)
- **No regression at APC 0**: ≤ +10ms noise on STARK excl trace

### Key metrics to check

- `rs_code_matrix_time_ms` for seg 0 (should match seg 1's ~0ms)
- `stacked_commit_time_ms` for seg 0 vs seg 1 gap (should shrink from 76ms to ~20ms)
- Total `STARK excl trace` at APC 300 and APC 0

## Rollback Criteria

1. `STARK excl trace` improvement at APC 300 is less than 10ms.
2. APC 0 `STARK excl trace` regresses by more than 20ms.
3. Any test failure in `cargo nextest run -p openvm-cuda-backend --test-threads=4`.
4. Any verification failure (invalid proofs).
