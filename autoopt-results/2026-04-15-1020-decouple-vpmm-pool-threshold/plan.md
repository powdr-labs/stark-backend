# Plan: Decouple VPMM Pool Threshold from Page Size

## Goal

Introduce a configurable `pool_threshold` parameter that determines the minimum allocation size routed to the VPMM virtual-memory system, independent of the physical `page_size`. By raising this threshold from 16 MiB (current page_size) to 64 MiB, allocations in the 16–64 MiB range will use `cudaMallocAsync` instead of the VPMM, avoiding per-allocation `cuMemCreate` calls, page tracking, and defragmentation overhead. The page_size remains at 16 MiB for large VPMM allocations.

This follows the pattern established by the `vpmm-bulk-page-creation` task, which showed that routing medium-sized allocations (2–16 MiB) through `cudaMallocAsync` instead of VPMM gave −136 ms on LogUp GKR + Constraints at APC 300 — far beyond the expected savings from reduced `cuMemCreate` calls alone.

## Current Code Path

### Allocation routing (`crates/cuda-common/src/memory_manager/mod.rs:71-92`)

```rust
fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
    let mut tracked_size = size;
    let ptr = if size < self.pool.page_size {     // ← threshold = page_size
        // cudaMallocAsync path
        cudaMallocAsync(&mut ptr, size, cudaStreamPerThread);
        self.allocated_ptrs.insert(ptr, size);
        ptr
    } else {
        // VPMM path
        tracked_size = size.next_multiple_of(self.pool.page_size);
        self.pool.malloc_internal(tracked_size, stream_id)?
    };
    // ...
}
```

### Deallocation routing (`crates/cuda-common/src/memory_manager/mod.rs:104-119`)

```rust
unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
    if let Some(size) = self.allocated_ptrs.remove(&nn) {
        // cudaMallocAsync path — ptr was in allocated_ptrs
        cudaFreeAsync(ptr, cudaStreamPerThread);
    } else {
        // VPMM path — ptr was in pool.malloc_regions
        self.pool.free_internal(ptr, stream_id)?;
    }
}
```

The deallocation path is routing-agnostic: it checks `allocated_ptrs` first, falls through to VPMM. No change needed here.

### Configuration (`crates/cuda-common/src/memory_manager/vm_pool.rs:35-86`)

`VpmmConfig` has fields `page_size`, `va_size`, `initial_pages`. It loads from environment variables in `from_env()`. `VirtualMemoryPool` stores `page_size` at field `vm_pool.rs:134`.

### Key allocations in the 16–64 MiB range

1. **GKR intermediates** (`crates/cuda-backend/src/logup_zerocheck/gkr_input.rs:308`): `DeviceBuffer::with_capacity(max_intermediates_len)` where `max_intermediates_len = TASK_SIZE(65536) × buffer_size × sizeof(EF)(16)`. Typical buffer_size = 10–40 → 10–40 MiB per thread. With 8 threads, 8 allocations of 10–40 MiB each, all currently routed through VPMM.

2. **Various temporary DeviceBuffers** allocated during proving (fold intermediates, evaluation buffers) that may fall in the 16–64 MiB range.

### VPMM overhead per allocation

For a 40 MiB VPMM allocation with 16 MiB pages:
- `ceil(40/16) = 3` physical pages created via `cuMemCreate` (~0.4 ms each) = 1.2 ms
- `cuMemMap` + `cuMemSetAccess` per page: ~0.1 ms each = 0.3 ms
- Free-region search + zombie cleanup: ~0.01 ms
- Total: ~1.5 ms per allocation

For 8 GKR intermediates alone: 8 × 1.5 ms ≈ 12 ms pure allocation overhead.

Additionally, the previous `vpmm-bulk-page-creation` task showed that VPMM management causes secondary effects (likely TLB/VA-layout degradation for bandwidth-bound kernels), contributing −136 ms beyond the direct allocation savings at APC 300.

## Changes

### Change 1: Add `pool_threshold` to `VpmmConfig`

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

Add a new field to `VpmmConfig`:

```rust
pub struct VpmmConfig {
    pub page_size: Option<usize>,
    pub va_size: usize,
    pub initial_pages: usize,
    pub pool_threshold: Option<usize>,  // NEW
}
```

Update `Default`:
```rust
impl Default for VpmmConfig {
    fn default() -> Self {
        Self {
            page_size: None,
            va_size: DEFAULT_VA_SIZE,
            initial_pages: 0,
            pool_threshold: None,  // NEW: resolved in VirtualMemoryPool::new()
        }
    }
}
```

Update `from_env()` to load `VPMM_POOL_THRESHOLD`:
```rust
let pool_threshold = std::env::var("VPMM_POOL_THRESHOLD").ok().map(|val| {
    let size: usize = val.parse().expect("VPMM_POOL_THRESHOLD must be a valid number");
    assert!(size > 0, "VPMM_POOL_THRESHOLD must be > 0");
    size
});
```

### Change 2: Store `pool_threshold` in `VirtualMemoryPool`

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

Add field to struct `VirtualMemoryPool` (at line 134):
```rust
pub(super) pool_threshold: usize,
```

In `VirtualMemoryPool::new()`, after resolving `page_size` (line 178), resolve `pool_threshold`:
```rust
let pool_threshold = match config.pool_threshold {
    Some(t) => {
        assert!(
            t >= page_size,
            "VPMM_POOL_THRESHOLD ({}) must be >= page_size ({})",
            t, page_size
        );
        t
    }
    // Default: 4x page_size (typically 64 MiB).
    // Allocations below this go through cudaMallocAsync.
    None => 4 * page_size,
};
```

Add to the `Self { ... }` initializer (line 204-219):
```rust
pool_threshold,
```

Add to the debug log (near line 190):
```rust
tracing::debug!(
    "VPMM: pool_threshold={}",
    ByteSize::b(pool_threshold as u64)
);
```

### Change 3: Use `pool_threshold` for allocation routing

**File**: `crates/cuda-common/src/memory_manager/mod.rs`

Change line 75 from:
```rust
let ptr = if size < self.pool.page_size {
```
to:
```rust
let ptr = if size < self.pool.pool_threshold {
```

No other changes needed. The deallocation path (`d_free`) already correctly routes based on the `allocated_ptrs` HashMap presence, which tracks all `cudaMallocAsync` allocations regardless of size.

### Change 4: Handle VPMM-unsupported fallback (overflow guard)

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

When VPMM is not supported by the GPU, `page_size` is set to `usize::MAX` (line 198) so all allocations use `cudaMallocAsync`. Computing `4 * usize::MAX` would overflow. Guard the default resolution in Change 2:

```rust
None => page_size.saturating_mul(4),
```

When `page_size == usize::MAX` (VPMM not supported), `saturating_mul(4)` produces `usize::MAX`, preserving the invariant that all allocations route through `cudaMallocAsync`. Otherwise produces `4 * page_size` (typically 64 MiB).

### Change 5: Update test callsites

**File**: `crates/cuda-common/src/memory_manager/tests.rs`

Four `VpmmConfig` struct-literal constructions (at lines 55-58, 117-121, 390-393, 614-617) must include the new `pool_threshold` field. Add `pool_threshold: None` to each, preserving the existing test behavior (default resolution).

## Invariants

1. **Allocation-deallocation consistency**: Every allocation through `cudaMallocAsync` is tracked in `allocated_ptrs`; every VPMM allocation is tracked in `pool.malloc_regions`. The `d_free` routing checks `allocated_ptrs` first. As long as the threshold is consistent between allocation and deallocation (which it is — a given pointer was allocated through exactly one path), correctness is maintained.

2. **VPMM page alignment**: VPMM allocations are still rounded up to `page_size` multiples (line 89). This is unchanged because only allocations ≥ `pool_threshold` (≥ 64 MiB) go through VPMM. Since `pool_threshold >= page_size`, the rounding is always valid.

3. **pool_threshold >= page_size**: Enforced by the assertion in Change 2. This ensures that VPMM never receives allocations smaller than a page.

4. **No change to VPMM internals**: The VPMM pool's page creation, mapping, defragmentation, free-region management, and zombie cleanup are all unchanged. Only the routing decision is affected.

5. **Proof correctness**: This optimization only changes allocation routing — the mathematical operations and data flow are identical. Any allocation size works correctly with both `cudaMallocAsync` and VPMM.

## Measurement Plan

1. Run `openvm-riscv/scripts/run_pairing.sh` for APC {0, 100, 300} in the powdr repo.
2. Analyze with `spec.py`. Compare STARK excl trace, LogUp GKR, Round 0, MLE Rounds, Trace Commit at all APC configs.
3. Verify no regression at APC 0 (threshold: +20 ms).
4. Collect nsight profile at APC 300 for GPU kernel timeline comparison.
5. All proof configurations must complete with `--recursion` (correctness check).

If regression at APC 0 > 20 ms (following the pattern seen in `cache-codeword-buffer-across-segments` and `multistream-stacked-reduction-round0` where pool state changes caused GKR regressions):
- Try `pool_threshold` = 2 × page_size (32 MiB) as a more conservative alternative.
- If still regressing, try `pool_threshold` = page_size (no change, confirming the regression is from the threshold increase).

## Rollback Criteria

- APC 0 STARK excl trace regression > 20 ms → revert.
- APC 300 STARK excl trace improvement < 10 ms → revert (not worth the added complexity).
- Any proof correctness failure → revert.
- If the 64 MiB threshold doesn't help but 32 MiB does, keep the 32 MiB version.
