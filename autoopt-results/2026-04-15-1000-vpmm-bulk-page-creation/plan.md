# Plan: VPMM Bulk Page Creation

## Goal

Eliminate the ~58ms cold-start overhead in the first segment's `rs_code_matrix` allocation at APC 300 by batching the VPMM's physical page creation. Currently, `defragment_or_create_new_pages` creates ~128 pages one at a time (each a separate `cuMemCreate` + `cuMemMap` driver call). This plan replaces the per-page `cuMemCreate` loop with a single bulk `cuMemCreate` call for the total needed size, then maps individual page-sized chunks from the bulk allocation using offset-based `cuMemMap`.

This is fundamentally different from the failed pool-warming approach (`cache-codeword-buffer-across-segments`), which allocated and freed a buffer before proving, creating a free region that disrupted VPMM pool state and caused a 177ms GKR regression at APC 0. The bulk page creation approach does not create or free any buffer — it changes only the internal mechanics of how physical pages are created when the pool genuinely needs to grow. The resulting pool state (VA addresses, page locations, free regions) is identical to today.

## Current Code Path

### Allocation trigger

1. `stacked_commit()` in `crates/cuda-backend/src/stacked_pcs.rs:48` calls `rs_code_matrix()` at line 70.
2. `rs_code_matrix()` at line 182 allocates `DeviceBuffer::<F>::with_capacity(codeword_height * width)` at line 194. At APC 300 seg 0, this is ~192MB.
3. `DeviceBuffer::with_capacity` calls into the VPMM via `MEMORY_MANAGER.lock().alloc()`.

### VPMM allocation flow

4. `VirtualMemoryPool::alloc()` (`crates/cuda-common/src/memory_manager/vm_pool.rs:200-270`) looks for a free region. On cold start, none exist.
5. Calls `defragment_or_create_new_pages(alloc_size, stream_id)` at line 217.
6. `defragment_or_create_new_pages()` (line 510) computes `allocate_size` = bytes needing new pages.
7. **The hot loop** (lines 538-576): creates pages one at a time:
   ```
   while allocated_dst < dst + allocate_size {
       handle = vpmm_create_physical(device_id, page_size)  // cuMemCreate — ~0.4ms each
       vpmm_map(allocated_dst, page_size, handle)            // cuMemMap   — ~0.05ms each
       active_pages.insert(allocated_dst, handle)
       allocated_dst += page_size
   }
   ```
   With page_size=2MB and allocate_size=256MB: 128 iterations × ~0.45ms = ~58ms.
8. After the loop, a single `vpmm_set_access()` call sets read-write permissions on the entire range.

### Page tracking and lifecycle

- `active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>` maps each page's VA to its physical allocation handle. Currently, each page has a unique handle.
- Pages are never released during normal operation — only on pool `Drop` or allocation rollback.
- `remap_regions()` (line 687) double-maps pages during defragmentation, looking up each page's handle via `active_pages[va]` and calling `vpmm_map(new_va, page_size, handle)` with implicit offset=0.
- `Drop` (line 757) iterates `active_pages` and calls `vpmm_unmap(va, page_size)` + `vpmm_release(handle)` per page.
- `rollback_new_pages()` (line 481) does the same for partially-created allocations on failure.

### Why the current code is slow

`cuMemCreate` (called via `vpmm_create_physical`) is the dominant cost at ~0.4ms per call because it interacts with the GPU physical memory allocator. With 128 calls, total `cuMemCreate` time is ~51ms. `cuMemMap` is cheap (~0.05ms each) because it only updates the GPU page table.

### CUDA API detail

The `cuMemMap(ptr, size, offset, handle, flags)` API supports an `offset` parameter that maps a sub-range of a physical allocation. The current C shim (`crates/cuda-common/cuda/src/vpmm_shim.cu:53`) hardcodes offset=0:
```c
int _vpmm_map(CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle h) {
    return (int)cuMemMap(va, bytes, 0, h, 0);  // offset=0
}
```

### Granularity safety

`allocate_size` is always a multiple of `page_size` (enforced by the assert at line 515), and `page_size` is always a multiple of the CUDA allocation granularity (validated during pool construction at lines 162-166). Therefore `allocate_size` is always a valid size argument for `cuMemCreate`.

## Changes

### Change 1: Add offset-aware `vpmm_map` C shim

**File**: `crates/cuda-common/cuda/src/vpmm_shim.cu`

Add a new function `_vpmm_map_offset` alongside the existing `_vpmm_map`:
```c
int _vpmm_map_offset(CUdeviceptr va, size_t bytes, size_t offset, CUmemGenericAllocationHandle h) {
    return (int)cuMemMap(va, bytes, offset, h, 0);
}
```

**Why**: Enables mapping page-sized chunks from a single bulk physical allocation at different offsets.

### Change 2: Add Rust FFI binding for offset-aware map

**File**: `crates/cuda-common/src/memory_manager/cuda.rs`

Add extern declaration and safe wrapper:
```rust
extern "C" {
    fn _vpmm_map_offset(va: CUdeviceptr, bytes: usize, offset: usize, h: CUmemGenericAllocationHandle) -> i32;
}

pub(super) unsafe fn vpmm_map_offset(
    va: CUdeviceptr, bytes: usize, offset: usize, h: CUmemGenericAllocationHandle,
) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_map_offset(va, bytes, offset, h))
}
```

**Why**: Exposes the offset-aware mapping to the pool implementation.

### Change 3: Extend page tracking with offset; deduplicate handles on release

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

Change `active_pages` value type from bare handle to `(handle, offset)` tuple:
```rust
active_pages: HashMap<CUdeviceptr, (CUmemGenericAllocationHandle, usize)>,
// (handle, offset_within_handle)
```

For single-page allocations (existing behavior), offset is always 0. For bulk-allocated pages, offset = `i * page_size`.

**No `PageInfo` struct or ref-counting needed.** Handle deduplication is done at release time using `HashSet` (see Changes 7 and 8). This avoids the borrow-conflict issue of calling a `release_handle(&mut self)` method during a `drain()` iteration.

All existing code that reads from `active_pages` must destructure the tuple. There are exactly 4 read sites:
1. `defragment_or_create_new_pages` page creation loop — writes via `active_pages.insert(va, (handle, 0))` for single-page, `(bulk_handle, offset)` for bulk.
2. `remap_regions` page lookup (line 711-727) — reads `(handle, offset)`, remaps with offset.
3. `Drop` (line 768) — reads `(handle, offset)`, unmaps, collects handles for dedup release.
4. `rollback_new_pages` (line 487-497) — handles rollback of failed allocations.

### Change 4: Bulk page creation in `defragment_or_create_new_pages`

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`, function `defragment_or_create_new_pages` (line 510)

Replace **only the page-creation while loop and its `allocated_pages` vector** (lines 537-576). The code after the loop — `vpmm_set_access` (line 586), `free_region_insert` (line 601), the `allocated_dst` debug-assert, and the logging — remains unchanged. The variable `allocated_dst` is set to `dst + allocate_size as u64` at the end of both bulk and single-page paths (matching the current post-loop state).

Replace with a bulk-or-single dispatch:

```rust
const BULK_THRESHOLD: usize = 16; // Only bulk for >=16 pages (>=32MB with 2MB pages)
// Chosen conservatively: targets the 128-page rs_code_matrix allocation while
// leaving smaller allocations on the existing per-page path. Can be tuned down
// after verifying no regressions.

let num_new_pages = allocate_size / self.page_size;
let mut allocated_pages: Vec<(CUdeviceptr, CUmemGenericAllocationHandle, usize)> = Vec::new();

if num_new_pages >= BULK_THRESHOLD {
    // Bulk path: one cuMemCreate for the entire allocation, N cuMemMap with offsets
    match unsafe { vpmm_create_physical(self.device_id, allocate_size) } {
        Ok(bulk_handle) => {
            for i in 0..num_new_pages {
                let offset = i * self.page_size;
                let va = dst + offset as u64;
                if let Err(e) = unsafe {
                    vpmm_map_offset(va, self.page_size, offset, bulk_handle)
                } {
                    // Rollback: unmap pages mapped so far, release bulk handle
                    for &(rollback_va, _, _) in &allocated_pages {
                        let _ = unsafe { vpmm_unmap(rollback_va, self.page_size) };
                        self.active_pages.remove(&rollback_va);
                    }
                    let _ = unsafe { vpmm_release(bulk_handle) };
                    // Return the full reserved VA span (requested, not allocate_size)
                    // because take_unmapped_region reserved `requested` bytes at `dst`.
                    self.insert_unmapped_region(dst, requested);
                    return Err(MemoryError::from(e));
                }
                self.active_pages.insert(va, (bulk_handle, offset));
                allocated_pages.push((va, bulk_handle, offset));
            }
        }
        Err(e) if e.is_out_of_memory() => {
            // Fall back to single-page creation below
            // (The bulk allocation requested the full size; individual pages may still fit)
            for i in 0..num_new_pages {
                let va = dst + (i * self.page_size) as u64;
                let handle = match unsafe {
                    vpmm_create_physical(self.device_id, self.page_size)
                } {
                    Ok(h) => h,
                    Err(e) => {
                        if e.is_out_of_memory() {
                            // Use `requested` for the VA reservation size, not `allocate_size`
                            self.rollback_new_pages_bulk(dst, requested, &allocated_pages);
                            return Err(MemoryError::OutOfMemory {
                                requested: allocate_size,
                                available: i * self.page_size,
                            });
                        }
                        return Err(MemoryError::from(e));
                    }
                };
                unsafe { vpmm_map(va, self.page_size, handle)?; }
                self.active_pages.insert(va, (handle, 0));
                allocated_pages.push((va, handle, 0));
            }
        }
        Err(e) => return Err(MemoryError::from(e)),
    }
} else {
    // Single-page path: existing behavior for small allocations
    for i in 0..num_new_pages {
        let va = dst + (i * self.page_size) as u64;
        let handle = match unsafe {
            vpmm_create_physical(self.device_id, self.page_size)
        } {
            Ok(h) => h,
            Err(e) => {
                if e.is_out_of_memory() {
                    // Use `requested` for the VA reservation size, not `allocate_size`
                    self.rollback_new_pages_bulk(dst, requested, &allocated_pages);
                    return Err(MemoryError::OutOfMemory {
                        requested: allocate_size,
                        available: i * self.page_size,
                    });
                }
                return Err(MemoryError::from(e));
            }
        };
        unsafe { vpmm_map(va, self.page_size, handle)?; }
        self.active_pages.insert(va, (handle, 0));
        allocated_pages.push((va, handle, 0));
    }
}
// Set allocated_dst for the post-loop code (vpmm_set_access, free_region_insert, etc.)
allocated_dst = dst + allocate_size as u64;
```

Use `BULK_THRESHOLD = 16` (32MB with 2MB pages). This targets the large `rs_code_matrix` allocation (~256MB = 128 pages) while leaving smaller allocations on the existing fast path, reducing blast radius.

**Important**: All three rollback call sites pass `requested` (not `allocate_size`) as the `reserved_size` argument. This is because `take_unmapped_region(requested)` at line 529 reserved `requested` bytes at `dst`, and rollback must return the full reserved span. `allocate_size = requested.saturating_sub(total_free_size)` can be smaller when existing free regions partially cover the request.

**Why**: Reduces `cuMemCreate` calls from 128 to 1 for the rs_code_matrix allocation. Estimated saving: 128 × 0.4ms - 1 × 0.4ms = ~51ms. Total new cost: 0.4ms + 128 × 0.05ms ≈ 6.8ms vs current ~58ms.

### Change 5: Update `rollback_new_pages` to handle new tuple type

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

Replace the existing `rollback_new_pages` function (line 481) with a version that accepts the new tuple type and deduplicates handles:

```rust
fn rollback_new_pages_bulk(
    &mut self,
    reserved_ptr: CUdeviceptr,
    reserved_size: usize,
    allocated_pages: &[(CUdeviceptr, CUmemGenericAllocationHandle, usize)],
) {
    let mut released_handles: HashSet<CUmemGenericAllocationHandle> = HashSet::new();
    for &(addr, handle, _offset) in allocated_pages {
        if let Err(e) = unsafe { vpmm_unmap(addr, self.page_size) } {
            tracing::error!(
                "rollback: vpmm_unmap failed: addr={:#x}, size={}: {:?}",
                addr, self.page_size, e
            );
        }
        self.active_pages.remove(&addr);
        if released_handles.insert(handle) {
            if let Err(e) = unsafe { vpmm_release(handle) } {
                tracing::error!("rollback: vpmm_release failed: handle={}: {:?}", handle, e);
            }
        }
    }
    self.insert_unmapped_region(reserved_ptr, reserved_size);
}
```

The `HashSet<CUmemGenericAllocationHandle>` ensures each handle (whether single-page or bulk) is released exactly once. For single-page allocations each handle is unique so the set has no effect. For bulk allocations, only the first page triggers the release.

**Why**: The existing `rollback_new_pages` takes `&[(CUdeviceptr, CUmemGenericAllocationHandle)]`, which doesn't include the offset. Rather than modifying the old function's signature (which would require updating all callers), add a new function with the correct type. The old `rollback_new_pages` can be removed if no other callers exist.

### Change 6: Update `remap_regions` for offset-aware mapping

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`, function `remap_regions` (line 687)

Change the per-page remap (lines 711-727) from:
```rust
let handle = self.active_pages.remove(&page).expect("...");
vpmm_map(curr_dst, self.page_size, handle)?;
self.active_pages.insert(curr_dst, handle);
```

To:
```rust
let (handle, offset) = self.active_pages.remove(&page).expect("...");
unsafe { vpmm_map_offset(curr_dst, self.page_size, offset, handle)?; }
self.active_pages.insert(curr_dst, (handle, offset));
```

**Why**: Preserves correct physical-to-virtual mapping when pages from a bulk allocation are remapped during defragmentation. For single-page allocations, offset=0 so the behavior is identical to the current `vpmm_map` call.

### Change 7: Update `Drop` with handle deduplication

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`, `impl Drop for VirtualMemoryPool` (line 757)

Change from:
```rust
for (ptr, handle) in self.active_pages.drain() {
    vpmm_unmap(ptr, self.page_size).unwrap();
    vpmm_release(handle).unwrap();
}
```

To:
```rust
// Phase 1: unmap all pages and collect unique handles
let mut handles_to_release: HashSet<CUmemGenericAllocationHandle> = HashSet::new();
for (ptr, (handle, _offset)) in self.active_pages.drain() {
    unsafe { vpmm_unmap(ptr, self.page_size).unwrap(); }
    handles_to_release.insert(handle);
}
// Phase 2: release each unique handle exactly once
for handle in handles_to_release {
    unsafe { vpmm_release(handle).unwrap(); }
}
```

The two-phase approach avoids the borrow conflict: `drain()` consumes `active_pages` in phase 1, then phase 2 operates on the collected `HashSet` with no outstanding borrows.

**Why**: Prevents double-freeing a bulk handle when multiple pages share it. The `HashSet` naturally deduplicates: for N bulk-allocated pages sharing one handle, the handle is inserted once and released once.

### Change 8: Add `HashSet` import

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`

Add `HashSet` to the existing `std::collections` import:
```rust
use std::collections::{BTreeMap, HashMap, HashSet};
```

## Invariants

1. **Pool state identity**: After bulk page creation, `active_pages` contains the same set of VA keys as the current code would produce. The free regions, allocated regions, and VA layout are identical. Only the handle values differ (shared vs unique), which is invisible to all code outside the VPMM.
2. **No pool state disruption**: No buffer is allocated and freed before proving. The pool's free-region topology at proving time is unchanged. This avoids the GKR regression pattern seen in `cache-codeword-buffer-across-segments`.
3. **Physical memory layout**: The bulk `cuMemCreate` produces a contiguous physical allocation. Individual pages are mapped at their natural offsets within it. GPU TLB behavior should be identical or better (contiguous physical memory may improve TLB hit rate).
4. **Handle lifecycle**: Each `cuMemCreate` call produces exactly one handle. Each handle is released exactly once via `cuMemRelease`. Deduplication in `Drop` and `rollback_new_pages_bulk` ensures this: the `HashSet` guarantees at-most-once release per handle.
5. **Defragmentation correctness**: `remap_regions` uses the stored offset per page, so remapping a bulk-allocated page to a new VA preserves the correct physical offset within the bulk handle.
6. **Fallback safety**: If bulk `cuMemCreate` fails with OOM, the code falls back to single-page creation. This ensures the optimization never makes allocation less robust than today.
7. **Verifier correctness**: No changes to the prover's mathematical behavior. Only the VPMM's internal page creation speed changes.
8. **Granularity alignment**: `allocate_size` is always a multiple of `page_size`, which is always a multiple of the CUDA allocation granularity. Therefore `allocate_size` is always a valid argument for `cuMemCreate`.

## Measurement Plan

Run the standard benchmark suite:
```bash
cd /home/georg/powdr/results/pairing
# APC 300
$PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics <results>/after_apc300.json --recursion
# APC 0
$PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics <results>/after_apc000.json --recursion
```

Verify:
1. **APC 300**: STARK excl trace should decrease by ~40-50ms (from ~1301ms to ~1250-1260ms). Trace Commit should decrease by ~40-50ms. The 58ms cold-start is entirely within the Trace Commit / `prover.stacked_commit` span (called from `stark_prove_excluding_trace` → `prove` → `stacked_commit` → `rs_code_matrix`), so the savings map directly to STARK excl trace.
2. **APC 0**: STARK excl trace must NOT regress by more than 20ms. LogUp GKR must NOT regress by more than 20ms (this is the specific regression indicator from the failed warmup approach).
3. **Correctness**: Both APC 0 and APC 300 proofs must verify successfully.

Run each configuration twice to confirm consistency.

## Rollback Criteria

- Revert if APC 300 STARK excl trace improvement is less than 20ms.
- Revert if APC 0 STARK excl trace regresses by more than 20ms.
- Revert if APC 0 LogUp GKR regresses by more than 20ms (the specific regression indicator from the previous attempt).
- Revert if any proof fails to verify.
