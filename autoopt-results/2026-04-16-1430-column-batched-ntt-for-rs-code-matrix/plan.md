# Plan: Column-Batched NTT for RS Code Matrix

## Goal

Improve the NTT throughput in `rs_code_matrix` by splitting the column-wise forward NTT into L2-cache-sized batches. At APC 300, the NTT processes ~53K columns × 2^14 height simultaneously, creating a 3.4GB working set that dwarfs the 72MB L2 cache. NTT butterfly stages at large strides (2^7–2^13 elements) read from DRAM on every access, achieving ~30% of peak bandwidth. By processing ~1000 columns per batch (~64MB, fitting in L2), the second NTT kernel step (stages 8–14) reads data that's still L2-resident from the first step (stages 1–7), improving effective bandwidth through better TLB hit rates and DRAM page locality. APC 300 benefits disproportionately: its 64KB columns fit well in L2 per batch, while APC 0's 4MB columns see less benefit.

## Current Code Path

### `rs_code_matrix` in `crates/cuda-backend/src/stacked_pcs.rs`, lines 181-279

The function computes Reed-Solomon codewords for the stacked trace matrix:

1. **Lines 197-226**: `batch_expand_pad` or `stack_traces_into_expanded` + optional `batch_ntt_small` — prepares the codeword buffer with stacked data, expanded to codeword height.
2. **Lines 236-252**: Optional `mle_interpolate_stages` — subset-zeta transform for `l_skip > 0`.
3. **Lines 254-264**: `bit_rev` — in-place bit-reversal permutation of all columns.
4. **Lines 267-274**: `batch_ntt` — forward NTT, the **most expensive step** (~47ms per segment at APC 300).

### `batch_ntt` in `crates/cuda-backend/src/ntt.rs`, lines 87-130

```rust
pub fn batch_ntt(buffer, log_trace_height, log_blowup, width, bit_reverse, is_intt) {
    let padded_poly_size = 1 << (log_trace_height + log_blowup);
    if bit_reverse { ntt::bit_rev(buffer, buffer, ...); }
    let mut _impl = NttImpl::new(buffer, log_trace_height, padded_poly_size, width, is_intt);
    // For log_trace_height in 11..=17 (APC 300 case, ~14):
    let step = log_trace_height / 2;
    _impl.step(step + log_trace_height % 2);  // kernel launch 1: stages 0..7
    _impl.step(step);                          // kernel launch 2: stages 7..14
}
```

Each `_impl.step(iterations)` calls `ntt::ct_mixed_radix_narrow` (`crates/cuda-backend/src/cuda/ntt.rs`, line 141) which is an FFI call to `_ct_mixed_radix_narrow`. The CUDA kernel receives:
- `d_inout`: raw device pointer (`DeviceBuffer::as_mut_raw_ptr()`)
- `poly_count`: number of independent column-NTTs = `width`
- `padded_poly_size`: elements per column
- Stage/radix parameters

The kernel processes all `poly_count` columns in parallel. Each column's NTT is independent.

### Per-segment numbers at APC 300

- `width` ≈ 53K columns
- `codeword_height` = 2^14 = 16384 (`log_trace_height` ≈ 14, after `l_skip` correction; `log_blowup` = 0 since expansion was pre-applied)
- Column size: 16384 × 4 bytes = 64 KB
- Total data: 53K × 64KB = 3.4 GB
- L2 cache: 72 MB on RTX 4090
- Columns per L2 batch: floor(64MB / 64KB) ≈ 1000 columns
- NTT kernel launches per batch: 2 (since 14 ≤ 17 → two steps)

### Per-segment numbers at APC 0

- `width` ≈ 784 columns
- `codeword_height` = 2^20 = 1048576
- Column size: 1048576 × 4 = 4 MB
- Total data: 784 × 4MB = 3.1 GB
- Columns per L2 batch: floor(64MB / 4MB) = 16 columns
- NTT kernel launches per batch: 3 (since 20 ≤ 30 → three steps)
- Number of batches: ceil(784/16) = 49

## Changes

### Change 1: Add `batch_ntt_column_batched` helper in `crates/cuda-backend/src/ntt.rs`

Add a new function alongside `batch_ntt` that splits the NTT into L2-sized column batches:

```rust
/// Like `batch_ntt`, but processes columns in L2-cache-sized batches for better
/// memory locality. Falls back to `batch_ntt` when the total working set already
/// fits in L2.
pub fn batch_ntt_column_batched(
    buffer: &DeviceBuffer<F>,
    log_trace_height: u32,
    log_blowup: u32,
    width: u32,
    bit_reverse: bool,
    is_intt: bool,
) {
    if log_trace_height == 0 {
        return;
    }
    let padded_poly_size = 1u64 << (log_trace_height + log_blowup);
    let col_bytes = padded_poly_size as usize * std::mem::size_of::<F>();
    let total_bytes = col_bytes * width as usize;

    // If the entire working set fits in ~60MB (leaving headroom in the 72MB L2),
    // there's no benefit to batching — use the standard path.
    const L2_BUDGET: usize = 60 * 1024 * 1024;
    if total_bytes <= L2_BUDGET {
        batch_ntt(buffer, log_trace_height, log_blowup, width, bit_reverse, is_intt);
        return;
    }

    let batch_cols = (L2_BUDGET / col_bytes).max(1) as u32;

    // Bit-reverse entire buffer first (one pass) — the batched NTT steps
    // expect bit-reversed input within each column.
    if bit_reverse {
        unsafe {
            ntt::bit_rev(
                buffer,
                buffer,
                log_trace_height,
                padded_poly_size as u32,
                width,
            )
            .unwrap();
        }
    }

    // Process NTT in column batches
    for batch_start in (0..width).step_by(batch_cols as usize) {
        let batch_width = (width - batch_start).min(batch_cols);
        let element_offset = batch_start as usize * padded_poly_size as usize;
        // SAFETY: offset is within buffer bounds; non-owning view is valid for
        // the duration of the NTT call; NTT operates only within the view.
        let sub_buffer = unsafe {
            DeviceBuffer::non_owning(
                buffer.as_ptr().add(element_offset) as *mut F,
                batch_width as usize * padded_poly_size as usize,
            )
        };
        let mut ntt_impl = NttImpl::new(
            &sub_buffer,
            log_trace_height,
            padded_poly_size as u32,
            batch_width,
            is_intt,
        );
        // Replicate the same step schedule as batch_ntt
        if log_trace_height <= 10 {
            ntt_impl.step(log_trace_height);
        } else if log_trace_height <= 17 {
            let step = log_trace_height / 2;
            ntt_impl.step(step + log_trace_height % 2);
            ntt_impl.step(step);
        } else if log_trace_height <= 30 {
            let step = log_trace_height / 3;
            let rem = log_trace_height % 3;
            ntt_impl.step(step);
            ntt_impl.step(step + (if log_trace_height == 29 { 1 } else { 0 }));
            ntt_impl.step(step + (if log_trace_height == 29 { 1 } else { rem }));
        } else if log_trace_height <= 40 {
            let step = log_trace_height / 4;
            let rem = log_trace_height % 4;
            ntt_impl.step(step);
            ntt_impl.step(step + (if rem > 2 { 1 } else { 0 }));
            ntt_impl.step(step + (if rem > 1 { 1 } else { 0 }));
            ntt_impl.step(step + (if rem > 0 { 1 } else { 0 }));
        }
    }
}
```

**Why this helps**: Reducing the concurrent NTT working set from 3.4GB to ~64MB improves DRAM page locality and TLB hit rates for the large-stride butterfly accesses in the NTT. Whether data from the first NTT step remains fully L2-resident for the second step depends on the kernel's thread-to-element mapping — all threads access the batch concurrently, so the full batch is the effective working set. The primary benefit is from reduced DRAM row buffer conflicts and improved TLB coverage when fewer pages are concurrently active. Expected improvement: 5-15ms on the NTT portion of Trace Commit.

**Note**: The `NttImpl` struct and its `step` method are already defined in `ntt.rs` (lines 32-77) and are `pub(crate)` accessible. The `ensure_initialized(is_intt)` call happens inside `NttImpl::new`.

### Change 2: Use `batch_ntt_column_batched` in `rs_code_matrix`

**File**: `crates/cuda-backend/src/stacked_pcs.rs`, lines 254-274

Replace:
```rust
unsafe {
    bit_rev(&codewords, &codewords, log_codeword_height as u32, codeword_height as u32, width as u32)?;
}
batch_ntt(&codewords, log_codeword_height as u32, 0u32, width as u32, false, false);
```

With:
```rust
batch_ntt_column_batched(
    &codewords,
    log_codeword_height as u32,
    0u32,
    width as u32,
    true,   // let the batched function handle bit_rev
    false,
);
```

This also eliminates the separate `bit_rev` call (it's done inside `batch_ntt_column_batched`), keeping the code cleaner. The bit_rev is NOT batched — it still runs on all columns at once. The only batched operation is the NTT kernel steps.

**Why bit_rev is not batched**: Bit-reversal is a simple scatter/gather with no reuse between elements. L2 caching provides no benefit because each element is read once and written once. The strided access pattern for bit_rev is inherent to the permutation. Batching it would add kernel launch overhead without improving bandwidth.

## Invariants

1. **Mathematical correctness**: Each column's NTT is independent. Processing columns in batches of 1000 produces the same result as processing all 53K at once — the NTT butterfly operations within each column only access elements of that same column.

2. **Memory safety**: The `non_owning` DeviceBuffer view points into the middle of the `codewords` buffer. It's only used within the for-loop body and dropped before the next iteration. The `codewords` buffer outlives all views.

3. **APC 0 behavior**: At APC 0, `total_bytes = 784 × 4MB = 3.1GB > 60MB`, so batching IS activated. But with `batch_cols = 16`, there are 49 batches × 3 steps = 147 kernel launches (vs 3 currently). The additional kernel launch overhead is 147 × ~10μs = 1.5ms. The L2 benefit for 4MB columns is moderate — large-stride butterflies (stride > 1MB) exceed L2 even within a batch. Expected net effect: neutral to slightly positive.

4. **Kernel launch overhead**: At APC 300, 53 batches × 2 steps = 106 kernel launches (vs 2 currently). Additional launch overhead: ~1ms. The expected TLB/DRAM-page-locality benefit (5-15ms) exceeds this overhead.

5. **No VPMM interaction**: The optimization only changes how existing NTT kernels are called — no new allocations, no buffer layout changes, no pool state disruption.

## Measurement Plan

0. **Before implementing**: Instrument the NTT time in isolation by adding `cudaEventRecord` + `cudaEventSynchronize` around just the `batch_ntt` call in `rs_code_matrix`. This establishes the actual NTT baseline (estimated 50-80ms of the ~195ms Trace Commit) and determines whether the optimization targets a sufficiently large span.
1. Run `run_pairing.sh` for APC 0, 100, 300 before and after the change.
2. Primary metric: `STARK (excl. trace)` at APC 300 — expect 5-15ms improvement (conservative estimate based on TLB and DRAM page locality gains; up to 30ms if L2 reuse materializes for the second NTT step). The primary benefit is from reducing concurrent working set from 3.4GB to 64MB, improving DRAM page hit rates for the strided butterfly accesses.
3. Secondary metric: `Trace Commit` breakdown — expect the NTT portion to decrease.
4. APC 0 regression check: STARK excl trace within ±20ms.
5. Sweep `L2_BUDGET` values (32MB, 48MB, 64MB, 128MB) to find the optimal batch size. The optimal point balances L2 residency against kernel launch overhead.

## Rollback Criteria

- Less than 5ms improvement on Trace Commit at APC 300 (measured per-component, not STARK total, since the NTT is a sub-span).
- Regression > 20ms on STARK excl trace at APC 0.
- Proof verification failure at any APC configuration.
