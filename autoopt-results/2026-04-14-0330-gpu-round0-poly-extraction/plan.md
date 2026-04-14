# Plan: GPU-side Round 0 Polynomial Extraction

## Goal

Eliminate per-AIR `cudaStreamSynchronize` calls and CPU post-processing in the Round 0 multi-stream loop by moving the polynomial extraction (transpose + iDFT + Lagrange interpolation + coefficient adjustment) into a small CUDA kernel that runs on the GPU stream immediately after each evaluation kernel. The extracted polynomial coefficients are written directly into a device-side batch array, and a single D2H copy at the end replaces the current ~1246 per-AIR pipeline drains.

**Target metric**: Round 0 at APC 300: 243ms → ~190ms (save ~50ms). STARK excl trace at APC 300: 1376ms → ~1326ms.

Note on estimate conservatism: the per-AIR D2H sync (1246 `cudaStreamSynchronize` calls) and CPU post-processing are eliminated, but per-AIR H2D copies (4 per AIR for `d_main_parts`, `d_numer_weights`, `d_denom_weights`, `d_rules`) and the interaction DAG construction inside `evaluate_round0_interactions_gpu` remain. These residual per-AIR costs cap the improvement.

## Current Code Path

### Entry point
`LogupZerocheckGpu::sumcheck_uni_round0_polys` in `crates/cuda-backend/src/logup_zerocheck/mod.rs:900`.

### Per-AIR processing (function `process_air_round0`, lines 152-322)

For each AIR (78 per thread at APC 300, 8 threads):

1. **CPU setup** (lines 156-175): Build `SymbolicConstraints`, compute omega_root, collect main_parts, upload `d_main_parts` via `to_device()`.

2. **Zerocheck GPU kernel** (lines 197-212): `evaluate_round0_constraints_gpu()` → produces `sum_buffer: DeviceBuffer<EF>` of size `num_cosets_zc * skip_domain` (where `num_cosets_zc = d - 1`, `skip_domain = 1 << l_skip = 16`). Size ranges from 16 EF values (d=2) to 48 EF values (d=4), i.e., 256-768 bytes.

3. **Zerocheck D2H + CPU post-processing** (lines 214-242):
   - `sum_buffer.to_host_on_current_stream()` — **PIPELINE DRAIN**: calls `cudaMemcpyAsync` + `cudaStreamSynchronize(cudaStreamPerThread)`.
   - Transpose: rearrange from `(coset_idx, l_skip_idx)` to `(l_skip_idx, coset_idx)` layout.
   - `UnivariatePoly::from_geometric_cosets_evals_idft(values, omega_root, omega_root)`:
     - `Radix2BowersSerial::idft_batch()` on 16-element columns.
     - Coefficient unshifting by `(omega_root^{-(col+1)})^row`.
     - Lagrange interpolation across `num_cosets_zc` cosets.
   - Coefficient adjustment: `coeffs[i] = -q[i] + (q[i - skip_domain] if i >= skip_domain)`.
   - Result: `UnivariatePoly` of degree `d * (2^l_skip - 1)`, with `height * width` coefficients stored in interleaved layout `coeffs[coset_idx * height + row_idx]` (see `poly.rs:678`).

4. **Logup GPU kernel** (lines 262-279): `evaluate_round0_interactions_gpu()` → produces `sum: DeviceBuffer<Frac<EF>>` of size `num_cosets_logup * skip_domain` (where `num_cosets_logup = d`). Size ranges from 32 to 64 FracExt values (1024-2048 bytes).

5. **Logup D2H + CPU post-processing** (lines 281-314):
   - `sum.to_host_on_current_stream()` — **PIPELINE DRAIN**.
   - Unpack `FracExt` into `(numer, denom)` vectors.
   - Optional normalization when `n < 0`: multiply numer by `(1/2^|n|)`.
   - Transpose (same pattern as zerocheck).
   - Two calls to `from_geometric_cosets_evals_idft`: one for numer (`init = F::ONE`), one for denom (`init = F::ONE`).
   - Result: two `UnivariatePoly` with `height * width` coefficients in interleaved layout.

6. **Return** `Round0AirResult { trace_idx, zerocheck_poly, logup_numer_poly, logup_denom_poly }`. Each polynomial is `Option<UnivariatePoly<EF>>` — `None` when the AIR has no constraints or no interactions.

### Post-loop scatter (lines 1179-1190)
Results are placed into `batch_sp_poly[3 * num_present_airs]`:
- `batch_sp_poly[2 * num_present_airs + trace_idx]` = zerocheck poly (only if `Some`)
- `batch_sp_poly[2 * trace_idx]` = logup numer poly (only if `Some`)
- `batch_sp_poly[2 * trace_idx + 1]` = logup denom poly (only if `Some`)

Unset slots remain as `UnivariatePoly::new(vec![])` (empty, length 0).

### Why it's slow
Each of the 623 AIRs performs 2 `to_host_on_current_stream` calls, each calling `cudaStreamSynchronize`. This:
- Creates 1246 GPU pipeline drains across the benchmark.
- Forces the CPU thread to block while the GPU kernel completes.
- Inserts ~0.5ms of CPU post-processing (transpose + iDFT) between each pair of kernel launches, during which the thread's GPU stream has no pending work.

## Changes

### Change 1: New CUDA kernel `round0_extract_poly`

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/round0_extract.cu` (new file)

Two `__global__` kernels, each launched as `<<<1,1>>>`:

**`round0_extract_zerocheck_kernel`**:
- Input: `const FpExt* d_evals` (evaluation output, length `num_cosets * skip_domain`), parameters, pre-computed tables, output pointer.
- Operations:
  1. Read evaluation values from device memory into registers/local memory.
  2. Transpose from `(coset, l_skip)` to `(l_skip, coset)` layout: `values[i * num_cosets + c] = d_evals[c * skip_domain + i]`.
  3. In-place inverse DFT on each column (size `skip_domain`). Must match `Radix2BowersSerial::idft_batch` exactly (reference: `p3_dft::radix_2_bowers_serial`, method `idft_batch` which calls three steps in sequence):
     - **(3a) `bowers_g_t` butterflies**: The Bowers-network inverse DFT butterfly pattern, NOT standard Cooley-Tukey. Twiddle factors are powers of `F::two_adic_generator(log2(skip_domain)).inverse()`, applied in the Bowers permutation order.
     - **(3b) Divide by height**: Multiply ALL elements by `F::from_canonical_u32(skip_domain).inverse()`. This is the `1/N` scaling factor of the inverse DFT.
     - **(3c) Bit-reversal permutation on rows**: Permute row indices by reversing their `log2(skip_domain)` low bits. For `skip_domain=16`, this is a 4-bit reversal: row 1 ↔ row 8, row 2 ↔ row 4, row 3 ↔ row 12, etc. After this step, rows are in natural (non-bit-reversed) order.
     The simplest correct implementation: hardcode the butterfly + bit-reversal pattern for skip_domain=16 (the only value used in the leaf config), using compile-time constants for the BabyBear two-adic inverse generators at each stage.
  4. Coefficient unshifting: for each `(row, col)`, multiply by `shift_invs_table[col * skip_domain + row]` (pre-computed on host and uploaded; `shift_invs_table[col * H + t] = (init^{-1} * shift^{-col})^t` where `H = skip_domain`).
  5. Lagrange interpolation: for each row `t`, for each output coset index `k` in `0..width`:
     `output[k * skip_domain + t] = sum_{i=0}^{width-1} values[t * width + i] * lagrange_basis[i * width + k]`
     This produces coefficients in the interleaved layout `coeffs[coset_idx * height + row_idx]` matching `poly.rs:678`.
  6. Zerocheck coefficient adjustment: `sp_0_deg = d * (skip_domain - 1)` where `d = num_cosets + 1` (since `num_cosets = d - 1`). For `i in 0..=sp_0_deg`:
     `result[i] = -output[i] + (output[i - skip_domain] if i >= skip_domain else 0)`.
     Note: `output` has `(d-1) * skip_domain` = `num_cosets * skip_domain` coefficients from step 5 (interleaved layout). `sp_0_deg + 1` may exceed this (e.g., d=4: output has 48 coeffs, sp_0_deg=60 needs index 60). The values at indices `>= num_cosets * skip_domain` in `output` are zero (from the zero-filled batch buffer), and the adjustment formula correctly references `output[i - skip_domain]` which is within bounds.
  7. Write `result[0..=sp_0_deg]` (= `sp_0_deg + 1` coefficients) to the designated batch array slot.

**`round0_extract_logup_kernel`**:
- Same as above but:
  - Input is `const FracExt* d_evals`. First unpack into separate numer/denom `FpExt` arrays (in registers).
  - Apply optional normalization: `numer[i] *= norm_factor` before transpose.
  - Run steps 2-5 twice (once for numer, once for denom), using the logup Lagrange basis (where `init = F::ONE`).
  - No coefficient adjustment step (logup polynomials are stored as-is after interpolation).
  - Write numer coefficients to numer batch slot, denom coefficients to denom batch slot.

**Extern "C" launcher functions**:
```c
extern "C" int _round0_extract_zerocheck_poly(
    FpExt* d_batch_out,         // destination slot (offset into device batch array)
    const FpExt* d_evals,       // evaluation kernel output
    uint32_t num_cosets,        // = local_constraint_deg - 1
    uint32_t skip_domain,       // = 1 << l_skip (always 16 for leaf)
    uint32_t sp_0_deg,          // = d * (skip_domain - 1), output poly degree
    const FpExt* d_lagrange,    // pre-computed [width * width] Lagrange basis
    const FpExt* d_shift_invs   // pre-computed [width * skip_domain] shift inverse powers
);

extern "C" int _round0_extract_logup_polys(
    FpExt* d_batch_numer,       // numer destination slot
    FpExt* d_batch_denom,       // denom destination slot
    const FracExt* d_evals,     // evaluation kernel output
    uint32_t num_cosets,        // = local_constraint_deg
    uint32_t skip_domain,       // = 1 << l_skip
    FpExt norm_factor,          // EF::ONE normally, EF::from(F::from_u32(1<<|n|).inverse()) when n<0
    const FpExt* d_lagrange,    // pre-computed Lagrange basis for logup (init=F::ONE)
    const FpExt* d_shift_invs   // pre-computed shift inverse powers for logup
);
```

### Change 2: Rust FFI bindings

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `extern "C"` declarations and safe wrapper functions for the two new kernels. The wrappers take `*mut EF` / `*const EF` raw pointers (not `&DeviceBuffer`) since the callers pass offsets into a shared device buffer.

### Change 3: Pre-compute and upload Lagrange basis tables

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, inside `sumcheck_uni_round0_polys` (before the multi-stream loop, around line 949)

`l_skip` is fixed per proof (set in `SystemParams`), not per AIR. `omega_root` varies by AIR via `F::two_adic_generator(log2_ceil(local_constraint_deg << l_skip))`.

Before Phase 1 (work item construction):
1. Collect unique `local_constraint_deg` values across all AIRs.
2. For each unique `d`, compute:
   - `omega_root = F::two_adic_generator(log2_ceil_usize(d << l_skip))`
   - `coset_base = omega_root.exp_power_of_2(l_skip)` (since `log_height = l_skip`)
   - For **zerocheck** (`num_cosets = d-1`, `init = omega_root`):
     - `lagrange_basis = lagrange_basis_from_geometric_points(coset_base, d-1, omega_root.exp_power_of_2(l_skip))`
     - `shift_invs`: for each `col in 0..d-1`, `row in 0..skip_domain`: `(omega_root^{-(col+1)})^row`
   - For **logup** (`num_cosets = d`, `init = F::ONE`):
     - `lagrange_basis = lagrange_basis_from_geometric_points(coset_base, d, F::ONE.exp_power_of_2(l_skip))`
     - `shift_invs`: for each `col in 0..d`, `row in 0..skip_domain`: `(F::ONE^{-1} * omega_root^{-col})^row = omega_root^{-col*row}`
3. Flatten each Lagrange basis into `Vec<EF>` of size `width * width` (row-major), and each shift_invs table into `Vec<EF>` of size `width * skip_domain`. Upload to GPU.
4. Store in a `HashMap<usize, PrecomputedR0Tables>` keyed by `local_constraint_deg`, where `PrecomputedR0Tables` contains 4 `DeviceBuffer<EF>` fields: `{zc_lagrange, zc_shift_invs, logup_lagrange, logup_shift_invs}`.

Expected: 2-3 unique `d` values, ~400 EF values total (~6KB GPU memory).

### Change 4: Pre-allocate device batch polynomial array

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, inside `sumcheck_uni_round0_polys`

After work item construction (around line 1036):
1. Compute per-AIR polynomial lengths:
   - For each `trace_idx`, record `poly_len_zc = sumcheck_round0_deg(l_skip, d) + 1 = d * (skip_domain - 1) + 1` if AIR has constraints, else 0. (From `prover/sumcheck.rs:178`: `sumcheck_round0_deg(l_skip, d) = d * ((1 << l_skip) - 1)`.)
   - `poly_len_logup = d * skip_domain` if AIR has interactions, else 0.
   - Store these in a `Vec<(usize, usize, usize)>` (one per trace_idx: `(zc_len, numer_len, denom_len)`).
2. Compute `max_poly_len`: maximum of all lengths. With `d_max=4`, `l_skip=4`: `max(4*(16-1)+1, 4*16) = max(61, 64) = 64`.
3. Allocate `DeviceBuffer<EF>::with_capacity(3 * num_present_airs * max_poly_len)`.
4. `fill_zero` the buffer (zero coefficients = zero polynomial, functionally equivalent to empty).
5. Compute the per-trace offset mapping as `usize` values:
   - `zc_offset[trace_idx] = (2 * num_present_airs + trace_idx) * max_poly_len`
   - `numer_offset[trace_idx] = (2 * trace_idx) * max_poly_len`
   - `denom_offset[trace_idx] = (2 * trace_idx + 1) * max_poly_len`
6. Pass offset mapping and raw device pointer to worker threads.

Guard: if `num_present_airs == 0`, skip the allocation and return an empty `batch_sp_poly` immediately (avoids `DeviceBuffer::with_capacity(0)` panic).

Memory: `3 * 623 * 64 * 16 = ~1.9MB`. Negligible.

### Change 5: Modify `process_air_round0` to skip D2H and CPU post-processing

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, function `process_air_round0`

Add new parameters: `d_batch_ptr: *mut EF` (raw device pointer to batch array), `zc_offset: usize`, `numer_offset: usize`, `denom_offset: usize`, pre-computed table references, `sp_0_deg: u32`.

Replace the D2H + CPU iDFT blocks (lines 214-242 and 281-314) with GPU extraction kernel launches:

```rust
// After zerocheck kernel:
let sum_buffer = evaluate_round0_constraints_gpu(...)?;
if !sum_buffer.is_empty() {
    unsafe {
        _round0_extract_zerocheck_poly(
            d_batch_ptr.add(w.zc_offset),  // raw pointer offset
            sum_buffer.as_ptr(),
            num_cosets_zc as u32,
            (1 << w.l_skip) as u32,
            sp_0_deg,
            tables.zc_lagrange.as_ptr(),
            tables.zc_shift_invs.as_ptr(),
        );
    }
}

// After logup kernel:
let sum = evaluate_round0_interactions_gpu(...)?;
if !sum.is_empty() {
    let norm_factor = if w.n.is_negative() {
        EF::from(F::from_u32(1 << w.n.unsigned_abs()).inverse())
    } else {
        EF::ONE
    };
    unsafe {
        _round0_extract_logup_polys(
            d_batch_ptr.add(w.numer_offset),
            d_batch_ptr.add(w.denom_offset),
            sum.as_raw_ptr() as *const _,
            num_cosets_logup as u32,
            (1 << w.l_skip) as u32,
            norm_factor.into(),  // convert to GPU FpExt repr
            tables.logup_lagrange.as_ptr(),
            tables.logup_shift_invs.as_ptr(),
        );
    }
}
```

`process_air_round0` no longer returns `Round0AirResult` with polynomials — it returns `Result<(), ...>` (or just a bool indicating success).

**Thread safety for `d_batch_ptr`**: The `DeviceBuffer` is allocated on the main thread. Worker threads receive the raw `*mut EF` pointer (obtained via `.as_mut_ptr()` before spawning). Each thread writes to non-overlapping offsets (by unique `trace_idx`). CUDA guarantees non-overlapping device writes from different streams are safe. Since `*mut EF` is not `Send`, transmit it as a `usize` (via `as usize`) and recast in the thread closure.

### Change 6: Replace post-loop scatter with single D2H

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`, lines 1179-1193

After all threads complete:
1. `current_stream_sync()` to ensure all extraction kernels from all streams are done. (Worker threads' `cudaStreamPerThread` kernels may still be pending on the main thread's view — the existing `std::thread::scope` join already ensures threads have returned, and since the extraction kernels are the last operations on each thread's stream, joining suffices for stream completion.)
2. D2H copy the device batch array to host: `d_batch_array.to_host()`.
3. Reconstruct `batch_sp_poly: Vec<UnivariatePoly<EF>>` from the flat host array using the pre-computed per-AIR polynomial lengths (from Change 4 step 1):
   - For each slot `i` in `0..3*num_present_airs`:
     - Determine the known polynomial length for this slot from `poly_lens[i]`.
     - If length is 0, set `batch_sp_poly[i] = UnivariatePoly::new(vec![])`.
     - Otherwise, slice `host_array[i * max_poly_len .. i * max_poly_len + poly_len]` and construct `UnivariatePoly::new(slice.to_vec())`.
   This preserves the exact same semantics as the current code: absent polynomials have length 0, present ones have the correct length.

### Change 7: Add `round0_extract.cu` to the build

**File**: `crates/cuda-backend/build.rs` (or the CUDA build configuration)

Add `cuda/src/logup_zerocheck/round0_extract.cu` to the list of compiled CUDA sources.

## Invariants

1. **Polynomial coefficients must exactly match the CPU-computed values.** The GPU iDFT must use the Bowers-network DFT pattern (matching `Radix2BowersSerial`), not standard Cooley-Tukey. Verify with a debug assertion that runs both paths on the first segment and compares coefficient-by-coefficient.

2. **Output coefficient layout must be interleaved**: `coeffs[coset_idx * height + row_idx]` matching `poly.rs:678`. The GPU kernel must write in this layout.

3. **The batch polynomial array layout must match the scatter logic.** Slot `2 * num_present_airs + trace_idx` = zerocheck, `2 * trace_idx` = numer, `2 * trace_idx + 1` = denom. The flat device array uses `max_poly_len` stride.

4. **Absent polynomials must be preserved as empty.** When an AIR has no constraints (empty `sum_buffer`) or no interactions (empty `eq_3bs`), no extraction kernel is launched, and the zero-filled slot in the device array is reconstructed as `UnivariatePoly::new(vec![])` (length 0) on the host, NOT as a zero-filled polynomial of length `max_poly_len`. Downstream code may check `.coeffs().len() == 0`.

5. **Pre-allocated intermediate buffers must still work.** The evaluation kernels are unchanged. Only the post-processing changes.

6. **APC 0 path must not regress.** At APC 0 (<100 AIRs), Round 0 runs single-threaded. The GPU extraction path applies equally.

7. **FracExt unpacking and normalization must be correct.** The logup kernel must split FracExt (p, q) and apply `norm_factor` to numerator when `n < 0`.

8. **Thread safety for shared device buffer.** Worker threads write to non-overlapping offsets of the shared batch array via raw pointer arithmetic. CUDA guarantees this is safe for non-overlapping addresses across streams.

## Measurement Plan

1. Build and run APC 300 benchmark: `cd /home/georg/powdr/results/pairing && RUST_LOG=info powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics <path>/after_apc300.json --recursion`
2. Run APC 0 benchmark similarly.
3. Analyze with `spec.py`: compare Round 0 and STARK excl trace.
4. Run nsys profile on APC 300: verify that D2H copy count drops significantly (from ~5686 to ~4440, eliminating ~1246 per-AIR copies).
5. Add a debug assertion (gated behind `#[cfg(debug_assertions)]`) that compares GPU-extracted coefficients with CPU-computed ones for the first segment.

**Expected results**:
- Round 0 at APC 300: 243ms → ~190ms (20% reduction, ~50ms savings).
- STARK excl trace at APC 300: 1376ms → ~1326ms (1.85x vs baseline).
- Round 0 at APC 0: 177ms → ~160ms (10% reduction).
- No regression in any other metric.

## Rollback Criteria

- If Round 0 improvement at APC 300 is **less than 25ms**, revert.
- If any other STARK excl trace component regresses by more than 15ms at APC 300, revert.
- If APC 0 STARK excl trace regresses by more than 20ms, revert.
- If polynomial coefficients do not match CPU-computed values (correctness failure), revert.
