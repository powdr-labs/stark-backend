# Plan: Batch Round 0 Coset-Parallel Kernel Launches

## Goal

Reduce Round 0 GPU time at APC 300 by batching ~935 sequential coset-parallel kernel launches into 2 batched launches (one for zerocheck, one for logup). Currently each small AIR launches its own kernel with 3-12 thread blocks on a GPU with 108+ SMs, resulting in 3-11% SM utilization per launch. Batching executes all small AIRs simultaneously in a single kernel, targeting near-100% SM utilization.

Primary target: Round 0 APC 300 wall time < 400ms (currently 570ms).
Secondary target: STARK excl trace APC 300 < 2050ms (currently 2205ms).

## Current Code Path

### Rust orchestration

`sumcheck_uni_round0_polys` (`crates/cuda-backend/src/logup_zerocheck/mod.rs:599-928`) runs in two phases:

- **Phase 1** (lines 744-841): Iterates all AIR instances sequentially. For each AIR:
  1. Builds `d_main_parts` (device array of trace column pointers) — line 781-786
  2. Calls `evaluate_round0_constraints_gpu` (zerocheck) — line 795-808
  3. Calls `evaluate_round0_interactions_gpu` (logup) — line 812-827
  4. Stores `DeviceBuffer` results in `Round0Pending` vec — line 830-841
- **Phase 2** (lines 844-924): `current_stream_sync()`, then D2H + polynomial construction per AIR.

### Per-AIR zerocheck path

`evaluate_round0_constraints_gpu` (`round0.rs:31-124`):
1. Early-returns if no constraints or `num_cosets == 0`
2. Reads pre-built rules from `pk.other_data.zerocheck_round0.inner` (d_rules, d_used_nodes, buffer_size)
3. Computes intermediates and temp_sums buffer sizes via FFI (`_zerocheck_r0_intermediates_buffer_size`)
4. Allocates `intermediates`, `temp_sums_buffer`, `sp_evals` on device
5. Calls FFI `zerocheck_ntt_eval_constraints` → `_zerocheck_ntt_eval_constraints` (zerocheck_round0.cu:667-717)

The C dispatcher checks `use_coset_parallel_mode(num_x, skip_domain)` (threshold: `num_x * skip_domain < 32768`):
- Small AIRs → `launch_zerocheck_coset_parallel` with grid `(x_blocks, num_cosets)` — each block handles ONE coset
- Large AIRs → `dispatch_zerocheck` with lockstep kernel — each thread handles ALL cosets

Each launcher issues 2 GPU kernel launches: evaluation kernel + `final_reduce_block_sums`.

### Per-AIR logup path

`evaluate_round0_interactions_gpu` (`round0.rs:132-275`):
1. Early-returns if no interactions (`eq_3bs.is_empty()`)
2. **Per-AIR CPU work**: Builds symbolic DAG from `symbolic.interactions`, compiles to `SymbolicRulesGpu`, computes `numer_weights` and `denom_weights`, computes `denom_sum_init`
3. Uploads `d_rules`, `d_numer_weights`, `d_denom_weights` to device (per-AIR H2D)
4. Allocates intermediates, temp_sums, output buffers
5. Calls FFI `logup_bary_eval_interactions_round0` → `_logup_bary_eval_interactions_round0` (logup_round0.cu:658-810)

Same coset-parallel dispatch as zerocheck, same 2-kernel launch pattern.

### Bottleneck summary

At APC 300: ~539 zerocheck + ~396 logup coset-parallel launches = ~935 evaluation kernels + ~935 reductions = **~1870 sequential GPU kernel launches**, each using 3-11% of SMs. Total coset-parallel GPU time: ~464ms (81% of Round 0's 570ms wall time).

## Changes

### Change 1: CUDA — Metadata structs

**File: `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`**

Add in the `zerocheck_round0` namespace:

```cpp
struct ZerocheckBatchMeta {
    const Fp *selectors_cube;          // [3][num_x]
    const Fp *preprocessed;            // per-AIR, may be null
    const Fp *const *main_parts;       // per-AIR pointer array on device
    const FpExt *eq_cube;              // [num_x]
    const Fp *public_values;           // per-AIR
    const Rule *d_rules;               // per-AIR constraint DAG
    const size_t *d_used_nodes;        // per-AIR used node indices
    size_t rules_len;
    size_t used_nodes_len;
    uint32_t buffer_size;              // must be <= BUFFER_THRESHOLD for batching
    uint32_t num_x;                    // 1 << n_lift
    uint32_t height;                   // trace height
    uint32_t num_cosets;               // constraint_degree - 1
    Fp g_shift;                        // omega_root for this AIR
};
```

**File: `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`**

Add in the `logup_round0` namespace:

```cpp
struct LogupBatchMeta {
    const Fp *selectors_cube;
    const Fp *preprocessed;
    const Fp *const *main_parts;
    const FpExt *eq_cube;
    const Fp *public_values;
    const FpExt *numer_weights;        // per-AIR, freshly computed
    const FpExt *denom_weights;        // per-AIR, freshly computed
    const Rule *d_rules;               // per-AIR, freshly compiled DAG
    size_t rules_len;
    FpExt denom_sum_init;              // per-AIR scalar
    uint32_t buffer_size;              // must be <= BUFFER_THRESHOLD for batching
    uint32_t num_x;
    uint32_t height;
    uint32_t num_cosets;               // constraint_degree
    Fp g_shift;
};
```

### Change 2: CUDA — Batched evaluation kernels

**File: `zerocheck_round0.cu`**

Add `zerocheck_ntt_evaluate_constraints_coset_parallel_batched_kernel<NEEDS_SHMEM>`:

- **Grid**: `(total_x_blocks, max_num_cosets)` — `total_x_blocks` = sum of per-AIR `ceil(num_x * skip_domain / blockDim.x)` across all batched AIRs
- **Block**: `(MAX_THREADS=128, 1, 1)` — uniform for all blocks
- **Template**: `GLOBAL=false` only (local intermediates). `NEEDS_SHMEM` dispatched at runtime based on `skip_domain > WARP_SIZE`
- **Additional kernel parameters** (beyond the per-AIR metadata):
  - `const ZerocheckBatchMeta *air_metas` — [num_airs] on device
  - `const uint32_t *air_for_block` — [total_x_blocks] maps each blockIdx.x to an AIR index
  - `const uint32_t *segment_offsets` — [num_airs + 1] cumulative x_blocks per AIR
  - `const FpExt *d_lambda_pows` — shared across all AIRs
  - `size_t lambda_len` — shared
  - `uint32_t skip_domain` — shared (constant across all AIRs)
  - `uint32_t d` — output stride = `max_num_cosets * skip_domain`

**Kernel body (derived from existing coset_parallel kernel at lines 335-467):**

1. Load AIR identity: `air_idx = air_for_block[blockIdx.x]`, `meta = air_metas[air_idx]`
2. Compute local block position: `local_block_x = blockIdx.x - segment_offsets[air_idx]`
3. Extract coset: `coset_idx = blockIdx.y`
4. **Bounds check**: if `coset_idx >= meta.num_cosets`, return early (pre-zeroed buffer handles the padding)
5. Compute thread indexing using `local_block_x` and `meta.num_x`:
   - `tidx = threadIdx.x + local_block_x * blockDim.x`
   - `ntt_idx = tidx & (skip_domain - 1)`
   - `x_int_base = tidx >> l_skip`
   - `air_x_blocks = div_ceil(meta.num_x * skip_domain, blockDim.x)`
   - `x_int_stride = (air_x_blocks * blockDim.x) >> l_skip`
6. Precompute coset values using `meta.g_shift`, `coset_idx` — identical to existing kernel lines 387-393
7. Set up `NttEvalContext<1>` using `meta.*` fields — identical to existing lines 414-426
8. Main loop over `x_int` — identical to existing lines 428-448, using `meta.*` for selectors/rules
9. **Zerofier division and in-block reduction** — identical to existing lines 450-466:
   - Compute `eval_point = g_coset * omega_skip_ntt` where `g_coset = pow(meta.g_shift, coset_idx + 1)` (per-AIR g_shift from metadata)
   - Compute `zerofier = exp_power_of_2(eval_point, l_skip) - Fp::one()`
   - Write `shared_sum[threadIdx.x] = sum * inv(zerofier)`, then block-reduce across lanes
   - The zerofier is never zero because `g_shift` is not in the skip domain and we use `(coset_idx + 1)` power
10. **Output**: `tmp_sums_buffer[blockIdx.x * d + coset_idx * skip_domain + ntt_idx] = tile_sum`
    - Uses global `blockIdx.x` (not local_block_x) so blocks from different AIRs write to non-overlapping regions

**File: `logup_round0.cu`**

Add `logup_r0_ntt_eval_interactions_coset_parallel_batched_kernel<NEEDS_SHMEM>`:

Same structure as zerocheck batched kernel, with these differences:
- Uses `FracExt` output type (shared_sum is `FracExt *`)
- **Identity-coset special case** (existing logup_round0.cu:361-381): when `coset_idx == 0`, the logup kernel uses `g_coset = Fp::one()` (identity shift), `eval_point = omega_skip_ntt`, and `omega_shift = Fp::one()`. For `coset_idx > 0`, it uses `g_coset = pow(meta.g_shift, coset_idx)` and standard shift computation. This branching must be preserved exactly as in the existing coset-parallel kernel.
- Calls `acc_interactions<1, NEEDS_SHMEM, false>` with `is_identity_coset = (coset_idx == 0)` as `skip_ntt` runtime flag — identical to existing lines 425-437
- Per-AIR data includes `meta.numer_weights`, `meta.denom_weights`, `meta.denom_sum_init` (note: logup does NOT use `d_used_nodes` — it indexes weights by rule index, not constraint index, so `LogupBatchMeta` correctly omits this field)
- No zerofier division in the reduction (logup doesn't divide by zerofier, unlike zerocheck)
- Shared memory: `sizeof(FracExt) * blockDim.x + (NEEDS_SHMEM ? sizeof(Fp) * blockDim.x : 0)`
- Output: `tmp_sums_buffer[blockIdx.x * d + coset_idx * skip_domain + ntt_idx] = tile_sum` where `tile_sum` is `FracExt`

### Change 3: CUDA — Batched launcher functions (extern "C")

**File: `zerocheck_round0.cu`**

Add launcher that combines evaluation kernel + batched reduction:

```cpp
extern "C" int _zerocheck_ntt_eval_constraints_batched(
    FpExt *tmp_sums_buffer,                  // pre-zeroed, [total_x_blocks][d]
    FpExt *output,                           // [num_airs][d]
    const ZerocheckBatchMeta *metas,         // [num_airs], device
    const uint32_t *air_for_block,           // [total_x_blocks], device
    const uint32_t *segment_offsets,         // [num_airs + 1], device
    const FpExt *d_lambda_pows,              // shared
    size_t lambda_len,
    uint32_t num_airs,
    uint32_t total_x_blocks,
    uint32_t max_num_cosets,
    uint32_t skip_domain
);
```

Logic:
1. `d = max_num_cosets * skip_domain`
2. `grid = dim3(total_x_blocks, max_num_cosets)`, `block = dim3(MAX_THREADS)`
3. Compute shmem: `sizeof(FpExt) * block.x + (needs_shmem ? sizeof(Fp) * block.x : 0)`
4. Dispatch on `needs_shmem = skip_domain > WARP_SIZE`: launch batched evaluation kernel
5. Launch `batched_final_reduce_block_sums` (from sumcheck.cuh:261-291):
   - `reduce_grid = dim3(num_airs, d)`, `reduce_block` from `kernel_launch_params(max_blocks_per_segment)`
     where `max_blocks_per_segment` = max over all AIRs of their x_blocks count
   - `reduce_shmem = div_ceil(reduce_block.x, WARP_SIZE) * sizeof(FpExt)`
   - Parameters: `tmp_sums_buffer, output, segment_offsets, d`

**File: `logup_round0.cu`**

Add analogous launcher:

```cpp
extern "C" int _logup_bary_eval_interactions_round0_batched(
    FracExt *tmp_sums_buffer,                // pre-zeroed, [total_x_blocks][d_frac]
    FracExt *output,                         // [num_airs][d_frac]
    const LogupBatchMeta *metas,             // [num_airs], device
    const uint32_t *air_for_block,           // [total_x_blocks], device
    const uint32_t *segment_offsets,         // [num_airs + 1], device
    uint32_t num_airs,
    uint32_t total_x_blocks,
    uint32_t max_num_cosets,
    uint32_t skip_domain
);
```

Same pattern. For the reduction, reinterpret `FracExt` as `FpExt` (2 components):
- `d_fpext = 2 * max_num_cosets * skip_domain`
- Launch `batched_final_reduce_block_sums` with `dim3(num_airs, d_fpext)` and `d = d_fpext`

### Change 4: Rust FFI declarations

**File: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`**

Add to the `extern "C"` block:

```rust
fn _zerocheck_ntt_eval_constraints_batched(
    tmp_sums_buffer: *mut EF,
    output: *mut EF,
    metas: *const std::ffi::c_void,    // ZerocheckBatchMeta*
    air_for_block: *const u32,
    segment_offsets: *const u32,
    d_lambda_pows: *const EF,
    lambda_len: usize,
    num_airs: u32,
    total_x_blocks: u32,
    max_num_cosets: u32,
    skip_domain: u32,
) -> i32;

fn _logup_bary_eval_interactions_round0_batched(
    tmp_sums_buffer: *mut Frac<EF>,
    output: *mut Frac<EF>,
    metas: *const std::ffi::c_void,    // LogupBatchMeta*
    air_for_block: *const u32,
    segment_offsets: *const u32,
    num_airs: u32,
    total_x_blocks: u32,
    max_num_cosets: u32,
    skip_domain: u32,
) -> i32;
```

Add safe Rust wrapper functions following the existing pattern (see `zerocheck_ntt_eval_constraints` at lines 924-967).

### Change 5: Rust metadata structs

**File: `crates/cuda-backend/src/logup_zerocheck/round0.rs`**

Add `#[repr(C)]` structs matching the CUDA definitions exactly (same field order and types):

```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ZerocheckBatchMeta {
    pub selectors_cube: *const F,
    pub preprocessed: *const F,
    pub main_parts: *const *const F,
    pub eq_cube: *const EF,
    pub public_values: *const F,
    pub d_rules: *const std::ffi::c_void,   // Rule*
    pub d_used_nodes: *const usize,
    pub rules_len: usize,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
    pub num_x: u32,
    pub height: u32,
    pub num_cosets: u32,
    pub g_shift: F,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LogupBatchMeta {
    pub selectors_cube: *const F,
    pub preprocessed: *const F,
    pub main_parts: *const *const F,
    pub eq_cube: *const EF,
    pub public_values: *const F,
    pub numer_weights: *const EF,
    pub denom_weights: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub denom_sum_init: EF,
    pub buffer_size: u32,
    pub num_x: u32,
    pub height: u32,
    pub num_cosets: u32,
    pub g_shift: F,
}
```

Both structs must be `unsafe impl Send` (contains raw pointers to device memory).

### Change 6: Rust orchestration

**File: `crates/cuda-backend/src/logup_zerocheck/mod.rs`**

Modify `sumcheck_uni_round0_polys` Phase 1 (lines 744-841). Replace the single per-AIR loop with three sub-phases:

#### Sub-phase A: Classify AIRs and collect metadata

Iterate all AIR instances (same loop structure as current lines 750-757). For each AIR:

1. Compute `num_x = 1 << n_lift`, `skip_domain = 1 << l_skip`
2. Check `is_coset_parallel = (num_x * skip_domain) < 32768`
3. Build `d_main_parts` (same as current lines 781-786). Store in a keep-alive vec.

**For zerocheck:**
4. Check `num_cosets_zc > 0` and has constraints (same as `evaluate_round0_constraints_gpu` early-return checks)
5. Read `buffer_size = pk.other_data.zerocheck_round0.inner.buffer_size`
6. Check `batch_eligible_zc = is_coset_parallel && buffer_size <= 16`
7. If eligible: build `ZerocheckBatchMeta` from existing device data (pk fields, selectors, eq_cube, etc.)
   - Compute `x_blocks = ceil(num_x * skip_domain / 128)`
   - Add to `zc_batch_metas` vec, record `x_blocks`
8. If not eligible: mark for individual launch

**For logup:**
9. Check `!eq_3bs.is_empty()` (has interactions)
10. Perform the per-AIR CPU work: build symbolic DAG, compute weights, compile rules (lines 161-207 of round0.rs)
11. Upload `d_rules`, `d_numer_weights`, `d_denom_weights` to device. Store in keep-alive vecs.
12. Read `buffer_size` from compiled rules
13. Check `batch_eligible_logup = is_coset_parallel && buffer_size <= 16`
14. If eligible: build `LogupBatchMeta` with device pointers from step 11
    - Compute `x_blocks`, add to `logup_batch_metas` vec
15. If not eligible: mark for individual launch

Compute for each batch:
- `total_x_blocks` = sum of per-AIR x_blocks
- `max_num_cosets` = max of per-AIR num_cosets
- `segment_offsets` = prefix sum of per-AIR x_blocks (length num_airs + 1)
- `air_for_block` = flat mapping, e.g., if AIR 0 has 3 blocks and AIR 1 has 2 blocks: `[0, 0, 0, 1, 1]`

#### Sub-phase B: Batched launches

**Zerocheck batch** (if `zc_batch_metas` non-empty):
1. Upload `zc_batch_metas`, `air_for_block`, `segment_offsets` to device
2. Allocate `d_zc_tmp` with capacity `total_x_blocks * d` (where `d = max_num_cosets * skip_domain`), zero-initialize via `fill_zero()`
3. Allocate `d_zc_output` with capacity `num_airs * d`
4. Call `zerocheck_ntt_eval_constraints_batched(d_zc_tmp, d_zc_output, ...)`

**Logup batch** (if `logup_batch_metas` non-empty):
1. Upload `logup_batch_metas`, mapping arrays to device
2. Allocate `d_logup_tmp` with capacity `total_x_blocks * d_logup` (where `d_logup = max_num_cosets_logup * skip_domain`), zero-initialize
3. Allocate `d_logup_output` with capacity `num_airs_logup * d_logup`
4. Call `logup_bary_eval_interactions_round0_batched(d_logup_tmp, d_logup_output, ...)`

#### Sub-phase C: Individual launches

For non-batch-eligible AIRs, call existing `evaluate_round0_constraints_gpu` and/or `evaluate_round0_interactions_gpu` as before.

#### Phase 2 modifications

After `current_stream_sync()`:
1. D2H the batched output buffers (`d_zc_output.to_host()`, `d_logup_output.to_host()`)
2. Drop keep-alive vecs (per-AIR logup DeviceBuffers, d_main_parts)
3. For each batched AIR, extract its slice from the flat output:
   - Zerocheck: `&zc_host_output[batch_idx * d .. batch_idx * d + actual_num_cosets * skip_domain]`
   - Logup: `&logup_host_output[batch_idx * d_logup .. batch_idx * d_logup + actual_num_cosets_logup * skip_domain]`
4. Feed into existing polynomial construction (lines 850-923) — **this code is unchanged**
5. Individual-launch AIR results handled via existing `Round0Pending` path

### Design decisions

1. **2D grid vs 3D grid**: Using 2D grid `(total_x_blocks, max_num_cosets)` with flat `air_for_block` mapping. A 3D grid `(max_x_blocks, max_num_cosets, num_airs)` would waste blocks when AIRs have varying x_block counts (e.g., 256 max_x_blocks but most AIRs have 1-2 → 98%+ blocks return early).

2. **GLOBAL=false only**: Only batching AIRs with `buffer_size <= BUFFER_THRESHOLD (16)`. The GLOBAL=true path requires per-AIR global intermediate buffers whose aggregate size could be significant (~73 MB for 500 AIRs). GLOBAL=true AIRs continue with individual launches. This captures the majority of small AIRs (large DAGs requiring buffer_size > 16 are uncommon for small traces).

3. **Uniform blockDim.x = 128**: All batched blocks use MAX_THREADS. AIRs with fewer threads (small num_x) have idle threads that contribute zero in the reduction. The cost is negligible — these idle threads' shared_sum writes are zero and don't affect correctness.

4. **Pre-zeroed tmp_sums_buffer**: Blocks with `coset_idx >= meta.num_cosets` return early without writing. Zero-initialization ensures the reduction reads zeros for padding positions. Cost: < 0.1ms for ~200 KB.

5. **Reuse `batched_final_reduce_block_sums`**: Available in `sumcheck.cuh:261-291`. Already supports segment offsets for multi-segment reduction. Already actively used by the MLE rounds kernels (batch_mle.cu, batch_mle_monomial.cu) — battle-tested.

6. **Separate zerocheck and logup batches**: Different kernel logic (constraint accumulation vs interaction numerator/denominator), different output types (`FpExt` vs `FracExt`), and different per-AIR data preparation (pre-built rules vs per-AIR DAG compilation).

7. **Per-AIR logup CPU work unchanged**: DAG compilation, weight computation, and per-AIR H2D uploads still happen sequentially per AIR. The optimization targets GPU execution time (sequential kernel launches → concurrent), not CPU-side preparation. Note: if the batched kernel reduces GPU time by 3-5x, the sequential CPU preparation (~396 AIRs worth of DAG building and weight computation) could become visible in the profile. If so, rayon-parallelizing the logup CPU preparation is a natural follow-up.

8. **Sub-batching by `num_cosets` as follow-up**: The current plan uses `max_num_cosets` for the grid's y-dimension, causing AIRs with fewer cosets to waste coset-dimension blocks (early-return). If profiling shows this waste is significant (e.g., `max_num_cosets = 4` but most AIRs have `num_cosets = 2`), grouping AIRs by identical `num_cosets` into separate sub-batches would eliminate the padding. This adds at most 2-3 extra launches (one per distinct `num_cosets` value) while simplifying the kernel (no per-AIR coset bounds check). Deferring this to a follow-up since the waste factor is at most 2-3x on the coset dimension, which is small relative to the 30-100x improvement from batching the x-block dimension.

## Invariants

1. **Per-AIR output values are bit-identical** to the current non-batched code. The batched kernel executes the same arithmetic per (AIR, coset, x_int, ntt_idx) as the existing coset-parallel kernel. Only the launch geometry and reduction strategy differ.

2. **Non-coset-parallel AIRs are completely unaffected**. They continue using the existing per-AIR lockstep kernel path.

3. **Phase 2 polynomial construction is unchanged**. It receives the same per-AIR `q_evals` / `s_evals` arrays regardless of whether they came from batched or individual launches.

4. **Memory safety**: All per-AIR DeviceBuffers referenced by metadata pointers are kept alive until after stream sync. The VPMM's stream-ordered deallocation ensures device memory is valid during kernel execution.

5. **COSET_PARALLEL_THRESHOLD is unchanged** (32768). The same AIRs that currently use coset-parallel mode will be batched.

6. **BUFFER_THRESHOLD is unchanged** (16). AIRs with buffer_size > 16 continue using individual launches with global intermediates.

## Measurement Plan

### Correctness verification

1. Run the full test suite:
   ```bash
   cargo nextest run -p openvm-cuda-backend --test-threads=4
   ```
   All 94 tests must pass.

2. Run pairing benchmark for APC {0, 100, 300} using the powdr repo:
   ```bash
   openvm-riscv/scripts/run_pairing.sh
   ```
   Proof verification must succeed for all configurations.

### Performance measurement

1. Collect metrics for APC {0, 100, 300}:
   - Before: use latest `autoopt-results/2026-04-10-2145-batch-degenerate-stacked-reduction/results/` as baseline
   - After: run pairing for each APC config

2. Analyze with `spec.py`:
   ```
   python spec.py <metrics_json> <experiment_name>
   ```
   Compare "Round 0" and "STARK (excl. trace)" across APC configurations.

3. Use Nsight Systems to verify GPU utilization improvement:
   ```bash
   nsys profile --stats=true <command>
   ```
   Check that the ~935 individual `coset_parallel` kernel launches are replaced by 2 batched launches + 2 batched reductions.

### Diagnostic: batch coverage

Add a temporary `debug!` log line counting how many AIRs are batch-eligible vs individual for zerocheck and logup at each APC config. Expected: >90% of coset-parallel AIRs are batch-eligible (GLOBAL=false). If <70% are eligible, consider adding GLOBAL=true batching as a follow-up.

## Rollback Criteria

1. **Any test failure**: Revert immediately. The batched kernel must produce bit-identical results.
2. **Round 0 APC 300 > 500ms**: Less than 12% improvement doesn't justify the added code complexity. Investigate whether GLOBAL=true AIRs dominate and consider adding that path.
3. **STARK excl trace APC 300 regression for APC 0**: The optimization should be neutral or positive for APC 0 (few small AIRs). Any regression indicates a bug.
4. **Peak GPU memory increase > 100 MB**: The batched buffers should total < 5 MB. Significant memory increase indicates a sizing error.
