# Plan: Sub-batch Round 0 Coset-Parallel Kernel Launches

## Goal

Batch multiple small coset-parallel AIR instances into single GPU kernel launches in Round 0,
using sub-batching with a memory budget to avoid the all-or-nothing limitation of the previous
attempt (commit 74134b4e, reverted at 61dc6616). This directly improves SM utilization by
packing many more thread blocks per kernel call — instead of ~1,470 serial kernel launches
each using 3-11% of 128 SMs, we get ~30-50 batched launches that saturate the GPU.

### Critical correction to the task scope

The task proposes batching only `GLOBAL=false` instances (buffer_size <= 16) and leaving
`GLOBAL=true` instances (buffer_size > 16) as individual launches. However, the nsys profiling
data shows the workload is dominated by `GLOBAL=true` coset-parallel kernels:

| Kernel variant | Instances | GPU time | % of coset-parallel |
|---|---|---|---|
| zerocheck `<GLOBAL=true, NEEDS_SHMEM=false>` | 539 | 221ms | 47% |
| logup `<GLOBAL=true, NEEDS_SHMEM=false>` | 396 | 196ms | 42% |
| zerocheck `<GLOBAL=false, NEEDS_SHMEM=false>` | 196 | 20ms | 4% |
| logup `<GLOBAL=false, NEEDS_SHMEM=false>` | 339 | 29ms | 6% |
| **Total** | **1,470** | **466ms** | **100%** |

Batching only `GLOBAL=false` instances saves at most ~49ms — far short of the task's expected
270-370ms improvement. To achieve the target, `GLOBAL=true` instances (417ms) must also be
batched. This plan includes both.

### Why the previous all-or-nothing approach was reverted

The previous attempt (74134b4e) was reverted not because of a memory budget failure, but
because it could not be measured — a pre-existing GPU OOM bug in
`gkr_input::log_gkr_input_evals` prevented the pairing benchmark from completing. The budget
check (128MB) did NOT disable batching for the pairing benchmark (skip_domain=32, estimated
temp ≈ 2.4MB). The optimization was correct (94/94 tests pass) but unmeasurable.

Sub-batching is still valuable for robustness with larger `skip_domain` values, and we extend
scope to include `GLOBAL=true` instances which the reverted commit did not handle.

## Current Code Path

Entry point: `sumcheck_uni_round0_polys()` in
`crates/cuda-backend/src/logup_zerocheck/mod.rs:599-928`.

```
sumcheck_uni_round0_polys()
  ├─ Lines 610-720: Setup (lambda_pows, eq_3b weights, eq_xi trees, selectors)
  ├─ Phase 1 (lines 750-841): Serial per-AIR kernel launch
  │   └─ For each of ~623 AIRs (APC 300):
  │       ├─ evaluate_round0_constraints_gpu() → zerocheck kernel
  │       │   (round0.rs:31-124 → FFI: cuda/logup_zerocheck.rs:924-967)
  │       │   └─ CUDA dispatch (zerocheck_round0.cu:667-716):
  │       │       ├─ coset_parallel if (num_x * skip_domain < 32768)
  │       │       │   → grid (x_blocks, num_cosets), ~410μs avg
  │       │       └─ lockstep otherwise → grid (x_blocks, 1)
  │       │
  │       └─ evaluate_round0_interactions_gpu() → logup kernel
  │           (round0.rs:132-275 → FFI: cuda/logup_zerocheck.rs:977-1020)
  │           └─ CUDA dispatch (logup_round0.cu:658-810):
  │               ├─ coset_parallel if (num_x * skip_domain < 32768)
  │               │   → grid (x_blocks, num_cosets), ~494μs avg
  │               └─ lockstep otherwise
  │
  ├─ current_stream_sync() (line 844)
  │
  └─ Phase 2 (lines 849-924): D2H + polynomial construction
      └─ For each pending: to_host(), transpose, IDFT, build sp_0
```

Key constants (defined in both .cu files):
- `BUFFER_THRESHOLD = 16` — GLOBAL=true above, GLOBAL=false at or below
- `COSET_PARALLEL_THRESHOLD = 32768` — coset-parallel below, lockstep at or above
- `MAX_THREADS = 128` — block size for coset-parallel kernels

The `batched_final_reduce_block_sums` kernel already exists in `cuda/include/sumcheck.cuh:261`
and is used by batch_mle.cu and batch_mle_monomial.cu. We reuse it for round 0 batching.

## Changes

### Change 1: CUDA — Re-introduce GLOBAL=false batched kernels from 74134b4e

**Files:**
- `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`
- `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

**What:** Cherry-pick the following from commit 74134b4e:

1. `ZerocheckBatchMeta` struct (zerocheck_round0.cu, after line 466)
2. `zerocheck_ntt_evaluate_constraints_coset_parallel_batched_kernel<NEEDS_SHMEM>` kernel
3. `_zerocheck_ntt_eval_constraints_batched` extern "C" launcher
4. `LogupBatchMeta` struct (logup_round0.cu, after line 459)
5. `logup_r0_ntt_eval_interactions_coset_parallel_batched_kernel<NEEDS_SHMEM>` kernel
6. `_logup_bary_eval_interactions_round0_batched` extern "C" launcher

These are exactly as in 74134b4e. They handle `GLOBAL=false` instances only (local register
buffers, `buffer_stride=1`). The grid is `(total_x_blocks, max_num_cosets)` with per-AIR
lookup via `air_for_block[blockIdx.x]`. The reduction uses
`sumcheck::batched_final_reduce_block_sums`.

**Why:** Proven correct (94/94 tests). Batching 535 GLOBAL=false instances eliminates ~49ms
GPU time and ~535 kernel launch calls.

### Change 2: CUDA — Add GLOBAL=true batched kernels (new)

**Files:**
- `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`
- `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

**What:** Add two new CUDA resources per file:

1. **Extended metadata structs** (`ZerocheckBatchMetaGlobal`, `LogupBatchMetaGlobal`):
   Same as the GLOBAL=false metadata but with one additional field:
   ```cpp
   Fp *intermediates;  // pre-allocated device buffer base for this AIR's intermediates
   ```

2. **Batched kernel** `zerocheck_ntt_evaluate_constraints_coset_parallel_batched_global_kernel<NEEDS_SHMEM>`:
   Structurally identical to the GLOBAL=false batched kernel, with the intermediates setup
   matching the non-batched coset-parallel kernel's strided layout
   (zerocheck_round0.cu:399-403):

   ```cpp
   // Non-batched GLOBAL=true layout (for reference):
   //   global_tidx = coset_idx * gridDim.x * blockDim.x + tidx;
   //   inter_buffer = d_intermediates + global_tidx;
   //   buffer_stride = gridDim.x * gridDim.y * blockDim.x;
   //
   // Batched equivalent (per-AIR gridDim replaced by per-AIR block counts):
   uint32_t air_x_blocks = segment_offsets[air_idx + 1] - segment_offsets[air_idx];
   uint32_t air_stride = air_x_blocks * meta.num_cosets * blockDim.x;
   uint32_t global_tidx = coset_idx * air_x_blocks * blockDim.x + tidx;
   Fp *inter_buffer = meta.intermediates + global_tidx;
   uint32_t buffer_stride = air_stride;
   ```

   The intermediates layout is `[buffer_size][num_cosets * threads_per_coset]` — a "planes"
   layout where each DAG intermediate index (z_index) selects a plane of `air_stride`
   elements. The access pattern in `acc_constraints` (eval_config.cuh:121) is
   `inter_buffer[z_index * buffer_stride + c]` where `c` is the coset index (0 for
   coset-parallel mode with NUM_COSETS=1). This must match the non-batched kernel exactly.

   Key difference from GLOBAL=false: `Fp local_buffer[1]` (dummy, not used) instead of
   `Fp local_buffer[BUFFER_THRESHOLD]`.

   Same for logup: `logup_r0_ntt_eval_interactions_coset_parallel_batched_global_kernel<NEEDS_SHMEM>`.

3. **Launcher functions** `_zerocheck_ntt_eval_constraints_batched_global` and
   `_logup_bary_eval_interactions_round0_batched_global`:
   Same signature as the GLOBAL=false launchers. The intermediates pointers are embedded in
   the per-AIR metadata struct, so no additional launcher parameter is needed.

**Why:** GLOBAL=true instances account for 89% (417ms) of coset-parallel GPU time. Without
batching these, the optimization cannot achieve the target improvement. The kernel modification
is minimal — the same `acc_constraints` / `acc_interactions` functions are reused, only the
intermediates setup differs.

**Per-AIR intermediates size:** For each GLOBAL=true AIR in a sub-batch:
```
intermed_elems = buffer_size * air_x_blocks * meta.num_cosets * blockDim.x
intermed_bytes = intermed_elems * sizeof(Fp)
```
Typical values at APC 300: buffer_size ~20-30, air_x_blocks=1, num_cosets=3-4, blockDim.x=128.
Per AIR: ~30 * 1 * 4 * 128 * 4 bytes = ~61 KB. For 539 AIRs: ~32 MB.

### Change 3: Rust — Metadata structs and constants

**File:** `crates/cuda-backend/src/logup_zerocheck/round0.rs`

**What:** Re-introduce from 74134b4e:
- `BUFFER_THRESHOLD`, `COSET_PARALLEL_THRESHOLD`, `MAX_THREADS_ROUND0` constants
- `ZerocheckBatchMeta`, `LogupBatchMeta` `#[repr(C)]` structs with `unsafe impl Send`

Add new:
- `ZerocheckBatchMetaGlobal`, `LogupBatchMetaGlobal` `#[repr(C)]` structs — same as the
  non-global variants but with `pub intermediates: *mut F` field. Example:
  ```rust
  #[repr(C)]
  #[derive(Clone, Copy)]
  pub(crate) struct ZerocheckBatchMetaGlobal {
      // ... all fields from ZerocheckBatchMeta ...
      pub intermediates: *mut F,  // base pointer into shared intermediates buffer
  }
  unsafe impl Send for ZerocheckBatchMetaGlobal {}
  ```

**Why:** These must match the CUDA-side struct layout exactly (`#[repr(C)]`). Separating
GLOBAL and non-GLOBAL metadata structs avoids a wasted pointer field in non-GLOBAL metadata
and keeps the CUDA kernels simple (no runtime GLOBAL check). The `intermediates` pointer is
set on the Rust side during Sub-phase C after allocating the shared intermediates buffer.

### Change 4: Rust — FFI wrappers

**File:** `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

**What:** Re-introduce from 74134b4e:
- `extern "C"` declarations for `_zerocheck_ntt_eval_constraints_batched` and
  `_logup_bary_eval_interactions_round0_batched`
- Safe wrapper functions `zerocheck_ntt_eval_constraints_batched` and
  `logup_bary_eval_interactions_round0_batched`

Add new:
- Same for the GLOBAL=true variants: `_zerocheck_ntt_eval_constraints_batched_global`,
  `_logup_bary_eval_interactions_round0_batched_global`, and their safe wrappers

**Why:** FFI boundary between Rust orchestration and CUDA kernels.

### Change 5: Rust — Sub-batch orchestration in `sumcheck_uni_round0_polys`

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`, function
`sumcheck_uni_round0_polys` (lines 599-928)

This is the main logic change. Replace the per-AIR loop in Phase 1 with a three-sub-phase
structure:

**Sub-phase A: Classify AIRs (lines 750-841 replacement)**

Iterate over all AIRs. For each, determine:
1. `is_coset_parallel = (num_x * skip_domain) < COSET_PARALLEL_THRESHOLD`
2. `has_constraints` / `has_interactions` (same as current)
3. `is_global_zc = zc_buffer_size > BUFFER_THRESHOLD`
4. `is_global_logup = logup_buffer_size > BUFFER_THRESHOLD`

Classify into categories:
- **Batch-eligible GLOBAL=false zerocheck**: `is_coset_parallel && has_constraints && !is_global_zc`
- **Batch-eligible GLOBAL=true zerocheck**: `is_coset_parallel && has_constraints && is_global_zc`
- **Batch-eligible GLOBAL=false logup**: `is_coset_parallel && has_interactions && !is_global_logup`
- **Batch-eligible GLOBAL=true logup**: `is_coset_parallel && has_interactions && is_global_logup`
- **Individual**: `!is_coset_parallel` (lockstep) — launch immediately as today

For batch-eligible AIRs, collect metadata into vectors (same structure as 74134b4e Sub-phase A,
extended with GLOBAL=true metadata). For logup, compute per-AIR DAG, weights, and d_rules
inline (same as 74134b4e).

Store per-AIR x_blocks for grouping: `x_blocks = (num_x * skip_domain).div_ceil(block_x)`.

For GLOBAL=true AIRs, also record the intermediates size needed per AIR:
- `intermed_elems = buffer_size * x_blocks * num_cosets * block_x`
  (matching the strided layout: `[buffer_size][num_cosets * threads_per_coset]`)
- Intermediates buffer is allocated per sub-batch in Sub-phase C (not here)

For batch-eligible logup AIRs (both GLOBAL=true and GLOBAL=false), hoist the per-AIR DAG
compilation and weight computation that currently lives inside
`evaluate_round0_interactions_gpu` (round0.rs:161-207). Build the DAG, compute
`numer_weights`, `denom_weights`, `denom_sum_init`, encode rules, and H2D transfer the device
buffers (`d_rules`, `d_numer_weights`, `d_denom_weights`). Store all device buffers in
keep-alive vectors (same pattern as 74134b4e, lines 800-870 in the reverted mod.rs diff).

**Sub-phase B: Form sub-batches**

For each of the 4 batch categories (zerocheck/logup x GLOBAL/non-GLOBAL):

```rust
const PER_BATCH_MEM_BUDGET: usize = 64 * 1024 * 1024; // 64 MB

let d = max_num_cosets * skip_domain;
let output_elem_size = size_of::<EF>(); // or size_of::<Frac<EF>>() for logup

let mut sub_batches = vec![];
let mut current_batch = vec![];
let mut current_total_x_blocks: u32 = 0;
let mut current_intermed_elems: usize = 0; // only for GLOBAL=true categories

for (idx, meta, x_blocks, intermed_elems) in eligible_airs {
    // tmp_sums_buffer: total_x_blocks * d * sizeof(output_elem)
    let projected_tmp = (current_total_x_blocks + x_blocks) as usize * d * output_elem_size;
    // intermediates: sum of per-AIR intermed_elems * sizeof(Fp) (GLOBAL=true only, 0 otherwise)
    let projected_intermed = (current_intermed_elems + intermed_elems) * size_of::<F>();
    let projected_total = projected_tmp + projected_intermed;

    if projected_total > PER_BATCH_MEM_BUDGET && !current_batch.is_empty() {
        sub_batches.push(current_batch);
        current_batch = vec![];
        current_total_x_blocks = 0;
        current_intermed_elems = 0;
    }
    current_batch.push((idx, meta, x_blocks, intermed_elems));
    current_total_x_blocks += x_blocks;
    current_intermed_elems += intermed_elems;
}
if !current_batch.is_empty() {
    sub_batches.push(current_batch);
}
```

The budget includes both `tmp_sums_buffer` and `intermediates` memory. For GLOBAL=false
categories, `intermed_elems` is always 0, so the budget degenerates to the tmp-only formula.

For the pairing benchmark at APC 300 (skip_domain=32, d≈128):
- GLOBAL=true zerocheck (539 AIRs): total_x_blocks ≈ 539 (most have 1 block)
  - tmp per batch = 539 * 128 * 16 = 1.1 MB → 1 sub-batch
- GLOBAL=true logup (396 AIRs): tmp ≈ 0.8 MB → 1 sub-batch
- GLOBAL=false zerocheck (196 AIRs): tmp ≈ 0.4 MB → 1 sub-batch
- GLOBAL=false logup (339 AIRs): tmp ≈ 0.7 MB → 1 sub-batch
- **Total: ~4 batched launches** (plus ~30-40 lockstep individual launches)

For large skip_domain (e.g., skip_domain=2048, d=8192):
- tmp per 32 AIRs = 32 * 8192 * 16 = 4.2 MB → many more sub-batches
- Sub-batching ensures memory stays bounded

**Sub-phase C: Launch sub-batches**

For each sub-batch, reusing the pattern from 74134b4e:

1. Build `air_for_block` and `segment_offsets` arrays (same `build_batch_arrays` helper)
2. H2D transfer metadata, air_for_block, segment_offsets (small, <10μs each)
3. Allocate or reuse temp buffer (`fill_zero()` required)
4. Launch batched evaluation kernel (GLOBAL=false or GLOBAL=true variant)
5. Launch `batched_final_reduce_block_sums` reduction kernel
   - Output into global output buffer at `batch_air_offset * d`
6. Drop temp buffer (freed for next sub-batch)

For GLOBAL=true sub-batches, additionally:
7. Allocate shared intermediates buffer sized as
   `sum(buffer_size_i * x_blocks_i * num_cosets_i * blockDim_x) * sizeof(Fp)`.
   Compute per-AIR offsets as a cumulative sum:
   ```rust
   let mut intermed_offset = 0usize;
   for air in &sub_batch {
       air.meta.intermediates = shared_intermed_ptr.add(intermed_offset);
       intermed_offset += air.buffer_size * air.x_blocks * air.num_cosets * block_x;
   }
   ```
8. H2D transfer the updated metadata (with intermediates pointers set)
9. Push the intermediates DeviceBuffer into a keep-alive vector — it must NOT be dropped until
   after `current_stream_sync()` because the kernel launch is asynchronous

The output buffer is pre-allocated for all batched AIRs in each category:
- `d_zc_global_output: DeviceBuffer<EF>` of size `num_zc_global_batched * d`
- `d_zc_local_output: DeviceBuffer<EF>` of size `num_zc_local_batched * d`
- Same for logup with `Frac<EF>`

**Temp buffer reuse optimization:** Pre-allocate one temp buffer sized for the largest
sub-batch. Reuse across sub-batches with `fill_zero()` between launches. This avoids
repeated `DeviceBuffer::with_capacity` calls. Cost: one `fill_zero()` per sub-batch (~0.1ms).

**Phase 2 modification:** Extend the `Round0Pending` struct with batch category info:

```rust
struct Round0Pending<F, EF> {
    // ... existing fields ...
    zc_batch_info: Option<(BatchCategory, usize)>,  // (category, index_in_category)
    logup_batch_info: Option<(BatchCategory, usize)>,
}

enum BatchCategory { GlobalTrue, GlobalFalse }
```

In Phase 2, after `current_stream_sync()`:
1. D2H all 4 output buffers (zc_global, zc_local, logup_global, logup_local)
2. For each pending AIR, extract the correct slice based on batch_info
3. Feed into existing polynomial construction (unchanged)

### Change 6: Keep-alive buffers

**File:** `crates/cuda-backend/src/logup_zerocheck/mod.rs`

For all batch-eligible AIRs, device buffers containing per-AIR data must remain alive until
`current_stream_sync()`:
- `d_main_parts_vec` — already handled in current code
- Logup per-AIR `d_rules`, `d_numer_weights`, `d_denom_weights` — keep-alive vectors as in
  74134b4e
- GLOBAL=true shared `intermediates` buffers — one per sub-batch, held in a Vec until sync

Drop all keep-alive buffers after `current_stream_sync()` and before Phase 2 D2H transfers.

## Invariants

1. **Correctness**: For every AIR, the batched kernel must produce the same output as the
   individual kernel for the same inputs. This follows from the batched kernel being
   structurally identical to the single-AIR kernel — same `acc_constraints`/`acc_interactions`
   logic, same NTT evaluation, same reduction. The only difference is thread block assignment.

2. **Output layout**: Batched output is `[num_airs_in_batch][d]` where
   `d = max_num_cosets * skip_domain`. For AIRs with `num_cosets < max_num_cosets`, the extra
   coset entries are zero (early-return blocks write nothing, temp buffer is zero-filled).
   Phase 2 reads only `num_cosets * skip_domain` elements per AIR, so padding is harmless.

3. **Memory safety**: All device pointers in metadata structs (selectors_cube, preprocessed,
   main_parts, eq_cube, d_rules, etc.) must remain valid until `current_stream_sync()`.
   This is guaranteed by the keep-alive vectors.

4. **GLOBAL=true intermediates**: The strided layout must exactly match the non-batched
   coset-parallel kernel (zerocheck_round0.cu:399-403). Specifically:
   - `global_tidx = coset_idx * air_x_blocks * blockDim.x + tidx`
   - `inter_buffer = meta.intermediates + global_tidx`
   - `buffer_stride = air_x_blocks * meta.num_cosets * blockDim.x`
   The access pattern in `acc_constraints` (eval_config.cuh:121) is
   `inter_buffer[z_index * buffer_stride + c]`. For coset-parallel (NUM_COSETS=1), c=0
   always, so each z_index selects a plane of `buffer_stride` elements.
   Per-AIR intermediates size: `buffer_size * air_x_blocks * num_cosets * blockDim.x * sizeof(Fp)`.
   The shared intermediates buffer must be at least the sum across all AIRs in the sub-batch.

5. **Intermediates lifetime**: All intermediates DeviceBuffers must remain alive (not dropped)
   until after `current_stream_sync()`. CUDA kernel launches are asynchronous — the kernel
   is still reading intermediates when the launch call returns. Dropping early is use-after-free.

6. **Lockstep fallback**: AIRs with `num_x * skip_domain >= COSET_PARALLEL_THRESHOLD` are
   NOT batched and continue to use individual lockstep kernel launches. This preserves
   correctness for large AIRs.

7. **Verifier unchanged**: No protocol changes. The polynomial values produced in Phase 2 are
   identical; only the GPU computation path differs.

## Measurement Plan

In the powdr repo:

```bash
# Before changes: capture baseline for this task
for apc in 0 100 300; do
  openvm-riscv/scripts/run_pairing.sh  # with APC=$apc
  cp results/pairing/apc${apc}/metrics.json \
     ../stark-backend/autoopt-results/2026-04-11-1145-subbatch-round0-coset-parallel/results/before_apc${apc}.json
done

# After changes: capture results
for apc in 0 100 300; do
  openvm-riscv/scripts/run_pairing.sh  # with APC=$apc
  cp results/pairing/apc${apc}/metrics.json \
     ../stark-backend/autoopt-results/2026-04-11-1145-subbatch-round0-coset-parallel/results/after_apc${apc}.json
done

# Combined metrics
python basic_metrics.py \
  autoopt-results/2026-04-11-1145-subbatch-round0-coset-parallel/results/after_apc*.json \
  > autoopt-results/2026-04-11-1145-subbatch-round0-coset-parallel/results/metrics_combined.json

# Analysis
python spec.py <combined_json> <experiment_name>
```

Run all tests:
```bash
cargo nextest run -p openvm-cuda-backend --test-threads=4
```

Nsight profiling (if needed for detailed kernel analysis):
```bash
nsys profile --trace=cuda --output=nsys_after \
  <pairing_benchmark_command>
nsys stats nsys_after.nsys-rep --report cuda_gpu_kern_sum
```

Key metrics to compare:
- Round 0 wall time (from metrics.json)
- STARK excl trace (from metrics.json)
- Number of kernel launches (from nsight)
- SM occupancy (from nsight)

## Rollback Criteria

Revert if any of these conditions hold after full implementation:

1. **No improvement**: Round 0 APC 300 >= 540ms (within 5% of current 569ms)
2. **Regression at APC 0**: STARK excl trace APC 0 increases by > 20ms
3. **Test failures**: Any of the 94 `openvm-cuda-backend` tests fail
4. **GPU OOM**: Batched temp/intermediates buffers cause out-of-memory on RTX 4090 (24GB)
5. **Proof verification failure**: Generated proofs fail verification for any APC config
