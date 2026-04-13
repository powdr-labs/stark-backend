# Plan: Batch Round 0 Descriptor Arrays

## Goal

Replace per-AIR Round 0 kernel launches (zerocheck constraint eval + logup interaction eval) with batched descriptor-array kernel launches, eliminating ~1500 kernel launches per segment at APC 300. This directly targets the worst-scaling component: Round 0 takes 294ms at APC 300 vs 177ms at APC 0 (gets SLOWER with more APCs), with nsys profiling showing 76% of wall time is per-AIR overhead (buffer allocation, kernel launch, H2D/D2H copies, CPU DAG construction), not GPU compute.

## Current Code Path

### Entry Point
`crates/cuda-backend/src/logup_zerocheck/mod.rs` line 899: Round 0 evaluation within `prove_zerocheck_and_logup_gpu`.

### Main Loop (lines 899-966)
1. **Phase 1 (lines 899-926)**: Build `Vec<Round0AirWorkItem>` — one per trace. Pure CPU.
2. **Line 929**: Sort work items by descending height.
3. **Lines 939-943**: Choose `num_threads = 8` if ≥ 100 AIRs, else 1.
4. **Lines 944-966**: Multi-thread via `std::thread::scope`. Each thread processes its chunk of AIRs sequentially via `process_air_round0()`.

### Per-AIR Processing: `process_air_round0()` (lines 140-272)
For each AIR, sequentially:
1. `SymbolicConstraints::from()` — CPU: build constraint representation from proving key (line 144)
2. `main_parts.to_device()` — H2D: upload trace pointers (line 162)
3. `evaluate_round0_constraints_gpu()` — GPU: 1 eval kernel + 1 reduction kernel + 3 buffer allocations (intermediates, temp_sums, sp_evals) (lines 168-181, defined in round0.rs lines 31-124)
4. `sum_buffer.to_host_on_current_stream()` — D2H (line 184)
5. CPU: transpose + iDFT + polynomial construction (lines 185-208)
6. `evaluate_round0_interactions_gpu()` — CPU: build synthetic interaction DAG + compute weights (round0.rs lines 161-206). GPU: 1 eval kernel + 1 reduction kernel + 5+ buffer allocations (round0.rs lines 209-272)
7. `sum.to_host_on_current_stream()` — D2H (line 232)
8. CPU: unzip, transpose, iDFT, polynomial construction (lines 233-261)

### Bottleneck Analysis (APC 300, 310 AIRs/segment)
Per segment, the current code executes:
- ~620 kernel launches (310 AIRs × 2 evals, each with eval + reduction)
- ~2000+ DeviceBuffer allocations (via global MemoryManager mutex)
- ~620 H2D copies (main_parts, rules, weights per AIR)
- ~620 D2H copies (results per AIR)
- ~310 CPU DAG constructions for interaction eval (SymbolicDagBuilder + SymbolicRulesGpu::new)

Nsys data: Total Round 0 GPU kernel time = 564ms across all streams. With 8 threads, ideal wall time would be ~30ms. Actual = 294ms. Overhead = 264ms (76%).

### CUDA Kernels
- `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`: `zerocheck_ntt_evaluate_constraints_coset_parallel_kernel` — 539 instances, 238ms total GPU time. Block size determined by `MAX_THREADS = 128` (line 472) and `eval_config.cuh` `kernel_launch_params()`.
- `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`: `logup_r0_ntt_eval_interactions_coset_parallel_kernel` — 396+339=735 instances, 249ms total GPU time. Block size determined by `MAX_THREADS = 128` (line 466).

### Key Data Structures in Current Kernels
The coset-parallel kernels use `NttEvalContext<1>` (from `dag_entry.cuh`), which operates on base-field `Fp` data with NTT-domain-specific fields: `omega_shifts`, `ntt_buffer`, `ntt_idx`, `buffer_stride`. This is fundamentally different from `EvalCoreCtx` (from `eval_ctx.cuh`), which operates on extension-field `FpExt` data in the MLE batch kernels.

The intermediates buffer layout is `[buffer_size][total_threads_for_air]` where `total_threads_for_air = num_cosets * gridDim.x * blockDim.x`. Each thread accesses `d_intermediates + global_tidx` with stride `buffer_stride = total_threads_for_air` (see `zerocheck_round0.cu` lines 399-403).

### Existing Pattern to Follow
`crates/cuda-backend/src/logup_zerocheck/batch_mle.rs` lines 113-218: `ZerocheckMleBatchBuilder` — collects per-AIR `BlockCtx` + `ZerocheckCtx` on CPU, uploads once, launches one batched kernel for all AIRs. Uses `air_block_offsets` for the reduction phase.

## Changes

### Change 1: Cache interaction DAG structure at keygen

**File**: `crates/cuda-backend/src/pkey.rs`

Add a new struct to `AirDataGpu`:
```rust
pub struct Round0InteractionRules {
    /// Encoded interaction DAG rules, pre-computed and uploaded at keygen.
    pub d_rules: DeviceBuffer<u128>,
    pub buffer_size: u32,
    /// For each interaction: the rule_idx of its `count` expression.
    pub count_rule_idxs: Vec<usize>,
    /// For each interaction: the rule_idxs of its message field expressions.
    /// message_rule_idxs[interaction_idx][field_idx] = rule_idx
    pub message_rule_idxs: Vec<Vec<usize>>,
    /// Number of rules (= length of weight vectors).
    pub rules_len: usize,
}
```

Add field `pub interaction_round0: Option<Round0InteractionRules>` to `AirDataGpu`.

**File**: `crates/cuda-backend/src/pkey.rs` (in the keygen function that populates `AirDataGpu`)

For each AIR with interactions, run the same DAG construction currently in `evaluate_round0_interactions_gpu` (round0.rs lines 161-206):
1. Build `SymbolicDagBuilder` from `symbolic.interactions`
2. Create `SymbolicRulesGpu::new()`
3. Encode and upload rules to device
4. Extract `dag_idx_to_rule_idx` mapping for count and message expressions
5. Store as `Round0InteractionRules`

This moves ~310 DAG constructions per segment from prove-time to keygen-time (once per AIR, amortized).

### Change 2: Add NTT-compatible descriptor structs

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`

Add per-AIR descriptor struct using base-field `Fp` types (NOT `EvalCoreCtx` which uses `FpExt`):
```cuda
struct Round0ZcCtx {
    // NTT-domain base-field pointers (matching NttEvalContext<1> needs)
    const Fp *selectors_cube;        // [3][num_x]
    const Fp *preprocessed;          // preprocessed trace (nullable)
    const Fp *const *main_parts;     // array of base-field trace buffer pointers
    const FpExt *eq_cube;            // eq(x) evaluations [num_x]
    const FpExt *lambda_pows;        // lambda challenge powers
    const Fp *public_values;         // per-AIR public inputs
    const Rule *d_rules;             // constraint DAG rules (from pkey)
    const size_t *d_used_nodes;      // constraint node indices (from pkey)
    size_t rules_len;
    size_t used_nodes_len;
    size_t lambda_len;
    uint32_t buffer_size;            // DAG intermediate buffer size
    Fp *d_intermediates;             // offset into shared intermediates buffer
    uint32_t buffer_stride;          // total threads for this AIR = num_cosets * num_x_blocks * block_x
    uint32_t num_x;
    uint32_t height;
    Fp g_shift;                      // coset generator
};
```

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

Analogous `Round0LogupCtx` with additional fields:
```cuda
struct Round0LogupCtx {
    // Same base-field pointers as Round0ZcCtx
    const Fp *selectors_cube;
    const Fp *preprocessed;
    const Fp *const *main_parts;
    const FpExt *eq_cube;
    const Fp *public_values;
    const Rule *d_rules;             // interaction DAG rules (from keygen cache)
    size_t rules_len;
    uint32_t buffer_size;
    Fp *d_intermediates;
    uint32_t buffer_stride;
    const FpExt *numer_weights;      // per-rule numerator weights (uploaded per segment)
    const FpExt *denom_weights;      // per-rule denominator weights (uploaded per segment)
    FpExt denom_sum_init;            // scalar per AIR
    uint32_t num_x;
    uint32_t height;
    Fp g_shift;
};
```

Both files: reuse the existing `BlockCtx` struct from `batch_mle.cu` (or define a compatible `struct { uint32_t local_block_idx; uint32_t air_idx; }`).

### Change 3: Sub-batch by (skip_domain, num_cosets) groups

**Critical design decision**: Different AIRs have different `skip_domain` (= `1 << l_skip`, but `l_skip` is global so this is actually uniform) and different `num_cosets` (= `constraint_degree - 1` for zerocheck, `constraint_degree` for logup). Within a batch, all AIRs must share the same `num_cosets` to:
- Allow uniform block size (`skip_domain * x_per_block`)
- Allow uniform `d = num_cosets * skip_domain` for the reduction kernel
- Simplify intermediate buffer stride computation

**Grouping strategy**:
1. Partition small AIRs by `num_cosets_zc` for constraint eval groups, and by `num_cosets_logup` for interaction eval groups.
2. At APC 300, `num_cosets` is typically 2-4 (constraint degree 3-5). Expected ~3 groups.
3. Each group gets one batched kernel launch + one reduction kernel launch.
4. Total: ~6 constraint launches + ~6 interaction launches = ~12 launches vs current ~1500.

**Block size per group**: `block_x = min(skip_domain * floor(MAX_THREADS / skip_domain), skip_domain * max_x_for_group)` where `MAX_THREADS = 128` (from zerocheck_round0.cu line 472) and `max_x_for_group = max(num_x)` across AIRs in the group. For `skip_domain = 8`, `block_x = min(128, 8 * max_x) = 128` typically (since `max_x >= 16` for most groups). `x_per_block = block_x / skip_domain = 16`.

### Change 4: Write batched Round 0 zerocheck constraint eval kernel

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/zerocheck_round0.cu`

New kernel: `batched_zerocheck_r0_coset_parallel_kernel`
- **Grid**: `(total_blocks, 1)` — 1D grid
- **Block**: `block_x` threads (uniform within group, computed per Change 3)
- **Template parameter**: `<bool GLOBAL>` — always `true` for the batched kernel (use device memory for intermediates; no shared memory variant needed)
- **Per-block logic**:
  1. Load `BlockCtx` from `d_block_ctxs[blockIdx.x]` → get `air_idx`, `local_block_idx`
  2. Load `Round0ZcCtx` from `d_zc_ctxs[air_idx]`
  3. From group-uniform `num_cosets` and per-AIR `num_x`: compute `num_x_blocks = ceil(num_x / x_per_block)`, then `coset_idx = local_block_idx / num_x_blocks`, `x_block_idx = local_block_idx % num_x_blocks`
  4. `x_per_block = blockDim.x / skip_domain` (known from block size)
  5. Thread maps: `ntt_idx = threadIdx.x % skip_domain`, `x_offset = threadIdx.x / skip_domain + x_block_idx * x_per_block`
  6. Compute NTT evaluation context: `inter_buffer = ctx.d_intermediates + (coset_idx * num_x_blocks * blockDim.x + x_block_idx * blockDim.x + threadIdx.x)`. Stride = `ctx.buffer_stride`.
  7. Compute coset-specific values: `g_coset = g_shift^(2^coset_shift_exp)` for `coset_idx`. Precompute `is_first_mult`, `is_last_mult` for this coset.
  8. Main loop: iterate `x_int` from `x_offset` to `num_x` stepping by `x_per_block`. Call `acc_constraints<1, false>(...)` (GLOBAL=true, no shmem) per iteration. Accumulate into `sum`.
  9. Block-level reduction across `skip_domain` threads, write to `tmp_sums_buffer[block_offset + coset_idx * skip_domain + ntt_idx]` where `block_offset` is the per-AIR offset.

New launcher: `_batched_zerocheck_r0_eval_constraints`
- Parameters: `d_block_ctxs`, `d_zc_ctxs`, `total_blocks`, `block_x`, `num_cosets`, `skip_domain`, `d_air_offsets`, `d_tmp_sums`, `d_output`, `num_airs`
- Launches batched eval kernel + `final_reduce_block_sums` per-AIR reduction with uniform `d = num_cosets * skip_domain`

**Reduction kernel**: Use the existing `sumcheck::batched_final_reduce_block_sums` from `sumcheck.cuh` directly. Since all AIRs in a group share the same `num_cosets` and `skip_domain`, `d = num_cosets * skip_domain` is uniform. Grid: `(num_airs, d)`. The `air_block_offsets` array gives the block range per AIR.

### Change 5: Write batched Round 0 logup interaction eval kernel

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/logup_round0.cu`

New kernel: `batched_logup_r0_coset_parallel_kernel`
- Same 1D grid structure as Change 4
- Per-block: load `Round0LogupCtx`, compute coset/x indices
- **Identity-coset handling**: coset_idx 0 is the identity coset. The kernel computes `bool is_identity_coset = (coset_idx == 0)` and passes it as the runtime `skip_ntt` flag to `acc_interactions<1, false, false>(...)`, matching the existing coset-parallel logup kernel logic (logup_round0.cu lines 361-442). For coset_idx 0: `g_coset = Fp::one()` and `skip_ntt = true`. For coset_idx > 0: `g_coset = g_shift^(...)` and `skip_ntt = false`.
- Block-level reduction writes `FracExt` (numer + denom) to `tmp_sums_buffer`

New launcher: `_batched_logup_r0_eval_interactions`
- Same structure as Change 4 launcher. Reduction uses `batched_final_reduce_block_sums` for `FracExt` type (the reduction kernel already supports `FracExt` via the stacked reduction usage).

### Change 6: Rust orchestration — batched Round 0

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

Replace the multi-threaded per-AIR loop (lines 935-966) with a batched approach. New helper function `evaluate_round0_batched()`:

**Phase A: Classify and group AIRs (single thread, CPU only)**
For each work item:
1. Compute `num_x = 1 << n_lift`, `num_cosets_zc = local_constraint_deg - 1`, `num_cosets_logup = local_constraint_deg`
2. Check coset-parallel threshold: `num_x * skip_domain < 32768` (from `use_coset_parallel_mode()` in zerocheck_round0.cu line 477). If false → "large AIR", process individually via existing per-AIR `process_air_round0()` (Change 7).
3. Group small AIRs by `num_cosets_zc` for constraint eval groups.
4. Group small AIRs by `num_cosets_logup` for interaction eval groups.
5. Within each group, sort by num_x descending for load balance.

**Phase B: Build descriptors per group (single thread, CPU only)**
For each constraint eval group (all AIRs with same `num_cosets_zc`):
1. Compute `block_x = min(MAX_THREADS, skip_domain * max_num_x)` where `MAX_THREADS = 128`. Ensure `block_x` is a multiple of `skip_domain`.
2. `x_per_block = block_x / skip_domain`
3. For each AIR in group:
   a. `num_x_blocks = ceil(num_x / x_per_block)`
   b. `blocks_per_air = num_x_blocks * num_cosets_zc`
   c. Build `BlockCtx` entries (local_block_idx 0..blocks_per_air, air_idx)
   d. Compute intermediate buffer size: call `_zerocheck_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets_zc, PER_GROUP_TEMP_BYTES)`. Accumulate total.
   e. Compute `buffer_stride = num_cosets_zc * num_x_blocks * block_x`
   f. Build `Round0ZcCtx` with all pointers from work item + pkey, store placeholder for `d_intermediates` (backfilled in Phase C)
   g. Compute tmp_sums buffer contribution: `num_blocks_for_this_air * num_cosets_zc * skip_domain`
4. Record `air_offsets` (cumulative block counts for reduction)

For each interaction eval group (all AIRs with same `num_cosets_logup`):
1. Same block sizing as above with `num_cosets_logup`
2. For each AIR:
   a. Load cached rules from `pk.other_data.interaction_round0`
   b. Compute dynamic weights from `eq_3bs` and `beta_pows` using cached `count_rule_idxs` and `message_rule_idxs`:
      ```rust
      let mut numer_weights = vec![EF::ZERO; cached.rules_len];
      let mut denom_weights = vec![EF::ZERO; cached.rules_len];
      let mut denom_sum_init = EF::ZERO;
      for (i, interaction) in symbolic.interactions.iter().enumerate() {
          numer_weights[cached.count_rule_idxs[i]] += eq_3bs[i];
          denom_sum_init += eq_3bs[i] * beta_pows[interaction.message.len()]
              * F::from_u32(interaction.bus_index as u32 + 1);
          for (j, _) in interaction.message.iter().enumerate() {
              denom_weights[cached.message_rule_idxs[i][j]] += eq_3bs[i] * beta_pows[j];
          }
      }
      ```
   c. Append weights to flat host-side weight arrays
   d. Build `Round0LogupCtx` with pointers from work item + pkey + weight offsets

**Phase C: Allocate, backfill, and upload per group**
For each group:
1. **Memory budget check**: If total intermediate buffer > 2 GB, split group into sub-groups. The 2 GB threshold is chosen because: the prealloc-round0-buffers failure showed ~1 GB per thread (8 threads = 8 GB total) degraded all phases. Here, 2 GB is a single allocation, no per-thread multiplication. This is conservative relative to 24 GB GPU memory minus ~5-8 GB working set.
2. Allocate `intermediates: DeviceBuffer<F>` with total capacity for all AIRs in (sub-)group
3. Backfill `d_intermediates` pointers in context structs using running offset
4. Allocate `tmp_sums_buffer` and `output_buffer`
5. Upload: `d_block_ctxs`, `d_zc_ctxs` (or `d_logup_ctxs`), `d_air_offsets`, weight buffers (for logup groups)

**Phase D: Launch kernels per group**
1. Launch `_batched_zerocheck_r0_eval_constraints` (or `_batched_logup_r0_eval_interactions`)
2. Launch reduction kernel (same `batched_final_reduce_block_sums` with uniform `d`)
3. D2H copy output buffer

**Phase E: CPU post-processing (parallel via rayon)**
For each small AIR:
1. Extract constraint result slice from group's output buffer (offset = `air_idx * num_cosets_zc * skip_domain`)
2. Transpose + iDFT → zerocheck polynomial (same as current lines 185-208)
3. Extract interaction result slice similarly
4. Unzip Frac, transpose + iDFT → logup numer/denom polynomials (same as current lines 233-261)

### Change 7: Fallback for large AIRs

**File**: `crates/cuda-backend/src/logup_zerocheck/mod.rs`

AIRs failing the coset-parallel threshold (`num_x * skip_domain >= 32768`) use the existing per-AIR `process_air_round0()` sequentially on the main thread. These are few in number (typically < 20 at APC 300, ~20 at APC 0) and already saturate the GPU individually.

Process large AIRs BEFORE the batched groups, sequentially. This ensures they get full GPU bandwidth before the batched launches occupy it.

### Change 8: Rust FFI bindings

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add `#[repr(C)]` Rust structs matching the CUDA descriptors:
```rust
#[repr(C)]
pub struct Round0ZcCtx {
    pub selectors_cube: *const F,
    pub preprocessed: *const F,
    pub main_parts: *const *const F,
    pub eq_cube: *const EF,
    pub lambda_pows: *const EF,
    pub public_values: *const F,
    pub d_rules: *const u128,          // Rule encoded as u128
    pub d_used_nodes: *const usize,
    pub rules_len: usize,
    pub used_nodes_len: usize,
    pub lambda_len: usize,
    pub buffer_size: u32,
    pub d_intermediates: *mut F,
    pub buffer_stride: u32,
    pub num_x: u32,
    pub height: u32,
    pub g_shift: F,
}

#[repr(C)]
pub struct Round0LogupCtx {
    pub selectors_cube: *const F,
    pub preprocessed: *const F,
    pub main_parts: *const *const F,
    pub eq_cube: *const EF,
    pub public_values: *const F,
    pub d_rules: *const u128,
    pub rules_len: usize,
    pub buffer_size: u32,
    pub d_intermediates: *mut F,
    pub buffer_stride: u32,
    pub numer_weights: *const EF,
    pub denom_weights: *const EF,
    pub denom_sum_init: EF,
    pub num_x: u32,
    pub height: u32,
    pub g_shift: F,
}
```

Add FFI extern declarations and safe wrapper functions for the batched launchers.

## Invariants

1. **Correctness**: The batched kernel must produce identical results to the per-AIR kernel for every AIR. The computation per block is identical to one block of the existing coset-parallel kernel. The `buffer_stride` per-AIR context field ensures each AIR's blocks index the intermediates buffer without collision.
2. **Memory safety**: All device pointers in context structs must be valid for the duration of the kernel launch. The shared intermediates buffer must be sized to hold all AIRs simultaneously within a sub-group. Weight buffers for logup must outlive the kernel launch.
3. **No regression at APC 0**: Large AIRs (lockstep mode) use the existing per-AIR path. The batched path only handles small AIRs. At APC 0 with ~20 AIRs/segment, most are large → minimal batching → no regression.
4. **Numerical equivalence**: The batched kernel uses GLOBAL intermediates mode only (no shared memory optimization). For most AIRs at APC 300, buffer_size is small enough that GLOBAL mode was already used. Verify via prove+verify correctness test.
5. **Keygen compatibility**: The new `Round0InteractionRules` field in `AirDataGpu` is computed at keygen. Existing serialized proving keys are not affected (this field is not serialized; it's computed in-memory).
6. **Identity coset**: The batched logup kernel handles coset_idx=0 as the identity coset with `g_coset = Fp::one()` and `skip_ntt = true`, matching the existing coset-parallel logup kernel behavior (logup_round0.cu lines 361-436).
7. **Uniform reduction**: Within each `(num_cosets)` group, all AIRs share the same `d = num_cosets * skip_domain`, so `batched_final_reduce_block_sums` works without modification.

## Measurement Plan

1. Run `run_pairing.sh` for APC {0, 100, 300} before and after the change.
2. Analyze with `spec.py` to compare STARK excl trace and Round 0 sub-metrics.
3. Run nsys profiling for APC 300 to verify kernel launch count reduction (~1500 → ~12-20).
4. Verify prove+verify passes for all 3 APC configs.

Expected results (conservative):
- Round 0 APC 300: 294ms → 140-180ms (1.6-2.1x improvement)
- Round 0 APC 0: 177ms → ≤ 177ms (no regression; large AIRs use existing path)
- STARK excl trace APC 300: 1409ms → 1225-1275ms (130-180ms improvement)

New descriptor overhead (per group: ~290 CPU struct builds, ~10KB H2D upload, ~1ms compute) is minor compared to eliminated overhead (~1500 kernel launches, ~2000 buffer allocations, ~310 CPU DAG constructions).

## Rollback Criteria

1. Round 0 at APC 300 improves by less than 60ms (< 20% of current 294ms)
2. STARK excl trace at APC 0 regresses by more than 30ms
3. Prove+verify fails for any APC configuration
4. GPU OOM error during batched kernel launch
