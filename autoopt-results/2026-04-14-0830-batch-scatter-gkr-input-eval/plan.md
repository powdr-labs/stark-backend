# Plan: Batch SCATTER-mode GKR Input Evaluation

## Goal

Reduce LogUp GKR input evaluation wall-clock time at APC 300 by batching SCATTER-mode AIR evaluations (buffer_size <= 10) into a single descriptor-array CUDA kernel per segment, launched on a dedicated background thread concurrent with the existing multi-stream GLOBAL-mode processing.

LogUp GKR is 534ms (40.7% of STARK excl trace at APC 300). The GKR input eval sub-phase is ~302ms across 2 segments (~242ms seg0, ~62ms seg1). From nsight profiling: 486 GLOBAL-mode instances (464ms GPU time, median 760us) + 305 SCATTER-mode instances (63ms GPU time, median 111us). SCATTER AIRs each launch a tiny kernel with 4-16 blocks that underutilizes the 128 SMs.

**Key mechanism**: Remove ~150 SCATTER AIRs per segment from the 8 multi-stream worker threads. Each thread processes ~20 GLOBAL-only AIRs instead of ~39 mixed AIRs. The SCATTER batch runs on a 9th background thread, hidden behind the GLOBAL processing. Each GLOBAL worker thread saves ~19 per-AIR overheads: CPU preparation (~25us), kernel launch (~10us), and between-kernel scheduling gap (~50us). Per-thread savings: ~1.6ms. The critical-path thread improves because its total serial work is shorter.

Additionally, the batched SCATTER kernel (one launch, ~1500 blocks) processes all SCATTER AIRs with optimal GPU block scheduling, vs the current approach where tiny 4-16 block kernels leave most SMs idle during each launch.

**Performance risk**: The prior `gkr-input-round-robin` task (8ms, reverted) showed that scheduling improvements at this phase have limited impact because aggregate kernel execution time dominates. This plan targets a different mechanism (removing work from threads rather than rebalancing), but the per-thread savings (~1.6ms) are modest. The improvement is expected to be 10-20ms, close to the rollback threshold.

## Current Code Path

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

1. **`log_gkr_input_evals()`** (line 215): Entry point.
   - Line 232-252: Build `Vec<GkrInputWorkItem>`, sort by descending height
   - Line 254-279: Compute max buffer sizes for pre-allocation
   - Line 282: Barrier sync
   - Line 285-289: Set num_threads (8 if >= 100 AIRs, else 1)
   - Line 291-303: Memory budget check (halve num_threads if pre-alloc > 2GB)
   - Line 306-321: Pre-allocate per-thread `GkrThreadBuffers`
   - Line 328-348: Multi-threaded processing: contiguous chunks, all AIR types mixed

2. **`process_gkr_input_air()`** (line 112): Per-AIR processing.
   - Line 125-129: Build partition pointers from cached_mains + common_main
   - Line 141-148: Upload public_values to device via `copy_to`
   - Line 152-156: Upload partition_ptrs to device via `copy_to`
   - Line 158-160: `is_global = buffer_size > 10` — determines SCATTER vs GLOBAL mode
   - Line 173-187: Single FFI call: `logup_gkr_input_eval(is_global, ...)`
   - Line 189-208: Conditional lifting: if height != lifted_height, scale + repeat

3. **CUDA kernel**: `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu`
   - Line 65-208: `evaluate_interactions_gkr_kernel<GLOBAL>` template
   - SCATTER (GLOBAL=false, line 89-92): `FpExt intermediates[10]` — local array, max 10 nodes
   - GLOBAL (GLOBAL=true, line 85-87): `d_intermediates` global buffer, stride = task_stride
   - Line 17-63: `evaluate_dag_entry_gkr()` — switch on SourceInfo.type to load data
   - Line 94-207: Per-row loop: iterate d_used_nodes, evaluate DAG, write to d_fracs
   - Line 216-265: Launcher: `grid = kernel_launch_params(is_global ? TASK_SIZE : height, 256)`

4. **FFI**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`
   - Line 278-292: extern declaration
   - Line 938-968: safe wrapper `logup_gkr_input_eval()`

**Existing batched pattern reference**: `ZerocheckCtx` (line 48-58 of `cuda/logup_zerocheck.rs`) + `BlockCtx` (line 17-22) used by `zerocheck_batch_mle_kernel` in `cuda/src/logup_zerocheck/batch_mle.cu`.

## Changes

### Change 1: Define GkrInputScatterCtx descriptor struct

**Files**: `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu` (CUDA) and `crates/cuda-backend/src/cuda/logup_zerocheck.rs` (Rust)

CUDA side:
```cuda
struct GkrInputScatterCtx {
    FracExt *d_fracs;               // Per-AIR output pointer (into leaves or tmp buffer)
    const Fp *d_preprocessed;       // Per-AIR preprocessed trace (or null ptr)
    const uint64_t *d_main;         // Per-AIR: pointer into concatenated partition-ptr array
    const Fp *d_public_values;      // Per-AIR: pointer into concatenated public-values array (or null)
    const FpExt *d_challenges;      // Shared challenge vector (same for all AIRs)
    const Rule *d_rules;            // Per-AIR: already on device in pk_air
    const size_t *d_used_nodes;     // Per-AIR: already on device in pk_air
    const uint32_t *d_pair_idxs;    // Per-AIR: already on device in pk_air
    size_t used_nodes_len;          // Per-AIR
    uint32_t permutation_height;    // Per-AIR trace height
    uint32_t num_blocks;            // Blocks assigned to this AIR
};
```

Rust side:
```rust
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GkrInputScatterCtx {
    pub d_fracs: *mut Frac<EF>,
    pub d_preprocessed: *const F,
    pub d_main: *const u64,
    pub d_public_values: *const F,
    pub d_challenges: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub d_used_nodes: *const usize,
    pub d_pair_idxs: *const u32,
    pub used_nodes_len: usize,
    pub permutation_height: u32,
    pub num_blocks: u32,
}
unsafe impl Send for GkrInputScatterCtx {}
unsafe impl Sync for GkrInputScatterCtx {}
```

### Change 2: Concatenated H2D upload and descriptor construction (on background thread)

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

**Stream ordering**: All SCATTER H2D uploads, descriptor construction, and kernel launch happen on the **background thread's** CUDA stream (`cudaStreamPerThread`). This avoids cross-stream ordering issues — the background thread's stream serializes its own H2D uploads before the kernel launch automatically.

The background thread receives only the `scatter_indices` (computed on the main thread), plus shared references to `work_items` and `d_challenges`. It does all prep work internally:

```rust
// Inside the background thread closure:
// Step 1: Concatenate host arrays for partition pointers and public values
let mut all_partition_ptrs: Vec<u64> = Vec::new();
let mut all_public_values: Vec<F> = Vec::new();
let mut scatter_partition_offsets: Vec<usize> = Vec::new();
let mut scatter_pv_offsets: Vec<usize> = Vec::new();

for &work_idx in &scatter_indices {
    let w = &work_items[work_idx];
    let air_ctx = w.air_ctx;

    scatter_partition_offsets.push(all_partition_ptrs.len());
    for committed in &air_ctx.cached_mains {
        all_partition_ptrs.push(committed.trace.buffer().as_ptr() as u64);
    }
    all_partition_ptrs.push(air_ctx.common_main.buffer().as_ptr() as u64);

    scatter_pv_offsets.push(all_public_values.len());
    all_public_values.extend_from_slice(&air_ctx.public_values);
}

// 2 bulk H2D uploads (on this thread's stream)
let d_all_partition_ptrs = all_partition_ptrs.to_device()?;
let d_all_public_values = if all_public_values.is_empty() {
    DeviceBuffer::new()
} else {
    all_public_values.to_device()?
};

// Step 2: Build BlockCtx and GkrInputScatterCtx arrays
let mut block_ctxs: Vec<BlockCtx> = Vec::new();
let mut air_ctxs: Vec<GkrInputScatterCtx> = Vec::new();

for (air_local_idx, &work_idx) in scatter_indices.iter().enumerate() {
    let w = &work_items[work_idx];
    let air_ctx = w.air_ctx;
    let pk_air = w.pk_air;
    let height = air_ctx.height() as u32;
    let num_air_blocks = height.div_ceil(256);
    let rules = &pk_air.other_data.interaction_rules;

    // BlockCtx entries: one per block for this AIR
    for local_block in 0..num_air_blocks {
        block_ctxs.push(BlockCtx {
            local_block_idx_x: local_block,
            air_idx: air_local_idx as u32,
        });
    }

    // Per-AIR descriptor
    let preprocessed_ptr = pk_air.preprocessed_data
        .as_ref()
        .map(|c| c.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());
    let pv_ptr = if air_ctx.public_values.is_empty() {
        std::ptr::null()
    } else {
        unsafe { d_all_public_values.as_ptr().add(scatter_pv_offsets[air_local_idx]) }
    };
    let height_usize = air_ctx.height();
    let lifted_height = max(height_usize, 1 << w.l_skip);
    let fracs_ptr = if height_usize != lifted_height {
        d_scatter_tmp.as_mut_ptr()  // lifted AIR → write to tmp
    } else {
        w.leaves_ptr.0  // non-lifted → write directly to leaves
    };

    air_ctxs.push(GkrInputScatterCtx {
        d_fracs: fracs_ptr,
        d_preprocessed: preprocessed_ptr,
        d_main: unsafe { d_all_partition_ptrs.as_ptr().add(scatter_partition_offsets[air_local_idx]) },
        d_public_values: pv_ptr,
        d_challenges: d_challenges_ref.as_ptr(),
        d_rules: rules.inner.d_rules.as_ptr() as *const _,
        d_used_nodes: rules.inner.d_used_nodes.as_ptr(),
        d_pair_idxs: rules.d_pair_idxs.as_ptr(),
        used_nodes_len: rules.inner.d_used_nodes.len(),
        permutation_height: height,
        num_blocks: num_air_blocks,
    });
}

let total_scatter_blocks = block_ctxs.len() as u32;
let d_block_ctxs = block_ctxs.to_device()?;
let d_air_ctxs = air_ctxs.to_device()?;

// Step 3: Launch batched kernel (on this thread's stream, after uploads)
unsafe {
    batched_gkr_input_eval_scatter(&d_block_ctxs, &d_air_ctxs, total_scatter_blocks)?;
}

// Step 4: Sequential lifting for SCATTER AIRs that need it
// (see Change 3)
```

This approach ensures all GPU work (H2D uploads → kernel launch → lifting) runs on the same CUDA stream, with correct ordering guaranteed by stream semantics.

### Change 3: Handle lifting for SCATTER AIRs

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

SCATTER AIRs where `height != lifted_height` (height < 2^l_skip) need post-processing: scale by norm_factor and vertically repeat. Strategy:

1. During descriptor construction, check if the AIR needs lifting.
2. For **non-lifted** SCATTER AIRs: `d_fracs` in the descriptor points directly to the AIR's slice in the `leaves` buffer.
3. For **lifted** SCATTER AIRs: `d_fracs` points to a shared tmp buffer. After the batched kernel completes, the background thread processes lifted AIRs sequentially — calling `frac_vector_scalar_multiply_ext_fp` + `frac_matrix_vertically_repeat` for each, reusing the same tmp buffer between AIRs.

Tmp buffer allocation:
```rust
// Compute max tmp size across lifted SCATTER AIRs
let max_scatter_tmp = scatter_indices.iter()
    .filter_map(|&idx| {
        let w = &work_items[idx];
        let height = w.air_ctx.height();
        let lifted = max(height, 1 << w.l_skip);
        let n_int = w.pk_air.vk.symbolic_constraints.interactions.len();
        if height != lifted { Some(height * n_int) } else { None }
    })
    .max()
    .unwrap_or(0);

let d_scatter_tmp = if max_scatter_tmp > 0 {
    DeviceBuffer::<Frac<EF>>::with_capacity(max_scatter_tmp)
} else {
    DeviceBuffer::new()
};
```

For each lifted SCATTER AIR in the descriptor: set `d_fracs = d_scatter_tmp.as_mut_ptr()`. After the batched kernel completes on the background thread, process lifted AIRs sequentially:

```rust
// On background thread, after batch kernel completes:
for &work_idx in &scatter_indices {
    let w = &work_items[work_idx];
    let height = w.air_ctx.height();
    let lifted_height = max(height, 1 << w.l_skip);
    if height != lifted_height {
        let n_int = w.pk_air.vk.symbolic_constraints.interactions.len();
        let norm_factor = F::from_usize(lifted_height / height).inverse();
        unsafe {
            frac_vector_scalar_multiply_ext_fp(
                d_scatter_tmp.as_mut_ptr(), norm_factor, (height * n_int) as u32
            )?;
            frac_matrix_vertically_repeat(
                w.leaves_ptr.0, d_scatter_tmp.as_ptr(),
                n_int as u32, lifted_height as u32, height as u32
            )?;
        }
    }
}
```

**Correctness**: Only one lifted AIR uses `d_scatter_tmp` at a time (sequential processing). Non-lifted AIRs write directly to leaves. The tmp buffer is reused between lifted AIRs (kernel outputs are consumed by the lifting kernels before the next lifted AIR starts).

### Change 4: New batched SCATTER kernel

**File**: `crates/cuda-backend/cuda/src/logup_zerocheck/gkr_input.cu` (append)

```cuda
__global__ void batched_evaluate_interactions_scatter_kernel(
    const BlockCtx *__restrict__ d_block_ctxs,
    const GkrInputScatterCtx *__restrict__ d_air_ctxs
) {
    BlockCtx block_ctx = d_block_ctxs[blockIdx.x];
    GkrInputScatterCtx ctx = d_air_ctxs[block_ctx.air_idx];

    uint32_t local_block = block_ctx.local_block_idx_x;
    uint32_t task_offset = local_block * blockDim.x + threadIdx.x;
    uint32_t task_stride = ctx.num_blocks * blockDim.x;

    FpExt intermediates[10];
    uint32_t intermediate_stride = 1;

    // Guard: boundary check
    if (task_offset >= ctx.permutation_height) return;

    // DAG evaluation loop — same logic as gkr_input.cu lines 94-207
    // but reading from ctx.d_rules, ctx.d_used_nodes, ctx.d_pair_idxs,
    // ctx.d_preprocessed, ctx.d_main, ctx.d_public_values, ctx.d_challenges
    // and writing to ctx.d_fracs
    // num_rows_per_tile = 1 for SCATTER mode (height <= TASK_SIZE)
    // ...
}

extern "C" int _batched_gkr_input_eval_scatter(
    const BlockCtx *d_block_ctxs,
    const GkrInputScatterCtx *d_air_ctxs,
    uint32_t num_blocks
) {
    if (num_blocks == 0) return 0;
    dim3 grid(num_blocks);
    dim3 block(256);
    batched_evaluate_interactions_scatter_kernel<<<grid, block>>>(
        d_block_ctxs, d_air_ctxs
    );
    return CHECK_KERNEL();
}
```

### Change 5: Rust FFI + safe wrapper

**File**: `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

```rust
extern "C" {
    fn _batched_gkr_input_eval_scatter(
        block_ctxs: *const BlockCtx,
        air_ctxs: *const GkrInputScatterCtx,
        num_blocks: u32,
    ) -> i32;
}

pub unsafe fn batched_gkr_input_eval_scatter(
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<GkrInputScatterCtx>,
    num_blocks: u32,
) -> Result<(), CudaError> {
    check_error(_batched_gkr_input_eval_scatter(
        block_ctxs.as_ptr(), air_ctxs.as_ptr(), num_blocks,
    ))
}
```

**CUDA header include**: The batched kernel in `gkr_input.cu` uses `BlockCtx` which is defined in `eval_ctx.cuh` under namespace `logup_zerocheck_mle`. Add `#include "eval_ctx.cuh"` at the top of `gkr_input.cu` and use `using logup_zerocheck_mle::BlockCtx;` inside the kernel, or define a local `GkrBlockCtx` struct with the same layout.

### Change 6: Restructure `log_gkr_input_evals` orchestration

**File**: `crates/cuda-backend/src/logup_zerocheck/gkr_input.rs`

After sorting work_items (line 252), partition into SCATTER and GLOBAL groups. Then in the thread::scope block:

- If `work_items.len() < 100`: use existing single-threaded path for ALL AIRs (unchanged, no batching). This preserves the APC 0 fallback exactly.
- If `work_items.len() >= 100`:
  - Compute SCATTER/GLOBAL indices on main thread (pure CPU, trivial)
  - Allocate `d_scatter_tmp` on main thread (single DeviceBuffer::with_capacity)
  - Spawn background thread for SCATTER batch — this thread does ALL SCATTER prep internally: concatenated H2D uploads, descriptor construction, kernel launch, lifting (Change 2 + Change 3). Only `scatter_indices`, `work_items`, `d_challenges`, and `d_scatter_tmp` are borrowed from the parent scope.
  - Spawn 8 worker threads for GLOBAL AIRs only (using contiguous chunks of height-sorted GLOBAL items, same existing pattern)
  - Join all threads

Thread count for GLOBAL workers: `min(NUM_GKR_INPUT_STREAMS, global_indices.len())`. Memory budget check uses only GLOBAL AIRs' max buffer sizes (SCATTER AIRs don't need `GkrThreadBuffers`).

## Invariants

1. **Correctness**: Every AIR processed exactly once. SCATTER → batched kernel, GLOBAL → multi-stream. Non-overlapping output regions in `leaves`.

2. **Output equivalence**: Batched SCATTER kernel uses identical DAG evaluation logic as `evaluate_interactions_gkr_kernel<false>`.

3. **APC 0 fallback**: Guard `work_items.len() >= 100` ensures APC 0 (99 AIRs) uses the unchanged single-threaded path. No SCATTER batching at APC 0.

4. **Lifting correctness**: Lifted SCATTER AIRs write to a shared `d_scatter_tmp` buffer (one at a time, sequentially on the background thread). Non-lifted SCATTER AIRs write directly to `leaves`. The tmp buffer is large enough for the biggest lifted SCATTER AIR.

5. **Memory safety**: All descriptor pointers reference device data valid for the `log_gkr_input_evals` lifetime. SCATTER intermediates are local arrays (10 elements, no device buffer needed). The `d_scatter_tmp` buffer is allocated on the main thread and borrowed by the background thread within `thread::scope`.

6. **GLOBAL path unchanged**: GLOBAL-mode processing is identical except it receives fewer work items.

## Measurement Plan

Run the standard benchmark:
```bash
cd /home/georg/powdr/results/pairing
/home/georg/powdr/target/release/powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics <path> --recursion
```

Key metrics (APC 300):
- **LogUp GKR**: current ~534ms, expect 10-20ms improvement
- **STARK excl trace**: current ~1312ms, expect proportional improvement
- No regression in Round 0, MLE Rounds, Trace Commit, Openings

Key metrics (APC 000): must not regress (single-threaded fallback unchanged).

Run each configuration at least twice.

## Rollback Criteria

Revert if ANY of:
- LogUp GKR at APC 300 improves by less than 10ms (median of 2+ runs)
- STARK excl trace at APC 300 regresses by more than 20ms
- Any APC 0 metric regresses by more than 30ms
- Proof verification fails
