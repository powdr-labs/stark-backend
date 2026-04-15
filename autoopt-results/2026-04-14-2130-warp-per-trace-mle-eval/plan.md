# Plan: Warp-per-trace monomial MLE evaluation for small traces

## Goal

Improve MLE Rounds GPU utilization at APC 300 by introducing a warp-per-trace monomial evaluation path for traces with few monomials. Currently, `zerocheck_monomial_kernel` (used for `low_early` + `late_eval` traces where `num_y ≤ 512`) parallelizes over monomials with `THREADS_PER_BLOCK = 256`. Since most AIRs have only 5–20 monomials, each block has 256 threads but only 5–20 are active (2–8% utilization). The optimization introduces a warp-per-trace kernel that flips the parallelism axis: each thread handles one y-value and loops over all monomials, packing 4 traces per 128-thread block with warp-level reduction.

Target: reduce MLE Rounds at APC 300 from 159ms toward ~135–145ms (15–25ms improvement), improving scaling from 0.70x toward ~0.77–0.83x.

## Current Code Path

### Routing: which traces reach which kernel

In `sumcheck_polys_batch_eval` (`mod.rs:1640–1946`):
- **late_eval** traces (round > n_lift, effectively num_y=1): handled by `ZerocheckMonomialBatch` with `evaluate(1)` (line 1876) → launches `zerocheck_monomial_kernel`.
- **low_early** traces (num_y ≤ monomial_num_y_threshold=512): handled by `ZerocheckMonomialBatch` with `evaluate(num_x)` (line 1876) → launches `zerocheck_monomial_kernel`.
- **high_mono_traces** (num_y > 512, low monomial ratio): `ZerocheckMonomialParYBatch` → launches `zerocheck_monomial_par_y_kernel`.
- **high_dag_traces** (num_y > 512, high monomial ratio): `ZerocheckMleBatchBuilder` → launches `zerocheck_batch_mle_kernel`.

For logup, analogous: `LogupMonomialBatch` (num_y ≤ 512), `LogupMleBatchBuilder` (num_y > 512).

At APC 300 with 623 small AIR instances: the vast majority have num_y ≤ 512 in early rounds, and ALL traces transition to late_eval (num_y=1) as MLE rounds progress. The `zerocheck_monomial_kernel` and `LogupMonomialBatch` handle most of the work.

### zerocheck_monomial_kernel (`batch_mle_monomial.cu:102–151`)

Grid: `(num_blocks, num_x)`. Block: `(threads_per_block=256)`.

Block assignment (Rust, `batch_mle_monomial.rs:152–168`):
```
mono_blocks = ceil(num_monomials / 256)
total_blocks = mono_blocks * num_y   // one set of mono_blocks per y-value
```

Kernel logic:
- Decodes `y_int = local_block_idx_x / mono_blocks` and `mono_block = local_block_idx_x % mono_blocks`.
- Thread `m = mono_block * 256 + threadIdx.x` evaluates monomial `m` if `m < num_monomials`.
- Evaluates product of variables for that monomial, multiplies by `lambda_combinations[m]`.
- `block_reduce_sum` sums across all threads → per-(y_int, x_int) partial sum.
- Thread 0 multiplies by `eq_xi[y_int]` and writes to `tmp_sums[blockIdx.x * num_x + x_int]`.

Secondary reduction: `batched_final_reduce_block_sums` reduces across mono_blocks for same (air, y_int).

**Waste at APC 300**: A typical APC AIR has `num_monomials ≈ 5–15`. With `threads_per_block = 256`:
- `mono_blocks = 1` (ceil(15/256) = 1)
- 256 threads per block, 15 active → 5.9% utilization
- Per trace: `num_y` blocks of 256 threads each, totaling `num_y × 256` threads with `num_y × 15` useful FMAs

nsight profile: `zerocheck_monomial_kernel`: 178 instances, 17ms total, 0.09ms avg.

### LogupMonomialBatch (`batch_mle_monomial.rs:588–793`)

Same pattern. `THREADS_PER_BLOCK_LOGUP = 128`. Block assignment differs slightly (accounts for both numerator and denominator interactions). Same utilization problem.

## Changes

### Change 1: New CUDA kernel — warp-per-trace monomial evaluation

**File:** `crates/cuda-backend/cuda/src/logup_zerocheck/batch_mle_monomial.cu`

Add `warp_zerocheck_monomial_kernel`:

```cuda
// Warp-per-trace: each warp handles one trace.
// Thread = one y-value. Each thread loops over ALL monomials for its y-value.
// Constraint: num_y <= WARP_SIZE and num_monomials <= warp_mono_limit (e.g. 32).
__global__ void warp_zerocheck_monomial_kernel(
    FpExt *__restrict__ output,                    // [total_traces * num_x], scatter output
    const MonomialAirCtx *__restrict__ air_ctxs,
    const uint32_t *__restrict__ trace_ids,        // [num_warps] → air_ctxs index
    const uint32_t *__restrict__ output_offsets,   // [num_warps] → output position
    uint32_t num_traces
) {
    constexpr uint32_t WARP = 32;
    uint32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    uint32_t lane = threadIdx.x % WARP;
    uint32_t num_x = gridDim.y;
    uint32_t x_int = blockIdx.y;

    if (warp_global >= num_traces) return;
    MonomialAirCtx actx = air_ctxs[trace_ids[warp_global]];

    uint32_t y_int = lane;
    bool active = (y_int < actx.num_y);

    FpExt sum(Fp::zero());
    if (active) {
        uint32_t height = num_x * actx.num_y;
        uint32_t row = x_int * actx.num_y + y_int;

        // Loop over ALL monomials (small count, so serial is fine)
        for (uint32_t m = 0; m < actx.num_monomials; ++m) {
            MonomialHeader hdr = actx.d_headers[m];
            FpExt product(Fp::one());
            for (uint16_t v = 0; v < hdr.num_vars; ++v) {
                PackedVar var = actx.d_variables[hdr.var_offset + v];
                product *= eval_variable(var, row, actx.eval_ctx, height);
            }
            sum += product * actx.d_lambda_combinations[m];
        }
        sum *= actx.d_eq_xi[y_int];
    }

    // Warp reduction: sum across y-values in this trace
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        uint32_t out_pos = output_offsets[warp_global];
        output[out_pos * num_x + x_int] = sum;
    }
}
```

The kernel takes an additional `output_offsets` parameter (`const uint32_t *__restrict__`) that maps each warp to its position in the output buffer. This enables scatter writes that preserve the original trace ordering.

Key design choices:
- **Parallelism axis flipped**: threads map to y-values (not monomials). Each thread serially evaluates all monomials — this is correct because `num_monomials ≤ 32`, so the serial loop is short.
- **Direct scatter output**: writes to the correct position in the output buffer via `output_offsets`. No `tmp_sums` intermediary, no secondary `batched_final_reduce_block_sums` launch.
- **Warp reduction**: `warp_reduce_sum` sums across y-values within the trace. No shared memory needed.
- **4 warps per block** (128 threads): 4 traces processed per block.
- **`trace_ids` indirection**: maps warp index → `air_ctxs` slot, allowing the same `air_ctxs` array to serve both the warp kernel (small traces) and the existing kernel (large traces).
- **`output_offsets` scatter**: preserves original trace ordering in the output buffer, avoiding consumer code changes.

Add C launcher `_warp_zerocheck_monomial_batched`:
- Grid: `(ceil(num_traces / 4), num_x)`, Block: `(128)`.
- No tmp_sums allocation, no secondary reduce launch.

### Change 2: New CUDA kernel — warp-per-trace logup monomial evaluation

**File:** `crates/cuda-backend/cuda/src/logup_zerocheck/batch_mle_monomial.cu`

Add `warp_logup_monomial_kernel` — same warp-per-trace pattern, but accumulates both `numer_sum` and `denom_sum` (FracExt output). Two `warp_reduce_sum` calls (no `__syncthreads` needed — single warp).

The logup monomial kernel (`logup_monomial_kernel`, `batch_mle_monomial.cu`) iterates over interactions and evaluates numerator/denominator expressions. The warp kernel maintains the same logic but with each thread covering one y-value and looping over interactions.

Add C launcher `_warp_logup_monomial_batched`.

### Change 3: Rust FFI bindings

**File:** `crates/cuda-backend/src/cuda/logup_zerocheck.rs`

Add extern "C" declarations and safe wrappers:
- `warp_zerocheck_monomial_batched(output, air_ctxs, trace_ids, num_traces, num_x) -> Result`
- `warp_logup_monomial_batched(output, logup_air_ctxs, trace_ids, num_traces, num_x) -> Result`

### Change 4: Partition in ZerocheckMonomialBatch (preserving trace order)

**File:** `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs`

Modify `ZerocheckMonomialBatch::new`:

```rust
const WARP_SIZE: u32 = 32;

// Classify each trace as warp-eligible or block-eligible.
// DO NOT reorder self.traces — keep original input order to preserve consumer indexing.
let mut warp_indices: Vec<u32> = Vec::new();  // indices into self.traces for warp-path
let mut warp_output_offsets: Vec<u32> = Vec::new();  // output position for each warp trace
let mut block_indices: Vec<usize> = Vec::new();  // indices into self.traces for block-path

for (i, t) in traces.iter().enumerate() {
    let nm = pk.per_air[t.air_idx].other_data.zerocheck_monomials
        .as_ref().unwrap().num_monomials;
    if t.num_y <= WARP_SIZE && nm <= WARP_SIZE {
        warp_output_offsets.push(i as u32);  // write to position i in output
        warp_indices.push(i as u32);         // air_ctxs index = i (original position)
    } else {
        block_indices.push(i);
    }
}
```

New builder fields:
```rust
// Warp-path fields:
warp_trace_ids: DeviceBuffer<u32>,        // [num_warp_traces]: air_ctxs index for each warp
warp_output_offsets: DeviceBuffer<u32>,   // [num_warp_traces]: output buffer position (× num_x)
num_warp_traces: u32,
// Block-path fields (existing, for non-warp traces):
block_ctxs: DeviceBuffer<BlockCtx>,
block_air_offsets: DeviceBuffer<u32>,
block_output_offsets: DeviceBuffer<u32>,  // [num_block_airs]: output buffer position (× num_x)
num_block_traces: u32,
// Shared:
air_ctxs: DeviceBuffer<MonomialAirCtx>,  // all traces in original order
```

**Key invariant**: `self.traces` is NOT reordered. The `air_ctxs` array contains ALL traces in original input order. The warp kernel uses `warp_trace_ids[warp_id]` to index into `air_ctxs` and `warp_output_offsets[warp_id]` to write to the correct position in the output buffer. The block kernel uses `BlockCtx.air_idx` to index into `air_ctxs` and `block_output_offsets[air_local]` for output positioning.

This preserves the existing `trace_indices()` → `enumerate()` → `host[i * num_x..]` consumption pattern used by callers in `batch_mle.rs:494` and `batch_mle.rs:530-532`, because position `i` in the output buffer corresponds to position `i` in `self.traces` (unchanged order).

### Change 5: Two-launch evaluate (scatter output)

**File:** `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs`

Modify `ZerocheckMonomialBatch::evaluate`:

```rust
pub fn evaluate(&self, num_x: u32) -> Result<DeviceBuffer<EF>, KernelError> {
    let total = self.traces.len();
    let mut output = DeviceBuffer::<EF>::with_capacity(total * num_x as usize);

    // Launch 1: warp-per-trace for small traces
    // Writes to scattered positions in output via warp_output_offsets
    if self.num_warp_traces > 0 {
        unsafe {
            warp_zerocheck_monomial_batched(
                &mut output,
                &self.air_ctxs,
                &self.warp_trace_ids,
                &self.warp_output_offsets,  // scatter: warp i writes to output[offsets[i] * num_x]
                self.num_warp_traces, num_x,
            )?;
        }
    }

    // Launch 2: existing block kernel for remaining traces
    // Uses existing kernel but with output scatter via block_output_offsets
    if self.num_block_traces > 0 {
        let num_blocks = self.block_ctxs.len();
        let mut tmp_sums = DeviceBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
        unsafe {
            zerocheck_monomial_batched_scatter(
                &mut tmp_sums,
                &mut output,           // shared output buffer
                &self.block_ctxs,
                &self.air_ctxs,
                &self.block_air_offsets,
                &self.block_output_offsets, // scatter: air i writes to output[offsets[i] * num_x]
                num_blocks as u32, num_x,
                self.num_block_traces, THREADS_PER_BLOCK,
            )?;
        }
    }

    Ok(output)
}
```

**Output layout**: trace `i` in `self.traces` (original order) corresponds to `output[i * num_x .. (i+1) * num_x]`. Both kernels write to the correct positions via their respective output offset arrays. `trace_indices()` iterates `self.traces` in original order, so `host[i * num_x..]` reads the correct trace. **No consumer code changes needed.**

The block-path reduction kernel (`batched_final_reduce_block_sums`) needs a minor variant that writes to scattered output positions instead of contiguous ones. This is a small modification to the reduction launcher: pass `block_output_offsets` and have thread 0 of each reduction block write to `output[offsets[air_idx] * num_x + x_int]` instead of `output[air_idx * num_x + x_int]`.

### Change 6: Apply same partition to LogupMonomialBatch

**File:** `crates/cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs`

Same pattern for `LogupMonomialBatch::new` and `evaluate`, using `warp_logup_monomial_batched`. The logup monomial kernel has `THREADS_PER_BLOCK_LOGUP = 128`, similar underutilization.

## Invariants

1. **Correctness**: The warp kernel evaluates the same monomials × lambda_combinations for the same (x_int, y_int) pairs, producing the same per-trace sums. The serial monomial loop in the warp kernel is mathematically equivalent to the parallel-monomial approach in the block kernel, because addition is commutative and associative over the field. `warp_reduce_sum` over y-values produces the same result as the block kernel's `block_reduce_sum` over mono-blocks followed by `batched_final_reduce_block_sums` over (y_int × mono_blocks).

2. **Threshold safety**: Only traces with `num_y <= WARP_SIZE=32` AND `num_monomials <= WARP_SIZE=32` use the warp path. The `num_y` guard ensures no y-values are missed (thread with `lane >= num_y` contributes zero). The `num_monomials` bound ensures the serial loop is short enough that per-thread work doesn't become a bottleneck.

3. **No intermediates needed**: The monomial kernel does NOT use intermediates buffers (unlike the DAG batch kernel). Each monomial is evaluated independently from trace data. This means the warp kernel needs no intermediates allocation at all.

4. **No APC 0 regression**: At APC 0 (99 AIRs, few segments), most traces have large num_y (>32) → they go to the block kernel path. The warp partition has 0–5 traces, producing negligible overhead.

5. **Memory**: The `warp_trace_ids` buffer is at most `num_warp_traces * 4` bytes (≈2KB). The `tmp_sums` buffer is now smaller (only allocated for block-path traces). Net memory is roughly unchanged.

## Measurement Plan

1. Build: `cd /home/georg/powdr && cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"`
2. Verify correctness: `cargo nextest run -p openvm-cuda-backend --test-threads=4` — all tests must pass.
3. Measure APC 300:
   ```bash
   RUST_LOG=info powdr_openvm_riscv prove --artifact apc300.cbor --input 0 --metrics current_apc300.json --recursion
   python3 spec.py current_apc300.json apc300
   ```
   Check: MLE Rounds < 148ms, STARK excl trace < 1273ms.
4. Measure APC 0:
   ```bash
   RUST_LOG=info powdr_openvm_riscv prove --artifact apc000.cbor --input 0 --metrics current_apc000.json --recursion
   python3 spec.py current_apc000.json apc000
   ```
   Check: No regression (STARK excl trace ≤ 2150ms).
5. Profile APC 300 with nsight: confirm `warp_zerocheck_monomial_kernel` runs with reduced block count. Compare `zerocheck_monomial_kernel` instance count (should be lower — block-path only).

## Rollback Criteria

- MLE Rounds at APC 300 improves by < 10ms: revert.
- STARK excl trace at APC 300 improves by < 10ms: revert.
- Any regression at APC 0 (STARK excl trace > 2160ms): revert.
- Any test failure: revert.
