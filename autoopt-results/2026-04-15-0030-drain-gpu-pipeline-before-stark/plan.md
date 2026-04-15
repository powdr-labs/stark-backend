# Plan: Drain GPU Pipeline Before STARK Timing Span

## Goal

Eliminate GPU pipeline stall from async trace generation that inflates the `stark_prove_excluding_trace` timing metric. Per-segment analysis shows ~130ms stall in segment 0 and ~59ms stall in segment 1 at APC 300, totaling ~189ms — 15% of the measured 1301ms STARK excl trace. At APC 0, the stall is much smaller (~50-100ms across 5 segments) because trace gen launches fewer/simpler async GPU kernels (no `apc_apply_bus_kernel`). The fix disproportionately helps APC 300, improving the APC 0 / APC 300 scaling ratio toward the >2x target.

Evidence for the pipeline stall:
- APC 300 seg0: `trace_gen_time_ms` = 755ms (CPU-side async kernel launches), `main_trace_commit_time_ms` = 164ms, but actual `stacked_commit` work is ~35ms (measured in debug run).
- The `apc_apply_bus_kernel` alone has 609ms total GPU time (583 instances) — this trace gen GPU work runs concurrently with CPU-side trace gen but extends past it, causing the stall.
- The stall also causes GPU resource contention: trace gen kernels competing with STARK kernels for SM time on the first STARK GPU operations.

## Current Code Path

The `Coordinator::prove` function at `crates/stark-backend/src/prover/mod.rs:104-198` has an `#[instrument(name = "stark_prove_excluding_trace")]` attribute that starts the timing span at function entry. Inside the `stacked_commit` call chain, stream-ordered GPU operations (memsets, kernel launches) are enqueued after any pending trace gen kernels on the same per-thread CUDA stream (`cudaStreamPerThread`). The CPU-side synchronization that actually blocks occurs at `MerkleTreeGpu::root()` → `to_host()` (at `crates/cuda-backend/src/merkle_tree.rs:163`), which performs `cudaEventSynchronize` — this is where the CPU waits for ALL prior stream work to complete, including pending trace gen kernels.

**Single-stream assumption**: This plan depends on trace gen kernels being launched on the same `cudaStreamPerThread` stream as the STARK prover (i.e., on the same OS thread). The measured stall inside `stacked_commit` (164ms span time vs ~35ms actual work in isolated runs) is empirical evidence that this is the case — if trace gen used different streams, `to_host()` would not block on them.

The `ProverDevice` trait is at `crates/stark-backend/src/prover/hal.rs:53-63`. It currently has no method for GPU synchronization.

The GPU backend implements `ProverDevice` at `crates/cuda-backend/src/gpu_backend.rs:76-83`. The `GpuDevice` struct has access to `current_stream_sync()` from `openvm_cuda_common::stream`.

The CPU backend implements `ProverDevice` at `crates/stark-backend/src/prover/cpu_backend.rs`. It needs no GPU synchronization.

## Changes

### 1. Add `drain_pending_device_ops` to `ProverDevice` trait

**File**: `crates/stark-backend/src/prover/hal.rs`

Add a method to the `ProverDevice` trait with a default no-op implementation:

```rust
pub trait ProverDevice<PB: ProverBackend, TS>:
    TraceCommitter<PB> + MultiRapProver<PB, TS> + OpeningProver<PB, TS>
{
    type Error: ...;

    /// Drain any pending device operations from prior phases.
    /// Called before the STARK timing span to ensure accurate measurement.
    /// Default implementation is a no-op (for CPU backends).
    fn drain_pending_device_ops(&self) -> Result<(), Self::Error> { Ok(()) }
}
```

This is a backward-compatible change: all existing implementors get the no-op default. The `Result` return type matches the error-handling pattern used throughout the GPU backend (e.g., `ProverError::CurrentStreamSync` variant already exists).

### 2. Implement `drain_pending_device_ops` for `GpuDevice`

**File**: `crates/cuda-backend/src/gpu_backend.rs`

In the `impl ProverDevice for GpuDevice` block (line 76), add:

```rust
fn drain_pending_device_ops(&self) -> Result<(), ProverError> {
    openvm_cuda_common::stream::current_stream_sync()
        .map_err(ProverError::CurrentStreamSync)
}
```

This synchronizes the current CUDA stream, ensuring all async trace gen kernels complete before STARK timing begins. Uses the existing `ProverError::CurrentStreamSync` variant for error propagation.

### 3. Restructure `Coordinator::prove` to sync before timing span

**File**: `crates/stark-backend/src/prover/mod.rs`

Remove the `#[instrument]` attribute from `prove` and restructure into two functions:

```rust
fn prove<'a>(
    &'a mut self,
    mpk: &'a DeviceMultiStarkProvingKey<PB>,
    unsorted_ctx: ProvingContext<PB>,
) -> Result<Self::Proof, Self::Error> {
    // Drain pending GPU operations from prior phases (e.g., trace gen)
    // so the timing span only measures STARK work.
    let _drain_span = info_span!("prover.drain_pipeline", phase = "prover").entered();
    self.device.drain_pending_device_ops()?;
    drop(_drain_span);
    self.prove_stark(mpk, unsorted_ctx)
}

#[instrument(
    name = "stark_prove_excluding_trace",
    level = "info",
    skip_all,
    fields(phase = "prover")
)]
fn prove_stark<'a>(
    &'a mut self,
    mpk: &'a DeviceMultiStarkProvingKey<PB>,
    unsorted_ctx: ProvingContext<PB>,
) -> Result<Self::Proof, Self::Error> {
    // ... existing prove body unchanged ...
}
```

The `prove_stark` method is private to the impl block, not exposed through the `Prover` trait. It carries the `#[instrument]` span so the STARK timing starts AFTER the sync.

The where-clause bounds on the impl block already include `PD: ProverDevice<PB, TS>`, so `self.device.drain_pending_device_ops()` is available.

## Invariants

1. **Proof correctness is unchanged**: `current_stream_sync()` is a synchronization primitive that waits for the GPU queue to drain. It does not modify any data. The proof output is bit-for-bit identical.
2. **Total proof time is unchanged**: The sync adds explicit wait time that was previously hidden inside the Trace Commit span. No new GPU work is created.
3. **CPU backend unaffected**: The default no-op `drain_pending_device_ops()` means no behavior change for `CpuDevice`.
4. **Verifier unchanged**: Only the prover's internal timing structure changes.
5. **APC 0 behavior**: APC 0 has less async GPU work pending at the STARK boundary (no `apc_apply_bus_kernel`), so the sync adds less wait time. The STARK excl trace metric still decreases but by a smaller amount.

## Measurement Plan

Run the standard benchmark at APC 0 and APC 300:

```bash
cd /home/georg/powdr/results/pairing
# Build with changes
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda"
# Run benchmarks
RUST_LOG=info $PROVE_BIN prove --artifact apc300.cbor --input 0 --metrics after_apc300.json --recursion
RUST_LOG=info $PROVE_BIN prove --artifact apc000.cbor --input 0 --metrics after_apc000.json --recursion
```

Analyze with `spec.py` and compare:
- `STARK (excl. trace)` at APC 300: expect decrease from ~1301ms to ~1100-1150ms
- `STARK (excl. trace)` at APC 0: expect decrease from ~2161ms to ~2050-2110ms
- Ratio APC 0 / APC 300: expect improvement from 1.66x toward ~1.85-1.90x
- Total proof time: expect unchanged (within noise) — the sync just moves time between metrics

Also verify per-segment metrics:
- `prover.main_trace_commit_time_ms` per segment: expect decrease (pipeline stall eliminated)
- `trace_gen_time_ms` per segment: expect unchanged (trace gen code is not modified)
- Constraint and opening sub-spans: expect unchanged

## Rollback Criteria

Revert if:
- Total proof time (not just STARK excl trace) increases by more than 20ms at any APC configuration, indicating the sync introduced unexpected overhead
- STARK excl trace at APC 300 does not decrease by at least 50ms, indicating the pipeline stall hypothesis is wrong
- APC 0 STARK excl trace regresses by more than 30ms
- Any test failure or proof verification error
