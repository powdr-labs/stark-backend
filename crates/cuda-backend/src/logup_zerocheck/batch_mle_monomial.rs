//! Batched monomial-based MLE evaluation for zerocheck and logup.
//!
//! This module provides batch evaluators for monomial evaluations across multiple AIRs,
//! enabling efficient GPU kernel launches that process multiple traces in a single launch.

use openvm_cuda_common::{
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    error::{CudaError, MemCopyError},
};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceMultiStarkProvingKey};
use p3_field::PrimeCharacteristicRing;
use tracing::debug;

use crate::{
    cuda::logup_zerocheck::{
        logup_monomial_batched, precompute_lambda_combinations,
        precompute_logup_denom_combinations, precompute_logup_numer_combinations,
        scatter_fpext_blocks, scatter_frac_blocks, warp_logup_monomial_batched,
        warp_zerocheck_monomial_batched, zerocheck_monomial_batched,
        zerocheck_monomial_par_y_batched, BlockCtx, EvalCoreCtx, LogupMonomialCommonCtx,
        LogupMonomialCtx, MonomialAirCtx,
    },
    error::KernelError,
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::batch_mle::TraceCtx,
    prelude::EF,
};

const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;

/// Upload a Vec to device, returning an empty DeviceBuffer if the Vec is empty.
fn to_device_or_empty<T>(v: Vec<T>) -> Result<DeviceBuffer<T>, MemCopyError> {
    if v.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        v.to_device()
    }
}

/// Returns true if the trace can use the monomial evaluation path.
///
/// A trace is eligible if it has constraints and the AIR has expanded monomials.
pub(crate) fn trace_has_monomials<HS: GpuHashScheme>(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
) -> bool {
    trace.has_constraints
        && pk.per_air[trace.air_idx]
            .other_data
            .zerocheck_monomials
            .as_ref()
            .map(|m| m.num_monomials > 0)
            .unwrap_or(false)
}

/// Get the number of monomials for a trace. Returns 0 if the trace has no monomials.
pub(crate) fn get_num_monomials<HS: GpuHashScheme>(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
) -> u32 {
    pk.per_air[trace.air_idx]
        .other_data
        .zerocheck_monomials
        .as_ref()
        .map(|m| m.num_monomials)
        .unwrap_or(0)
}

/// Get the rules_len for a trace's zerocheck DAG.
pub(crate) fn get_zerocheck_rules_len<HS: GpuHashScheme>(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
) -> usize {
    pk.per_air[trace.air_idx]
        .other_data
        .zerocheck_mle
        .inner
        .d_rules
        .len()
}

/// Precompute lambda combinations for a single AIR's monomials.
///
/// Returns a buffer of length `num_monomials` where each element is
/// `sum_l(coefficient_l * lambda_pows[constraint_idx_l])` for that monomial.
///
/// The AIR must have nonempty monomials.
pub(crate) fn compute_lambda_combinations<HS: GpuHashScheme>(
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    air_idx: usize,
    lambda_pows: &DeviceBuffer<EF>,
) -> Result<DeviceBuffer<EF>, CudaError> {
    let monomials = pk.per_air[air_idx]
        .other_data
        .zerocheck_monomials
        .as_ref()
        .expect("AIR must have monomials");
    let mut buf = DeviceBuffer::<EF>::with_capacity(monomials.num_monomials as usize);
    unsafe {
        precompute_lambda_combinations(
            &mut buf,
            monomials.d_headers.as_ptr(),
            monomials.d_lambda_terms.as_ptr(),
            lambda_pows,
            monomials.num_monomials,
        )?;
    }
    Ok(buf)
}

/// Batch evaluator for monomial-based zerocheck MLE evaluation.
///
/// Pre-builds GPU contexts for all traces, then evaluates using a two-path strategy:
/// - **Warp path**: traces with `num_y <= 32` and `num_monomials <= 32` use a warp-per-trace
///   kernel where each thread handles one y-value and loops over all monomials.
/// - **Block path**: remaining traces use the existing block kernel with block reduction.
///
/// The caller must filter traces using [`trace_has_monomials`] before constructing.
/// The batch must contain at least one trace.
pub(crate) struct ZerocheckMonomialBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    // Warp path: air_ctxs in original order, trace_ids/output_offsets for scatter
    warp_air_ctxs: DeviceBuffer<MonomialAirCtx>,
    warp_trace_ids: DeviceBuffer<u32>,
    warp_output_offsets: DeviceBuffer<u32>,
    num_warp_traces: u32,
    // Block path: block-local air_ctxs + block_ctxs + air_offsets
    block_air_ctxs: DeviceBuffer<MonomialAirCtx>,
    block_ctxs: DeviceBuffer<BlockCtx>,
    block_air_offsets: DeviceBuffer<u32>,
    block_output_offsets: DeviceBuffer<u32>,
    num_block_traces: u32,
}

fn build_monomial_air_ctx(t: &TraceCtx, monomials: &crate::pkey::ZerocheckMonomials, lc: &DeviceBuffer<EF>) -> MonomialAirCtx {
    MonomialAirCtx {
        d_headers: monomials.d_headers.as_ptr(),
        d_variables: monomials.d_variables.as_ptr(),
        d_lambda_combinations: lc.as_ptr(),
        num_monomials: monomials.num_monomials,
        eval_ctx: EvalCoreCtx {
            d_selectors: t.sels_ptr,
            d_preprocessed: t.prep_ptr,
            d_main: t.main_ptrs_dev.as_ptr(),
            d_public: t.public_ptr,
        },
        d_eq_xi: t.eq_xi_ptr,
        num_y: t.num_y,
    }
}

impl<'a> ZerocheckMonomialBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `lambda_combinations` must contain one buffer per trace (in iteration order),
    /// each precomputed via [`compute_lambda_combinations`].
    ///
    /// # Panics
    ///
    /// Panics if `traces` is empty or if `lambda_combinations` length doesn't match.
    pub fn new<HS: GpuHashScheme>(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        lambda_combinations: &[&DeviceBuffer<EF>],
    ) -> Result<Self, MemCopyError> {
        let traces: Vec<_> = traces.into_iter().collect();
        assert!(
            !traces.is_empty(),
            "ZerocheckMonomialBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            lambda_combinations.len(),
            "lambda_combinations must have one buffer per trace"
        );

        // Partition traces into warp-eligible and block-eligible, building air_ctxs for each
        let mut warp_air_ctxs_h: Vec<MonomialAirCtx> = Vec::new();
        let mut warp_trace_ids_h: Vec<u32> = Vec::new();
        let mut warp_output_offsets_h: Vec<u32> = Vec::new();

        let mut block_air_ctxs_h: Vec<MonomialAirCtx> = Vec::new();
        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut block_air_offsets_h: Vec<u32> = vec![0];
        let mut block_output_offsets_h: Vec<u32> = Vec::new();

        for (i, (t, lc)) in traces.iter().zip(lambda_combinations).enumerate() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .zerocheck_monomials
                .as_ref()
                .unwrap();

            let air_ctx = build_monomial_air_ctx(t, monomials, lc);

            if t.num_y <= WARP_SIZE && monomials.num_monomials <= WARP_SIZE {
                // Warp path: store air_ctx, trace_id indexes into warp_air_ctxs
                let warp_local = warp_air_ctxs_h.len() as u32;
                warp_air_ctxs_h.push(air_ctx);
                warp_trace_ids_h.push(warp_local);
                warp_output_offsets_h.push(i as u32);
            } else {
                // Block path: block-local air_ctxs + block assignments
                let block_local = block_air_ctxs_h.len() as u32;
                block_air_ctxs_h.push(air_ctx);
                let mono_blocks = monomials.num_monomials.div_ceil(THREADS_PER_BLOCK);
                let total_blocks = mono_blocks * t.num_y;
                for local_idx in 0..total_blocks {
                    block_ctxs_h.push(BlockCtx {
                        local_block_idx_x: local_idx,
                        air_idx: block_local,
                    });
                }
                block_air_offsets_h.push(block_ctxs_h.len() as u32);
                block_output_offsets_h.push(i as u32);
            }
        }

        let num_warp_traces = warp_air_ctxs_h.len() as u32;
        let num_block_traces = block_air_ctxs_h.len() as u32;

        // Upload to device (handle empty partitions)
        let warp_air_ctxs = to_device_or_empty(warp_air_ctxs_h)?;
        let warp_trace_ids = to_device_or_empty(warp_trace_ids_h)?;
        let warp_output_offsets = to_device_or_empty(warp_output_offsets_h)?;
        let block_air_ctxs = to_device_or_empty(block_air_ctxs_h)?;
        let block_ctxs = to_device_or_empty(block_ctxs_h)?;
        let block_air_offsets = to_device_or_empty(block_air_offsets_h)?;
        let block_output_offsets = to_device_or_empty(block_output_offsets_h)?;

        debug!(
            num_airs = traces.len(),
            num_warp_traces,
            num_block_traces,
            num_block_ctxs = block_ctxs.len(),
            "ZerocheckMonomialBatch created"
        );

        Ok(Self {
            traces,
            warp_air_ctxs,
            warp_trace_ids,
            warp_output_offsets,
            num_warp_traces,
            block_air_ctxs,
            block_ctxs,
            block_air_offsets,
            block_output_offsets,
            num_block_traces,
        })
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    pub fn evaluate(&self, num_x: u32) -> Result<DeviceBuffer<EF>, KernelError> {
        let total_traces = self.traces.len();
        let mut output = DeviceBuffer::<EF>::with_capacity(total_traces * num_x as usize);

        // Launch 1: warp kernel for small traces (scatter output via output_offsets)
        if self.num_warp_traces > 0 {
            unsafe {
                warp_zerocheck_monomial_batched(
                    &mut output,
                    &self.warp_air_ctxs,
                    &self.warp_trace_ids,
                    &self.warp_output_offsets,
                    self.num_warp_traces,
                    num_x,
                )?;
            }
        }

        // Launch 2: block kernel for larger traces (contiguous output, then scatter)
        if self.num_block_traces > 0 {
            let num_blocks = self.block_ctxs.len();
            let mut tmp_sums = DeviceBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
            let mut block_output =
                DeviceBuffer::<EF>::with_capacity(self.num_block_traces as usize * num_x as usize);

            unsafe {
                zerocheck_monomial_batched(
                    &mut tmp_sums,
                    &mut block_output,
                    &self.block_ctxs,
                    &self.block_air_ctxs,
                    &self.block_air_offsets,
                    num_blocks as u32,
                    num_x,
                    self.num_block_traces,
                    THREADS_PER_BLOCK,
                )?;
            }

            // Scatter block results into correct positions in the shared output
            unsafe {
                scatter_fpext_blocks(
                    &mut output,
                    &block_output,
                    &self.block_output_offsets,
                    self.num_block_traces,
                    num_x,
                )?;
            }
        }

        Ok(output)
    }
}

// Constants for par-y kernel
const THREADS_PER_BLOCK_PAR_Y: u32 = 128;
const DEFAULT_MAX_MONOMIALS_PER_THREAD: u32 = 64;
const WAVES_TARGET: u32 = 4;

/// Batch evaluator for monomial-based zerocheck MLE evaluation, parallelizing over y_int.
///
/// This variant is optimized for traces with high `num_y`: each thread handles one y_int
/// and loops over a chunk of monomials. The chunk size is auto-tuned based on SM count.
///
/// The caller must filter traces using [`trace_has_monomials`] before constructing.
/// The batch must contain at least one trace.
pub(crate) struct ZerocheckMonomialParYBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: DeviceBuffer<BlockCtx>,
    air_ctxs: DeviceBuffer<MonomialAirCtx>,
    air_offsets: DeviceBuffer<u32>,
    num_blocks: u32,
    chunk_size: u32,
}

impl<'a> ZerocheckMonomialParYBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `lambda_combinations` must contain one buffer per trace (in iteration order),
    /// each precomputed via [`compute_lambda_combinations`].
    ///
    /// The `sm_count` and `num_x` parameters are used to auto-tune the chunk size
    /// for optimal SM utilization.
    ///
    /// # Panics
    ///
    /// Panics if `traces` is empty or if `lambda_combinations` length doesn't match.
    #[allow(clippy::too_many_arguments)]
    pub fn new<HS: GpuHashScheme>(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        lambda_combinations: &[&DeviceBuffer<EF>],
        sm_count: u32,
        num_x: u32,
        max_monomials_per_thread: Option<u32>,
    ) -> Result<Self, MemCopyError> {
        let traces: Vec<_> = traces.into_iter().collect();
        assert!(
            !traces.is_empty(),
            "ZerocheckMonomialParYBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            lambda_combinations.len(),
            "lambda_combinations must have one buffer per trace"
        );

        let threads_per_block = THREADS_PER_BLOCK_PAR_Y;
        let max_mono_per_thread =
            max_monomials_per_thread.unwrap_or(DEFAULT_MAX_MONOMIALS_PER_THREAD);

        // First pass: collect per-AIR info
        let mut per_air_info: Vec<(u32, u32)> = Vec::new(); // (y_blocks, num_monomials)
        let mut max_monomials = 0u32;

        for t in traces.iter() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .zerocheck_monomials
                .as_ref()
                .expect("AIR with constraints must have monomials");

            let y_blocks = t.num_y.div_ceil(threads_per_block);
            max_monomials = max_monomials.max(monomials.num_monomials);
            per_air_info.push((y_blocks, monomials.num_monomials));
        }

        // Determine chunk_size based on SM utilization and cap
        // We want: total_blocks * num_x >= sm_count * WAVES_TARGET
        // total_blocks = sum of (y_blocks * air_mono_chunks) per AIR
        // where air_mono_chunks = ceil(num_monomials / chunk_size)
        //
        // Also: chunk_size <= max_mono_per_thread
        //
        // Start with chunk_size = max_mono_per_thread and adjust if needed
        let target_blocks = sm_count * WAVES_TARGET;

        // Initial estimate: use max_mono_per_thread as chunk_size
        let mut chunk_size = max_mono_per_thread;
        loop {
            let total_blocks: u32 = per_air_info
                .iter()
                .map(|(y_blocks, num_mono)| {
                    let air_mono_chunks = num_mono.div_ceil(chunk_size);
                    y_blocks * air_mono_chunks
                })
                .sum();

            if total_blocks * num_x >= target_blocks || chunk_size <= 1 {
                break;
            }
            // Need more blocks: reduce chunk_size
            chunk_size = (chunk_size / 2).max(1);
        }

        // Build block_ctxs with per-AIR encoding
        // Encoding: local_block_idx_x = y_block * air_mono_chunks + mono_chunk
        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, (y_blocks, num_mono)) in per_air_info.iter().enumerate() {
            let air_mono_chunks = num_mono.div_ceil(chunk_size);

            for y_block in 0..*y_blocks {
                for mono_chunk in 0..air_mono_chunks {
                    let local_idx = y_block * air_mono_chunks + mono_chunk;
                    block_ctxs_h.push(BlockCtx {
                        local_block_idx_x: local_idx,
                        air_idx: local_air as u32,
                    });
                }
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        let num_blocks = block_ctxs_h.len() as u32;

        // Build MonomialAirCtx for each trace
        let air_ctxs_h: Vec<MonomialAirCtx> = traces
            .iter()
            .zip(lambda_combinations)
            .map(|(t, lc)| {
                let monomials = pk.per_air[t.air_idx]
                    .other_data
                    .zerocheck_monomials
                    .as_ref()
                    .unwrap();

                let eval_ctx = EvalCoreCtx {
                    d_selectors: t.sels_ptr,
                    d_preprocessed: t.prep_ptr,
                    d_main: t.main_ptrs_dev.as_ptr(),
                    d_public: t.public_ptr,
                };

                MonomialAirCtx {
                    d_headers: monomials.d_headers.as_ptr(),
                    d_variables: monomials.d_variables.as_ptr(),
                    d_lambda_combinations: lc.as_ptr(),
                    num_monomials: monomials.num_monomials,
                    eval_ctx,
                    d_eq_xi: t.eq_xi_ptr,
                    num_y: t.num_y,
                }
            })
            .collect();

        // Upload to device
        let block_ctxs = block_ctxs_h.to_device()?;
        let air_ctxs = air_ctxs_h.to_device()?;
        let air_offsets = air_offsets.to_device()?;

        debug!(
            num_airs = traces.len(),
            num_blocks, chunk_size, max_monomials, "ZerocheckMonomialParYBatch created"
        );

        Ok(Self {
            traces,
            block_ctxs,
            air_ctxs,
            air_offsets,
            num_blocks,
            chunk_size,
        })
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    /// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
    pub fn evaluate(&self, num_x: u32) -> Result<DeviceBuffer<EF>, KernelError> {
        let num_airs = self.air_ctxs.len();

        debug!(
            num_blocks = %self.num_blocks,
            %num_x,
            %num_airs,
            chunk_size = %self.chunk_size,
            "zerocheck_monomial_par_y_batched"
        );

        let mut tmp_sums =
            DeviceBuffer::<EF>::with_capacity(self.num_blocks as usize * num_x as usize);
        let mut output = DeviceBuffer::<EF>::with_capacity(num_airs * num_x as usize);

        debug_assert_eq!(
            self.air_offsets.len(),
            num_airs + 1,
            "air_offsets must have num_airs + 1 elements"
        );
        // SAFETY: All device pointers in block_ctxs and air_ctxs were constructed from
        // valid DeviceBuffers that outlive this call (TraceCtx references, pk monomial data,
        // lambda_combinations). The air_offsets buffer has length num_airs + 1 as required.
        unsafe {
            zerocheck_monomial_par_y_batched(
                &mut tmp_sums,
                &mut output,
                &self.block_ctxs,
                &self.air_ctxs,
                &self.air_offsets,
                self.num_blocks,
                num_x,
                num_airs as u32,
                self.chunk_size,
                THREADS_PER_BLOCK_PAR_Y,
            )?;
        }

        Ok(output)
    }
}

// ============================================================================
// LOGUP MONOMIAL EVALUATION
// ============================================================================

/// Precomputed logup combinations for a single AIR.
pub struct LogupCombinations {
    pub d_numer_combinations: DeviceBuffer<EF>,
    pub d_denom_combinations: DeviceBuffer<EF>,
    pub bus_term_sum: EF,
}

/// Precompute logup combinations for a single AIR's interaction monomials.
///
/// The AIR must have nonempty interaction monomials.
pub(crate) fn compute_logup_combinations<HS: GpuHashScheme>(
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    air_idx: usize,
    d_beta_pows: &DeviceBuffer<EF>,
    d_eq_3bs: &DeviceBuffer<EF>,
    eq_3bs_host: &[EF],
    beta_pows_host: &[EF],
) -> Result<LogupCombinations, CudaError> {
    let monomials = pk.per_air[air_idx]
        .other_data
        .interaction_monomials
        .as_ref()
        .expect("AIR must have interaction monomials");

    // Precompute numerator combinations: sum_i(coeff_i * eq_3bs[interaction_idx_i])
    let mut d_numer_combinations = if monomials.num_numer_monomials > 0 {
        DeviceBuffer::<EF>::with_capacity(monomials.num_numer_monomials as usize)
    } else {
        DeviceBuffer::new()
    };
    if monomials.num_numer_monomials > 0 {
        unsafe {
            precompute_logup_numer_combinations(
                &mut d_numer_combinations,
                monomials.d_numer_headers.as_ptr(),
                monomials.d_numer_terms.as_ptr(),
                d_eq_3bs,
                monomials.num_numer_monomials,
            )?;
        }
    }

    // Precompute denominator combinations: sum_i(coeff_i * beta_pows[field_idx_i] *
    // eq_3bs[interaction_idx_i])
    let mut d_denom_combinations = if monomials.num_denom_monomials > 0 {
        DeviceBuffer::<EF>::with_capacity(monomials.num_denom_monomials as usize)
    } else {
        DeviceBuffer::new()
    };
    if monomials.num_denom_monomials > 0 {
        unsafe {
            precompute_logup_denom_combinations(
                &mut d_denom_combinations,
                monomials.d_denom_headers.as_ptr(),
                monomials.d_denom_terms.as_ptr(),
                d_beta_pows,
                d_eq_3bs,
                monomials.num_denom_monomials,
            )?;
        }
    }

    // Compute bus_term_sum on CPU: sum_i(beta_pows[message_len_i] * (bus_idx[i]+1) * eq_3bs[i])
    let interactions = &pk.per_air[air_idx].vk.symbolic_constraints.interactions;
    debug_assert_eq!(
        interactions.len(),
        eq_3bs_host.len(),
        "interaction count must match eq_3bs"
    );
    let mut bus_term_sum = EF::ZERO;
    for (i, interaction) in interactions.iter().enumerate() {
        let beta_len = beta_pows_host[interaction.message.len()];
        let bus_idx = interaction.bus_index as u32;
        bus_term_sum += beta_len * EF::from_u32(bus_idx + 1) * eq_3bs_host[i];
    }

    Ok(LogupCombinations {
        d_numer_combinations,
        d_denom_combinations,
        bus_term_sum,
    })
}

const THREADS_PER_BLOCK_LOGUP: u32 = 128;

/// Batch evaluator for logup monomial MLE evaluation.
///
/// Uses a two-path strategy like [`ZerocheckMonomialBatch`]:
/// - **Warp path**: traces with small num_y and few monomials use a warp-per-trace kernel.
/// - **Block path**: remaining traces use the existing block kernel with block reduction.
pub(crate) struct LogupMonomialBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    // Warp path
    warp_common_ctxs: DeviceBuffer<LogupMonomialCommonCtx>,
    warp_numer_ctxs: DeviceBuffer<LogupMonomialCtx>,
    warp_denom_ctxs: DeviceBuffer<LogupMonomialCtx>,
    warp_trace_ids: DeviceBuffer<u32>,
    warp_output_offsets: DeviceBuffer<u32>,
    num_warp_traces: u32,
    // Block path
    block_common_ctxs: DeviceBuffer<LogupMonomialCommonCtx>,
    block_numer_ctxs: DeviceBuffer<LogupMonomialCtx>,
    block_denom_ctxs: DeviceBuffer<LogupMonomialCtx>,
    block_ctxs: DeviceBuffer<BlockCtx>,
    block_air_offsets: DeviceBuffer<u32>,
    block_output_offsets: DeviceBuffer<u32>,
    block_num_blocks: u32,
    num_block_traces: u32,
}

impl<'a> LogupMonomialBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `logup_combinations` must contain one `LogupCombinations` per trace (in iteration order),
    /// each precomputed via [`compute_logup_combinations`].
    ///
    /// # Panics
    ///
    /// Panics if `traces` is empty or if `logup_combinations` length doesn't match.
    pub fn new<HS: GpuHashScheme>(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        logup_combinations: &[&LogupCombinations],
    ) -> Result<Self, MemCopyError> {
        let traces: Vec<_> = traces.into_iter().collect();
        assert!(
            !traces.is_empty(),
            "LogupMonomialBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            logup_combinations.len(),
            "logup_combinations must have one entry per trace"
        );

        let threads_per_block = THREADS_PER_BLOCK_LOGUP;

        // Partition into warp-eligible and block-eligible
        let mut warp_common_h: Vec<LogupMonomialCommonCtx> = Vec::new();
        let mut warp_numer_h: Vec<LogupMonomialCtx> = Vec::new();
        let mut warp_denom_h: Vec<LogupMonomialCtx> = Vec::new();
        let mut warp_trace_ids_h: Vec<u32> = Vec::new();
        let mut warp_output_offsets_h: Vec<u32> = Vec::new();

        let mut block_common_h: Vec<LogupMonomialCommonCtx> = Vec::new();
        let mut block_numer_h: Vec<LogupMonomialCtx> = Vec::new();
        let mut block_denom_h: Vec<LogupMonomialCtx> = Vec::new();
        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut block_air_offsets_h: Vec<u32> = vec![0];
        let mut block_output_offsets_h: Vec<u32> = Vec::new();

        for (i, (t, lc)) in traces.iter().zip(logup_combinations).enumerate() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .interaction_monomials
                .as_ref()
                .unwrap();
            let max_monomials = monomials
                .num_numer_monomials
                .max(monomials.num_denom_monomials);

            let eval_ctx = EvalCoreCtx {
                d_selectors: t.sels_ptr,
                d_preprocessed: t.prep_ptr,
                d_main: t.main_ptrs_dev.as_ptr(),
                d_public: t.public_ptr,
            };

            let numer_ctx = LogupMonomialCtx {
                d_headers: monomials.d_numer_headers.as_ptr(),
                d_variables: monomials.d_numer_variables.as_ptr(),
                d_combinations: lc.d_numer_combinations.as_ptr(),
                num_monomials: monomials.num_numer_monomials,
            };
            let denom_ctx = LogupMonomialCtx {
                d_headers: monomials.d_denom_headers.as_ptr(),
                d_variables: monomials.d_denom_variables.as_ptr(),
                d_combinations: lc.d_denom_combinations.as_ptr(),
                num_monomials: monomials.num_denom_monomials,
            };

            if t.num_y <= WARP_SIZE && max_monomials <= WARP_SIZE {
                let warp_local = warp_common_h.len() as u32;
                let mono_blocks = max_monomials.div_ceil(threads_per_block).max(1);
                warp_common_h.push(LogupMonomialCommonCtx {
                    eval_ctx,
                    d_eq_xi: t.eq_xi_ptr,
                    bus_term_sum: lc.bus_term_sum,
                    num_y: t.num_y,
                    mono_blocks,
                });
                warp_numer_h.push(numer_ctx);
                warp_denom_h.push(denom_ctx);
                warp_trace_ids_h.push(warp_local);
                warp_output_offsets_h.push(i as u32);
            } else {
                let block_local = block_common_h.len() as u32;
                let mono_blocks = max_monomials.div_ceil(threads_per_block).max(1);
                block_common_h.push(LogupMonomialCommonCtx {
                    eval_ctx,
                    d_eq_xi: t.eq_xi_ptr,
                    bus_term_sum: lc.bus_term_sum,
                    num_y: t.num_y,
                    mono_blocks,
                });
                block_numer_h.push(numer_ctx);
                block_denom_h.push(denom_ctx);
                for y_int in 0..t.num_y {
                    for mono_block in 0..mono_blocks {
                        block_ctxs_h.push(BlockCtx {
                            local_block_idx_x: y_int * mono_blocks + mono_block,
                            air_idx: block_local,
                        });
                    }
                }
                block_air_offsets_h.push(block_ctxs_h.len() as u32);
                block_output_offsets_h.push(i as u32);
            }
        }

        let num_warp_traces = warp_common_h.len() as u32;
        let num_block_traces = block_common_h.len() as u32;
        let block_num_blocks = block_ctxs_h.len() as u32;

        // Upload to device (handle empty partitions)
        let warp_common_ctxs = to_device_or_empty(warp_common_h)?;
        let warp_numer_ctxs = to_device_or_empty(warp_numer_h)?;
        let warp_denom_ctxs = to_device_or_empty(warp_denom_h)?;
        let warp_trace_ids = to_device_or_empty(warp_trace_ids_h)?;
        let warp_output_offsets = to_device_or_empty(warp_output_offsets_h)?;
        let block_common_ctxs = to_device_or_empty(block_common_h)?;
        let block_numer_ctxs = to_device_or_empty(block_numer_h)?;
        let block_denom_ctxs = to_device_or_empty(block_denom_h)?;
        let block_ctxs = to_device_or_empty(block_ctxs_h)?;
        let block_air_offsets = to_device_or_empty(block_air_offsets_h)?;
        let block_output_offsets = to_device_or_empty(block_output_offsets_h)?;

        debug!(
            num_airs = traces.len(),
            num_warp_traces,
            num_block_traces,
            block_num_blocks,
            "LogupMonomialBatch created"
        );

        Ok(Self {
            traces,
            warp_common_ctxs,
            warp_numer_ctxs,
            warp_denom_ctxs,
            warp_trace_ids,
            warp_output_offsets,
            num_warp_traces,
            block_common_ctxs,
            block_numer_ctxs,
            block_denom_ctxs,
            block_ctxs,
            block_air_offsets,
            block_output_offsets,
            block_num_blocks,
            num_block_traces,
        })
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` FracExt elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    pub fn evaluate(&self, num_x: u32) -> Result<DeviceBuffer<Frac<EF>>, KernelError> {
        let total_traces = self.traces.len();
        let mut output = DeviceBuffer::<Frac<EF>>::with_capacity(total_traces * num_x as usize);

        // Launch 1: warp kernel for small traces (scatter output)
        if self.num_warp_traces > 0 {
            unsafe {
                warp_logup_monomial_batched(
                    &mut output,
                    &self.warp_common_ctxs,
                    &self.warp_numer_ctxs,
                    &self.warp_denom_ctxs,
                    &self.warp_trace_ids,
                    &self.warp_output_offsets,
                    self.num_warp_traces,
                    num_x,
                )?;
            }
        }

        // Launch 2: block kernel for larger traces (contiguous output, then scatter)
        if self.num_block_traces > 0 {
            let num_blocks = self.block_num_blocks;
            let mut tmp_sums =
                DeviceBuffer::<Frac<EF>>::with_capacity(num_blocks as usize * num_x as usize);
            let mut block_output =
                DeviceBuffer::<Frac<EF>>::with_capacity(self.num_block_traces as usize * num_x as usize);

            unsafe {
                logup_monomial_batched(
                    &mut tmp_sums,
                    &mut block_output,
                    &self.block_ctxs,
                    &self.block_common_ctxs,
                    &self.block_numer_ctxs,
                    &self.block_denom_ctxs,
                    &self.block_air_offsets,
                    num_blocks,
                    num_x,
                    self.num_block_traces,
                    THREADS_PER_BLOCK_LOGUP,
                )?;
            }

            // Scatter block results into correct positions in the shared output
            unsafe {
                scatter_frac_blocks(
                    &mut output,
                    &block_output,
                    &self.block_output_offsets,
                    self.num_block_traces,
                    num_x,
                )?;
            }
        }

        Ok(output)
    }
}
