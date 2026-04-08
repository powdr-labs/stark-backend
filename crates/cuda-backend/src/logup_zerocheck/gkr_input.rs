use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::{
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
    DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
};
use p3_field::{Field, PrimeCharacteristicRing};
use tracing::{info, instrument};

use super::errors::InteractionGpuError;
use crate::{
    cuda::logup_zerocheck::{
        frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp, logup_gkr_input_eval,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    prelude::{EF, F},
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub layout_slices: Vec<StackedSlice>,
}

// TODO[jpw]: revisit if this function is needed
pub fn collect_trace_interactions<HS: GpuHashScheme>(
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    layout: &StackedLayout,
) -> Vec<Option<TraceInteractionMeta>> {
    // Pre-group layout slices by trace to avoid repeated scans later.
    let mut slices_by_trace: Vec<Vec<(usize, StackedSlice)>> =
        vec![Vec::new(); ctx.per_trace.len()];
    for &(trace_idx, interaction_idx, ref slice) in &layout.sorted_cols {
        if let Some(entries) = slices_by_trace.get_mut(trace_idx) {
            entries.push((interaction_idx, *slice));
        }
    }

    ctx.per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, (air_idx, _))| {
            let vk = &pk.per_air[*air_idx].vk;
            if !vk.has_interaction() {
                return None;
            }

            let mut layout_entries = vec![None; vk.num_interactions()];
            for (interaction_idx, slice) in &slices_by_trace[trace_idx] {
                if let Some(slot) = layout_entries.get_mut(*interaction_idx) {
                    *slot = Some(*slice);
                }
            }

            let layout_slices = layout_entries
                .into_iter()
                .enumerate()
                .map(|(idx, maybe_slice)| {
                    maybe_slice.unwrap_or_else(|| {
                        panic!(
                            "missing stacked slice for interaction {} of trace {}",
                            idx, trace_idx
                        )
                    })
                })
                .collect_vec();

            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                layout_slices,
            })
        })
        .collect()
}

/// Evaluate interactions from trace evaluation matrices to get (p, q) fractional sumcheck input.
/// Returns leaves buffer (WITHOUT alpha applied) and alpha value to be applied in first tree layer.
#[instrument(name = "prover.rap_constraints.logup_gkr.input_evals", skip_all)]
pub fn log_gkr_input_evals<HS: GpuHashScheme>(
    trace_interactions: &[Option<TraceInteractionMeta>],
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    l_skip: usize,
    alpha_logup: EF,
    d_challenges: &DeviceBuffer<EF>,
    total_leaves: usize,
) -> Result<(DeviceBuffer<Frac<EF>>, EF), InteractionGpuError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok((DeviceBuffer::new(), alpha_logup));
    }

    let leaves = DeviceBuffer::<Frac<EF>>::with_capacity(total_leaves);
    leaves.fill_zero()?; // Correctness: untouched slots must be zero
    let null_preprocessed = DeviceBuffer::<F>::new();

    // ====================================================================
    // Batched GKR input evaluation
    // ====================================================================
    let mut gkr_batched = vec![false; trace_interactions.len()];
    {
        use crate::cuda::logup_zerocheck::{
            gkr_input_eval_batched, GkrInputBlockCtx, GkrInputCtx,
        };
        use openvm_cuda_common::stream::current_stream_sync;

        const GKR_GLOBAL_THRESHOLD: u32 = 10;
        const GKR_TASK_SIZE: usize = 1 << 16;

        // Group traces by GLOBAL
        for is_global in [true, false] {
            // Collect traces for this group
            let group: Vec<(usize, &TraceInteractionMeta)> = trace_interactions
                .iter()
                .enumerate()
                .filter_map(|(idx, opt)| {
                    let meta = opt.as_ref()?;
                    let pk_air = &pk.per_air[meta.air_idx];
                    let bs = pk_air.other_data.interaction_rules.inner.buffer_size;
                    if (bs > GKR_GLOBAL_THRESHOLD) == is_global {
                        Some((idx, meta))
                    } else {
                        None
                    }
                })
                .collect();

            if group.is_empty() {
                continue;
            }

            // Build BlockCtx and trace contexts
            let mut block_ctxs: Vec<GkrInputBlockCtx> = Vec::new();
            let mut trace_ctxs: Vec<GkrInputCtx> = Vec::new();
            let mut _keepalive_intermed: Vec<DeviceBuffer<EF>> = Vec::new();
            let mut _keepalive_partitions: Vec<DeviceBuffer<u64>> = Vec::new();
            let mut _keepalive_pubvals: Vec<DeviceBuffer<F>> = Vec::new();
            // Track lifted traces: (group_local_idx, tmp_buf_idx, meta data for lifting)
            let mut lifted_traces: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
            let mut _keepalive_tmp_output: Vec<DeviceBuffer<Frac<EF>>> = Vec::new();

            for (local_idx, (trace_idx, meta)) in group.iter().enumerate() {
                let air_ctx = &ctx.per_trace[meta.trace_idx].1;
                let pk_air = &pk.per_air[meta.air_idx];
                let rules = &pk_air.other_data.interaction_rules;
                let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();
                let height = air_ctx.height();

                let blocks_per_trace = if is_global {
                    ((GKR_TASK_SIZE + 255) / 256) as u32
                } else {
                    ((height + 255) / 256).max(1) as u32
                };

                for b in 0..blocks_per_trace {
                    block_ctxs.push(GkrInputBlockCtx {
                        local_block_idx_x: b,
                        air_idx: local_idx as u32,
                    });
                }

                // Partition pointers
                let partition_ptrs: Vec<u64> = air_ctx
                    .cached_mains
                    .iter()
                    .map(|c| c.trace.buffer().as_ptr() as u64)
                    .chain(std::iter::once(air_ctx.common_main.buffer().as_ptr() as u64))
                    .collect();
                let d_partition = partition_ptrs.to_device()?;

                let d_pub = if air_ctx.public_values.is_empty() {
                    DeviceBuffer::<F>::new()
                } else {
                    air_ctx.public_values.to_device()?
                };

                let d_intermediates_ptr = if is_global && rules.inner.buffer_size > 0 {
                    let cap = GKR_TASK_SIZE * rules.inner.buffer_size as usize;
                    let buf = DeviceBuffer::<EF>::with_capacity(cap);
                    let ptr = buf.as_mut_ptr();
                    _keepalive_intermed.push(buf);
                    ptr
                } else {
                    std::ptr::null_mut()
                };

                // Output pointer
                let slice = meta.layout_slices.first().unwrap();
                debug_assert_eq!(slice.col_idx, 0);
                let dst_offset = slice.row_idx;
                let lifted_height = max(height, 1 << l_skip);
                let needs_lifting = height != lifted_height;

                let d_output = if needs_lifting {
                    let required = height * num_interactions;
                    let buf = DeviceBuffer::<Frac<EF>>::with_capacity(required);
                    let ptr = buf.as_mut_ptr();
                    let tmp_idx = _keepalive_tmp_output.len();
                    _keepalive_tmp_output.push(buf);
                    lifted_traces.push((
                        local_idx, tmp_idx, dst_offset, lifted_height, num_interactions,
                    ));
                    ptr
                } else {
                    unsafe { leaves.as_mut_ptr().add(dst_offset) }
                };

                let d_preprocessed_ptr = pk_air
                    .preprocessed_data
                    .as_ref()
                    .map(|cd| cd.trace.buffer().as_ptr())
                    .unwrap_or(std::ptr::null());

                let num_rows_per_tile = height.div_ceil(GKR_TASK_SIZE).max(1);

                trace_ctxs.push(GkrInputCtx {
                    d_preprocessed: d_preprocessed_ptr,
                    d_main: d_partition.as_ptr(),
                    d_public_values: d_pub.as_ptr(),
                    d_rules: rules.inner.d_rules.as_raw_ptr(),
                    d_used_nodes: rules.inner.d_used_nodes.as_ptr(),
                    d_pair_idxs: rules.d_pair_idxs.as_ptr(),
                    used_nodes_len: rules.inner.d_used_nodes.len(),
                    permutation_height: height as u32,
                    num_rows_per_tile: num_rows_per_tile as u32,
                    buffer_size: rules.inner.buffer_size,
                    blocks_per_trace,
                    d_intermediates: d_intermediates_ptr,
                    d_output,
                });

                _keepalive_partitions.push(d_partition);
                _keepalive_pubvals.push(d_pub);
            }

            let total_blocks = block_ctxs.len() as u32;
            let d_block_ctxs = block_ctxs.to_device()?;
            let d_trace_ctxs = trace_ctxs.to_device()?;

            unsafe {
                gkr_input_eval_batched(
                    is_global,
                    &d_block_ctxs,
                    &d_trace_ctxs,
                    d_challenges,
                    total_blocks,
                )?;
            }

            // Sync before lifting
            current_stream_sync()?;

            // Post-kernel lifting for lifted traces
            for &(_local_idx, tmp_idx, dst_offset, lifted_height, num_interactions) in
                &lifted_traces
            {
                let tmp_buf = &_keepalive_tmp_output[tmp_idx];
                let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };
                let height = tmp_buf.len() / num_interactions;
                let norm_factor_denom = lifted_height / height;
                let norm_factor = F::from_usize(norm_factor_denom).inverse();
                unsafe {
                    frac_vector_scalar_multiply_ext_fp(
                        tmp_buf.as_mut_ptr(),
                        norm_factor,
                        tmp_buf.len() as u32,
                    )?;
                    frac_matrix_vertically_repeat(
                        leaves_ptr,
                        tmp_buf.as_ptr(),
                        num_interactions as u32,
                        lifted_height as u32,
                        height as u32,
                    )?;
                }
            }

            // Mark all traces in this group as batched
            for (trace_idx, _) in &group {
                gkr_batched[*trace_idx] = true;
            }
        }

        let batched_count = gkr_batched.iter().filter(|&&b| b).count();
        if batched_count > 0 {
            info!("batched gkr input: {batched_count} traces");
        }
    }

    // Sequential fallback for any non-batched traces
    let mut d_partition_ptrs = DeviceBuffer::<u64>::new();
    let mut tmp = DeviceBuffer::<Frac<EF>>::new();
    for (meta_idx, meta) in trace_interactions.iter().enumerate().filter_map(|(i, m)| m.as_ref().map(|m| (i, m))) {
        if gkr_batched[meta_idx] {
            continue;
        }
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];

        let preprocessed_matrix = pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| &committed.trace);

        let mut partitioned_main = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
        for committed in &air_ctx.cached_mains {
            partitioned_main.push(&committed.trace);
        }
        partitioned_main.push(&air_ctx.common_main);

        let rules = &pk_air.other_data.interaction_rules;
        let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_preprocessed);
        let d_public_values = if air_ctx.public_values.is_empty() {
            DeviceBuffer::<F>::new()
        } else {
            air_ctx.public_values.to_device()?
        };

        let height = air_ctx.height();
        debug_assert_eq!(height, partitioned_main[0].height());
        let partition_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect_vec();
        if partition_ptrs.len() > d_partition_ptrs.len() {
            d_partition_ptrs = DeviceBuffer::with_capacity(partition_ptrs.len());
        }
        partition_ptrs.copy_to(&mut d_partition_ptrs)?;

        let buffer_size = rules.inner.buffer_size;
        // TODO[jpw]: remove magic 10
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size as usize)
        } else {
            DeviceBuffer::<EF>::with_capacity(1)
        };

        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionGpuError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);
        // SAFETY: by definition of interactions stacked layout, `leaves` has enough capacity
        let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };

        let trace_output = if height != lifted_height {
            let required = height * num_interactions;
            if required > tmp.len() {
                tmp = DeviceBuffer::with_capacity(required);
            }
            tmp.as_mut_ptr()
        } else {
            leaves_ptr
        };
        unsafe {
            logup_gkr_input_eval(
                is_global,
                trace_output,
                d_preprocessed,
                &d_partition_ptrs,
                &d_public_values,
                d_challenges,
                &intermediates,
                &rules.inner.d_rules,
                &rules.inner.d_used_nodes,
                &rules.d_pair_idxs,
                height as u32,
                num_rows_per_tile as u32,
            )?;
        }
        if height != lifted_height {
            debug_assert_eq!(lifted_height % height, 0);
            debug_assert!(!tmp.is_empty());
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_usize(norm_factor_denom).inverse();
            unsafe {
                // SAFETY: scaling within buffer length
                frac_vector_scalar_multiply_ext_fp(
                    tmp.as_mut_ptr(),
                    norm_factor,
                    tmp.len() as u32,
                )?;
                // SAFETY: stacked interaction layout is defined with respect to lifted height so
                // lifting (i.e., vertically repeating) stays within bounds
                frac_matrix_vertically_repeat(
                    leaves_ptr,
                    tmp.as_ptr(),
                    num_interactions as u32,
                    lifted_height as u32,
                    height as u32,
                )?;
            }
        }
    }

    // NOTE: alpha is NO LONGER applied here - it will be fused into the first tree layer
    // in fractional_sumcheck_gpu for better performance (eliminates one memory pass)
    Ok((leaves, alpha_logup))
}
