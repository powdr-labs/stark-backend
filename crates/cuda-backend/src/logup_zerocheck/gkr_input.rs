use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::{
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
    DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
};
use p3_field::{Field, PrimeCharacteristicRing};
use tracing::instrument;

use super::errors::InteractionGpuError;
use crate::{
    cuda::logup_zerocheck::{
        frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp,
        logup_gkr_input_eval, logup_gkr_input_eval_batched, GkrBlockCtx, GkrInputCtx,
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

/// Per-AIR metadata collected in the preparation pass, used for batched dispatch and post-processing.
struct AirPrepData {
    is_global: bool,
    height: usize,
    lifted_height: usize,
    num_interactions: usize,
    dst_offset: usize,
    /// Offset into the shared lift tmp buffer (0 if no lifting needed).
    lift_tmp_offset: usize,
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
    leaves.fill_zero()?;
    let null_preprocessed = DeviceBuffer::<F>::new();

    // ── Pass 1: Prepare per-AIR data, compute lift tmp offsets ──
    let mut air_prep: Vec<AirPrepData> = Vec::new();
    let mut total_lift_tmp: usize = 0;

    // Keep device buffers alive until after kernel launch.
    let mut keepalive_partition_ptrs: Vec<DeviceBuffer<u64>> = Vec::new();
    let mut keepalive_public_values: Vec<DeviceBuffer<F>> = Vec::new();
    let mut keepalive_intermediates: Vec<DeviceBuffer<EF>> = Vec::new();

    // Collect per-AIR metadata (first pass: no kernel launch yet)
    struct AirDeviceData {
        d_preprocessed_ptr: *const F,
        d_partition_ptrs_ptr: *const u64,
        d_public_values_ptr: *const F,
        d_intermediates_ptr: *mut EF,
        d_rules_ptr: *const std::ffi::c_void,
        d_used_nodes_ptr: *const usize,
        d_pair_idxs_ptr: *const u32,
        used_nodes_len: usize,
        num_blocks: usize,
        task_stride: u32,
        num_rows_per_tile: u32,
    }
    let mut air_device_data: Vec<AirDeviceData> = Vec::new();

    for meta in trace_interactions.iter().flatten() {
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
        let mut d_partition_ptrs = DeviceBuffer::with_capacity(partition_ptrs.len());
        partition_ptrs.copy_to(&mut d_partition_ptrs)?;

        let buffer_size = rules.inner.buffer_size;
        let is_global = buffer_size > 10;

        let task_count = if is_global {
            TASK_SIZE as usize
        } else {
            height
        };
        let num_blocks = task_count.div_ceil(256);
        let task_stride = (num_blocks * 256) as u32;
        let intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size as usize)
        } else {
            DeviceBuffer::<EF>::with_capacity(1)
        };
        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1) as u32;

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionGpuError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);

        let lift_tmp_offset = if height != lifted_height {
            let offset = total_lift_tmp;
            total_lift_tmp += height * num_interactions;
            offset
        } else {
            0
        };

        air_prep.push(AirPrepData {
            is_global,
            height,
            lifted_height,
            num_interactions,
            dst_offset,
            lift_tmp_offset,
        });
        air_device_data.push(AirDeviceData {
            d_preprocessed_ptr: d_preprocessed.as_ptr(),
            d_partition_ptrs_ptr: d_partition_ptrs.as_ptr(),
            d_public_values_ptr: d_public_values.as_ptr(),
            d_intermediates_ptr: intermediates.as_mut_ptr(),
            d_rules_ptr: rules.inner.d_rules.as_raw_ptr(),
            d_used_nodes_ptr: rules.inner.d_used_nodes.as_ptr(),
            d_pair_idxs_ptr: rules.d_pair_idxs.as_ptr(),
            used_nodes_len: rules.inner.d_used_nodes.len(),
            num_blocks,
            task_stride,
            num_rows_per_tile,
        });

        keepalive_partition_ptrs.push(d_partition_ptrs);
        keepalive_public_values.push(d_public_values);
        keepalive_intermediates.push(intermediates);
    }

    // Allocate shared temp buffer for all AIRs needing lifting
    let lift_tmp = if total_lift_tmp > 0 {
        DeviceBuffer::<Frac<EF>>::with_capacity(total_lift_tmp)
    } else {
        DeviceBuffer::<Frac<EF>>::new()
    };

    // ── Pass 2: Build GkrInputCtx / GkrBlockCtx arrays, grouped by GLOBAL flag ──
    let mut global_air_ctxs: Vec<GkrInputCtx> = Vec::new();
    let mut global_block_ctxs: Vec<GkrBlockCtx> = Vec::new();
    let mut local_air_ctxs: Vec<GkrInputCtx> = Vec::new();
    let mut local_block_ctxs: Vec<GkrBlockCtx> = Vec::new();

    for (prep, dev) in air_prep.iter().zip(air_device_data.iter()) {
        let d_fracs = if prep.height != prep.lifted_height {
            unsafe { lift_tmp.as_mut_ptr().add(prep.lift_tmp_offset) }
        } else {
            unsafe { leaves.as_mut_ptr().add(prep.dst_offset) }
        };

        let air_ctx_gpu = GkrInputCtx {
            d_fracs,
            d_preprocessed: dev.d_preprocessed_ptr,
            d_main: dev.d_partition_ptrs_ptr,
            d_public_values: dev.d_public_values_ptr,
            d_challenges: d_challenges.as_ptr(),
            d_intermediates: dev.d_intermediates_ptr,
            d_rules: dev.d_rules_ptr,
            d_used_nodes: dev.d_used_nodes_ptr,
            d_pair_idxs: dev.d_pair_idxs_ptr,
            used_nodes_len: dev.used_nodes_len,
            permutation_height: prep.height as u32,
            num_rows_per_tile: dev.num_rows_per_tile,
            task_stride: dev.task_stride,
        };

        let (air_ctxs, block_ctxs) = if prep.is_global {
            (&mut global_air_ctxs, &mut global_block_ctxs)
        } else {
            (&mut local_air_ctxs, &mut local_block_ctxs)
        };
        let air_idx = air_ctxs.len() as u32;
        for local_block in 0..dev.num_blocks {
            block_ctxs.push(GkrBlockCtx {
                local_block_idx_x: local_block as u32,
                air_idx,
            });
        }
        air_ctxs.push(air_ctx_gpu);
    }
    let _ = air_device_data; // consumed

    // ── Phase 3: Launch batched kernels ──
    if !global_air_ctxs.is_empty() {
        let d_air_ctxs = global_air_ctxs.to_device()?;
        let d_block_ctxs = global_block_ctxs.to_device()?;
        unsafe {
            logup_gkr_input_eval_batched(
                true,
                &d_block_ctxs,
                &d_air_ctxs,
                global_block_ctxs.len() as u32,
            )?;
        }
    }

    if !local_air_ctxs.is_empty() {
        let d_air_ctxs = local_air_ctxs.to_device()?;
        let d_block_ctxs = local_block_ctxs.to_device()?;
        unsafe {
            logup_gkr_input_eval_batched(
                false,
                &d_block_ctxs,
                &d_air_ctxs,
                local_block_ctxs.len() as u32,
            )?;
        }
    }

    // ── Phase 4: Apply lifting for AIRs where height < lifted_height ──
    for prep in &air_prep {
        if prep.height == prep.lifted_height {
            continue;
        }
        let leaves_ptr = unsafe { leaves.as_mut_ptr().add(prep.dst_offset) };
        let tmp_ptr = unsafe { lift_tmp.as_ptr().add(prep.lift_tmp_offset) };
        let tmp_mut_ptr = unsafe { lift_tmp.as_mut_ptr().add(prep.lift_tmp_offset) };
        let norm_factor = F::from_usize(prep.lifted_height / prep.height).inverse();
        unsafe {
            frac_vector_scalar_multiply_ext_fp(
                tmp_mut_ptr,
                norm_factor,
                (prep.height * prep.num_interactions) as u32,
            )?;
            frac_matrix_vertically_repeat(
                leaves_ptr,
                tmp_ptr,
                prep.num_interactions as u32,
                prep.lifted_height as u32,
                prep.height as u32,
            )?;
        }
    }

    // NOTE: alpha is NO LONGER applied here - it will be fused into the first tree layer
    // in fractional_sumcheck_gpu for better performance (eliminates one memory pass)
    Ok((leaves, alpha_logup))
}
