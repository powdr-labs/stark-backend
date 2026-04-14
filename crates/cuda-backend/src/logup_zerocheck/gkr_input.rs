use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    stream::current_stream_sync,
};
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
        batched_gkr_input_eval_scatter, frac_matrix_vertically_repeat,
        frac_vector_scalar_multiply_ext_fp, logup_gkr_input_eval, BlockCtx,
        GkrInputScatterCtx,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    prelude::{EF, F},
};

const TASK_SIZE: u32 = 65536;

/// Number of OS threads (and thus CUDA streams) for parallel GKR input evaluation.
const NUM_GKR_INPUT_STREAMS: usize = 8;

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

struct GkrThreadBuffers {
    intermediates: DeviceBuffer<EF>,
    public_values: DeviceBuffer<F>,
    partition_ptrs: DeviceBuffer<u64>,
    tmp: DeviceBuffer<Frac<EF>>,
}

struct GkrInputWorkItem<'a, HS: GpuHashScheme> {
    air_ctx: &'a openvm_stark_backend::prover::AirProvingContext<GenericGpuBackend<HS>>,
    pk_air: &'a openvm_stark_backend::prover::DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    l_skip: usize,
    d_challenges: &'a DeviceBuffer<EF>,
    leaves_ptr: SendPtr<Frac<EF>>,
}

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

fn process_gkr_input_air<HS: GpuHashScheme>(
    w: &GkrInputWorkItem<HS>,
    buffers: &mut GkrThreadBuffers,
) -> Result<(), InteractionGpuError> {
    let null_preprocessed = DeviceBuffer::<F>::new();
    let air_ctx = w.air_ctx;
    let pk_air = w.pk_air;

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

    let null_pv = DeviceBuffer::<F>::new();
    let has_public_values = !air_ctx.public_values.is_empty();
    if has_public_values {
        air_ctx.public_values.copy_to(&mut buffers.public_values)?;
    }
    let d_public_values = if has_public_values {
        &buffers.public_values
    } else {
        &null_pv
    };

    let height = air_ctx.height();
    debug_assert_eq!(height, partitioned_main[0].height());
    let partition_ptrs = partitioned_main
        .iter()
        .map(|m| m.buffer().as_ptr() as u64)
        .collect_vec();
    partition_ptrs.copy_to(&mut buffers.partition_ptrs)?;

    let buffer_size = rules.inner.buffer_size;
    // TODO[jpw]: remove magic 10
    let is_global = buffer_size > 10;

    let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

    let leaves_ptr = w.leaves_ptr.0;
    let l_skip = w.l_skip;
    let lifted_height = max(height, 1 << l_skip);

    let trace_output = if height != lifted_height {
        buffers.tmp.as_mut_ptr()
    } else {
        leaves_ptr
    };
    unsafe {
        logup_gkr_input_eval(
            is_global,
            trace_output,
            d_preprocessed,
            &buffers.partition_ptrs,
            d_public_values,
            w.d_challenges,
            &buffers.intermediates,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            &rules.d_pair_idxs,
            height as u32,
            num_rows_per_tile as u32,
        )?;
    }
    if height != lifted_height {
        debug_assert_eq!(lifted_height % height, 0);
        debug_assert!(!buffers.tmp.is_empty());
        let norm_factor_denom = lifted_height / height;
        let norm_factor = F::from_usize(norm_factor_denom).inverse();
        unsafe {
            frac_vector_scalar_multiply_ext_fp(
                buffers.tmp.as_mut_ptr(),
                norm_factor,
                (height * num_interactions) as u32,
            )?;
            frac_matrix_vertically_repeat(
                leaves_ptr,
                buffers.tmp.as_ptr(),
                num_interactions as u32,
                lifted_height as u32,
                height as u32,
            )?;
        }
    }
    Ok(())
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

    // Phase 1: Build work items and sort by descending height for load balance.
    let mut work_items: Vec<GkrInputWorkItem<HS>> = trace_interactions
        .iter()
        .flatten()
        .map(|meta| {
            let air_ctx = &ctx.per_trace[meta.trace_idx].1;
            let pk_air = &pk.per_air[meta.air_idx];
            let slice = meta.layout_slices.first().unwrap();
            assert_eq!(slice.col_idx, 0);
            let dst_offset = slice.row_idx;
            let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };
            GkrInputWorkItem {
                air_ctx,
                pk_air,
                l_skip,
                d_challenges,
                leaves_ptr: SendPtr(leaves_ptr),
            }
        })
        .collect();

    work_items.sort_by(|a, b| b.air_ctx.height().cmp(&a.air_ctx.height()));

    // Partition into SCATTER (buffer_size <= 10) and GLOBAL (buffer_size > 10) indices.
    let mut scatter_indices: Vec<usize> = Vec::new();
    let mut global_indices: Vec<usize> = Vec::new();
    for (i, w) in work_items.iter().enumerate() {
        if w.pk_air.other_data.interaction_rules.inner.buffer_size > 10 {
            global_indices.push(i);
        } else {
            scatter_indices.push(i);
        }
    }

    // Pre-compute max buffer sizes across all work items (needed for single-threaded fallback).
    let mut max_intermediates_len: usize = 1;
    let mut max_public_values_len: usize = 0;
    let mut max_partition_ptrs_len: usize = 0;
    let mut max_tmp_len: usize = 0;

    for w in &work_items {
        let rules = &w.pk_air.other_data.interaction_rules;
        let buffer_size = rules.inner.buffer_size as usize;
        let is_global = buffer_size > 10;
        if is_global {
            max_intermediates_len = max_intermediates_len.max(TASK_SIZE as usize * buffer_size);
        }

        max_public_values_len = max_public_values_len.max(w.air_ctx.public_values.len());

        let num_partitions = w.air_ctx.cached_mains.len() + 1;
        max_partition_ptrs_len = max_partition_ptrs_len.max(num_partitions);

        let height = w.air_ctx.height();
        let lifted_height = height.max(1 << w.l_skip);
        if height != lifted_height {
            let num_interactions = w.pk_air.vk.symbolic_constraints.interactions.len();
            max_tmp_len = max_tmp_len.max(height * num_interactions);
        }
    }

    // Compute scatter tmp layout: each lifted AIR gets its own section (non-overlapping).
    let mut scatter_tmp_offsets: Vec<Option<usize>> = vec![None; scatter_indices.len()];
    let mut total_scatter_tmp: usize = 0;
    for (local_idx, &idx) in scatter_indices.iter().enumerate() {
        let w = &work_items[idx];
        let height = w.air_ctx.height();
        let lifted = max(height, 1 << w.l_skip);
        if height != lifted {
            let n_int = w.pk_air.vk.symbolic_constraints.interactions.len();
            scatter_tmp_offsets[local_idx] = Some(total_scatter_tmp);
            total_scatter_tmp += height * n_int;
        }
    }

    // Barrier: ensure fill_zero() and all prior GPU work is visible to worker thread streams.
    current_stream_sync().map_err(InteractionGpuError::from)?;

    if work_items.len() < 100 {
        // Single-threaded path: process ALL AIRs (unchanged for APC 0 fallback).
        let mut bufs = GkrThreadBuffers {
            intermediates: DeviceBuffer::with_capacity(max_intermediates_len),
            public_values: if max_public_values_len > 0 {
                DeviceBuffer::with_capacity(max_public_values_len)
            } else {
                DeviceBuffer::new()
            },
            partition_ptrs: DeviceBuffer::with_capacity(max_partition_ptrs_len),
            tmp: if max_tmp_len > 0 {
                DeviceBuffer::with_capacity(max_tmp_len)
            } else {
                DeviceBuffer::new()
            },
        };
        for w in &work_items {
            process_gkr_input_air(w, &mut bufs)?;
        }
    } else {
        // Multi-threaded path with SCATTER batching.
        let mut num_global_threads = NUM_GKR_INPUT_STREAMS.min(global_indices.len().max(1));

        // Memory budget check: reduce thread count if pre-allocation would exceed 2 GB.
        let per_thread_bytes = max_intermediates_len * std::mem::size_of::<EF>()
            + max_public_values_len * std::mem::size_of::<F>()
            + max_partition_ptrs_len * std::mem::size_of::<u64>()
            + max_tmp_len * std::mem::size_of::<Frac<EF>>();
        while num_global_threads > 1 && num_global_threads * per_thread_bytes > 2_000_000_000 {
            num_global_threads /= 2;
            tracing::warn!(
                "Reducing GKR input eval thread count to {} due to memory budget ({}MB per thread)",
                num_global_threads,
                per_thread_bytes / (1024 * 1024)
            );
        }

        // Pre-allocate per-thread buffer pools for GLOBAL workers.
        let thread_buffers: Vec<GkrThreadBuffers> = (0..num_global_threads)
            .map(|_| GkrThreadBuffers {
                intermediates: DeviceBuffer::with_capacity(max_intermediates_len),
                public_values: if max_public_values_len > 0 {
                    DeviceBuffer::with_capacity(max_public_values_len)
                } else {
                    DeviceBuffer::new()
                },
                partition_ptrs: DeviceBuffer::with_capacity(max_partition_ptrs_len),
                tmp: if max_tmp_len > 0 {
                    DeviceBuffer::with_capacity(max_tmp_len)
                } else {
                    DeviceBuffer::new()
                },
            })
            .collect();

        // Allocate scatter tmp buffer for lifted SCATTER AIRs.
        let d_scatter_tmp = if total_scatter_tmp > 0 {
            DeviceBuffer::<Frac<EF>>::with_capacity(total_scatter_tmp)
        } else {
            DeviceBuffer::<Frac<EF>>::new()
        };

        let work_items_ref = &work_items;
        let scatter_indices_ref = &scatter_indices;
        let scatter_tmp_offsets_ref = &scatter_tmp_offsets;
        let d_scatter_tmp_ref = &d_scatter_tmp;

        std::thread::scope(|s| {
            // Spawn background thread for batched SCATTER processing.
            let scatter_handle = if !scatter_indices.is_empty() {
                Some(s.spawn(move || -> Result<(), InteractionGpuError> {
                    // Step 1: Concatenate host arrays for partition pointers and
                    // public values.
                    let mut all_partition_ptrs: Vec<u64> = Vec::new();
                    let mut all_public_values: Vec<F> = Vec::new();
                    let mut scatter_partition_offsets: Vec<usize> = Vec::new();
                    let mut scatter_pv_offsets: Vec<usize> = Vec::new();

                    for &work_idx in scatter_indices_ref {
                        let w = &work_items_ref[work_idx];
                        let air_ctx = w.air_ctx;

                        scatter_partition_offsets.push(all_partition_ptrs.len());
                        for committed in &air_ctx.cached_mains {
                            all_partition_ptrs
                                .push(committed.trace.buffer().as_ptr() as u64);
                        }
                        all_partition_ptrs
                            .push(air_ctx.common_main.buffer().as_ptr() as u64);

                        scatter_pv_offsets.push(all_public_values.len());
                        all_public_values.extend_from_slice(&air_ctx.public_values);
                    }

                    // 2 bulk H2D uploads (on this thread's per-thread CUDA stream).
                    let d_all_partition_ptrs = all_partition_ptrs.to_device()?;
                    let d_all_public_values = if all_public_values.is_empty() {
                        DeviceBuffer::new()
                    } else {
                        all_public_values.to_device()?
                    };

                    // Step 2: Build BlockCtx and GkrInputScatterCtx arrays.
                    let mut block_ctxs: Vec<BlockCtx> = Vec::new();
                    let mut air_ctxs: Vec<GkrInputScatterCtx> = Vec::new();

                    for (air_local_idx, &work_idx) in
                        scatter_indices_ref.iter().enumerate()
                    {
                        let w = &work_items_ref[work_idx];
                        let air_ctx = w.air_ctx;
                        let pk_air = w.pk_air;
                        let height = air_ctx.height() as u32;
                        let num_air_blocks = height.div_ceil(256);
                        let rules = &pk_air.other_data.interaction_rules;

                        for local_block in 0..num_air_blocks {
                            block_ctxs.push(BlockCtx {
                                local_block_idx_x: local_block,
                                air_idx: air_local_idx as u32,
                            });
                        }

                        let preprocessed_ptr = pk_air
                            .preprocessed_data
                            .as_ref()
                            .map(|c| c.trace.buffer().as_ptr())
                            .unwrap_or(std::ptr::null());

                        let pv_ptr = if air_ctx.public_values.is_empty() {
                            std::ptr::null()
                        } else {
                            unsafe {
                                d_all_public_values
                                    .as_ptr()
                                    .add(scatter_pv_offsets[air_local_idx])
                            }
                        };

                        let fracs_ptr =
                            if let Some(offset) = scatter_tmp_offsets_ref[air_local_idx]
                            {
                                unsafe {
                                    d_scatter_tmp_ref.as_mut_ptr().add(offset)
                                }
                            } else {
                                w.leaves_ptr.0
                            };

                        air_ctxs.push(GkrInputScatterCtx {
                            d_fracs: fracs_ptr,
                            d_preprocessed: preprocessed_ptr,
                            d_main: unsafe {
                                d_all_partition_ptrs
                                    .as_ptr()
                                    .add(scatter_partition_offsets[air_local_idx])
                            },
                            d_public_values: pv_ptr,
                            d_challenges: w.d_challenges.as_ptr(),
                            d_rules: rules.inner.d_rules.as_raw_ptr(),
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

                    // Step 3: Launch batched kernel.
                    unsafe {
                        batched_gkr_input_eval_scatter(
                            &d_block_ctxs,
                            &d_air_ctxs,
                            total_scatter_blocks,
                        )?;
                    }

                    // Step 4: Sequential lifting for SCATTER AIRs that need it.
                    for (air_local_idx, &work_idx) in
                        scatter_indices_ref.iter().enumerate()
                    {
                        if let Some(offset) =
                            scatter_tmp_offsets_ref[air_local_idx]
                        {
                            let w = &work_items_ref[work_idx];
                            let height = w.air_ctx.height();
                            let lifted_height = max(height, 1 << w.l_skip);
                            let n_int = w
                                .pk_air
                                .vk
                                .symbolic_constraints
                                .interactions
                                .len();
                            let norm_factor =
                                F::from_usize(lifted_height / height).inverse();
                            let tmp_ptr = unsafe {
                                d_scatter_tmp_ref.as_mut_ptr().add(offset)
                            };
                            unsafe {
                                frac_vector_scalar_multiply_ext_fp(
                                    tmp_ptr,
                                    norm_factor,
                                    (height * n_int) as u32,
                                )?;
                                frac_matrix_vertically_repeat(
                                    w.leaves_ptr.0,
                                    tmp_ptr as *const _,
                                    n_int as u32,
                                    lifted_height as u32,
                                    height as u32,
                                )?;
                            }
                        }
                    }

                    current_stream_sync().map_err(InteractionGpuError::from)?;
                    Ok(())
                }))
            } else {
                None
            };

            // Spawn GLOBAL worker threads.
            let global_chunk_size = if global_indices.is_empty() {
                1
            } else {
                global_indices.len().div_ceil(num_global_threads)
            };
            let global_handles: Vec<_> = thread_buffers
                .into_iter()
                .zip(global_indices.chunks(global_chunk_size))
                .map(|(mut bufs, idx_chunk)| {
                    s.spawn(move || -> Result<(), InteractionGpuError> {
                        for &idx in idx_chunk {
                            process_gkr_input_air(&work_items_ref[idx], &mut bufs)?;
                        }
                        current_stream_sync().map_err(InteractionGpuError::from)?;
                        Ok(())
                    })
                })
                .collect();

            for handle in global_handles {
                handle.join().unwrap()?;
            }
            if let Some(handle) = scatter_handle {
                handle.join().unwrap()?;
            }
            Ok::<_, InteractionGpuError>(())
        })?;
    }

    // NOTE: alpha is NO LONGER applied here - it will be fused into the first tree layer
    // in fractional_sumcheck_gpu for better performance (eliminates one memory pass)
    Ok((leaves, alpha_logup))
}
