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
        batched_gkr_input_eval, frac_matrix_vertically_repeat,
        frac_vector_scalar_multiply_ext_fp, logup_gkr_input_eval, BatchGkrInputBlockCtx,
        BatchGkrInputDesc,
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

    // Phase 1.5: Partition work items into batch-eligible and non-eligible.
    // Batch-eligible: GLOBAL mode (buffer_size > 10) AND height <= TASK_SIZE (num_rows_per_tile == 1).
    let mut batch_indices = Vec::new();
    let mut non_batch_indices = Vec::new();
    for (i, w) in work_items.iter().enumerate() {
        let buffer_size = w.pk_air.other_data.interaction_rules.inner.buffer_size;
        let is_global = buffer_size > 10;
        let height = w.air_ctx.height();
        if is_global && height <= TASK_SIZE as usize {
            batch_indices.push(i);
        } else {
            non_batch_indices.push(i);
        }
    }

    // Build batched descriptors for eligible AIRs.
    let mut _total_batch_blocks: u32 = 0;
    let mut total_batch_intermediates: usize = 0;
    let mut total_batch_tmp: usize = 0; // for height != lifted_height temporaries
    let mut batch_needs_tmp = Vec::new(); // (work_item_idx, tmp_offset, height, num_interactions)

    // Collect per-AIR partition_ptrs and public_values into flat buffers for single H2D.
    let mut all_partition_ptrs: Vec<u64> = Vec::new();
    let mut all_public_values: Vec<F> = Vec::new();

    let mut host_descs: Vec<BatchGkrInputDesc> = Vec::with_capacity(batch_indices.len());
    let mut host_block_ctxs: Vec<BatchGkrInputBlockCtx> = Vec::new();

    for (desc_idx, &wi) in batch_indices.iter().enumerate() {
        let w = &work_items[wi];
        let height = w.air_ctx.height() as u32;
        let rules = &w.pk_air.other_data.interaction_rules;
        let buffer_size = rules.inner.buffer_size;

        let blocks_per_air = height.div_ceil(256);
        let intermediates_per_air = height as usize * buffer_size as usize;
        let intermediates_offset = total_batch_intermediates;

        // Collect partition pointers
        let _partition_ptrs_offset = all_partition_ptrs.len();
        let mut partitioned_main = Vec::with_capacity(w.air_ctx.cached_mains.len() + 1);
        for committed in &w.air_ctx.cached_mains {
            partitioned_main.push(&committed.trace);
        }
        partitioned_main.push(&w.air_ctx.common_main);
        for m in &partitioned_main {
            all_partition_ptrs.push(m.buffer().as_ptr() as u64);
        }

        // Collect public values
        let _pv_offset = all_public_values.len();
        let has_pv = !w.air_ctx.public_values.is_empty();
        if has_pv {
            all_public_values.extend_from_slice(&w.air_ctx.public_values);
        }

        // Determine output pointer: direct to leaves or via tmp buffer
        let lifted_height = max(height as usize, 1 << w.l_skip);
        let d_fracs_ptr = if height as usize != lifted_height {
            let num_interactions = w.pk_air.vk.symbolic_constraints.interactions.len();
            let tmp_offset = total_batch_tmp;
            batch_needs_tmp.push((wi, tmp_offset, height as usize, num_interactions));
            total_batch_tmp += height as usize * num_interactions;
            // Will be set after tmp buffer is allocated
            std::ptr::null_mut()
        } else {
            w.leaves_ptr.0
        };

        let null_preprocessed = DeviceBuffer::<F>::new();
        let preprocessed_matrix = w
            .pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| &committed.trace);
        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer().as_ptr())
            .unwrap_or(null_preprocessed.as_ptr());

        host_descs.push(BatchGkrInputDesc {
            d_fracs: d_fracs_ptr,
            d_preprocessed,
            d_main: std::ptr::null(), // will be patched after H2D upload
            d_public_values: if has_pv {
                std::ptr::null() // will be patched after H2D upload
            } else {
                std::ptr::null()
            },
            d_rules: rules.inner.d_rules.as_raw_ptr(),
            d_used_nodes: rules.inner.d_used_nodes.as_ptr(),
            d_pair_idxs: rules.d_pair_idxs.as_ptr(),
            used_nodes_len: rules.inner.d_used_nodes.len(),
            permutation_height: height,
            buffer_size,
            intermediates_offset: intermediates_offset as u32,
        });

        // Build block contexts for this AIR
        for b in 0..blocks_per_air {
            host_block_ctxs.push(BatchGkrInputBlockCtx {
                local_block_idx: b,
                air_idx: desc_idx as u32,
            });
        }

        _total_batch_blocks += blocks_per_air;
        total_batch_intermediates += intermediates_per_air;
    }

    // Upload flat partition_ptrs and public_values, then patch descriptor pointers.
    let d_all_partition_ptrs = if !all_partition_ptrs.is_empty() {
        Some(all_partition_ptrs.to_device()?)
    } else {
        None
    };
    let d_all_public_values = if !all_public_values.is_empty() {
        Some(all_public_values.to_device()?)
    } else {
        None
    };

    // Allocate tmp buffer for batch-eligible AIRs needing height normalization.
    let d_batch_tmp = if total_batch_tmp > 0 {
        Some(DeviceBuffer::<Frac<EF>>::with_capacity(total_batch_tmp))
    } else {
        None
    };

    // Patch descriptor pointers to point into uploaded device buffers.
    {
        let mut partition_ptrs_cursor = 0usize;
        let mut pv_cursor = 0usize;
        let mut tmp_cursor = 0usize;
        for (desc_idx, &wi) in batch_indices.iter().enumerate() {
            let w = &work_items[wi];
            let num_partitions = w.air_ctx.cached_mains.len() + 1;

            if let Some(ref d_pp) = d_all_partition_ptrs {
                host_descs[desc_idx].d_main =
                    unsafe { d_pp.as_ptr().add(partition_ptrs_cursor) };
            }
            partition_ptrs_cursor += num_partitions;

            let has_pv = !w.air_ctx.public_values.is_empty();
            if has_pv {
                if let Some(ref d_pv) = d_all_public_values {
                    host_descs[desc_idx].d_public_values =
                        unsafe { d_pv.as_ptr().add(pv_cursor) };
                }
                pv_cursor += w.air_ctx.public_values.len();
            }

            // Patch tmp buffer pointers for AIRs needing normalization.
            let height = w.air_ctx.height();
            let lifted_height = max(height, 1 << w.l_skip);
            if height != lifted_height {
                if let Some(ref d_tmp) = d_batch_tmp {
                    host_descs[desc_idx].d_fracs = unsafe { d_tmp.as_mut_ptr().add(tmp_cursor) };
                }
                let num_interactions = w.pk_air.vk.symbolic_constraints.interactions.len();
                tmp_cursor += height * num_interactions;
            }
        }
    }

    // Pre-compute max buffer sizes across ALL work items (needed for fallback path).
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

    // Barrier: ensure fill_zero() and all prior GPU work is visible to worker thread streams.
    current_stream_sync().map_err(InteractionGpuError::from)?;

    // Memory budget: cap at 64 MB so intermediates fit in L2 cache (72 MB on RTX 4090).
    // Exceeding L2 destroys the cache reuse that makes the per-stream approach fast.
    const MAX_BATCH_INTERMEDIATES_BYTES: usize = 64 * 1024 * 1024;
    let batch_intermediates_bytes = total_batch_intermediates * std::mem::size_of::<EF>();
    let use_batched =
        !batch_indices.is_empty() && batch_intermediates_bytes <= MAX_BATCH_INTERMEDIATES_BYTES;

    // If over budget, move batch-eligible items back to non-batch.
    if !use_batched && !batch_indices.is_empty() {
        non_batch_indices.clear();
        non_batch_indices.extend(0..work_items.len());
        batch_indices.clear();
    }

    // Phase 2: Process AIRs — batched kernel + multi-stream workers.
    let non_batch_work: Vec<&GkrInputWorkItem<HS>> =
        non_batch_indices.iter().map(|&i| &work_items[i]).collect();

    let mut num_threads = if non_batch_work.len() >= 100 {
        NUM_GKR_INPUT_STREAMS.min(non_batch_work.len())
    } else if non_batch_work.is_empty() {
        0
    } else {
        1
    };

    // Memory budget check: reduce num_threads if pre-allocation would exceed 2 GB.
    if num_threads > 0 {
        let per_thread_bytes = max_intermediates_len * std::mem::size_of::<EF>()
            + max_public_values_len * std::mem::size_of::<F>()
            + max_partition_ptrs_len * std::mem::size_of::<u64>()
            + max_tmp_len * std::mem::size_of::<Frac<EF>>();
        while num_threads > 1 && num_threads * per_thread_bytes > 2_000_000_000 {
            num_threads /= 2;
            tracing::warn!(
                "Reducing GKR input eval thread count to {} due to memory budget ({}MB per thread)",
                num_threads,
                per_thread_bytes / (1024 * 1024)
            );
        }
    }

    // Pre-allocate per-thread buffer pools for non-batch workers.
    let thread_buffers: Vec<GkrThreadBuffers> = (0..num_threads.max(1))
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

    // Upload batched descriptors and block contexts.
    let d_descs = if use_batched {
        Some(host_descs.to_device()?)
    } else {
        None
    };
    let d_block_ctxs = if use_batched {
        Some(host_block_ctxs.to_device()?)
    } else {
        None
    };
    let mut d_batch_intermediates = if use_batched && total_batch_intermediates > 0 {
        DeviceBuffer::<EF>::with_capacity(total_batch_intermediates)
    } else {
        DeviceBuffer::new()
    };

    // Launch batched kernel on main thread's stream, then spawn workers.
    if use_batched {
        if let (Some(ref d_d), Some(ref d_bc)) = (&d_descs, &d_block_ctxs) {
            unsafe {
                batched_gkr_input_eval(
                    d_d,
                    d_bc,
                    &mut d_batch_intermediates,
                    d_challenges,
                )?;
            }
        }
    }

    // Run non-batch AIRs on worker threads (or main thread if few).
    if !non_batch_work.is_empty() {
        if num_threads <= 1 {
            let mut bufs = thread_buffers.into_iter().next().unwrap();
            for w in &non_batch_work {
                process_gkr_input_air(w, &mut bufs)?;
            }
        } else {
            let chunk_size = non_batch_work.len().div_ceil(num_threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = thread_buffers
                    .into_iter()
                    .zip(non_batch_work.chunks(chunk_size))
                    .map(|(mut bufs, chunk)| {
                        s.spawn(move || -> Result<(), InteractionGpuError> {
                            for w in chunk {
                                process_gkr_input_air(w, &mut bufs)?;
                            }
                            current_stream_sync().map_err(InteractionGpuError::from)?;
                            Ok(())
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap()?;
                }
                Ok::<_, InteractionGpuError>(())
            })?;
        }
    }

    // Sync main thread's stream to ensure batched kernel has completed.
    if use_batched {
        current_stream_sync().map_err(InteractionGpuError::from)?;
    }

    // Height normalization for batch-eligible AIRs that wrote to tmp buffer.
    if use_batched {
        for &(wi, _tmp_offset, height, num_interactions) in &batch_needs_tmp {
            let w = &work_items[wi];
            let lifted_height = max(height, 1 << w.l_skip);
            debug_assert_ne!(height, lifted_height);
            debug_assert_eq!(lifted_height % height, 0);
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_usize(norm_factor_denom).inverse();

            // Find this AIR's descriptor to get its d_fracs (tmp buffer pointer).
            let desc_idx = batch_indices
                .iter()
                .position(|&x| x == wi)
                .unwrap();
            let tmp_ptr = host_descs[desc_idx].d_fracs;

            unsafe {
                frac_vector_scalar_multiply_ext_fp(
                    tmp_ptr,
                    norm_factor,
                    (height * num_interactions) as u32,
                )?;
                frac_matrix_vertically_repeat(
                    w.leaves_ptr.0,
                    tmp_ptr as *const Frac<EF>,
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
