use itertools::Itertools;
use rustc_hash::FxHashMap;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    prover::{
        fractional_sumcheck_gkr::Frac, AirProvingContext, DeviceMultiStarkProvingKey,
        DeviceStarkProvingKey,
    },
};
use p3_field::PrimeCharacteristicRing;
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    base::DeviceMatrix,
    cuda::logup_zerocheck::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    poly::EqEvalLayers,
    prelude::{EF, F},
};

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
///
/// `num_cosets` should equal `constraint_degree - 1` because we evaluate the quotient polynomial.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() || num_cosets == 0 {
        // No plain AIR constraints, return empty buffer
        return Ok(DeviceBuffer::new());
    }

    let rules = &pk.other_data.zerocheck_round0;

    let buffer_size: u32 = rules.inner.buffer_size;
    let intermed_capacity = unsafe {
        _zerocheck_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<F>::new()
    };

    let temp_sums_buffer_capacity = unsafe {
        _zerocheck_r0_temp_sums_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<EF>();
    if used_temp_bytes > max_temp_bytes {
        // We do not error if the required bytes is greater than the requested max, but this may
        // lead to unexpected peak memory usage.
        warn!("zerocheck used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})");
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut sp_evals =
        DeviceBuffer::<EF>::with_capacity(num_cosets as usize * skip_domain as usize);
    // SAFETY:
    // - No bounds checks are done in this kernel. It fully assumes that the Rules are trusted and
    //   all nodes are valid.
    unsafe {
        zerocheck_ntt_eval_constraints(
            &mut temp_sums_buffer,
            &mut sp_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            lambda_pows,
            public_values,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    Ok(sp_evals)
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// `constraints` includes interaction expressions for the AIR.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    symbolic: &SymbolicConstraints<F>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    beta_pows: &[EF],
    eq_3bs: &[EF],
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    // Check if this trace has interactions
    if eq_3bs.is_empty() {
        return Ok(DeviceBuffer::new());
    }
    let large_domain = num_cosets * skip_domain;

    // We create a new "interactions DAG" where the new .constraints are the interaction [count,
    // message_0, message_1, ..] expressions themselves, while the .interactions are empty
    // We track the indices with InteractionNode

    // Copied from build_symbolic_constraints_dag to handle sorting of constraints
    // NOTE: For logup round0, the kernel uses weights indexed by rule_idx, not constraint_idx.
    // So we deduplicate constraint_idx and use dag_idx_to_rule_idx for weight mapping.
    let (rules, d_numer_weights, d_denom_weights, denom_sum_init) = {
        let mut dag_builder = SymbolicDagBuilder::new();
        let mut sorted_used_dag_idxs = Vec::new();
        for interaction in &symbolic.interactions {
            let count = dag_builder.add_expr(&interaction.count);
            sorted_used_dag_idxs.push(count);
            sorted_used_dag_idxs.extend(
                interaction
                    .message
                    .iter()
                    .map(|field_expr| dag_builder.add_expr(field_expr)),
            );
        }
        sorted_used_dag_idxs.sort();
        // Deduplicate for the dag since logup round0 kernel doesn't use used_nodes
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesGpu::new(&dag, true);
        let mut numer_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_sum_init = EF::ZERO;
        for (interaction_idx, interaction) in symbolic.interactions.iter().enumerate() {
            // CAUTION: an expression node could be used in multiple interactions, and might even be
            // used as `count` in one, but message field in another. We only care about their
            // weighted sum with eq_3b, so we compute the weights ahead of time.
            let count_dag_idx =
                dag_builder.expr_to_idx[&(&interaction.count as *const SymbolicExpression<_>)];
            let count_rule_idx = rules.dag_idx_to_rule_idx[&count_dag_idx];
            numer_weights[count_rule_idx] += eq_3bs[interaction_idx];
            denom_sum_init += eq_3bs[interaction_idx]
                * beta_pows[interaction.message.len()]
                * F::from_u32(interaction.bus_index as u32 + 1);

            for (message_idx, message) in interaction.message.iter().enumerate() {
                let message_dag_idx =
                    dag_builder.expr_to_idx[&(message as *const SymbolicExpression<_>)];
                let message_rule_idx = rules.dag_idx_to_rule_idx[&message_dag_idx];
                denom_weights[message_rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
            }
        }
        let d_numer_weights = numer_weights.to_device()?;
        let d_denom_weights = denom_weights.to_device()?;
        (rules, d_numer_weights, d_denom_weights, denom_sum_init)
    };

    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity = unsafe {
        _logup_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup_r0:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<F>::new()
    };

    let temp_sums_buffer_capacity = unsafe {
        _logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
    };
    debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<Frac<EF>>();
    if used_temp_bytes > max_temp_bytes {
        warn!(
            "logup_round0 used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})"
        );
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut s_evals = DeviceBuffer::<Frac<EF>>::with_capacity(large_domain as usize);

    unsafe {
        logup_bary_eval_interactions_round0(
            &mut temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            public_values,
            &d_numer_weights,
            &d_denom_weights,
            denom_sum_init,
            &d_rules,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    Ok(s_evals)
}

// ============================================================================
// Batched round-0 zerocheck
// ============================================================================

/// Classification info for one trace during round-0 batching.
#[derive(Debug, Clone)]
pub struct Round0TraceInfo {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub height: usize,
    pub n_lift: usize,
    pub num_x: usize,
    pub num_cosets_zc: usize,
    pub local_constraint_deg: usize,
    pub omega_root: F,
    pub buffer_size_zc: u32,
}

/// Grouping key for batched round-0 zerocheck. All traces in a batch must share this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Round0GroupKey {
    pub num_x: usize,
    pub height: usize,
    pub num_cosets_zc: usize,
}

const COSET_PARALLEL_THRESHOLD: usize = 32768;
const BUFFER_THRESHOLD: u32 = 16;
const WARP_SIZE: u32 = 32;

impl Round0TraceInfo {
    /// Whether this trace qualifies for batched coset-parallel evaluation.
    pub fn is_batchable(&self, skip_domain: usize) -> bool {
        self.num_x * skip_domain < COSET_PARALLEL_THRESHOLD && self.num_cosets_zc > 0
    }

    pub fn group_key(&self) -> Round0GroupKey {
        Round0GroupKey {
            num_x: self.num_x,
            height: self.height,
            num_cosets_zc: self.num_cosets_zc,
        }
    }
}

/// Group a list of traces by their round-0 batching key.
/// Returns groups in arbitrary order; each group contains only batchable traces.
pub fn group_batchable_traces(
    traces: &[Round0TraceInfo],
    skip_domain: usize,
) -> FxHashMap<Round0GroupKey, Vec<usize>> {
    let mut groups: FxHashMap<Round0GroupKey, Vec<usize>> = FxHashMap::default();
    for (i, t) in traces.iter().enumerate() {
        if t.is_batchable(skip_domain) {
            groups.entry(t.group_key()).or_default().push(i);
        }
    }
    groups
}

/// Evaluate batched round-0 zerocheck constraints for a group of traces.
///
/// Returns a Vec of `(trace_idx, q_evals)` pairs, where `q_evals` has
/// `num_cosets * skip_domain` elements per trace.
///
/// # Arguments
/// * `group` - Indices into `all_traces` for traces in this batch group
/// * `all_traces` - Full trace info array
/// * `skip_domain` - `2^l_skip`
/// * `pk` - Proving key (for per-AIR rules)
/// * `selectors_base` - Per-trace base selector device matrices `[3][num_x]`
/// * `eq_xis` - Pre-built eq evaluation trees keyed by n_lift
/// * `public_values_per_trace` - Per-trace public values on device
/// * `d_lambda_pows` - Shared lambda powers on device
/// * `memory_limit_bytes` - Memory budget for temp allocations
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_zerocheck_batched<HS: GpuHashScheme>(
    group: &[usize],
    all_traces: &[Round0TraceInfo],
    skip_domain: usize,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_base: &[DeviceMatrix<F>],
    eq_xis: &FxHashMap<usize, EqEvalLayers<EF>>,
    public_values_per_trace: &[DeviceBuffer<F>],
    ctx_per_trace: &[(usize, AirProvingContext<GenericGpuBackend<HS>>)],
    d_lambda_pows: &DeviceBuffer<EF>,
    memory_limit_bytes: usize,
) -> Result<Vec<(usize, Vec<EF>)>, Round0EvalError> {
    use crate::cuda::logup_zerocheck::{
        zerocheck_r0_batched, zerocheck_r0_batched_launch_params, Round0BlockCtx,
        Round0ZerocheckCtx,
    };
    use openvm_cuda_common::copy::MemCopyD2H;

    if group.is_empty() {
        return Ok(vec![]);
    }

    let first = &all_traces[group[0]];
    let num_x = first.num_x;
    let height = first.height;
    let num_cosets = first.num_cosets_zc;
    let omega_root = first.omega_root;
    let skip_domain_u32 = skip_domain as u32;
    let d = num_cosets * skip_domain;

    // Compute launch params from worst-case buffer_size in group
    let max_buffer_size = group
        .iter()
        .map(|&i| all_traces[i].buffer_size_zc)
        .max()
        .unwrap();
    let is_global = max_buffer_size > BUFFER_THRESHOLD;
    let needs_shmem = skip_domain_u32 > WARP_SIZE;
    let (blocks_per_trace, threads_per_block) = zerocheck_r0_batched_launch_params(
        max_buffer_size,
        skip_domain_u32,
        num_x as u32,
        num_cosets as u32,
        memory_limit_bytes,
    );

    // Sub-batch by memory budget
    let per_trace_bytes = |idx: &usize| -> usize {
        let t = &all_traces[*idx];
        let pk_air = &pk.per_air[t.air_idx];
        let bs = pk_air.other_data.zerocheck_round0.inner.buffer_size as usize;
        let tmp_sums = blocks_per_trace as usize * d * std::mem::size_of::<EF>();
        let output = d * std::mem::size_of::<EF>();
        let intermed = if is_global {
            num_cosets * blocks_per_trace as usize * threads_per_block as usize * bs
                * std::mem::size_of::<F>()
        } else {
            0
        };
        tmp_sums + output + intermed
    };

    let mut all_results = Vec::with_capacity(group.len());
    let mut batch_start = 0;

    while batch_start < group.len() {
        // Determine sub-batch size
        let mut batch_end = batch_start;
        let mut batch_memory = 0usize;
        while batch_end < group.len() {
            let trace_mem = per_trace_bytes(&group[batch_end]);
            if batch_end > batch_start && batch_memory + trace_mem > memory_limit_bytes {
                break;
            }
            batch_memory += trace_mem;
            batch_end += 1;
        }
        // Ensure at least one trace per sub-batch
        if batch_end == batch_start {
            batch_end = batch_start + 1;
        }
        let sub_batch = &group[batch_start..batch_end];
        let num_traces = sub_batch.len();

        debug!(
            "zerocheck_r0_batched: sub-batch of {num_traces} traces, \
             height={height}, num_x={num_x}, num_cosets={num_cosets}, \
             blocks_per_trace={blocks_per_trace}, is_global={is_global}, \
             batch_memory={batch_memory}"
        );

        // Build BlockCtx array and segment_offsets
        let mut block_ctxs: Vec<Round0BlockCtx> = Vec::new();
        let mut segment_offsets: Vec<u32> = vec![0];
        let mut spatial_row_count: u32 = 0;
        for (local_idx, _trace_global_idx) in sub_batch.iter().enumerate() {
            let row_base = spatial_row_count;
            for coset in 0..num_cosets as u32 {
                for b in 0..blocks_per_trace {
                    block_ctxs.push(Round0BlockCtx {
                        local_block_idx_x: b,
                        air_idx: local_idx as u32,
                        coset_idx: coset,
                        row_base,
                    });
                }
            }
            spatial_row_count += blocks_per_trace;
            segment_offsets.push(spatial_row_count);
        }
        let total_blocks = block_ctxs.len() as u32;
        let total_spatial_rows = spatial_row_count as usize;

        // Build per-trace contexts
        let mut intermediates_keepalive: Vec<DeviceBuffer<F>> = Vec::new();
        let mut main_parts_keepalive: Vec<DeviceBuffer<*const F>> = Vec::new();
        let mut trace_ctxs: Vec<Round0ZerocheckCtx> = Vec::with_capacity(num_traces);

        for &trace_global_idx in sub_batch {
            let t = &all_traces[trace_global_idx];
            let pk_air = &pk.per_air[t.air_idx];
            let rules = &pk_air.other_data.zerocheck_round0;
            let (_, air_ctx) = &ctx_per_trace[t.trace_idx];

            // Upload main_parts pointer array
            let mut main_ptrs: Vec<*const F> = Vec::new();
            for committed in &air_ctx.cached_mains {
                main_ptrs.push(committed.trace.buffer().as_ptr());
            }
            main_ptrs.push(air_ctx.common_main.buffer().as_ptr());
            let d_main_parts = main_ptrs.to_device().map_err(Round0EvalError::Copy)?;

            // Allocate intermediates if GLOBAL
            let d_intermediates = if is_global && rules.inner.buffer_size > 0 {
                let cap = num_cosets
                    * blocks_per_trace as usize
                    * threads_per_block as usize
                    * rules.inner.buffer_size as usize;
                let buf = DeviceBuffer::<F>::with_capacity(cap);
                let ptr = buf.as_mut_ptr();
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            let preprocessed_ptr = pk_air
                .preprocessed_data
                .as_ref()
                .map(|cd| cd.trace.buffer().as_ptr())
                .unwrap_or(std::ptr::null());

            let eq_xi_tree = &eq_xis[&t.n_lift];

            trace_ctxs.push(Round0ZerocheckCtx {
                selectors_cube: selectors_base[t.trace_idx].buffer().as_ptr(),
                preprocessed: preprocessed_ptr,
                main_parts: d_main_parts.as_ptr(),
                eq_cube: eq_xi_tree.get_ptr(t.n_lift),
                public_values: public_values_per_trace[t.trace_idx].as_ptr(),
                d_rules: rules.inner.d_rules.as_raw_ptr(),
                d_used_nodes: rules.inner.d_used_nodes.as_ptr(),
                rules_len: rules.inner.d_rules.len(),
                used_nodes_len: rules.inner.d_used_nodes.len(),
                lambda_len: pk_air
                    .vk
                    .symbolic_constraints
                    .constraints
                    .constraint_idx
                    .len(),
                buffer_size: rules.inner.buffer_size,
                d_intermediates,
            });
            main_parts_keepalive.push(d_main_parts);
        }

        // Upload contexts
        let d_block_ctxs = block_ctxs.to_device().map_err(Round0EvalError::Copy)?;
        let d_trace_ctxs = trace_ctxs.to_device().map_err(Round0EvalError::Copy)?;
        let d_segment_offsets = segment_offsets.to_device().map_err(Round0EvalError::Copy)?;

        // Allocate output buffers
        let mut d_tmp_sums = DeviceBuffer::<EF>::with_capacity(total_spatial_rows * d);
        let mut d_output = DeviceBuffer::<EF>::with_capacity(num_traces * d);

        // Launch batched eval + reduction
        unsafe {
            zerocheck_r0_batched(
                is_global,
                needs_shmem,
                &mut d_tmp_sums,
                &mut d_output,
                &d_block_ctxs,
                &d_trace_ctxs,
                d_lambda_pows,
                &d_segment_offsets,
                skip_domain_u32,
                num_x as u32,
                height as u32,
                num_cosets as u32,
                blocks_per_trace,
                omega_root,
                total_blocks,
                threads_per_block,
                d as u32,
                num_traces as u32,
            )
            .map_err(Round0EvalError::Cuda)?;
        }

        // Single D2H for the sub-batch
        let all_q_evals = d_output.to_host().map_err(Round0EvalError::Copy)?;

        // Collect results
        for (i, &trace_global_idx) in sub_batch.iter().enumerate() {
            let offset = i * d;
            let q_evals = all_q_evals[offset..offset + d].to_vec();
            all_results.push((all_traces[trace_global_idx].trace_idx, q_evals));
        }

        // Drop keepalive buffers (safe: to_host synced the stream)
        drop(intermediates_keepalive);
        drop(main_parts_keepalive);

        batch_start = batch_end;
    }

    Ok(all_results)
}
