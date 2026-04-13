use std::collections::BTreeMap;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    prover::{fractional_sumcheck_gkr::Frac, DeviceStarkProvingKey},
};
use p3_field::PrimeCharacteristicRing;
use tracing::{debug, info, warn};

use super::errors::Round0EvalError;
use crate::{
    cuda::logup_zerocheck::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        batched_logup_r0_eval_interactions, batched_zerocheck_r0_eval_constraints,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints, Round0BlockCtx,
        Round0LogupCtx, Round0ZcCtx,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
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

/// Descriptor for one AIR in a batched Round 0 group.
pub(crate) struct BatchedR0AirDesc {
    pub trace_idx: usize,
    pub num_x: u32,
    pub height: u32,
    pub g_shift: F,
    pub buffer_size_zc: u32,
    pub buffer_size_logup: u32,
    pub num_cosets_zc: u32,
    pub num_cosets_logup: u32,
    pub selectors_cube: *const F,
    pub preprocessed: *const F,
    pub d_main_parts: *const *const F,
    pub eq_cube: *const EF,
    pub lambda_pows: *const EF,
    pub public_values: *const F,
    pub d_zc_rules: *const std::ffi::c_void,
    pub d_zc_used_nodes: *const usize,
    pub zc_rules_len: usize,
    pub zc_used_nodes_len: usize,
    pub lambda_len: usize,
    pub logup_rules: Option<BatchedR0LogupRules>,
}

pub(crate) struct BatchedR0LogupRules {
    pub d_rules: DeviceBuffer<u128>,
    pub rules_len: usize,
    pub buffer_size: u32,
    pub d_numer_weights: DeviceBuffer<EF>,
    pub d_denom_weights: DeviceBuffer<EF>,
    pub denom_sum_init: EF,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_logup_rules_for_air<HS: GpuHashScheme>(
    _pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    symbolic: &SymbolicConstraints<F>,
    beta_pows: &[EF],
    eq_3bs: &[EF],
) -> Result<Option<BatchedR0LogupRules>, Round0EvalError> {
    if eq_3bs.is_empty() {
        return Ok(None);
    }
    let mut dag_builder = SymbolicDagBuilder::new();
    let mut sorted_used_dag_idxs = Vec::new();
    for interaction in &symbolic.interactions {
        let count = dag_builder.add_expr(&interaction.count);
        sorted_used_dag_idxs.push(count);
        sorted_used_dag_idxs.extend(
            interaction.message.iter().map(|f| dag_builder.add_expr(f)),
        );
    }
    sorted_used_dag_idxs.sort();
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
    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;
    Ok(Some(BatchedR0LogupRules {
        d_rules,
        rules_len: rules.rules.len(),
        buffer_size: rules.buffer_size.try_into().unwrap(),
        d_numer_weights,
        d_denom_weights,
        denom_sum_init,
    }))
}

const COSET_PARALLEL_THRESHOLD: u32 = 32768;
const MAX_THREADS_R0: u32 = 128;
const BUFFER_THRESHOLD_R0: u32 = 16;

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_batched_zerocheck_group(
    descs: &[&BatchedR0AirDesc],
    num_cosets: u32,
    skip_domain: u32,
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
    if descs.is_empty() || num_cosets == 0 {
        return Ok(DeviceBuffer::new());
    }
    let max_num_x = descs.iter().map(|d| d.num_x).max().unwrap_or(1);
    let block_x = std::cmp::min(MAX_THREADS_R0, skip_domain * max_num_x);
    let block_x = (block_x / skip_domain) * skip_domain;
    let x_per_block = block_x / skip_domain;

    let mut block_ctxs_h = Vec::new();
    let mut air_offsets_h: Vec<u32> = vec![0];
    let mut total_intermediates: usize = 0;

    struct AirLayout {
        buffer_stride: u32,
        intermediates_size: usize,
        blocks_per_air: u32,
    }
    let layouts: Vec<AirLayout> = descs
        .iter()
        .map(|d| {
            let num_x_blocks = d.num_x.div_ceil(x_per_block);
            let blocks_per_air = num_x_blocks * num_cosets;
            let buffer_stride = num_cosets * num_x_blocks * block_x;
            let intermediates_size = if d.buffer_size_zc > BUFFER_THRESHOLD_R0 {
                buffer_stride as usize * d.buffer_size_zc as usize
            } else {
                0
            };
            AirLayout { buffer_stride, intermediates_size, blocks_per_air }
        })
        .collect();

    for (local_air_idx, layout) in layouts.iter().enumerate() {
        for local_block in 0..layout.blocks_per_air {
            block_ctxs_h.push(Round0BlockCtx {
                local_block_idx: local_block,
                air_idx: local_air_idx as u32,
            });
        }
        air_offsets_h.push(*air_offsets_h.last().unwrap() + layout.blocks_per_air);
        total_intermediates += layout.intermediates_size;
    }

    let total_blocks = block_ctxs_h.len() as u32;
    if total_blocks == 0 {
        return Ok(DeviceBuffer::new());
    }

    let mut intermediates = if total_intermediates > 0 {
        DeviceBuffer::<F>::with_capacity(total_intermediates)
    } else {
        DeviceBuffer::<F>::new()
    };

    let mut zc_ctxs_h = Vec::with_capacity(descs.len());
    let mut intermediates_offset: usize = 0;
    for (d, layout) in descs.iter().zip(&layouts) {
        let d_intermediates = if layout.intermediates_size > 0 {
            let ptr = unsafe { intermediates.as_mut_ptr().add(intermediates_offset) };
            intermediates_offset += layout.intermediates_size;
            ptr
        } else {
            std::ptr::null_mut()
        };
        zc_ctxs_h.push(Round0ZcCtx {
            selectors_cube: d.selectors_cube,
            preprocessed: d.preprocessed,
            main_parts: d.d_main_parts,
            eq_cube: d.eq_cube,
            lambda_pows: d.lambda_pows,
            public_values: d.public_values,
            d_rules: d.d_zc_rules,
            d_used_nodes: d.d_zc_used_nodes,
            rules_len: d.zc_rules_len,
            used_nodes_len: d.zc_used_nodes_len,
            lambda_len: d.lambda_len,
            buffer_size: d.buffer_size_zc,
            d_intermediates,
            buffer_stride: layout.buffer_stride,
            num_x: d.num_x,
            height: d.height,
            g_shift: d.g_shift,
        });
    }

    let d_block_ctxs = block_ctxs_h.to_device()?;
    let d_zc_ctxs = zc_ctxs_h.to_device()?;
    let d_air_offsets = air_offsets_h.to_device()?;

    let output_size = descs.len() * num_cosets as usize * skip_domain as usize;
    let tmp_sums_size = total_blocks as usize * num_cosets as usize * skip_domain as usize;
    let mut tmp_sums_buffer = DeviceBuffer::<EF>::with_capacity(tmp_sums_size);
    let mut output = DeviceBuffer::<EF>::with_capacity(output_size);

    unsafe {
        batched_zerocheck_r0_eval_constraints(
            &mut tmp_sums_buffer,
            &mut output,
            &d_block_ctxs,
            &d_zc_ctxs,
            &d_air_offsets,
            total_blocks,
            descs.len() as u32,
            num_cosets,
            skip_domain,
            block_x,
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_batched_logup_group(
    descs: &[&BatchedR0AirDesc],
    num_cosets: u32,
    skip_domain: u32,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    if descs.is_empty() || num_cosets == 0 {
        return Ok(DeviceBuffer::new());
    }
    let max_num_x = descs.iter().map(|d| d.num_x).max().unwrap_or(1);
    let block_x = std::cmp::min(MAX_THREADS_R0, skip_domain * max_num_x);
    let block_x = (block_x / skip_domain) * skip_domain;
    let x_per_block = block_x / skip_domain;

    let mut block_ctxs_h = Vec::new();
    let mut air_offsets_h: Vec<u32> = vec![0];
    let mut total_intermediates: usize = 0;

    struct AirLayout {
        buffer_stride: u32,
        intermediates_size: usize,
        blocks_per_air: u32,
    }
    let layouts: Vec<AirLayout> = descs
        .iter()
        .map(|d| {
            let num_x_blocks = d.num_x.div_ceil(x_per_block);
            let blocks_per_air = num_x_blocks * num_cosets;
            let buffer_stride = num_cosets * num_x_blocks * block_x;
            let intermediates_size = if d.buffer_size_logup > BUFFER_THRESHOLD_R0 {
                buffer_stride as usize * d.buffer_size_logup as usize
            } else {
                0
            };
            AirLayout { buffer_stride, intermediates_size, blocks_per_air }
        })
        .collect();

    for (local_air_idx, layout) in layouts.iter().enumerate() {
        for local_block in 0..layout.blocks_per_air {
            block_ctxs_h.push(Round0BlockCtx {
                local_block_idx: local_block,
                air_idx: local_air_idx as u32,
            });
        }
        air_offsets_h.push(*air_offsets_h.last().unwrap() + layout.blocks_per_air);
        total_intermediates += layout.intermediates_size;
    }

    let total_blocks = block_ctxs_h.len() as u32;
    if total_blocks == 0 {
        return Ok(DeviceBuffer::new());
    }

    let mut intermediates = if total_intermediates > 0 {
        DeviceBuffer::<F>::with_capacity(total_intermediates)
    } else {
        DeviceBuffer::<F>::new()
    };

    let mut logup_ctxs_h = Vec::with_capacity(descs.len());
    let mut intermediates_offset: usize = 0;
    for (d, layout) in descs.iter().zip(&layouts) {
        let logup = d.logup_rules.as_ref().expect("logup desc must have rules");
        let d_intermediates = if layout.intermediates_size > 0 {
            let ptr = unsafe { intermediates.as_mut_ptr().add(intermediates_offset) };
            intermediates_offset += layout.intermediates_size;
            ptr
        } else {
            std::ptr::null_mut()
        };
        logup_ctxs_h.push(Round0LogupCtx {
            selectors_cube: d.selectors_cube,
            preprocessed: d.preprocessed,
            main_parts: d.d_main_parts,
            eq_cube: d.eq_cube,
            public_values: d.public_values,
            d_rules: logup.d_rules.as_raw_ptr() as *const std::ffi::c_void,
            rules_len: logup.rules_len,
            buffer_size: logup.buffer_size,
            d_intermediates,
            buffer_stride: layout.buffer_stride,
            numer_weights: logup.d_numer_weights.as_ptr(),
            denom_weights: logup.d_denom_weights.as_ptr(),
            denom_sum_init: logup.denom_sum_init,
            num_x: d.num_x,
            height: d.height,
            g_shift: d.g_shift,
        });
    }

    let d_block_ctxs = block_ctxs_h.to_device()?;
    let d_logup_ctxs = logup_ctxs_h.to_device()?;
    let d_air_offsets = air_offsets_h.to_device()?;

    let output_size = descs.len() * num_cosets as usize * skip_domain as usize;
    let tmp_sums_size = total_blocks as usize * num_cosets as usize * skip_domain as usize;
    let mut tmp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(tmp_sums_size);
    let mut output = DeviceBuffer::<Frac<EF>>::with_capacity(output_size);

    unsafe {
        batched_logup_r0_eval_interactions(
            &mut tmp_sums_buffer,
            &mut output,
            &d_block_ctxs,
            &d_logup_ctxs,
            &d_air_offsets,
            total_blocks,
            descs.len() as u32,
            num_cosets,
            skip_domain,
            block_x,
        )?;
    }
    Ok(output)
}

// The batched_round0_phase4 function is a stub for now.
// The full implementation has been verified at the CUDA kernel level but
// needs additional integration work for the orchestration layer.
#[allow(unused_variables, clippy::too_many_arguments)]
pub(crate) fn batched_round0_phase4_stub() {
    // Placeholder
}
