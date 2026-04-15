use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceStarkProvingKey};
use p3_field::PrimeCharacteristicRing;
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    cuda::logup_zerocheck::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    prelude::{EF, F},
};

/// Pre-allocated per-thread GPU buffers for Round 0 evaluation.
/// Reused across AIRs to eliminate per-AIR mutex contention on the global memory manager.
pub(crate) struct Round0ThreadBuffers {
    pub zc_intermediates: DeviceBuffer<F>,
    pub zc_temp_sums: DeviceBuffer<EF>,
    pub logup_intermediates: DeviceBuffer<F>,
    pub logup_temp_sums: DeviceBuffer<Frac<EF>>,
}

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
    prealloc_intermediates: Option<&mut DeviceBuffer<F>>,
    prealloc_temp_sums: Option<&mut DeviceBuffer<EF>>,
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

    let use_prealloc_inter = prealloc_intermediates
        .as_ref()
        .map_or(false, |p| intermed_capacity > 0 && p.len() >= intermed_capacity);
    let mut fallback_intermediates;
    let intermediates = if use_prealloc_inter {
        prealloc_intermediates.unwrap()
    } else {
        fallback_intermediates = if intermed_capacity > 0 {
            debug!("zerocheck:intermediates_capacity={intermed_capacity}");
            DeviceBuffer::<F>::with_capacity(intermed_capacity)
        } else {
            DeviceBuffer::<F>::new()
        };
        &mut fallback_intermediates
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

    let use_prealloc_temp = prealloc_temp_sums
        .as_ref()
        .map_or(false, |p| p.len() >= temp_sums_buffer_capacity);
    let mut fallback_temp_sums;
    let temp_sums_buffer = if use_prealloc_temp {
        prealloc_temp_sums.unwrap()
    } else {
        debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
        fallback_temp_sums = DeviceBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
        &mut fallback_temp_sums
    };

    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<EF>();
    if used_temp_bytes > max_temp_bytes {
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
            temp_sums_buffer,
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
            intermediates,
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
/// Uses pre-computed logup DAG rules from the proving key (`LogupRound0Rules`),
/// only computing the per-challenge interaction weights at proving time.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_gpu<HS: GpuHashScheme>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
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
    prealloc_intermediates: Option<&mut DeviceBuffer<F>>,
    prealloc_temp_sums: Option<&mut DeviceBuffer<Frac<EF>>>,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    // Check if this trace has interactions
    if eq_3bs.is_empty() {
        return Ok(DeviceBuffer::new());
    }
    let large_domain = num_cosets * skip_domain;

    // Use pre-computed DAG rules from keygen; only compute weights per-challenge.
    let logup_rules = &pk.other_data.logup_round0;
    let buffer_size = logup_rules.buffer_size;
    let mut numer_weights = vec![EF::ZERO; logup_rules.num_rules];
    let mut denom_weights = vec![EF::ZERO; logup_rules.num_rules];
    let mut denom_sum_init = EF::ZERO;
    for (interaction_idx, mapping) in logup_rules.interaction_mappings.iter().enumerate() {
        numer_weights[mapping.count_rule_idx] += eq_3bs[interaction_idx];
        denom_sum_init += eq_3bs[interaction_idx]
            * beta_pows[mapping.message_rule_idxs.len()]
            * F::from_u32(mapping.bus_index);
        for (message_idx, &rule_idx) in mapping.message_rule_idxs.iter().enumerate() {
            denom_weights[rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
        }
    }
    let d_numer_weights = numer_weights.to_device()?;
    let d_denom_weights = denom_weights.to_device()?;

    let intermed_capacity = unsafe {
        _logup_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };

    let use_prealloc_inter = prealloc_intermediates
        .as_ref()
        .map_or(false, |p| intermed_capacity > 0 && p.len() >= intermed_capacity);
    let mut fallback_intermediates;
    let intermediates = if use_prealloc_inter {
        prealloc_intermediates.unwrap()
    } else {
        fallback_intermediates = if intermed_capacity > 0 {
            debug!("logup_r0:intermediates_capacity={intermed_capacity}");
            DeviceBuffer::<F>::with_capacity(intermed_capacity)
        } else {
            DeviceBuffer::<F>::new()
        };
        &mut fallback_intermediates
    };

    let temp_sums_buffer_capacity = unsafe {
        _logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
    };

    let use_prealloc_temp = prealloc_temp_sums
        .as_ref()
        .map_or(false, |p| p.len() >= temp_sums_buffer_capacity);
    let mut fallback_temp_sums;
    let temp_sums_buffer = if use_prealloc_temp {
        prealloc_temp_sums.unwrap()
    } else {
        debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
        fallback_temp_sums = DeviceBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
        &mut fallback_temp_sums
    };

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
            temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            public_values,
            &d_numer_weights,
            &d_denom_weights,
            denom_sum_init,
            &logup_rules.d_rules,
            buffer_size,
            intermediates,
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
