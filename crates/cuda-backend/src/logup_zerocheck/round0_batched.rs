use std::collections::BTreeMap;

use itertools::Itertools;
use openvm_cuda_common::{
    copy::{MemCopyD2HStreamSync, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::current_stream_sync,
};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    p3_matrix::dense::RowMajorMatrix,
    poly_common::UnivariatePoly,
    prover::{
        fractional_sumcheck_gkr::Frac, sumcheck::sumcheck_round0_deg,
        DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
    },
};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_ceil_usize;
use tracing::info;

use super::{errors::Round0EvalError, LogupZerocheckError, LogupZerocheckGpu, Round0AirResult};
use crate::{
    cuda::logup_zerocheck::{
        batched_logup_r0_eval_interactions, batched_zerocheck_r0_eval_constraints,
        Round0BlockCtx, Round0LogupCtx, Round0ZcCtx,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    prelude::{EF, F},
};

const COSET_PARALLEL_THRESHOLD: u32 = 32768;
const MAX_THREADS_R0: u32 = 128;
const MIN_BATCHABLE_AIRS: usize = 50;
/// Max intermediates per AIR in the batched path. AIRs exceeding this are left for the per-AIR
/// path which can tune grid dimensions based on max_temp_bytes.
const MAX_INTERMEDIATES_PER_AIR: usize = 16 * 1024 * 1024; // 16M Fp elements = 64MB

/// Identify which traces are eligible for the batched Round 0 path.
///
/// A trace is batchable if num_x * skip_domain < COSET_PARALLEL_THRESHOLD (uses coset-parallel
/// kernel). If fewer than MIN_BATCHABLE_AIRS traces qualify, returns all-false.
#[inline(never)]
pub fn identify_batchable_airs<HS: GpuHashScheme>(
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    n_per_trace: &[isize],
    l_skip: usize,
) -> Vec<bool> {
    let skip_domain = 1u32 << l_skip;
    let mut mask = vec![false; ctx.per_trace.len()];
    let mut count = 0usize;
    for (trace_idx, ((air_idx, _), &n)) in
        ctx.per_trace.iter().zip(n_per_trace).enumerate()
    {
        let n_lift = n.max(0) as usize;
        let num_x = 1u32 << n_lift;
        if num_x * skip_domain >= COSET_PARALLEL_THRESHOLD {
            continue;
        }
        // Check if intermediates per AIR would be too large for the batched path.
        let single_pk = &pk.per_air[*air_idx];
        let local_constraint_deg = single_pk.vk.max_constraint_degree as u32;
        let num_cosets_zc = local_constraint_deg.saturating_sub(1);
        let block_x = std::cmp::min(MAX_THREADS_R0, skip_domain * num_x);
        let block_x = (block_x / skip_domain) * skip_domain;
        let x_per_block = block_x / skip_domain;
        let num_x_blocks = num_x.div_ceil(x_per_block);
        let buffer_stride_zc = num_cosets_zc * num_x_blocks * block_x;
        let buffer_size_zc = single_pk.other_data.zerocheck_round0.inner.buffer_size;
        let intermediates_zc = buffer_stride_zc as usize * buffer_size_zc as usize;
        if intermediates_zc > MAX_INTERMEDIATES_PER_AIR {
            continue;
        }
        mask[trace_idx] = true;
        count += 1;
    }
    if count < MIN_BATCHABLE_AIRS {
        mask.fill(false);
    }
    mask
}

/// Holds pre-built logup DAG rules for one AIR.
struct LogupRulesForAir {
    d_rules: DeviceBuffer<u128>,
    rules_len: usize,
    buffer_size: u32,
    d_numer_weights: DeviceBuffer<EF>,
    d_denom_weights: DeviceBuffer<EF>,
    denom_sum_init: EF,
}

/// Build logup interaction DAG rules for a single AIR.
#[inline(never)]
fn build_logup_rules_for_air(
    symbolic: &SymbolicConstraints<F>,
    beta_pows: &[EF],
    eq_3bs: &[EF],
) -> Result<Option<LogupRulesForAir>, LogupZerocheckError> {
    if eq_3bs.is_empty() || symbolic.interactions.is_empty() {
        return Ok(None);
    }
    let mut dag_builder = SymbolicDagBuilder::new();
    let mut sorted_used_dag_idxs = Vec::new();
    for interaction in &symbolic.interactions {
        let count = dag_builder.add_expr(&interaction.count);
        sorted_used_dag_idxs.push(count);
        sorted_used_dag_idxs
            .extend(interaction.message.iter().map(|f| dag_builder.add_expr(f)));
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
    Ok(Some(LogupRulesForAir {
        d_rules,
        rules_len: rules.rules.len(),
        buffer_size: rules.buffer_size.try_into().unwrap(),
        d_numer_weights,
        d_denom_weights,
        denom_sum_init,
    }))
}

/// Per-AIR descriptor for batched Round 0 (internal to this module).
struct AirDesc {
    trace_idx: usize,
    num_x: u32,
    height: u32,
    n: isize,
    g_shift: F,
    has_zc_constraints: bool,
    num_cosets_zc: u32,
    num_cosets_logup: u32,
    buffer_size_zc: u32,
    // Device pointers (valid for kernel launch duration)
    selectors_cube: *const F,
    preprocessed: *const F,
    d_main_parts: *const *const F,
    eq_cube: *const EF,
    lambda_pows: *const EF,
    public_values: *const F,
    d_zc_rules: *const std::ffi::c_void,
    d_zc_used_nodes: *const usize,
    zc_rules_len: usize,
    zc_used_nodes_len: usize,
    lambda_len: usize,
    logup_rules: Option<LogupRulesForAir>,
}

// SAFETY: Device pointers are valid for the kernel launch duration. The batch function ensures
// all referenced device buffers outlive the kernel launches.
unsafe impl Send for AirDesc {}
unsafe impl Sync for AirDesc {}

const BUFFER_THRESHOLD_R0: u32 = 16;

/// Execute batched Round 0 for small AIRs.
///
/// Called AFTER `sumcheck_uni_round0_polys` returns, so all precomputed data on `prover` is
/// populated. This function is `#[inline(never)]` to prevent optimizer cross-contamination
/// with the existing per-AIR code path.
#[inline(never)]
pub fn batch_round0_small_airs<HS: GpuHashScheme>(
    prover: &mut LogupZerocheckGpu<HS>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    skip_mask: &[bool],
) -> Result<Vec<Round0AirResult>, LogupZerocheckError> {
    let l_skip = prover.l_skip;
    let skip_domain = 1u32 << l_skip;

    let batchable_count = skip_mask.iter().filter(|&&b| b).count();
    if batchable_count == 0 {
        return Ok(vec![]);
    }
    info!("batch_round0: processing {batchable_count} small AIRs via batched kernels");
    info!("batch_round0: processing {batchable_count} small AIRs via batched kernels");

    let d_lambda_pows = prover
        .lambda_pows
        .as_ref()
        .expect("lambda powers must be set before round-0 evaluation");

    // Step 0: Build per-AIR main_parts device pointers in a single batched upload.
    let mut all_main_ptrs: Vec<*const F> = Vec::new();
    let mut main_ptrs_offsets: Vec<usize> = Vec::new();

    for (trace_idx, (_air_idx, air_ctx)) in ctx.per_trace.iter().enumerate() {
        if !skip_mask[trace_idx] {
            main_ptrs_offsets.push(0);
            continue;
        }
        let offset = all_main_ptrs.len();
        main_ptrs_offsets.push(offset);
        for committed in &air_ctx.cached_mains {
            all_main_ptrs.push(committed.trace.buffer().as_ptr());
        }
        all_main_ptrs.push(air_ctx.common_main.buffer().as_ptr());
    }

    let d_all_main_ptrs = if all_main_ptrs.is_empty() {
        DeviceBuffer::new()
    } else {
        all_main_ptrs.to_device()?
    };

    // Build AirDesc for each batchable trace.
    let mut descs: Vec<AirDesc> = Vec::with_capacity(batchable_count);
    for (trace_idx, ((air_idx, _air_ctx), &n)) in
        ctx.per_trace.iter().zip(&prover.n_per_trace).enumerate()
    {
        if !skip_mask[trace_idx] {
            continue;
        }
        let single_pk = &prover.pk.per_air[*air_idx];
        let local_constraint_deg = single_pk.vk.max_constraint_degree as usize;
        let n_lift = n.max(0) as usize;
        let num_x = 1u32 << n_lift;
        let height = ctx.per_trace[trace_idx].1.common_main.height() as u32;
        let log_large_domain = log2_ceil_usize(local_constraint_deg << l_skip);
        let g_shift = F::two_adic_generator(log_large_domain);
        let num_cosets_zc = local_constraint_deg.saturating_sub(1) as u32;
        let num_cosets_logup = local_constraint_deg as u32;

        let zc_rules = &single_pk.other_data.zerocheck_round0;
        let buffer_size_zc = zc_rules.inner.buffer_size;

        let preprocessed = single_pk
            .preprocessed_data
            .as_ref()
            .map(|cd| cd.trace.buffer().as_ptr())
            .unwrap_or(std::ptr::null());

        let eq_xi_tree = &prover.eq_xis[&n_lift];

        let single_air_constraints =
            SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
        let logup_rules = build_logup_rules_for_air(
            &single_air_constraints,
            &prover.beta_pows,
            &prover.eq_3b_per_trace[trace_idx],
        )?;

        // Pointer into the batched main_ptrs device buffer
        let d_main_parts = if d_all_main_ptrs.is_empty() {
            std::ptr::null()
        } else {
            unsafe { d_all_main_ptrs.as_ptr().add(main_ptrs_offsets[trace_idx]) }
        };

        let has_zc_constraints = !single_pk
            .vk
            .symbolic_constraints
            .constraints
            .constraint_idx
            .is_empty();

        descs.push(AirDesc {
            trace_idx,
            num_x,
            height,
            n,
            g_shift,
            has_zc_constraints,
            num_cosets_zc,
            num_cosets_logup,
            buffer_size_zc,
            selectors_cube: prover.sels_per_trace_base[trace_idx].buffer().as_ptr(),
            preprocessed,
            d_main_parts,
            eq_cube: eq_xi_tree.get_ptr(n_lift),
            lambda_pows: d_lambda_pows.as_ptr(),
            public_values: prover.public_values_per_trace[trace_idx].as_ptr(),
            d_zc_rules: zc_rules.inner.d_rules.as_raw_ptr(),
            d_zc_used_nodes: zc_rules.inner.d_used_nodes.as_ptr(),
            zc_rules_len: zc_rules.inner.d_rules.len(),
            zc_used_nodes_len: zc_rules.inner.d_used_nodes.len(),
            lambda_len: d_lambda_pows.len(),
            logup_rules,
        });
    }

    // Step 1: Group by num_cosets_zc, launch batched zerocheck.
    let mut zc_groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, d) in descs.iter().enumerate() {
        if d.has_zc_constraints && d.num_cosets_zc > 0 {
            zc_groups.entry(d.num_cosets_zc).or_default().push(i);
        }
    }

    // Store zerocheck raw outputs per desc index
    let mut zc_raw_outputs: Vec<Option<Vec<EF>>> = vec![None; descs.len()];

    for (&num_cosets, air_indices) in &zc_groups {
        let group_descs: Vec<&AirDesc> = air_indices.iter().map(|&i| &descs[i]).collect();
        info!(
            "batch_round0: zc group num_cosets={num_cosets} airs={}",
            group_descs.len(),
        );
        let output =
            evaluate_batched_zerocheck_group(&group_descs, num_cosets, skip_domain, l_skip)?;
        if !output.is_empty() {
            // Sync to catch kernel errors early
            current_stream_sync().map_err(|e| Round0EvalError::Copy(e.into()))?;
            let host_output = output.to_host_on_current_stream()?;
            let stride = num_cosets as usize * skip_domain as usize;
            for (local_idx, &desc_idx) in air_indices.iter().enumerate() {
                let start = local_idx * stride;
                let end = start + stride;
                zc_raw_outputs[desc_idx] = Some(host_output[start..end].to_vec());
            }
        }
    }
    info!("batch_round0: zerocheck done OK");

    // Step 2: Group by num_cosets_logup, launch batched logup.
    let mut logup_groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, d) in descs.iter().enumerate() {
        if d.logup_rules.is_some() && d.num_cosets_logup > 0 {
            logup_groups.entry(d.num_cosets_logup).or_default().push(i);
        }
    }

    let mut logup_raw_outputs: Vec<Option<Vec<Frac<EF>>>> = vec![None; descs.len()];

    for (&num_cosets, air_indices) in &logup_groups {
        let group_descs: Vec<&AirDesc> = air_indices.iter().map(|&i| &descs[i]).collect();
        info!(
            "batch_round0: logup group num_cosets={num_cosets} airs={}",
            group_descs.len()
        );
        let output =
            evaluate_batched_logup_group(&group_descs, num_cosets, skip_domain, l_skip)?;
        if !output.is_empty() {
            current_stream_sync().map_err(|e| Round0EvalError::Copy(e.into()))?;
            let host_output = output.to_host_on_current_stream()?;
            let stride = num_cosets as usize * skip_domain as usize;
            for (local_idx, &desc_idx) in air_indices.iter().enumerate() {
                let start = local_idx * stride;
                let end = start + stride;
                logup_raw_outputs[desc_idx] = Some(host_output[start..end].to_vec());
            }
        }
    }

    // Step 4: CPU post-processing — transpose + iDFT for each batchable AIR.
    let results: Vec<Round0AirResult> = descs
        .iter()
        .enumerate()
        .map(|(desc_idx, d)| {
            let local_constraint_deg = (d.num_cosets_zc + 1) as usize;

            let zerocheck_poly = zc_raw_outputs[desc_idx].as_ref().map(|q_evals| {
                let num_cosets_zc = d.num_cosets_zc as usize;
                let omega_root = d.g_shift;
                let mut values = EF::zero_vec(num_cosets_zc << l_skip);
                for coset_idx in 0..num_cosets_zc {
                    for i in 0..1 << l_skip {
                        values[i * num_cosets_zc + coset_idx] =
                            q_evals[(coset_idx << l_skip) + i];
                    }
                }
                let q = UnivariatePoly::from_geometric_cosets_evals_idft(
                    RowMajorMatrix::new(values, num_cosets_zc),
                    omega_root,
                    omega_root,
                );
                let sp_0_deg = sumcheck_round0_deg(l_skip, local_constraint_deg);
                let coeffs = (0..=sp_0_deg)
                    .map(|i| {
                        let mut c = -*q.coeffs().get(i).unwrap_or(&EF::ZERO);
                        if i >= 1 << l_skip {
                            c += q.coeffs()[i - (1 << l_skip)];
                        }
                        c
                    })
                    .collect_vec();
                UnivariatePoly::new(coeffs)
            });

            let (logup_numer_poly, logup_denom_poly) =
                if let Some(evals) = logup_raw_outputs[desc_idx].as_ref() {
                    let num_cosets_logup = d.num_cosets_logup as usize;
                    let omega_root = d.g_shift;
                    let (mut numer, denom): (Vec<EF>, Vec<EF>) =
                        evals.iter().map(|frac| (frac.p, frac.q)).unzip();

                    // Negative-n normalization for very small AIRs
                    if d.n.is_negative() {
                        let norm_factor = F::from_u32(1 << d.n.unsigned_abs()).inverse();
                        for s in &mut numer {
                            *s *= norm_factor;
                        }
                    }

                    let mut numer_values = EF::zero_vec(num_cosets_logup << l_skip);
                    let mut denom_values = EF::zero_vec(num_cosets_logup << l_skip);
                    for coset_idx in 0..num_cosets_logup {
                        for i in 0..1 << l_skip {
                            let src = (coset_idx << l_skip) + i;
                            let dst = i * num_cosets_logup + coset_idx;
                            numer_values[dst] = numer[src];
                            denom_values[dst] = denom[src];
                        }
                    }
                    let numer_poly = UnivariatePoly::from_geometric_cosets_evals_idft(
                        RowMajorMatrix::new(numer_values, num_cosets_logup),
                        omega_root,
                        F::ONE,
                    );
                    let denom_poly = UnivariatePoly::from_geometric_cosets_evals_idft(
                        RowMajorMatrix::new(denom_values, num_cosets_logup),
                        omega_root,
                        F::ONE,
                    );
                    (Some(numer_poly), Some(denom_poly))
                } else {
                    (None, None)
                };

            Round0AirResult {
                trace_idx: d.trace_idx,
                zerocheck_poly,
                logup_numer_poly,
                logup_denom_poly,
            }
        })
        .collect();

    Ok(results)
}

/// Merge results from the per-AIR path (large AIRs) and the batched path (small AIRs).
#[inline(never)]
pub fn merge_results(
    mut batch_sp_poly: Vec<UnivariatePoly<EF>>,
    small_results: Vec<Round0AirResult>,
    num_present_airs: usize,
) -> Vec<UnivariatePoly<EF>> {
    for result in small_results {
        if let Some(poly) = result.zerocheck_poly {
            batch_sp_poly[2 * num_present_airs + result.trace_idx] = poly;
        }
        if let Some(poly) = result.logup_numer_poly {
            batch_sp_poly[2 * result.trace_idx] = poly;
        }
        if let Some(poly) = result.logup_denom_poly {
            batch_sp_poly[2 * result.trace_idx + 1] = poly;
        }
    }
    batch_sp_poly
}

// --- GPU kernel launch helpers ---

#[inline(never)]
fn evaluate_batched_zerocheck_group(
    descs: &[&AirDesc],
    num_cosets: u32,
    skip_domain: u32,
    _l_skip: usize,
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
            // In GLOBAL mode, each thread needs buffer_size entries with stride buffer_stride.
            // Always allocate when buffer_size > 0 since the batched kernel uses GLOBAL mode.
            // For buffer_size=0, no intermediates are needed at all.
            let intermediates_size = buffer_stride as usize
                * std::cmp::max(d.buffer_size_zc as usize, 1);
            AirLayout {
                buffer_stride,
                intermediates_size,
                blocks_per_air,
            }
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

    let intermediates = if total_intermediates > 0 {
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

#[inline(never)]
fn evaluate_batched_logup_group(
    descs: &[&AirDesc],
    num_cosets: u32,
    skip_domain: u32,
    _l_skip: usize,
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
            let logup = d.logup_rules.as_ref().unwrap();
            let num_x_blocks = d.num_x.div_ceil(x_per_block);
            let blocks_per_air = num_x_blocks * num_cosets;
            let buffer_stride = num_cosets * num_x_blocks * block_x;
            // Always allocate intermediates in GLOBAL mode (batched kernel always uses GLOBAL)
            let intermediates_size = buffer_stride as usize * logup.buffer_size as usize;
            AirLayout {
                buffer_stride,
                intermediates_size,
                blocks_per_air,
            }
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

    let intermediates = if total_intermediates > 0 {
        DeviceBuffer::<F>::with_capacity(total_intermediates)
    } else {
        DeviceBuffer::<F>::new()
    };

    let mut logup_ctxs_h = Vec::with_capacity(descs.len());
    let mut intermediates_offset: usize = 0;
    for (d, layout) in descs.iter().zip(&layouts) {
        let logup = d.logup_rules.as_ref().unwrap();
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
            d_rules: logup.d_rules.as_raw_ptr(),
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
