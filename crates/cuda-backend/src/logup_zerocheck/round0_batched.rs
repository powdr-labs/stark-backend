use std::collections::BTreeMap;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    prover::{fractional_sumcheck_gkr::Frac, MatrixDimensions},
};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

use openvm_cuda_common::error::CudaError;

use super::{errors::Round0EvalError, LogupZerocheckError, Round0AirWorkItem, Round0ExtractTables};
use crate::{
    cuda::logup_zerocheck::{
        _round0_extract_logup_polys, _round0_extract_zerocheck_poly,
        batched_logup_r0_eval_interactions, batched_zerocheck_r0_eval_constraints,
        Round0BlockCtx, Round0LogupCtx, Round0ZcCtx,
    },
    hash_scheme::GpuHashScheme,
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    prelude::{EF, F},
};

const COSET_PARALLEL_THRESHOLD: u32 = 32768;
const MAX_THREADS_R0: u32 = 128;
const MIN_BATCHABLE_AIRS: usize = 50;
/// Max intermediates per AIR (Fp elements). Keeps aggregate arena small enough
/// that L2-cache reuse across wave fronts remains effective.
const MAX_INTERMEDIATES_PER_AIR: usize = 16 * 1024 * 1024;
/// Max total aggregate intermediates (bytes) across all batched AIRs.
/// Must fit in L2 cache (72 MB on RTX 4090) to avoid cache thrashing.
const MAX_AGGREGATE_INTERMEDIATES_BYTES: usize = 64 * 1024 * 1024;

/// Identify which work items are eligible for the batched Round 0 path.
///
/// A work item is batchable if it uses the coset-parallel kernel variant
/// (num_x * skip_domain < COSET_PARALLEL_THRESHOLD) and its intermediates
/// fit in the per-AIR memory budget. Returns an all-false vec if fewer than
/// MIN_BATCHABLE_AIRS qualify.
#[inline(never)]
pub fn identify_batchable_airs<HS: GpuHashScheme>(
    work_items: &[Round0AirWorkItem<HS>],
    l_skip: usize,
) -> Vec<bool> {
    let skip_domain = 1u32 << l_skip;
    let mut mask = vec![false; work_items.len()];
    let mut count = 0usize;
    let mut total_intermed_bytes: usize = 0;

    // Compute per-AIR intermediates sizes (sorted by ascending size for greedy packing)
    let mut candidates: Vec<(usize, usize)> = Vec::new(); // (index, intermed_bytes)
    for (idx, w) in work_items.iter().enumerate() {
        let n_lift = w.n.max(0) as usize;
        let num_x = 1u32 << n_lift;
        if num_x * skip_domain >= COSET_PARALLEL_THRESHOLD {
            continue;
        }
        // Check per-AIR intermediates budget
        let local_constraint_deg = w.single_pk.vk.max_constraint_degree as u32;
        let num_cosets_zc = local_constraint_deg.saturating_sub(1);
        let block_x = std::cmp::min(MAX_THREADS_R0, skip_domain * num_x);
        let block_x = (block_x / skip_domain) * skip_domain;
        let x_per_block = block_x / skip_domain;
        let num_x_blocks = num_x.div_ceil(x_per_block);
        let buffer_stride = num_cosets_zc * num_x_blocks * block_x;
        let buffer_size = w.single_pk.other_data.zerocheck_round0.inner.buffer_size;
        let intermediates = buffer_stride as usize * std::cmp::max(buffer_size as usize, 1);
        if intermediates > MAX_INTERMEDIATES_PER_AIR {
            continue;
        }
        // Only batch AIRs with zero or small intermediates
        // to keep aggregate within L2 cache budget
        // Skip AIRs without zerocheck constraints AND without interactions
        if w.zc_intermed_cap == 0 && w.eq_3bs.is_empty() {
            continue;
        }
        let intermed_bytes = intermediates * std::mem::size_of::<F>();
        candidates.push((idx, intermed_bytes));
    }

    // Sort by ascending intermediates size and greedily add AIRs within budget
    candidates.sort_by_key(|&(_, bytes)| bytes);
    for &(idx, bytes) in &candidates {
        if total_intermed_bytes + bytes > MAX_AGGREGATE_INTERMEDIATES_BYTES {
            continue;
        }
        mask[idx] = true;
        count += 1;
        total_intermed_bytes += bytes;
    }

    if count < MIN_BATCHABLE_AIRS {
        mask.fill(false);
        count = 0;
        total_intermed_bytes = 0;
    }
    tracing::info!(
        "Batch Round 0: {} of {} candidates selected, aggregate intermediates: {} MB",
        count,
        candidates.len(),
        total_intermed_bytes / 1_000_000
    );
    mask
}

/// Pre-built logup DAG rules for one AIR, ready for the batched kernel.
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

/// Per-AIR descriptor used internally by the batch orchestration.
struct BatchAirDesc {
    work_item_idx: usize,
    num_x: u32,
    height: u32,
    n: isize,
    g_shift: F,
    has_zc_constraints: bool,
    num_cosets_zc: u32,
    num_cosets_logup: u32,
    buffer_size_zc: u32,
    // Device pointers
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
    // Batch offsets into d_batch_array
    zc_batch_offset: usize,
    numer_batch_offset: usize,
    denom_batch_offset: usize,
}

unsafe impl Send for BatchAirDesc {}
unsafe impl Sync for BatchAirDesc {}

/// State returned by `batch_round0_eval_only` containing per-group eval results.
/// Must be kept alive until all Phase 2 extraction is complete.
pub struct BatchEvalState {
    /// Zerocheck eval output per group: (num_cosets, output buffer, air indices into descs)
    zc_groups: Vec<(u32, DeviceBuffer<EF>, Vec<usize>)>,
    /// Logup eval output per group: (num_cosets, output buffer, air indices into descs)
    logup_groups: Vec<(u32, DeviceBuffer<Frac<EF>>, Vec<usize>)>,
    /// Per-AIR descriptors (kept alive for extraction metadata)
    descs: Vec<BatchAirDesc>,
    /// Batched main pointer upload (kept alive so device pointers remain valid)
    _d_all_main_ptrs: DeviceBuffer<*const F>,
    /// Intermediates arenas per zc group (kept alive so device pointers remain valid during extraction)
    _zc_intermediates: Vec<DeviceBuffer<F>>,
    /// Intermediates arenas per logup group
    _logup_intermediates: Vec<DeviceBuffer<F>>,
    /// Skip domain
    skip_domain: u32,
}

impl BatchEvalState {
    /// Drop intermediates arenas to free GPU memory before Phase 2.
    /// Safe to call after current_stream_sync() ensures all eval kernels are done.
    pub fn drop_intermediates(&mut self) {
        self._zc_intermediates.clear();
        self._logup_intermediates.clear();
    }
}

/// Execute batched Round 0 eval-only for all batchable AIRs.
///
/// Runs batched eval+reduce kernels for groups of small AIRs on the default
/// stream. Does NOT run extraction — that is deferred to Phase 2 worker threads
/// via `extract_batched_air()`. The returned `BatchEvalState` must be kept alive
/// until all extraction is complete.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn batch_round0_eval_only<HS: GpuHashScheme>(
    work_items: &[Round0AirWorkItem<HS>],
    batchable: &[bool],
    l_skip: usize,
    eq_xis: &rustc_hash::FxHashMap<usize, super::EqEvalLayers<EF>>,
    eq_3b_per_trace: &[Vec<EF>],
    beta_pows: &[EF],
) -> Result<BatchEvalState, LogupZerocheckError> {
    let skip_domain = 1u32 << l_skip;
    let batchable_count = batchable.iter().filter(|&&b| b).count();
    tracing::info!(" processing {batchable_count} small AIRs via batched kernels");

    // Build main_parts device pointers in a single batched upload.
    let mut all_main_ptrs: Vec<*const F> = Vec::new();
    let mut main_ptrs_offsets: Vec<usize> = Vec::with_capacity(work_items.len());
    for (idx, w) in work_items.iter().enumerate() {
        if !batchable[idx] {
            main_ptrs_offsets.push(0);
            continue;
        }
        let offset = all_main_ptrs.len();
        main_ptrs_offsets.push(offset);
        for committed in w.cached_mains {
            all_main_ptrs.push(committed.trace.buffer().as_ptr());
        }
        all_main_ptrs.push(w.common_main.buffer().as_ptr());
    }
    let d_all_main_ptrs = if all_main_ptrs.is_empty() {
        DeviceBuffer::new()
    } else {
        all_main_ptrs.to_device()?
    };

    // Build per-AIR descriptors.
    tracing::info!(" building descriptors for {batchable_count} AIRs...");
    let t_desc_start = std::time::Instant::now();
    let mut descs: Vec<BatchAirDesc> = Vec::with_capacity(batchable_count);
    for (idx, w) in work_items.iter().enumerate() {
        if !batchable[idx] {
            continue;
        }
        let single_pk = w.single_pk;
        let local_constraint_deg = single_pk.vk.max_constraint_degree as usize;
        let n_lift = w.n.max(0) as usize;
        let num_x = 1u32 << n_lift;
        let height = w.common_main.height() as u32;
        let log_large_domain = p3_util::log2_ceil_usize(local_constraint_deg << l_skip);
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

        let single_air_constraints = SymbolicConstraints::from(&single_pk.vk.symbolic_constraints);
        let logup_rules = build_logup_rules_for_air(
            &single_air_constraints,
            beta_pows,
            &eq_3b_per_trace[w.trace_idx],
        )?;

        let d_main_parts = if d_all_main_ptrs.is_empty() {
            std::ptr::null()
        } else {
            unsafe { d_all_main_ptrs.as_ptr().add(main_ptrs_offsets[idx]) }
        };

        let has_zc_constraints = !single_pk
            .vk
            .symbolic_constraints
            .constraints
            .constraint_idx
            .is_empty();

        let d_lambda_pows = w.d_lambda_pows;

        descs.push(BatchAirDesc {
            work_item_idx: idx,
            num_x,
            height,
            n: w.n,
            g_shift,
            has_zc_constraints,
            num_cosets_zc,
            num_cosets_logup,
            buffer_size_zc,
            selectors_cube: w.selectors_cube.buffer().as_ptr(),
            preprocessed,
            d_main_parts,
            eq_cube: eq_xis[&n_lift].get_ptr(n_lift),
            lambda_pows: d_lambda_pows.as_ptr(),
            public_values: w.public_values.as_ptr(),
            d_zc_rules: zc_rules.inner.d_rules.as_raw_ptr(),
            d_zc_used_nodes: zc_rules.inner.d_used_nodes.as_ptr(),
            zc_rules_len: zc_rules.inner.d_rules.len(),
            zc_used_nodes_len: zc_rules.inner.d_used_nodes.len(),
            lambda_len: d_lambda_pows.len(),
            logup_rules,
            zc_batch_offset: w.zc_batch_offset,
            numer_batch_offset: w.numer_batch_offset,
            denom_batch_offset: w.denom_batch_offset,
        });
    }

    tracing::info!(" descriptors built in {}ms", t_desc_start.elapsed().as_millis());

    // Group by num_cosets_zc, launch batched zerocheck eval.
    let mut zc_group_map: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, d) in descs.iter().enumerate() {
        if d.has_zc_constraints && d.num_cosets_zc > 0 {
            zc_group_map.entry(d.num_cosets_zc).or_default().push(i);
        }
    }

    let mut zc_groups = Vec::new();
    let mut zc_intermediates = Vec::new();
    for (&num_cosets, air_indices) in &zc_group_map {
        let group_descs: Vec<&BatchAirDesc> = air_indices.iter().map(|&i| &descs[i]).collect();
        tracing::info!(" launching zc group num_cosets={num_cosets} airs={}", group_descs.len());
        let t0 = std::time::Instant::now();
        let (output, intermediates) =
            evaluate_batched_zerocheck_group(&group_descs, num_cosets, skip_domain)?;
        tracing::info!(" zc group done in {}ms", t0.elapsed().as_millis());
        zc_groups.push((num_cosets, output, air_indices.clone()));
        zc_intermediates.push(intermediates);
    }

    // Group by num_cosets_logup, launch batched logup eval.
    let mut logup_group_map: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, d) in descs.iter().enumerate() {
        if d.logup_rules.is_some() && d.num_cosets_logup > 0 {
            logup_group_map
                .entry(d.num_cosets_logup)
                .or_default()
                .push(i);
        }
    }

    let mut logup_groups = Vec::new();
    let mut logup_intermediates = Vec::new();
    for (&num_cosets, air_indices) in &logup_group_map {
        let group_descs: Vec<&BatchAirDesc> = air_indices.iter().map(|&i| &descs[i]).collect();
        tracing::info!(" launching logup group num_cosets={num_cosets} airs={}", group_descs.len());
        let t0 = std::time::Instant::now();
        let (output, intermediates) =
            evaluate_batched_logup_group(&group_descs, num_cosets, skip_domain)?;
        tracing::info!(" logup group done in {}ms", t0.elapsed().as_millis());
        logup_groups.push((num_cosets, output, air_indices.clone()));
        logup_intermediates.push(intermediates);
    }

    tracing::info!(" eval done for {batchable_count} AIRs");
    Ok(BatchEvalState {
        zc_groups,
        logup_groups,
        descs,
        _d_all_main_ptrs: d_all_main_ptrs,
        _zc_intermediates: zc_intermediates,
        _logup_intermediates: logup_intermediates,
        skip_domain,
    })
}

/// Extract polynomial coefficients for a single batched AIR into d_batch_array.
///
/// Called from Phase 2 worker threads. The `batch_eval_state` must have been
/// created by `batch_round0_eval_only()` and the default stream must have been
/// synced before worker threads begin.
#[inline(never)]
pub fn extract_batched_air(
    w_idx: usize,
    local_constraint_deg: usize,
    batch_eval_state: &BatchEvalState,
    d_batch_ptr: *mut EF,
    extract_tables: &rustc_hash::FxHashMap<usize, Round0ExtractTables>,
) -> Result<(), LogupZerocheckError> {
    // Find which desc this work item corresponds to
    let desc_idx = batch_eval_state
        .descs
        .iter()
        .position(|d| d.work_item_idx == w_idx)
        .expect("extract_batched_air called for non-batched work item");
    let d = &batch_eval_state.descs[desc_idx];

    // Extract zerocheck polynomial
    for (num_cosets, output, air_indices) in &batch_eval_state.zc_groups {
        if let Some(local_idx) = air_indices.iter().position(|&i| i == desc_idx) {
            let stride = *num_cosets as usize * batch_eval_state.skip_domain as usize;
            let tables = &extract_tables[&local_constraint_deg];
            let err = unsafe {
                _round0_extract_zerocheck_poly(
                    d_batch_ptr.add(d.zc_batch_offset),
                    output.as_ptr().add(local_idx * stride),
                    tables.zc_transform.as_ptr(),
                    tables.zc_input_size,
                    tables.zc_output_size,
                )
            };
            if err != 0 {
                return Err(Round0EvalError::Cuda(CudaError::new(err)).into());
            }
            break;
        }
    }

    // Extract logup polynomials
    for (num_cosets, output, air_indices) in &batch_eval_state.logup_groups {
        if let Some(local_idx) = air_indices.iter().position(|&i| i == desc_idx) {
            let stride = *num_cosets as usize * batch_eval_state.skip_domain as usize;
            let tables = &extract_tables[&local_constraint_deg];
            let norm_factor: EF = if d.n.is_negative() {
                EF::from(F::from_u32(1 << d.n.unsigned_abs()).inverse())
            } else {
                EF::ONE
            };
            let err = unsafe {
                _round0_extract_logup_polys(
                    d_batch_ptr.add(d.numer_batch_offset),
                    d_batch_ptr.add(d.denom_batch_offset),
                    output.as_ptr().add(local_idx * stride),
                    tables.logup_transform.as_ptr(),
                    tables.logup_size,
                    tables.logup_size,
                    norm_factor,
                )
            };
            if err != 0 {
                return Err(Round0EvalError::Cuda(CudaError::new(err)).into());
            }
            break;
        }
    }

    Ok(())
}

#[inline(never)]
fn evaluate_batched_zerocheck_group(
    descs: &[&BatchAirDesc],
    num_cosets: u32,
    skip_domain: u32,
) -> Result<(DeviceBuffer<EF>, DeviceBuffer<F>), Round0EvalError> {
    if descs.is_empty() || num_cosets == 0 {
        return Ok((DeviceBuffer::new(), DeviceBuffer::new()));
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
            let intermediates_size =
                buffer_stride as usize * std::cmp::max(d.buffer_size_zc as usize, 1);
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
        return Ok((DeviceBuffer::new(), DeviceBuffer::new()));
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
    Ok((output, intermediates))
}

#[inline(never)]
fn evaluate_batched_logup_group(
    descs: &[&BatchAirDesc],
    num_cosets: u32,
    skip_domain: u32,
) -> Result<(DeviceBuffer<Frac<EF>>, DeviceBuffer<F>), Round0EvalError> {
    if descs.is_empty() || num_cosets == 0 {
        return Ok((DeviceBuffer::new(), DeviceBuffer::new()));
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
        return Ok((DeviceBuffer::new(), DeviceBuffer::new()));
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
    Ok((output, intermediates))
}
