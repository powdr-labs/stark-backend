//! Defines the symbolic rule data to precompute and store in the GPU proving key
use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints, SymbolicDagBuilder, SymbolicExpressionDag,
    },
    keygen::types::StarkProvingKey,
    StarkProtocolConfig,
};
use p3_field::PrimeCharacteristicRing;

use crate::{
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    monomial::{
        ExpandedInteractionMonomials, ExpandedMonomials, InteractionMonomialTerm, LambdaTerm,
        MonomialHeader, PackedVar,
    },
    prelude::{EF, F},
};

pub struct AirDataGpu {
    pub interaction_rules: InteractionEvalRules,
    /// Whether to buffer vars depends on the performance and memory access patterns of the kernel.
    /// This may be tuned.
    pub zerocheck_round0: ConstraintOnlyRules<true>,
    pub zerocheck_mle: ConstraintOnlyRules<false>,
    pub zerocheck_monomials: Option<ZerocheckMonomials>,
    pub interaction_monomials: Option<InteractionMonomials>,
    /// Pre-computed Round 0 logup interaction rules.
    /// `None` for AIRs with no interactions.
    pub round0_interaction_rules: Option<Round0InteractionRules>,
}

/// Pre-computed Round 0 logup interaction rules.
/// The DAG structure and compiled rules depend only on the AIR's symbolic constraints
/// (fixed at keygen time). Only the weights (which depend on runtime eq_3b and beta_pows)
/// need recomputation at proving time.
pub struct Round0InteractionRules {
    /// Encoded rules on device, compiled with buffer_vars=true for Round 0 kernel.
    pub(crate) d_rules: DeviceBuffer<u128>,
    /// Buffer size for GPU intermediate values.
    pub(crate) buffer_size: u32,
    /// Number of compiled rules. Used to allocate weight vectors at runtime.
    pub(crate) num_rules: usize,
    /// Per-interaction metadata for fast weight computation at runtime.
    pub(crate) weight_map: Round0WeightMap,
}

/// Maps interaction indices to rule indices for runtime weight computation.
/// Replaces the expensive DAG pointer lookups with direct array indexing.
pub struct Round0WeightMap {
    /// count_rule_idxs[i] = rule index for interaction i's count expression.
    pub count_rule_idxs: Vec<usize>,
    /// message_offsets[i]..message_offsets[i+1] indexes into message_rule_idxs.
    pub message_offsets: Vec<usize>,
    /// Flat array of rule indices for all message fields across all interactions.
    pub message_rule_idxs: Vec<usize>,
    /// bus_indices[i] = bus index for interaction i.
    pub bus_indices: Vec<u16>,
}

/// Used for GKR input evaluation and logup MLE sumcheck rounds.
pub struct InteractionEvalRules {
    pub(crate) inner: EvalRules,
    /// Constraints consist of all `(numer, denom)` pairs **topologically sorted**. We map the
    /// constraint idx back to unsorted order. ```text
    /// constraint_idx => 2 * interaction_idx + is_denom
    /// ```
    pub(crate) d_pair_idxs: DeviceBuffer<u32>,
    pub(crate) max_fields_len: usize,
}

/// Constraints only, no interactions
pub struct ConstraintOnlyRules<const BUFFER_VARS: bool> {
    pub(crate) inner: EvalRules,
}

pub struct EvalRules {
    /// Encoded rules
    pub d_rules: DeviceBuffer<u128>,
    pub d_used_nodes: DeviceBuffer<usize>,
    pub buffer_size: u32,
}

pub struct ZerocheckMonomials {
    pub d_headers: DeviceBuffer<MonomialHeader>,
    pub d_variables: DeviceBuffer<PackedVar>,
    pub d_lambda_terms: DeviceBuffer<LambdaTerm<F>>,
    pub num_monomials: u32,
}

pub struct InteractionMonomials {
    pub d_numer_headers: DeviceBuffer<MonomialHeader>,
    pub d_numer_variables: DeviceBuffer<PackedVar>,
    pub d_numer_terms: DeviceBuffer<InteractionMonomialTerm<F>>,
    pub num_numer_monomials: u32,
    pub d_denom_headers: DeviceBuffer<MonomialHeader>,
    pub d_denom_variables: DeviceBuffer<PackedVar>,
    pub d_denom_terms: DeviceBuffer<InteractionMonomialTerm<F>>,
    pub num_denom_monomials: u32,
    pub max_fields_len: usize,
    pub num_interactions: u32,
}

fn to_device_or_empty<T>(data: &[T]) -> Result<DeviceBuffer<T>, MemCopyError> {
    if data.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        data.to_device()
    }
}

impl AirDataGpu {
    pub fn new<S: StarkProtocolConfig<F = F>>(
        pk: &StarkProvingKey<S>,
    ) -> Result<Self, MemCopyError> {
        let dag = &pk.vk.symbolic_constraints;
        let symbolic_constraints = SymbolicConstraints::from(dag);
        let interaction_rules = InteractionEvalRules::new(&symbolic_constraints)?;
        let zerocheck_round0 = ConstraintOnlyRules::<true>::new(&dag.constraints)?;
        let zerocheck_mle = ConstraintOnlyRules::<false>::new(&dag.constraints)?;

        let zerocheck_monomials = if dag.constraints.num_constraints() > 0 {
            let expanded = ExpandedMonomials::from_dag(&dag.constraints);
            Some(ZerocheckMonomials::from_expanded(&expanded)?)
        } else {
            None
        };
        let interaction_monomials = if !symbolic_constraints.interactions.is_empty() {
            let expanded =
                ExpandedInteractionMonomials::from_symbolic_constraints(&symbolic_constraints);
            Some(InteractionMonomials::from_expanded(&expanded)?)
        } else {
            None
        };
        let round0_interaction_rules = if !symbolic_constraints.interactions.is_empty() {
            Some(Round0InteractionRules::new(&symbolic_constraints)?)
        } else {
            None
        };
        Ok(Self {
            interaction_rules,
            zerocheck_round0,
            zerocheck_mle,
            zerocheck_monomials,
            interaction_monomials,
            round0_interaction_rules,
        })
    }
}

impl ZerocheckMonomials {
    pub fn from_expanded(expanded: &ExpandedMonomials<F>) -> Result<Self, MemCopyError> {
        // Validate bounds for all monomial headers to prevent out-of-bounds access in CUDA kernel
        let num_variables = expanded.variables.len();
        let num_lambda_terms = expanded.lambda_terms.len();
        for (i, hdr) in expanded.headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_variables,
                "Monomial {i}: var_offset ({}) + num_vars ({}) = {var_end} exceeds variables.len() ({num_variables})",
                hdr.var_offset,
                hdr.num_vars
            );
            assert!(
                term_end <= num_lambda_terms,
                "Monomial {i}: term_offset ({}) + num_terms ({}) = {term_end} exceeds lambda_terms.len() ({num_lambda_terms})",
                hdr.term_offset,
                hdr.num_terms
            );
        }

        Ok(Self {
            d_headers: expanded.headers.to_device()?,
            d_variables: expanded.variables.to_device()?,
            d_lambda_terms: expanded.lambda_terms.to_device()?,
            num_monomials: expanded.headers.len() as u32,
        })
    }
}

impl InteractionMonomials {
    pub fn from_expanded(expanded: &ExpandedInteractionMonomials<F>) -> Result<Self, MemCopyError> {
        // Validate numerator monomial headers
        let num_numer_vars = expanded.numer_variables.len();
        let num_numer_terms = expanded.numer_terms.len();
        for (i, hdr) in expanded.numer_headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_numer_vars,
                "Numer monomial {i}: var_offset + num_vars exceeds bounds"
            );
            assert!(
                term_end <= num_numer_terms,
                "Numer monomial {i}: term_offset + num_terms exceeds bounds"
            );
        }

        // Validate denominator monomial headers
        let num_denom_vars = expanded.denom_variables.len();
        let num_denom_terms = expanded.denom_terms.len();
        for (i, hdr) in expanded.denom_headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_denom_vars,
                "Denom monomial {i}: var_offset + num_vars exceeds bounds"
            );
            assert!(
                term_end <= num_denom_terms,
                "Denom monomial {i}: term_offset + num_terms exceeds bounds"
            );
        }

        Ok(Self {
            d_numer_headers: to_device_or_empty(&expanded.numer_headers)?,
            d_numer_variables: to_device_or_empty(&expanded.numer_variables)?,
            d_numer_terms: to_device_or_empty(&expanded.numer_terms)?,
            num_numer_monomials: expanded.numer_headers.len() as u32,
            d_denom_headers: to_device_or_empty(&expanded.denom_headers)?,
            d_denom_variables: to_device_or_empty(&expanded.denom_variables)?,
            d_denom_terms: to_device_or_empty(&expanded.denom_terms)?,
            num_denom_monomials: expanded.denom_headers.len() as u32,
            max_fields_len: expanded.max_fields_len,
            num_interactions: expanded.num_interactions,
        })
    }
}

impl InteractionEvalRules {
    pub fn new(symbolic_constraints: &SymbolicConstraints<F>) -> Result<Self, MemCopyError> {
        let interactions = &symbolic_constraints.interactions;
        let num_interactions = interactions.len();
        if num_interactions == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),

                max_fields_len: 0,
                d_pair_idxs: DeviceBuffer::new(),
            });
        }
        let max_fields_len = interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        // [alpha, beta^0, ..., beta^max_fields_len]
        let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..max_fields_len + 2)
            .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
            .collect();

        let mut frac_pairs = Vec::with_capacity(num_interactions * 2);
        for interaction in interactions.iter() {
            let numer = interaction.count.clone();
            let b = SymbolicExpression::from_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_u32(0);
            for (j, expr) in interaction.message.iter().enumerate() {
                denom += betas[j].clone() * expr.clone();
            }
            denom += betas[interaction.message.len()].clone() * b;
            frac_pairs.push(numer);
            frac_pairs.push(denom);
        }
        // build DAG without sorting constraint idxs:
        let (dag, pair_idxs) = {
            let mut dag_builder = SymbolicDagBuilder::new();
            let mut dag_pair_idxs: Vec<(usize, u32)> = frac_pairs
                .iter()
                .enumerate()
                .map(|(pair_idx, expr)| {
                    let dag_idx = dag_builder.add_expr(expr);
                    (dag_idx, pair_idx.try_into().unwrap())
                })
                .collect_vec();
            dag_pair_idxs.sort();
            let (constraint_idx, pair_idxs): (Vec<_>, Vec<_>) = dag_pair_idxs.into_iter().unzip();
            // NOTE: do not sort pair_idxs since we need to keep them in pairs
            let dag = SymbolicExpressionDag {
                nodes: dag_builder.nodes,
                constraint_idx,
            };
            (dag, pair_idxs)
        };
        let rules = SymbolicRulesGpu::new(&dag, false);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();
        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = used_nodes.to_device()?;
        let d_pair_idxs = pair_idxs.to_device()?;
        assert_eq!(
            used_nodes.len(),
            2 * num_interactions,
            "Rules come in (numer, denom) pairs"
        );

        let inner = EvalRules {
            d_rules,
            d_used_nodes,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
        };

        Ok(Self {
            inner,
            d_pair_idxs,
            max_fields_len,
        })
    }
}

impl<const BUFFER_VARS: bool> ConstraintOnlyRules<BUFFER_VARS> {
    pub fn new(dag: &SymbolicExpressionDag<F>) -> Result<Self, MemCopyError> {
        if dag.num_constraints() == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),
            });
        }

        let rules = SymbolicRulesGpu::new(dag, BUFFER_VARS);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();

        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = used_nodes.to_device()?;

        let inner = EvalRules {
            d_rules,
            d_used_nodes,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
        };
        Ok(Self { inner })
    }
}

impl EvalRules {
    pub fn dummy() -> Self {
        Self {
            d_rules: DeviceBuffer::new(),
            d_used_nodes: DeviceBuffer::new(),
            buffer_size: 0,
        }
    }
}

impl Round0InteractionRules {
    /// Build pre-computed Round 0 logup interaction rules from symbolic constraints.
    /// This performs the same DAG construction + rule compilation as
    /// `evaluate_round0_interactions_gpu` lines 162-210, but runs once at keygen time.
    pub fn new(symbolic_constraints: &SymbolicConstraints<F>) -> Result<Self, MemCopyError> {
        let interactions = &symbolic_constraints.interactions;
        assert!(!interactions.is_empty());

        // Build the interaction-only DAG (same as round0.rs:162-180)
        let mut dag_builder = SymbolicDagBuilder::new();
        let mut interaction_dag_idxs: Vec<(usize, Vec<usize>)> =
            Vec::with_capacity(interactions.len());
        let mut sorted_used_dag_idxs = Vec::new();

        for interaction in interactions {
            let count_idx = dag_builder.add_expr(&interaction.count);
            sorted_used_dag_idxs.push(count_idx);
            let message_idxs: Vec<usize> = interaction
                .message
                .iter()
                .map(|field_expr| {
                    let idx = dag_builder.add_expr(field_expr);
                    sorted_used_dag_idxs.push(idx);
                    idx
                })
                .collect();
            interaction_dag_idxs.push((count_idx, message_idxs));
        }

        sorted_used_dag_idxs.sort();
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesGpu::new(&dag, true);

        // Build the weight map using dag_idx_to_rule_idx
        let num_interactions = interactions.len();
        let mut count_rule_idxs = Vec::with_capacity(num_interactions);
        let mut message_offsets = Vec::with_capacity(num_interactions + 1);
        let mut message_rule_idxs = Vec::new();
        let mut bus_indices = Vec::with_capacity(num_interactions);

        for (i, (count_dag_idx, msg_dag_idxs)) in interaction_dag_idxs.iter().enumerate() {
            count_rule_idxs.push(rules.dag_idx_to_rule_idx[count_dag_idx]);
            message_offsets.push(message_rule_idxs.len());
            for msg_dag_idx in msg_dag_idxs {
                message_rule_idxs.push(rules.dag_idx_to_rule_idx[msg_dag_idx]);
            }
            bus_indices.push(interactions[i].bus_index);
        }
        message_offsets.push(message_rule_idxs.len());

        let num_rules = rules.rules.len();
        let encoded_rules: Vec<u128> = rules.rules.iter().map(|c| c.encode()).collect();
        let d_rules = encoded_rules.to_device()?;

        Ok(Self {
            d_rules,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
            num_rules,
            weight_map: Round0WeightMap {
                count_rule_idxs,
                message_offsets,
                message_rule_idxs,
                bus_indices,
            },
        })
    }

    /// Compute runtime-dependent weights for the Round 0 logup kernel.
    /// Uses the pre-computed weight map for direct array indexing instead of
    /// FxHashMap lookups through dag_builder.expr_to_idx and rules.dag_idx_to_rule_idx.
    pub fn compute_weights(
        &self,
        eq_3bs: &[EF],
        beta_pows: &[EF],
    ) -> Result<(DeviceBuffer<EF>, DeviceBuffer<EF>, EF), MemCopyError> {
        let map = &self.weight_map;
        let mut numer_weights = vec![EF::ZERO; self.num_rules];
        let mut denom_weights = vec![EF::ZERO; self.num_rules];
        let mut denom_sum_init = EF::ZERO;

        for i in 0..map.count_rule_idxs.len() {
            numer_weights[map.count_rule_idxs[i]] += eq_3bs[i];

            let msg_start = map.message_offsets[i];
            let msg_end = map.message_offsets[i + 1];
            let msg_len = msg_end - msg_start;

            denom_sum_init +=
                eq_3bs[i] * beta_pows[msg_len] * F::from_u32(map.bus_indices[i] as u32 + 1);

            for (j, &rule_idx) in map.message_rule_idxs[msg_start..msg_end].iter().enumerate() {
                denom_weights[rule_idx] += eq_3bs[i] * beta_pows[j];
            }
        }

        let d_numer = numer_weights.to_device()?;
        let d_denom = denom_weights.to_device()?;
        Ok((d_numer, d_denom, denom_sum_init))
    }
}
