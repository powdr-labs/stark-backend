use std::fmt::Debug;

use p3_challenger::CanObserve;
use p3_matrix::dense::RowMajorMatrix;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    air_builders::symbolic::{symbolic_expression::SymbolicExpression, SymbolicConstraints},
    interaction::fri_log_up::{STARK_LU_NUM_CHALLENGES, STARK_LU_NUM_EXPOSED_VALUES},
    prover::types::PairView,
    rap::AirBuilder,
};

/// Interaction debugging tools
pub mod debug;
pub mod fri_log_up;
pub mod rap;
pub mod trace;
mod utils;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InteractionType {
    Send,
    Receive,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Interaction<Expr> {
    pub fields: Vec<Expr>,
    pub count: Expr,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

pub type SymbolicInteraction<F> = Interaction<SymbolicExpression<F>>;

/// An [AirBuilder] with additional functionality to build special logUp arguments for
/// communication between AIRs across buses. These arguments use randomness to
/// add additional trace columns (in the extension field) and constraints to the AIR.
///
/// An interactive AIR is a AIR that can specify buses for sending and receiving data
/// to other AIRs. The original AIR is augmented by virtual columns determined by
/// the interactions to define a [RAP](crate::rap::Rap).
pub trait InteractionBuilder: AirBuilder {
    /// Stores a new send interaction in the builder.
    fn push_send<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Send);
    }

    /// Stores a new receive interaction in the builder.
    fn push_receive<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Receive);
    }

    /// Stores a new interaction in the builder.
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    );

    /// Returns the current number of interactions.
    fn num_interactions(&self) -> usize;

    /// Returns all interactions stored.
    fn all_interactions(&self) -> &[Interaction<Self::Expr>];
}

pub struct RapPhaseProverData<Challenge> {
    /// Challenges from the challenger in this phase that determine RAP constraints and exposed values.
    pub challenges: Vec<Challenge>,

    /// After challenge trace per air computed as a function of `challenges`.
    pub after_challenge_trace_per_air: Vec<Option<RowMajorMatrix<Challenge>>>,

    /// Public values of the phase that are functions of `challenges`.
    pub exposed_values_per_air: Vec<Option<Vec<Challenge>>>,
}

pub struct RapPhaseVerifierData<Challenge> {
    /// Challenges from the challenger in this phase that determine RAP constraints and exposed values.
    pub challenges_per_phase: Vec<Vec<Challenge>>,
}

#[derive(Debug)]
pub struct RapPhaseShape {
    pub num_challenges: usize,

    pub num_exposed_values: usize,

    /// Any additional rotations to open at in the permutation PCS round.
    ///
    /// Specifies that each `i` in `extra_opening_rots` should be opened at
    /// `zeta * g^i` (in addition to `zeta` and `zeta * g`).
    pub extra_opening_rots: Vec<usize>,
}

/// Supported challenge phases in a RAP.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum RapPhaseSeqKind {
    // GkrLogUp,
    /// Up to one phase with prover/verifier given by [[fri_log_up::FriLogUpPhase]] and
    /// constraints given by [[fri_log_up::eval_fri_log_up_phase]].
    FriLogUp,
    None,
}

impl RapPhaseSeqKind {
    pub fn shape(&self) -> Vec<RapPhaseShape> {
        match self {
            RapPhaseSeqKind::FriLogUp => vec![RapPhaseShape {
                num_challenges: STARK_LU_NUM_CHALLENGES,
                num_exposed_values: STARK_LU_NUM_EXPOSED_VALUES,
                extra_opening_rots: vec![],
            }],
            RapPhaseSeqKind::None => vec![],
        }
    }
}

/// Defines a particular protocol for the "after challenge" phase in a RAP.
///
/// A [RapPhaseSeq] is defined by the proving and verifying methods implemented in this trait,
/// as well as via some "eval" method that is determined by `RapPhaseId`.
pub trait RapPhaseSeq<F, Challenge, Challenger> {
    type PartialProof: Clone + Serialize + DeserializeOwned;
    /// Preprocessed data necessary for the RAP partial proving
    type PartialProvingKey: Clone + Serialize + DeserializeOwned;
    type Error: Debug;

    const ID: RapPhaseSeqKind;

    /// The protocol parameters for the challenge phases may depend on the AIR constraints.
    fn generate_pk_per_air(
        &self,
        symbolic_constraints_per_air: &[SymbolicConstraints<F>],
        max_constraint_degree: usize,
    ) -> Vec<Self::PartialProvingKey>;

    /// Partially prove the challenge phases,
    ///
    /// Samples challenges, generates after challenge traces and exposed values, and proves any
    /// extra-STARK part of the protocol.
    ///
    /// "Partial" refers to the fact that some STARK parts of the protocol---namely, the constraints
    /// on the after challenge traces returned in `RapPhaseProverData`---are handled external to
    /// this function.
    fn partially_prove(
        &self,
        challenger: &mut Challenger,
        constraints_per_air: &[&SymbolicConstraints<F>],
        params_per_air: &[&Self::PartialProvingKey],
        trace_view_per_air: &[PairTraceView<F>],
    ) -> Option<(Self::PartialProof, RapPhaseProverData<Challenge>)>;

    /// Partially verifies the challenge phases.
    ///
    /// Assumes the shape of `exposed_values_per_air_per_phase` is verified externally.
    ///
    /// An implementation of this function must sample challenges for the challenge phases and then
    /// observe the exposed values and commitment.
    fn partially_verify<Commitment: Clone>(
        &self,
        challenger: &mut Challenger,
        partial_proof: Option<&Self::PartialProof>,
        exposed_values_per_air_per_phase: &[Vec<Vec<Challenge>>],
        commitments_per_phase: &[Commitment],
        // per commitment, per matrix, per rotation, per column
        after_challenge_opened_values: &[Vec<Vec<Vec<Challenge>>>],
    ) -> (RapPhaseVerifierData<Challenge>, Result<(), Self::Error>)
    where
        Challenger: CanObserve<Commitment>;
}

type PairTraceView<'a, F> = PairView<&'a RowMajorMatrix<F>, F>;
