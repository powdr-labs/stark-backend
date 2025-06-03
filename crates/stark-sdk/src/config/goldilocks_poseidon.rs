use std::any::type_name;

use openvm_stark_backend::{
    config::StarkConfig,
    interaction::fri_log_up::FriLogUpPhase,
    p3_challenger::DuplexChallenger,
    p3_commit::ExtensionMmcs,
    p3_field::{extension::BinomialExtensionField, Field},
    prover::{
        cpu::{CpuBackend, CpuDevice},
        MultiTraceStarkProver,
    },
};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use rand::{rngs::StdRng, SeedableRng};

use super::instrument::{HashStatistics, Instrumented, StarkHashStatistics};
use crate::{
    assert_sc_compatible_with_serde,
    config::fri_params::SecurityParameters,
    engine::{StarkEngine, StarkEngineWithHashInstrumentation},
};

const RATE: usize = 4;
// permutation width
const WIDTH: usize = 8; // rate + capacity
const DIGEST_WIDTH: usize = 4;

type Val = Goldilocks;
type PackedVal = <Val as Field>::Packing;
type Challenge = BinomialExtensionField<Val, 2>;
type Perm = Poseidon<Val, MdsMatrixGoldilocks, WIDTH, 7>;
type InstrPerm = Instrumented<Perm>;

// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, DIGEST_WIDTH, WIDTH>;
type ValMmcs<P> =
    MerkleTreeMmcs<PackedVal, <Val as Field>::Packing, Hash<P>, Compress<P>, DIGEST_WIDTH>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
pub type Challenger<P> = DuplexChallenger<Val, P, WIDTH, RATE>;
type Dft = Radix2DitParallel<Val>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;
type RapPhase<P> = FriLogUpPhase<Val, Challenge, Challenger<P>>;

pub type GoldilocksPermutationConfig<P> =
    StarkConfig<Pcs<P>, RapPhase<P>, Challenge, Challenger<P>>;
pub type GoldilocksPoseidonConfig = GoldilocksPermutationConfig<Perm>;
pub type GoldilocksPoseidonEngine = GoldilocksPermutationEngine<Perm>;

assert_sc_compatible_with_serde!(GoldilocksPoseidonConfig);

pub struct GoldilocksPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    security_params: SecurityParameters,
    pub config: GoldilocksPermutationConfig<P>,
    pub perm: P,
    pub max_constraint_degree: usize,
}

impl<P> StarkEngine<GoldilocksPermutationConfig<P>> for GoldilocksPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    fn config(&self) -> &GoldilocksPermutationConfig<P> {
        &self.config
    }

    fn prover<'a>(&'a self) -> MultiTraceStarkProver<'a, GoldilocksPermutationConfig<P>>
    where
        Self: 'a,
    {
        MultiTraceStarkProver::new(
            CpuBackend::default(),
            CpuDevice::new(self.config(), self.security_params.fri_params.log_blowup),
            self.new_challenger(),
        )
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.max_constraint_degree)
    }

    fn new_challenger(&self) -> Challenger<P> {
        Challenger::new(self.perm.clone())
    }
}

impl<P> StarkEngineWithHashInstrumentation<GoldilocksPermutationConfig<Instrumented<P>>>
    for GoldilocksPermutationEngine<Instrumented<P>>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    fn clear_instruments(&mut self) {
        self.perm.input_lens_by_type.lock().unwrap().clear();
    }
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T> {
        let counter = self.perm.input_lens_by_type.lock().unwrap();
        let permutations = counter.iter().fold(0, |total, (name, lens)| {
            if name == type_name::<[Val; WIDTH]>() {
                let count: usize = lens.iter().sum();
                println!("Permutation: {name}, Count: {count}");
                total + count
            } else {
                panic!("Permutation type not yet supported: {}", name);
            }
        });

        StarkHashStatistics {
            name: type_name::<P>().to_string(),
            stats: HashStatistics { permutations },
            fri_params: self.security_params.fri_params,
            custom,
        }
    }
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> GoldilocksPoseidonEngine {
    let perm = random_perm();
    engine_from_perm(perm, SecurityParameters::standard_fast())
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm) -> GoldilocksPoseidonConfig {
    config_from_perm(perm, SecurityParameters::standard_fast())
}

pub fn engine_from_perm<P>(
    perm: P,
    security_params: SecurityParameters,
) -> GoldilocksPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    let max_constraint_degree = security_params.fri_params.max_constraint_degree();
    let config = config_from_perm(&perm, security_params.clone());
    GoldilocksPermutationEngine {
        config,
        perm,
        security_params,
        max_constraint_degree,
    }
}

pub fn config_from_perm<P>(
    perm: &P,
    security_params: SecurityParameters,
) -> GoldilocksPermutationConfig<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let SecurityParameters {
        fri_params,
        log_up_params,
    } = security_params;
    let fri_config = FriConfig {
        log_blowup: fri_params.log_blowup,
        log_final_poly_len: fri_params.log_final_poly_len,
        num_queries: fri_params.num_queries,
        proof_of_work_bits: fri_params.proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let rap_phase = FriLogUpPhase::new(log_up_params, fri_params.log_blowup);
    GoldilocksPermutationConfig::new(pcs, rap_phase)
}

pub fn random_perm() -> Perm {
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);
    Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rng)
}

pub fn random_instrumented_perm() -> InstrPerm {
    let perm = random_perm();
    Instrumented::new(perm)
}
