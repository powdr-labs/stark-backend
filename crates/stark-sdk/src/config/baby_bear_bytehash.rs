use openvm_stark_backend::{
    config::StarkConfig,
    interaction::fri_log_up::FriLogUpPhase,
    p3_challenger::{HashChallenger, SerializingChallenger32},
    p3_commit::ExtensionMmcs,
    p3_field::extension::BinomialExtensionField,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        MultiTraceStarkProver,
    },
};
use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, SerializingHasher32};

use super::FriParameters;
use crate::{
    config::{
        fri_params::SecurityParameters, log_up_params::log_up_security_params_baby_bear_100_bits,
    },
    engine::{StarkEngine, StarkFriEngine},
};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

// Generic over H: CryptographicHasher<u8, [u8; 32]>
type FieldHash<H> = SerializingHasher32<H>;
type Compress<H> = CompressionFunctionFromHasher<H, 2, 32>;
// type InstrCompress<H> = Instrumented<Compress<H>>;

type ValMmcs<H> = MerkleTreeMmcs<Val, u8, FieldHash<H>, Compress<H>, 32>;
type ChallengeMmcs<H> = ExtensionMmcs<Val, Challenge, ValMmcs<H>>;
type Dft = Radix2DitParallel<Val>;
type Challenger<H> = SerializingChallenger32<Val, HashChallenger<u8, H, 32>>;

type Pcs<H> = TwoAdicFriPcs<Val, Dft, ValMmcs<H>, ChallengeMmcs<H>>;

type RapPhase<H> = FriLogUpPhase<Val, Challenge, Challenger<H>>;

pub type BabyBearByteHashConfig<H> = StarkConfig<Pcs<H>, RapPhase<H>, Challenge, Challenger<H>>;

pub struct BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    pub fri_params: FriParameters,
    pub config: BabyBearByteHashConfig<H>,
    pub byte_hash: H,
    pub max_constraint_degree: usize,
}

impl<H> StarkEngine<BabyBearByteHashConfig<H>> for BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone + Send + Sync,
{
    fn config(&self) -> &BabyBearByteHashConfig<H> {
        &self.config
    }

    fn prover<'a>(&'a self) -> MultiTraceStarkProver<'a, BabyBearByteHashConfig<H>>
    where
        Self: 'a,
    {
        MultiTraceStarkProver::new(
            CpuBackend::default(),
            CpuDevice::new(self.config(), self.fri_params.log_blowup),
            self.new_challenger(),
        )
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.max_constraint_degree)
    }

    fn new_challenger(&self) -> Challenger<H> {
        Challenger::from_hasher(vec![], self.byte_hash.clone())
    }
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine<H>(byte_hash: H) -> BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    engine_from_byte_hash(byte_hash, SecurityParameters::standard_fast())
}

pub fn engine_from_byte_hash<H>(
    byte_hash: H,
    security_params: SecurityParameters,
) -> BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    let fri_params = security_params.fri_params;
    let max_constraint_degree = fri_params.max_constraint_degree();
    let config = config_from_byte_hash(byte_hash.clone(), security_params);
    BabyBearByteHashEngine {
        config,
        byte_hash,
        fri_params,
        max_constraint_degree,
    }
}

pub fn config_from_byte_hash<H>(
    byte_hash: H,
    security_params: SecurityParameters,
) -> BabyBearByteHashConfig<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    let field_hash = FieldHash::new(byte_hash.clone());
    let compress = Compress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress);
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
    BabyBearByteHashConfig::new(pcs, rap_phase)
}

pub trait BabyBearByteHashEngineWithDefaultHash<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    fn default_hash() -> H;
}

impl<H: CryptographicHasher<u8, [u8; 32]> + Clone + Send + Sync>
    StarkFriEngine<BabyBearByteHashConfig<H>> for BabyBearByteHashEngine<H>
where
    BabyBearByteHashEngine<H>: BabyBearByteHashEngineWithDefaultHash<H>,
{
    fn new(fri_params: FriParameters) -> Self {
        let security_params = SecurityParameters {
            fri_params,
            log_up_params: log_up_security_params_baby_bear_100_bits(),
        };
        engine_from_byte_hash(Self::default_hash(), security_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
