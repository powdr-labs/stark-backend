//! Benchmark for measuring proving time vs AIR complexity.
//!
//! Creates configurable synthetic AIRs with:
//! - Zero witness (all trace cells are 0)
//! - Boolean constraints: `x * (x - 1) = 0`
//! - Self-canceling bus interactions: `bus_send([x])` + `bus_receive([x])`
//!
//! Usage:
//!   cargo run -p openvm-benchmark-air-complexity --release -- \
//!     --num-airs 4 --cols-per-air 100 --constraints-per-col 2 --log-total-cells 24

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
    AirRef, ColumnsAir, PartitionedBaseAir, StarkEngine,
};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, baby_bear_poseidon2::BabyBearPoseidon2RefEngine,
    MAX_APP_LOG_STACKED_HEIGHT,
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_util::log2_ceil_usize;

type F = BabyBear;

#[derive(Parser)]
#[command(about = "Benchmark proving time vs AIR complexity")]
struct Args {
    /// Number of AIRs
    #[arg(long, default_value_t = 1)]
    num_airs: usize,

    /// Number of columns per AIR
    #[arg(long, default_value_t = 20)]
    cols_per_air: usize,

    /// Number of boolean constraints per column per AIR
    #[arg(long, default_value_t = 1)]
    constraints_per_col: usize,

    /// Number of send/receive bus interaction pairs per column per AIR.
    /// Each pair creates one `push_interaction(..., count=1)` and one
    /// `push_interaction(..., count=-1)` with the same message, so they cancel out.
    #[arg(long, default_value_t = 1)]
    interactions_per_col: usize,

    /// Log2 of target total trace cells across all AIRs. Actual count may differ
    /// due to rounding trace height to a power of 2.
    #[arg(long, default_value_t = 28)]
    log_total_cells: usize,
}

// ---------------------------------------------------------------------------
// BenchmarkAir
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BenchmarkAir {
    pub num_columns: usize,
    pub constraints_per_col: usize,
    pub interactions_per_col: usize,
}

impl<F> BaseAir<F> for BenchmarkAir {
    fn width(&self) -> usize {
        self.num_columns
    }
}
impl<F> BaseAirWithPublicValues<F> for BenchmarkAir {}
impl<F> ColumnsAir<F> for BenchmarkAir {}
impl<F> PartitionedBaseAir<F> for BenchmarkAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for BenchmarkAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();

        // Boolean constraints: col * (col - 1) == 0
        for col_idx in 0..self.num_columns {
            let col = local[col_idx];
            for _ in 0..self.constraints_per_col {
                builder.assert_bool(col);
            }
        }

        // Self-canceling bus interactions: place a send+receive pair on every
        // other column so the total push_interaction count equals
        // interactions_per_col * num_columns (for even num_columns).
        let num_interaction_pairs = self.num_columns / 2;
        for pair in 0..num_interaction_pairs {
            let field = vec![local[pair * 2]];
            for _ in 0..self.interactions_per_col {
                builder.push_interaction(0, field.clone(), AB::Expr::ONE, 0);
                builder.push_interaction(0, field.clone(), AB::Expr::NEG_ONE, 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    openvm_stark_sdk::utils::setup_tracing();

    let args = Args::parse();

    assert!(args.num_airs > 0);
    assert!(args.cols_per_air > 0);

    // Compute trace height (round up to power of 2)
    let target_total_cells = 1usize << args.log_total_cells;
    let cells_per_air = target_total_cells / args.num_airs;
    let rows_per_air = cells_per_air / args.cols_per_air;
    assert!(
        rows_per_air >= 2,
        "log_total_cells too small for the given num_airs and cols_per_air"
    );
    let log_trace_height = log2_ceil_usize(rows_per_air);
    let trace_height = 1usize << log_trace_height;
    let actual_total_cells = args.num_airs * args.cols_per_air * trace_height;

    let total_constraints = args.constraints_per_col * args.cols_per_air * args.num_airs;
    let bus_interactions_per_air = (args.cols_per_air / 2) * 2 * args.interactions_per_col;
    let total_bus_interactions = bus_interactions_per_air * args.num_airs;
    let constraint_instances = total_constraints * trace_height;
    let bus_interaction_messages = total_bus_interactions * trace_height;

    // Ratio of actual cells vs target
    let cells_ratio_str = if actual_total_cells == target_total_cells {
        "exact".to_string()
    } else if actual_total_cells < target_total_cells {
        format!(
            "{:.2}x below target",
            target_total_cells as f64 / actual_total_cells as f64
        )
    } else {
        format!(
            "{:.2}x above target",
            actual_total_cells as f64 / target_total_cells as f64
        )
    };

    println!("=== AIR Complexity Benchmark ===");
    println!("  num_airs:               {}", args.num_airs);
    println!("  cols_per_air:           {}", args.cols_per_air);
    println!("  constraints_per_col:    {}", args.constraints_per_col);
    println!("  interactions_per_col:   {}", args.interactions_per_col);
    println!("  trace_height:           {trace_height} (2^{log_trace_height})");
    println!("  trace_cells:            {actual_total_cells} ({cells_ratio_str})");
    println!("  constraints:            {total_constraints}");
    println!("  bus_interactions:       {total_bus_interactions}");
    println!("  constraint_instances:   {constraint_instances}");
    println!("  bus_interaction_msgs:   {bus_interaction_messages}");

    // Create AIRs
    let airs: Vec<AirRef<_>> = (0..args.num_airs)
        .map(|_| {
            Arc::new(BenchmarkAir {
                num_columns: args.cols_per_air,
                constraints_per_col: args.constraints_per_col,
                interactions_per_col: args.interactions_per_col,
            }) as AirRef<_>
        })
        .collect();

    // Use MAX_APP_LOG_STACKED_HEIGHT, matching default_app_config() in the openvm cli crate.
    let params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let engine: BabyBearPoseidon2RefEngine = StarkEngine::new(params);

    // Keygen
    println!("\nKeygen...");
    let start = Instant::now();
    let (pk, vk) = engine.keygen(&airs);
    println!("  time: {:?}", start.elapsed());

    // Generate zero traces
    let ctx = ProvingContext::new(
        (0..args.num_airs)
            .map(|i| {
                let trace = ColMajorMatrix::new(
                    vec![F::ZERO; trace_height * args.cols_per_air],
                    args.cols_per_air,
                );
                (i, AirProvingContext::simple_no_pis(trace))
            })
            .collect(),
    );

    // Prove
    let d_pk = engine.device().transport_pk_to_device(&pk);
    println!("Proving...");
    let start = Instant::now();
    let proof = engine.prove(&d_pk, ctx).unwrap();
    let prove_time = start.elapsed();
    println!("  time: {prove_time:?}");

    // Verify
    println!("Verifying...");
    let start = Instant::now();
    engine.verify(&vk, &proof).unwrap();
    println!("  time: {:?}", start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvm_stark_backend::SystemParams;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2RefEngine;

    fn run_benchmark_test(
        num_airs: usize,
        cols_per_air: usize,
        constraints_per_col: usize,
        interactions_per_col: usize,
        log_trace_height: usize,
    ) {
        let trace_height = 1usize << log_trace_height;

        let airs: Vec<AirRef<_>> = (0..num_airs)
            .map(|_| {
                Arc::new(BenchmarkAir {
                    num_columns: cols_per_air,
                    constraints_per_col,
                    interactions_per_col,
                }) as AirRef<_>
            })
            .collect();

        let params = SystemParams::new_for_testing(log_trace_height);
        let engine: BabyBearPoseidon2RefEngine = StarkEngine::new(params);
        let (pk, vk) = engine.keygen(&airs);

        let ctx = ProvingContext::new(
            (0..num_airs)
                .map(|i| {
                    let trace = ColMajorMatrix::new(
                        vec![F::ZERO; trace_height * cols_per_air],
                        cols_per_air,
                    );
                    (i, AirProvingContext::simple_no_pis(trace))
                })
                .collect(),
        );

        let d_pk = engine.device().transport_pk_to_device(&pk);
        let proof = engine.prove(&d_pk, ctx).unwrap();
        engine.verify(&vk, &proof).unwrap();
    }

    #[test]
    fn test_single_air_constraints_only() {
        run_benchmark_test(1, 10, 1, 0, 8);
    }

    #[test]
    fn test_single_air_with_interactions() {
        run_benchmark_test(1, 10, 1, 1, 8);
    }

    #[test]
    fn test_multi_air() {
        run_benchmark_test(3, 10, 1, 1, 8);
    }

    #[test]
    fn test_wide_air() {
        run_benchmark_test(1, 100, 2, 1, 8);
    }

    #[test]
    fn test_many_constraints_and_interactions() {
        run_benchmark_test(1, 5, 3, 3, 8);
    }
}
