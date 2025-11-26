//! Helper methods for testing use

use std::collections::HashMap;

use itertools::izip;

use crate::prover::{
    hal::{AdjustableMatrix, MatrixDimensions, ProverBackend},
    types::AirProvingContext,
};

impl<PB: ProverBackend> AirProvingContext<PB> {
    pub fn simple(trace: PB::Matrix, public_values: Vec<PB::Val>) -> Self {
        Self::new(vec![], Some(trace), public_values)
    }
    pub fn simple_no_pis(trace: PB::Matrix) -> Self {
        Self::simple(trace, vec![])
    }

    pub fn multiple_simple(traces: Vec<PB::Matrix>, public_values: Vec<Vec<PB::Val>>) -> Vec<Self> {
        izip!(traces, public_values)
            .map(|(trace, pis)| Self::simple(trace, pis))
            .collect()
    }

    pub fn multiple_simple_no_pis(traces: Vec<PB::Matrix>) -> Vec<Self> {
        traces.into_iter().map(Self::simple_no_pis).collect()
    }
    /// Return the height of the main trace.
    pub fn main_trace_height(&self) -> usize {
        if self.cached_mains.is_empty() {
            // An AIR must have a main trace.
            self.common_main.as_ref().unwrap().height()
        } else {
            self.cached_mains[0].trace.height()
        }
    }

    pub fn append(&mut self, other: Vec<(PB::Matrix, Vec<usize>)>) {
        self.common_main.as_mut().unwrap().append(other);
    }

    pub fn add_frequencies(&mut self, frequencies: HashMap<PB::Val, usize>) {
        let fixed = &self.cached_mains[0].trace;
        self.common_main.as_mut().unwrap().add_frequencies(frequencies, fixed);
    }
}
