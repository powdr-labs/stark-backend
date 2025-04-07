//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use std::{iter, sync::Arc};

use derivative::Derivative;
use itertools::izip;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    config::{StarkGenericConfig, Val},
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{
        cpu::CpuDevice,
        hal::TraceCommitter,
        types::{AirProofInput, AirProofRawInput, CommittedTraceData},
    },
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};

pub struct DummyInteractionCols;
impl DummyInteractionCols {
    pub fn count_col() -> usize {
        0
    }
    pub fn field_col(field_idx: usize) -> usize {
        field_idx + 1
    }
}

#[derive(Clone, Copy)]
pub struct DummyInteractionAir {
    field_width: usize,
    /// Send if true. Receive if false.
    pub is_send: bool,
    bus_index: BusIndex,
    pub count_weight: u32,
    /// If true, then | count | and | fields[..] | are in separate main trace partitions.
    pub partition: bool,
}

impl DummyInteractionAir {
    pub fn new(field_width: usize, is_send: bool, bus_index: BusIndex) -> Self {
        Self {
            field_width,
            is_send,
            bus_index,
            count_weight: 0,
            partition: false,
        }
    }

    pub fn partition(self) -> Self {
        Self {
            partition: true,
            ..self
        }
    }

    pub fn field_width(&self) -> usize {
        self.field_width
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for DummyInteractionAir {}
impl<F: Field> PartitionedBaseAir<F> for DummyInteractionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.partition {
            vec![self.field_width]
        } else {
            vec![]
        }
    }
    fn common_main_width(&self) -> usize {
        if self.partition {
            1
        } else {
            1 + self.field_width
        }
    }
}
impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        1 + self.field_width
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + PartitionedAirBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let (fields, count) = if self.partition {
            let local_0 = builder.common_main().row_slice(0);
            let local_1 = builder.cached_mains()[0].row_slice(0);
            let count = local_0[0];
            let fields = local_1.to_vec();
            (fields, count)
        } else {
            let main = builder.main();
            let local = main.row_slice(0);
            let count = local[DummyInteractionCols::count_col()];
            let fields: Vec<_> = (0..self.field_width)
                .map(|i| local[DummyInteractionCols::field_col(i)])
                .collect();
            (fields, count)
        };
        if self.is_send {
            builder.push_interaction(self.bus_index, fields, count, self.count_weight);
        } else {
            builder.push_interaction(
                self.bus_index,
                fields,
                AB::Expr::NEG_ONE * count,
                self.count_weight,
            );
        }
    }
}

/// Note: in principle, committing cached trace is out of scope of a chip. But this chip is for
/// usually testing, so we support it for convenience.
#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
pub struct DummyInteractionChip<'a, SC: StarkGenericConfig> {
    device: Option<CpuDevice<'a, SC>>,
    // common_main: Option<RowMajorMatrix<Val<SC>>>,
    data: Option<DummyInteractionData>,
    pub air: DummyInteractionAir,
}

#[derive(Debug, Clone)]
pub struct DummyInteractionData {
    pub count: Vec<u32>,
    pub fields: Vec<Vec<u32>>,
}

impl<'a, SC: StarkGenericConfig> DummyInteractionChip<'a, SC>
where
    Val<SC>: FieldAlgebra,
{
    pub fn new_without_partition(field_width: usize, is_send: bool, bus_index: BusIndex) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index);
        Self {
            device: None,
            data: None,
            air,
        }
    }
    pub fn new_with_partition(
        config: &'a SC,
        field_width: usize,
        is_send: bool,
        bus_index: BusIndex,
    ) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index).partition();
        Self {
            device: Some(CpuDevice::new(config)),
            data: None,
            air,
        }
    }
    pub fn load_data(&mut self, data: DummyInteractionData) {
        let DummyInteractionData { count, fields } = &data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        self.data = Some(data);
    }

    #[allow(clippy::type_complexity)]
    fn generate_traces_with_partition(
        &self,
        data: DummyInteractionData,
    ) -> (RowMajorMatrix<Val<SC>>, CommittedTraceData<SC>) {
        let DummyInteractionData {
            mut count,
            mut fields,
        } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let h = h.next_power_of_two();
        count.resize(h, 0);
        fields.resize(h, vec![0; w]);
        let common_main_val: Vec<_> = count
            .into_iter()
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        let cached_trace_val: Vec<_> = fields
            .into_iter()
            .flatten()
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        let cached_trace = Arc::new(RowMajorMatrix::new(cached_trace_val, w));
        let (commit, data) = self
            .device
            .as_ref()
            .unwrap()
            .commit(&[cached_trace.clone()]);
        (
            RowMajorMatrix::new(common_main_val, 1),
            CommittedTraceData {
                trace: cached_trace,
                commitment: commit,
                pcs_data: data.data,
            },
        )
    }

    fn generate_traces_without_partition(
        &self,
        data: DummyInteractionData,
    ) -> RowMajorMatrix<Val<SC>> {
        let DummyInteractionData { count, fields } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let common_main_val: Vec<_> = izip!(count, fields)
            .flat_map(|(count, fields)| iter::once(count).chain(fields))
            .chain(iter::repeat(0))
            .take((w + 1) * h.next_power_of_two())
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        RowMajorMatrix::new(common_main_val, w + 1)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for DummyInteractionChip<'_, SC> {
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        assert!(self.data.is_some());
        let data = self.data.clone().unwrap();
        if self.device.is_some() {
            let (common_main, cached) = self.generate_traces_with_partition(data);
            AirProofInput {
                cached_mains_pdata: vec![(cached.commitment, cached.pcs_data)],
                raw: AirProofRawInput {
                    cached_mains: vec![cached.trace],
                    common_main: Some(common_main),
                    public_values: vec![],
                },
            }
        } else {
            let common_main = self.generate_traces_without_partition(data);
            AirProofInput {
                cached_mains_pdata: vec![],
                raw: AirProofRawInput {
                    cached_mains: vec![],
                    common_main: Some(common_main),
                    public_values: vec![],
                },
            }
        }
    }
}

impl<SC: StarkGenericConfig> ChipUsageGetter for DummyInteractionChip<'_, SC> {
    fn air_name(&self) -> String {
        "DummyInteractionAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        if let Some(data) = &self.data {
            data.count.len()
        } else {
            0
        }
    }

    fn trace_width(&self) -> usize {
        self.air.field_width + 1
    }
}
