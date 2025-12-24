use std::{
    any::Any,
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::prover::{hal::ProverBackend, types::AirProvingContext};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use p3_baby_bear::BabyBear;

/// Context for APC (Aggregated Proving Context) trace generation.
/// Contains all device buffers and parameters needed for direct-to-APC trace generation.
#[derive(Clone)]
pub struct ApcTracingContext<'a> {
    /// Output trace buffer (column-major)
    pub d_trace: &'a DeviceBuffer<BabyBear>,
    /// Substitution indices for column remapping
    pub d_subs: &'a DeviceBuffer<u32>,
    /// Optimized widths for each sub-AIR
    pub d_opt_widths: &'a DeviceBuffer<u32>,
    /// Post-optimization column offsets
    pub d_post_opt_offsets: &'a DeviceBuffer<u32>,
    /// Number of calls packed per APC row
    pub calls_per_apc_row: u32,
    /// Height of the APC trace
    pub apc_height: usize,
    /// Width of the APC trace
    pub apc_width: usize,
}

impl<'a> ApcTracingContext<'a> {
    pub fn new(
        d_trace: &'a DeviceBuffer<BabyBear>,
        d_subs: &'a DeviceBuffer<u32>,
        d_opt_widths: &'a DeviceBuffer<u32>,
        d_post_opt_offsets: &'a DeviceBuffer<u32>,
        calls_per_apc_row: u32,
        apc_height: usize,
        apc_width: usize,
    ) -> Self {
        Self {
            d_trace,
            d_subs,
            d_opt_widths,
            d_post_opt_offsets,
            calls_per_apc_row,
            apc_height,
            apc_width,
        }
    }
}

/// A chip is a [ProverBackend]-specific object that converts execution logs (also referred to as
/// records) into a trace matrix.
///
/// A chip may be stateful and store state on either host or device, although it is preferred that
/// all state is received through records.
pub trait Chip<R, PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB>;

    /// Generate trace directly into the APC buffer.
    /// Implementors should write trace data using the context's device buffers.
    fn generate_proving_ctx_new(&self, _records: R, _ctx: &ApcTracingContext) {
        // Default no-op implementation for chips that don't support direct-to-APC
    }
}

/// Auto-implemented trait for downcasting of trait objects.
pub trait AnyChip<R, PB: ProverBackend>: Chip<R, PB> {
    fn as_any(&self) -> &dyn Any;
}

impl<R, PB: ProverBackend, C: Chip<R, PB> + 'static> AnyChip<R, PB> for C {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for RefCell<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.borrow().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Rc<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.as_ref().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Arc<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.as_ref().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Mutex<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.lock().unwrap().generate_proving_ctx(records)
    }
}

// TODO: consider deleting this
/// A trait to get chip usage information.
pub trait ChipUsageGetter {
    fn air_name(&self) -> String;
    /// If the chip has a state-independent trace height that is determined
    /// upon construction, return this height. This is used to distinguish
    /// "static" versus "dynamic" usage metrics.
    fn constant_trace_height(&self) -> Option<usize> {
        None
    }
    /// Height of used rows in the main trace.
    fn current_trace_height(&self) -> usize;
    /// Width of the main trace
    fn trace_width(&self) -> usize;
    /// For metrics collection
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for Rc<C> {
    fn air_name(&self) -> String {
        self.as_ref().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.as_ref().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.as_ref().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.as_ref().trace_width()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for RefCell<C> {
    fn air_name(&self) -> String {
        self.borrow().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.borrow().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for Mutex<C> {
    fn air_name(&self) -> String {
        self.lock().unwrap().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.lock().unwrap().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.lock().unwrap().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.lock().unwrap().trace_width()
    }
}
