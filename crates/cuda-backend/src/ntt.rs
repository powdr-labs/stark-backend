use std::sync::Once;

use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::{cuda::ntt, prelude::F};

const MAX_LG_DOMAIN_SIZE: usize = 27;
const LG_WINDOW_SIZE: usize = MAX_LG_DOMAIN_SIZE.div_ceil(5);
const WINDOW_SIZE: usize = 1 << LG_WINDOW_SIZE;
const WINDOW_NUM: usize = MAX_LG_DOMAIN_SIZE.div_ceil(LG_WINDOW_SIZE);

static INIT_FORWARD: Once = Once::new();
static INIT_INVERSE: Once = Once::new();

fn ensure_initialized(inverse: bool) {
    let once = if inverse {
        &INIT_INVERSE
    } else {
        &INIT_FORWARD
    };

    once.call_once(|| {
        let partial_twiddles = DeviceBuffer::<[F; WINDOW_SIZE]>::with_capacity(WINDOW_NUM);
        let twiddles = DeviceBuffer::<F>::with_capacity(32 + 64 + 128 + 256 + 512);
        unsafe {
            ntt::generate_all_twiddles(&twiddles, inverse).unwrap();
            ntt::generate_partial_twiddles(&partial_twiddles, inverse).unwrap();
        }
    });
}

struct NttImpl<'a> {
    buffer: &'a DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stage: u32,
}

impl<'a> NttImpl<'a> {
    fn new(
        buffer: &'a DeviceBuffer<F>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
    ) -> Self {
        ensure_initialized(is_intt);
        Self {
            buffer,
            lg_domain_size,
            padded_poly_size,
            poly_count,
            is_intt,
            stage: 0,
        }
    }

    fn step(&mut self, iterations: u32) {
        assert!(iterations <= 10);
        let radix = if iterations < 6 { 6 } else { iterations };
        unsafe {
            ntt::ct_mixed_radix_narrow(
                self.buffer,
                radix,
                self.lg_domain_size,
                self.stage,
                iterations,
                self.padded_poly_size,
                self.poly_count,
                self.is_intt,
            )
            .unwrap();
        }
        self.stage += iterations;
    }
}

/// Performs column-wise batch NTT on `buffer`, where `buffer` is assumed to be column-major with
/// columns of height `2^(log_trace_height + log_blowup)`. The NTT are performed on the first
/// `2^log_trace_height` elements of each column. If `bit_reverse` is true, then the input columns
/// are assumed to be ordered in **natural** ordering, and a bit-reversal permutation is applied for
/// the internal algorithm of the NTT. If `bit_reverse` is false, then the input columns are assumed
/// to be in bit-reverse ordering. If `is_intt` is true, the inverse NTT is performed; otherwise,
/// the forward NTT is performed.
pub fn batch_ntt(
    buffer: &DeviceBuffer<F>,
    log_trace_height: u32,
    log_blowup: u32,
    width: u32,
    bit_reverse: bool,
    is_intt: bool,
) {
    if log_trace_height == 0 {
        return;
    }

    let padded_poly_size = 1 << (log_trace_height + log_blowup);

    if bit_reverse {
        unsafe {
            ntt::bit_rev(buffer, buffer, log_trace_height, padded_poly_size, width).unwrap();
        }
    }

    let mut _impl = NttImpl::new(buffer, log_trace_height, padded_poly_size, width, is_intt);
    if log_trace_height <= 10 {
        _impl.step(log_trace_height);
    } else if log_trace_height <= 17 {
        let step = log_trace_height / 2;
        _impl.step(step + log_trace_height % 2);
        _impl.step(step);
    } else if log_trace_height <= 30 {
        let step = log_trace_height / 3;
        let rem = log_trace_height % 3;
        _impl.step(step);
        _impl.step(step + (if log_trace_height == 29 { 1 } else { 0 }));
        _impl.step(step + (if log_trace_height == 29 { 1 } else { rem }));
    } else if log_trace_height <= 40 {
        let step = log_trace_height / 4;
        let rem = log_trace_height % 4;
        _impl.step(step);
        _impl.step(step + (if rem > 2 { 1 } else { 0 }));
        _impl.step(step + (if rem > 1 { 1 } else { 0 }));
        _impl.step(step + (if rem > 0 { 1 } else { 0 }));
    } else {
        panic!("log_trace_height > 40 not supported");
    }
}

/// Like [`batch_ntt`], but processes columns in L2-cache-sized batches for better
/// memory locality. Falls back to [`batch_ntt`] when the total working set already
/// fits in L2.
pub fn batch_ntt_column_batched(
    buffer: &DeviceBuffer<F>,
    log_trace_height: u32,
    log_blowup: u32,
    width: u32,
    bit_reverse: bool,
    is_intt: bool,
) {
    if log_trace_height == 0 {
        return;
    }
    let padded_poly_size = 1u64 << (log_trace_height + log_blowup);
    let col_bytes = padded_poly_size as usize * std::mem::size_of::<F>();
    let total_bytes = col_bytes * width as usize;

    // If the entire working set fits in ~60MB (leaving headroom in the 72MB L2),
    // there's no benefit to batching — use the standard path.
    const L2_BUDGET: usize = 60 * 1024 * 1024;
    if total_bytes <= L2_BUDGET {
        batch_ntt(buffer, log_trace_height, log_blowup, width, bit_reverse, is_intt);
        return;
    }

    let batch_cols = (L2_BUDGET / col_bytes).max(1) as u32;

    // Bit-reverse entire buffer first (one pass) — the batched NTT steps
    // expect bit-reversed input within each column.
    if bit_reverse {
        unsafe {
            ntt::bit_rev(
                buffer,
                buffer,
                log_trace_height,
                padded_poly_size as u32,
                width,
            )
            .unwrap();
        }
    }

    // Process NTT in column batches
    for batch_start in (0..width).step_by(batch_cols as usize) {
        let batch_width = (width - batch_start).min(batch_cols);
        let element_offset = batch_start as usize * padded_poly_size as usize;
        // SAFETY: offset is within buffer bounds; non-owning view is valid for
        // the duration of the NTT call; NTT operates only within the view.
        let sub_buffer = unsafe {
            DeviceBuffer::non_owning(
                buffer.as_ptr().add(element_offset) as *mut F,
                batch_width as usize * padded_poly_size as usize,
            )
        };
        let mut ntt_impl = NttImpl::new(
            &sub_buffer,
            log_trace_height,
            padded_poly_size as u32,
            batch_width,
            is_intt,
        );
        // Replicate the same step schedule as batch_ntt
        if log_trace_height <= 10 {
            ntt_impl.step(log_trace_height);
        } else if log_trace_height <= 17 {
            let step = log_trace_height / 2;
            ntt_impl.step(step + log_trace_height % 2);
            ntt_impl.step(step);
        } else if log_trace_height <= 30 {
            let step = log_trace_height / 3;
            let rem = log_trace_height % 3;
            ntt_impl.step(step);
            ntt_impl.step(step + (if log_trace_height == 29 { 1 } else { 0 }));
            ntt_impl.step(step + (if log_trace_height == 29 { 1 } else { rem }));
        } else if log_trace_height <= 40 {
            let step = log_trace_height / 4;
            let rem = log_trace_height % 4;
            ntt_impl.step(step);
            ntt_impl.step(step + (if rem > 2 { 1 } else { 0 }));
            ntt_impl.step(step + (if rem > 1 { 1 } else { 0 }));
            ntt_impl.step(step + (if rem > 0 { 1 } else { 0 }));
        }
    }
}
