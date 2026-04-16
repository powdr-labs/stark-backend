use std::{cmp::max, sync::Arc};

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::MatrixDimensions;

use super::errors::FoldPleError;
use crate::{
    base::DeviceMatrix,
    cuda::logup_zerocheck::{batched_fold_ple_from_evals, fold_ple_from_evals, FoldPleDesc},
    prelude::{EF, F},
};

/// Folds plain using mixed coefficients, folds rotation from evals.
/// - `mixed` should be mixed coefficient form of the _lifted_ trace.
/// - `trace_evals` should be unlifted (the original trace).
pub fn fold_ple_evals_rotate(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    trace_evals: &DeviceMatrix<F>,
    d_inv_lagrange_denoms_r0: &DeviceBuffer<EF>,
    need_rot: bool,
) -> Result<DeviceMatrix<EF>, FoldPleError> {
    let width = trace_evals.width();
    let height = trace_evals.height();
    let num_x = max(height >> l_skip, 1);
    let out_width = width * if need_rot { 2 } else { 1 };
    let folded_buf = DeviceBuffer::<EF>::with_capacity(num_x * out_width);
    // SAFETY:
    // - We allocated `folded_buf` for `num_x * width * (1 or 2)` elements.
    // - `trace_evals` is `height x width` unlighted matrix
    unsafe {
        fold_ple_evals_gpu(
            l_skip,
            d_omega_skip_pows,
            trace_evals,
            folded_buf.as_mut_ptr(),
            d_inv_lagrange_denoms_r0,
            false,
        )?;

        if need_rot {
            // Fold the rotation from evals
            fold_ple_evals_gpu(
                l_skip,
                d_omega_skip_pows,
                trace_evals,
                folded_buf.as_mut_ptr().add(num_x * width),
                d_inv_lagrange_denoms_r0,
                true,
            )?;
        }
    }
    let folded = DeviceMatrix::new(Arc::new(folded_buf), num_x, out_width);
    Ok(folded)
}

/// Folds PLE evaluations by interpolating univariate polynomials on coset D and evaluating at r_0.
/// Returns a single matrix of width `width`.
///
/// When `rotate` is true, returns the folding of the lift of the rotated matrix.
/// When `rotate` is false, returns the folding of the lift of the original matrix.
///
/// # Assumptions
/// - `mat` should be the unlifted original matrix of trace evaluations.
/// - `output` should be a valid pointer to a buffer of size at least `num_x * width` where `num_x =
///   max(height / 2^l_skip, 1)`.
pub unsafe fn fold_ple_evals_gpu(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    mat: &DeviceMatrix<F>,
    output: *mut EF,
    d_inv_lagrange_denoms_r0: &DeviceBuffer<EF>,
    rotate: bool,
) -> Result<(), FoldPleError> {
    let height = mat.height();
    let width = mat.width();

    if height == 0 || width == 0 {
        return Ok(());
    }

    let skip_domain = d_omega_skip_pows.len();
    debug_assert_eq!(skip_domain, 1 << l_skip);
    let lifted_height = max(skip_domain, height);
    let num_x = lifted_height / skip_domain;

    // Launch kernel
    unsafe {
        fold_ple_from_evals(
            mat.buffer(),
            output,
            d_omega_skip_pows,
            d_inv_lagrange_denoms_r0,
            height as u32,
            width as u32,
            l_skip as u32,
            num_x as u32,
            rotate,
        )?;
    }
    Ok(())
}

/// Item describing one fold_ple operation for the batched path.
pub struct FoldPleItem<'a> {
    pub trace_evals: &'a DeviceMatrix<F>,
    pub need_rot: bool,
}

/// Result of a single batched fold_ple operation.
pub struct FoldPleResult {
    pub folded: DeviceMatrix<EF>,
}

/// Batch-launches fold_ple_from_evals for multiple matrices in 1-2 kernel launches
/// (one for rotate=false, one for rotate=true if any need_rot).
///
/// Returns one `FoldPleResult` per input item, in the same order.
pub fn batched_fold_ple_evals_rotate(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    d_inv_lagrange_denoms_r0: &DeviceBuffer<EF>,
    items: &[FoldPleItem<'_>],
) -> Result<Vec<FoldPleResult>, FoldPleError> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let skip_domain = d_omega_skip_pows.len();
    let block_size = max(256, skip_domain) as u32;
    let chunks_per_block = block_size / skip_domain as u32;

    // Pre-allocate output buffers for each item and compute dimensions
    struct ItemInfo {
        width: usize,
        height: usize,
        num_x: usize,
        out_width: usize,
    }

    let mut infos: Vec<ItemInfo> = Vec::with_capacity(items.len());
    let mut output_bufs: Vec<DeviceBuffer<EF>> = Vec::with_capacity(items.len());

    for item in items {
        let width = item.trace_evals.width();
        let height = item.trace_evals.height();
        let num_x = max(height >> l_skip, 1);
        let out_width = width * if item.need_rot { 2 } else { 1 };
        output_bufs.push(DeviceBuffer::<EF>::with_capacity(num_x * out_width));
        infos.push(ItemInfo {
            width,
            height,
            num_x,
            out_width,
        });
    }

    // Build descriptor arrays for rotate=false (all items) and rotate=true (items with need_rot)
    let mut no_rot_descs: Vec<FoldPleDesc> = Vec::new();
    let mut no_rot_total_blocks: u32 = 0;
    let mut rot_descs: Vec<FoldPleDesc> = Vec::new();
    let mut rot_total_blocks: u32 = 0;

    for (i, item) in items.iter().enumerate() {
        let info = &infos[i];
        if info.height == 0 || info.width == 0 {
            continue;
        }
        let blocks_per_col =
            (info.num_x as u32 + chunks_per_block - 1) / chunks_per_block;
        let item_blocks = blocks_per_col * info.width as u32;

        // rotate=false descriptor
        no_rot_descs.push(FoldPleDesc {
            src: item.trace_evals.buffer().as_ptr(),
            dst: output_bufs[i].as_mut_ptr(),
            height: info.height as u32,
            width: info.width as u32,
            num_x: info.num_x as u32,
            block_start: no_rot_total_blocks,
        });
        no_rot_total_blocks += item_blocks;

        // rotate=true descriptor (offset dst by num_x * width)
        if item.need_rot {
            rot_descs.push(FoldPleDesc {
                src: item.trace_evals.buffer().as_ptr(),
                dst: unsafe { output_bufs[i].as_mut_ptr().add(info.num_x * info.width) },
                height: info.height as u32,
                width: info.width as u32,
                num_x: info.num_x as u32,
                block_start: rot_total_blocks,
            });
            rot_total_blocks += item_blocks;
        }
    }

    // Launch batched kernels
    unsafe {
        if !no_rot_descs.is_empty() {
            let d_descs = no_rot_descs.to_device()?;
            batched_fold_ple_from_evals(
                &d_descs,
                no_rot_total_blocks,
                d_omega_skip_pows,
                d_inv_lagrange_denoms_r0,
                l_skip as u32,
                false,
            )?;
        }

        if !rot_descs.is_empty() {
            let d_descs = rot_descs.to_device()?;
            batched_fold_ple_from_evals(
                &d_descs,
                rot_total_blocks,
                d_omega_skip_pows,
                d_inv_lagrange_denoms_r0,
                l_skip as u32,
                true,
            )?;
        }
    }

    // Build results
    let results = infos
        .into_iter()
        .zip(output_bufs)
        .map(|(info, buf)| {
            let folded = DeviceMatrix::new(Arc::new(buf), info.num_x, info.out_width);
            FoldPleResult { folded }
        })
        .collect();

    Ok(results)
}
