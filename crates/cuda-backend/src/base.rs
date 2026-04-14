use std::{ffi::c_void, fmt::Debug, marker::PhantomData, sync::Arc};

use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2H},
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::current_stream_sync,
};
use openvm_stark_backend::prover::MatrixDimensions;

pub struct DeviceMatrix<T> {
    buffer: Arc<DeviceBuffer<T>>,
    height: usize,
    width: usize,
}

unsafe impl<T> Send for DeviceMatrix<T> {}
unsafe impl<T> Sync for DeviceMatrix<T> {}

impl<T> Clone for DeviceMatrix<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            height: self.height,
            width: self.width,
        }
    }
}

impl<T> Drop for DeviceMatrix<T> {
    fn drop(&mut self) {
        tracing::debug!(
            "Dropping DeviceMatrix of size {} with Arc strong count={}",
            self.buffer.len(),
            self.strong_count()
        );
    }
}

impl<T> DeviceMatrix<T> {
    pub fn new(buffer: Arc<DeviceBuffer<T>>, height: usize, width: usize) -> Self {
        assert_ne!(
            height * width,
            0,
            "Zero dimensions h {} w {} are wrong",
            height,
            width
        );
        assert_eq!(
            buffer.len(),
            height * width,
            "Buffer size must match dimensions"
        );
        Self {
            buffer,
            height,
            width,
        }
    }

    pub fn with_capacity(height: usize, width: usize) -> Self {
        Self {
            buffer: Arc::new(DeviceBuffer::with_capacity(height * width)),
            height,
            width,
        }
    }

    pub fn dummy() -> Self {
        Self {
            buffer: Arc::new(DeviceBuffer::new()),
            height: 0,
            width: 0,
        }
    }

    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
    }

    pub fn as_view(&self) -> DeviceMatrixView<'_, T> {
        // SAFETY: buffer is borrowed for lifetime 'a of the view
        unsafe { DeviceMatrixView::from_raw_parts(self.buffer.as_ptr(), self.height, self.width) }
    }
}

impl<T> MatrixDimensions for DeviceMatrix<T> {
    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }
}

impl<T> MemCopyD2H<T> for DeviceMatrix<T> {
    fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        self.buffer.to_host()
    }
}

impl<T: Debug> Debug for DeviceMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceMatrix (height = {}, width = {}): {:?}",
            self.height(),
            self.width(),
            self.buffer()
        )
    }
}

/// View of a device matrix. Dropping does not free memory.
#[derive(Clone, Copy)]
pub struct DeviceMatrixView<'a, T> {
    ptr: *const T,
    height: usize,
    width: usize,
    _ptr_lifetime: PhantomData<&'a T>,
}

unsafe impl<T> Send for DeviceMatrixView<'_, T> {}
unsafe impl<T> Sync for DeviceMatrixView<'_, T> {}

impl<T> DeviceMatrixView<'_, T> {
    /// # Safety
    /// - The pointer must be valid for the lifetime of the view.
    /// - The pointer must have memory allocated for the following `height * width` elements of `T`.
    pub unsafe fn from_raw_parts(ptr: *const T, height: usize, width: usize) -> Self {
        Self {
            ptr,
            height,
            width,
            _ptr_lifetime: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> MatrixDimensions for DeviceMatrixView<'_, T> {
    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }
}

/// Non-owning matrix view into an arena-allocated GPU buffer.
/// Does NOT free memory on drop — the backing `FoldArena` owns the memory.
#[derive(Clone, Copy)]
pub struct ArenaMatrix<T> {
    ptr: *mut T,
    height: usize,
    width: usize,
}

unsafe impl<T> Send for ArenaMatrix<T> {}
unsafe impl<T> Sync for ArenaMatrix<T> {}

impl<T> ArenaMatrix<T> {
    pub fn new(ptr: *mut T, height: usize, width: usize) -> Self {
        Self { ptr, height, width }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn buffer_len(&self) -> usize {
        self.height * self.width
    }

    pub fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        let len = self.buffer_len();
        let mut host_vec = Vec::with_capacity(len);
        let size_bytes = std::mem::size_of::<T>() * len;
        unsafe {
            cuda_memcpy::<true, false>(
                host_vec.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size_bytes,
            )?;
        }
        current_stream_sync().map_err(MemCopyError::from)?;
        unsafe {
            host_vec.set_len(len);
        }
        Ok(host_vec)
    }
}

impl<T> MatrixDimensions for ArenaMatrix<T> {
    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }
}

/// Transitional enum: holds either an owned `DeviceMatrix` (from `fold_ple_evals`)
/// or an arena-backed `ArenaMatrix` (from MLE fold rounds).
#[derive(Clone)]
pub enum MatrixRef<T> {
    Owned(DeviceMatrix<T>),
    Arena(ArenaMatrix<T>),
}

impl<T> MatrixRef<T> {
    pub fn as_ptr(&self) -> *const T {
        match self {
            MatrixRef::Owned(m) => m.buffer().as_ptr(),
            MatrixRef::Arena(m) => m.as_ptr(),
        }
    }

    pub fn buffer_len(&self) -> usize {
        match self {
            MatrixRef::Owned(m) => m.buffer().len(),
            MatrixRef::Arena(m) => m.buffer_len(),
        }
    }

    pub fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        match self {
            MatrixRef::Owned(m) => m.buffer().to_host(),
            MatrixRef::Arena(m) => m.to_host(),
        }
    }
}

impl<T> MatrixDimensions for MatrixRef<T> {
    #[inline]
    fn height(&self) -> usize {
        match self {
            MatrixRef::Owned(m) => m.height(),
            MatrixRef::Arena(m) => m.height(),
        }
    }

    #[inline]
    fn width(&self) -> usize {
        match self {
            MatrixRef::Owned(m) => m.width(),
            MatrixRef::Arena(m) => m.width(),
        }
    }
}

/// Arena allocator for MLE fold output buffers. Each `allocate_bulk` call
/// creates one large `DeviceBuffer` and returns a raw pointer into it.
/// All buffers are freed when the arena is dropped.
pub struct FoldArena<T> {
    buffers: Vec<DeviceBuffer<T>>,
}

impl<T> FoldArena<T> {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Allocate a contiguous buffer of `total_cells` elements and return a mutable pointer.
    pub fn allocate_bulk(&mut self, total_cells: usize) -> *mut T {
        let buf = DeviceBuffer::with_capacity(total_cells);
        let ptr = buf.as_mut_ptr();
        self.buffers.push(buf);
        ptr
    }
}

/// The following trait and types are borrowed from [halo2](https:://github.com/zcash/halo2).
/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

pub struct DevicePoly<T, B> {
    pub is_bit_reversed: bool,
    pub coeff: DeviceBuffer<T>,
    _marker: PhantomData<B>,
}

impl<T, B> DevicePoly<T, B> {
    pub fn new(is_bit_reversed: bool, coeff: DeviceBuffer<T>) -> Self {
        Self {
            is_bit_reversed,
            coeff,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeff.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_matrix() {
        let buffer = Arc::new(DeviceBuffer::<i32>::with_capacity(12));
        let matrix = DeviceMatrix::<i32>::new(buffer, 3, 4);
        assert_eq!(matrix.height(), 3);
        assert_eq!(matrix.width(), 4);
    }
}
