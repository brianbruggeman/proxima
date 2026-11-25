//! The runtime operations a card-C executor makes, behind one trait.
//!
//! The executor lowers once (CUDA) and runs on whichever runtime implements
//! [`GpuRuntime`]; HIP is a second runtime of the same executor, not a second
//! executor. Every method here is one call the CUDA executor already made on a
//! stream: allocate zeroed, upload, download, device-to-device copy, sync.
//!
//! Teaching pointer: launches are built with [`crate::kernel_params`], the one
//! place a launch's parameter pointers are made. Statically dispatched (no
//! `Box<dyn>`): `CudaPlan` names its runtime type, so each call is a direct call.

use core::ops::Range;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchArgs, LaunchConfig, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// An element a runtime moves between host and device.
///
/// Implemented per type, never blanket, so the set of element types a kernel
/// ABI accepts stays an explicit list.
pub trait DeviceElement: DeviceRepr + ValidAsZeroBits + Copy + 'static {}

impl DeviceElement for f32 {}
impl DeviceElement for u32 {}
impl DeviceElement for u8 {}

/// Grid, block and dynamic shared memory of one launch, in the units every runtime takes.
#[derive(Debug, Clone, Copy)]
pub struct LaunchShape {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// The device operations one executor runs on.
pub trait GpuRuntime {
    /// What a failed device call reports.
    type Error;
    /// A device allocation of `Element`s.
    type Buffer<Element: DeviceElement>;
    /// A compiled kernel entry point; cloning shares the loaded handle.
    type Function: Clone;
    /// A loaded module, from which functions are looked up.
    type Module;
    /// Module bytes a runtime loads: the compile output.
    type Image;
    /// The borrowed argument list of one launch, built in the launching function so each argument outlives the launch.
    type Launch<'a>
    where
        Self: 'a;

    /// A zero-filled device buffer of `len` elements.
    ///
    /// # Errors
    /// The device allocation failed.
    fn alloc_zeros<Element: DeviceElement>(&self, len: usize) -> Result<Self::Buffer<Element>, Self::Error>;

    /// A new device buffer holding `values`.
    ///
    /// # Errors
    /// The allocation or the host-to-device copy failed.
    fn upload<Element: DeviceElement>(&self, values: &[Element]) -> Result<Self::Buffer<Element>, Self::Error>;

    /// Overwrites the front of `buffer` with `values` in place.
    ///
    /// # Errors
    /// The host-to-device copy failed.
    fn upload_into<Element: DeviceElement>(&self, values: &[Element], buffer: &mut Self::Buffer<Element>) -> Result<(), Self::Error>;

    /// Overwrites `buffer` from element `offset` with `values` in place.
    ///
    /// # Errors
    /// The host-to-device copy failed.
    fn upload_at<Element: DeviceElement>(&self, values: &[Element], buffer: &mut Self::Buffer<Element>, offset: usize) -> Result<(), Self::Error>;

    /// All of `buffer`, copied to the host.
    ///
    /// # Errors
    /// The device-to-host copy failed.
    fn download<Element: DeviceElement>(&self, buffer: &Self::Buffer<Element>) -> Result<Vec<Element>, Self::Error>;

    /// The `range` of `buffer`, copied to the host.
    ///
    /// # Errors
    /// The device-to-host copy failed.
    fn download_range<Element: DeviceElement>(&self, buffer: &Self::Buffer<Element>, range: Range<usize>) -> Result<Vec<Element>, Self::Error>;

    /// Copies `source[range]` into `destination` starting at element `offset`, on the device.
    ///
    /// # Errors
    /// The device-to-device copy failed.
    fn copy_range<Element: DeviceElement>(
        &self,
        source: &Self::Buffer<Element>,
        range: Range<usize>,
        destination: &mut Self::Buffer<Element>,
        offset: usize,
    ) -> Result<(), Self::Error>;

    /// Wraps source text (PTX) as a loadable image.
    fn image_from_source(source: &str) -> Self::Image;

    /// Loads `image` into the device context.
    ///
    /// # Errors
    /// The driver rejected the image.
    fn load_module(&self, image: Self::Image) -> Result<Arc<Self::Module>, Self::Error>;

    /// Looks `entry` up in `module`.
    ///
    /// # Errors
    /// The module has no such entry.
    fn load_function(&self, module: &Arc<Self::Module>, entry: &str) -> Result<Self::Function, Self::Error>;

    /// Starts the argument list of a launch of `function`; push each argument by reference, then call [`Self::launch`].
    fn launch_arguments<'a>(&'a self, function: &'a Self::Function) -> Self::Launch<'a>;

    /// Launches with `arguments` on `shape`.
    ///
    /// # Safety
    /// The arguments must match the kernel's parameter list in count, type and order.
    ///
    /// # Errors
    /// The launch failed.
    unsafe fn launch(&self, arguments: Self::Launch<'_>, shape: LaunchShape) -> Result<(), Self::Error>;

    /// Blocks until every queued device operation finished.
    ///
    /// # Errors
    /// A queued operation failed.
    fn synchronize(&self) -> Result<(), Self::Error>;
}

/// The cudarc stream the CUDA executor always ran on, behind [`GpuRuntime`].
#[derive(Debug, Clone)]
pub struct CudarcRuntime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudarcRuntime {
    /// The cudarc stream, for the event recording only cudarc exposes.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    #[must_use]
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { context, stream }
    }
}

impl GpuRuntime for CudarcRuntime {
    type Error = DriverError;
    type Buffer<Element: DeviceElement> = CudaSlice<Element>;
    type Function = CudaFunction;
    type Module = CudaModule;
    type Image = Ptx;
    type Launch<'a> = LaunchArgs<'a>;

    fn alloc_zeros<Element: DeviceElement>(&self, len: usize) -> Result<CudaSlice<Element>, DriverError> {
        self.stream.alloc_zeros::<Element>(len)
    }

    fn upload<Element: DeviceElement>(&self, values: &[Element]) -> Result<CudaSlice<Element>, DriverError> {
        self.stream.clone_htod(values)
    }

    fn upload_into<Element: DeviceElement>(&self, values: &[Element], buffer: &mut CudaSlice<Element>) -> Result<(), DriverError> {
        self.stream.memcpy_htod(values, buffer)
    }

    fn upload_at<Element: DeviceElement>(&self, values: &[Element], buffer: &mut CudaSlice<Element>, offset: usize) -> Result<(), DriverError> {
        let mut window = buffer.slice_mut(offset..offset + values.len());
        self.stream.memcpy_htod(values, &mut window)
    }

    fn download<Element: DeviceElement>(&self, buffer: &CudaSlice<Element>) -> Result<Vec<Element>, DriverError> {
        self.stream.clone_dtoh(buffer)
    }

    fn download_range<Element: DeviceElement>(&self, buffer: &CudaSlice<Element>, range: Range<usize>) -> Result<Vec<Element>, DriverError> {
        self.stream.clone_dtoh(&buffer.slice(range))
    }

    fn copy_range<Element: DeviceElement>(
        &self,
        source: &CudaSlice<Element>,
        range: Range<usize>,
        destination: &mut CudaSlice<Element>,
        offset: usize,
    ) -> Result<(), DriverError> {
        let window_end = offset + range.len();
        self.stream.memcpy_dtod(&source.slice(range), &mut destination.slice_mut(offset..window_end))
    }

    fn image_from_source(source: &str) -> Ptx {
        Ptx::from_src(source)
    }

    fn load_module(&self, image: Ptx) -> Result<Arc<CudaModule>, DriverError> {
        self.context.load_module(image)
    }

    fn load_function(&self, module: &Arc<CudaModule>, entry: &str) -> Result<CudaFunction, DriverError> {
        module.load_function(entry)
    }

    fn launch_arguments<'a>(&'a self, function: &'a CudaFunction) -> LaunchArgs<'a> {
        self.stream.launch_builder(function)
    }

    unsafe fn launch(&self, mut arguments: LaunchArgs<'_>, shape: LaunchShape) -> Result<(), DriverError> {
        let config = LaunchConfig { grid_dim: shape.grid_dim, block_dim: shape.block_dim, shared_mem_bytes: shape.shared_mem_bytes };
        unsafe { arguments.launch(config) }.map(|_| ())
    }

    fn synchronize(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }
}
