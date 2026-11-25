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
use core::time::Duration;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaView, CudaViewMut, PushKernelArg, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchArgs, LaunchConfig, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use proxima_tensor::Executes;

use crate::cuda::target::Target;
use crate::cuda::{CudaKernel, EmitDevice, executes as cuda_executes};
use crate::cuda_driver::{CachedCudaModule, CompileCounters, CudaDriverError};
use proxima_primitives::sync::blocking::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;

/// An element a runtime moves between host and device.
///
/// Implemented per type, never blanket, so the set of element types a kernel
/// ABI accepts stays an explicit list.
pub trait DeviceElement: DeviceRepr + ValidAsZeroBits + Copy + core::fmt::Debug + 'static {}

impl DeviceElement for u32 {}
impl DeviceElement for u8 {}

/// Grid, block and dynamic shared memory of one launch, in the units every runtime takes.
#[derive(Debug, Clone, Copy)]
pub struct LaunchShape {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// The argument list of one launch: every argument is pushed by reference, so each outlives the launch.
///
/// Teaching pointer: the cudarc impl wraps its typed `LaunchArgs::arg`; a runtime without typed
/// arguments builds a [`crate::kernel_params::KernelParams`] instead.
pub trait LaunchArguments<'a, Runtime: GpuRuntime> {
    /// Pushes a buffer the kernel reads.
    fn push<Element: DeviceElement>(&mut self, buffer: &'a Runtime::Buffer<Element>) -> &mut Self;
    /// Pushes a buffer the kernel writes.
    fn push_mut<Element: DeviceElement>(&mut self, buffer: &'a mut Runtime::Buffer<Element>) -> &mut Self;
    /// Pushes a shared window.
    fn push_view<Element: DeviceElement>(&mut self, view: &'a Runtime::View<'_, Element>) -> &mut Self;
    /// Pushes a writable window.
    fn push_view_mut<Element: DeviceElement>(&mut self, view: &'a mut Runtime::ViewMut<'_, Element>) -> &mut Self;
    /// Pushes a scalar kernel parameter.
    fn push_value<Element: DeviceElement>(&mut self, value: &'a Element) -> &mut Self;
}

/// The device operations one executor runs on.
pub trait GpuRuntime: core::fmt::Debug + Clone {
    /// What a failed device call reports.
    type Error;
    /// What a failed kernel build reports: compile and cache failures as well as driver ones.
    type BuildError;
    /// A device allocation of `Element`s.
    type Buffer<Element: DeviceElement>: core::fmt::Debug + Clone;
    /// A shared window onto part of a buffer.
    type View<'a, Element: DeviceElement>
    where
        Self: 'a;
    /// A writable window onto part of a buffer.
    type ViewMut<'a, Element: DeviceElement>
    where
        Self: 'a;
    /// A point in the device queue, for timing.
    type Event;
    /// A compiled kernel entry point; cloning shares the loaded handle.
    type Function: core::fmt::Debug + Clone;
    /// A loaded module, from which functions are looked up.
    type Module;
    /// Module bytes a runtime loads: the compile output.
    type Image;
    /// The borrowed argument list of one launch, built in the launching function so each argument outlives the launch.
    type Launch<'a>: LaunchArguments<'a, Self>
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

    /// The loaded function for `kernel`, compiled or read from a cache on first use.
    ///
    /// # Errors
    /// The kernel failed to compile or load.
    fn build_function(&self, kernel: &CudaKernel) -> Result<Self::Function, Self::BuildError>;

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

    /// The operations this runtime's kernels execute, which the bind pass reads.
    fn executes(&self) -> Executes;

    /// The emit target this runtime's kernels are rendered for.
    fn emit_target(&self) -> Target;

    /// The device facts that shape emitted launch grids.
    fn emit_device(&self) -> EmitDevice;

    /// Elements in `buffer`.
    fn len<Element: DeviceElement>(buffer: &Self::Buffer<Element>) -> usize;

    /// A shared window onto `range` of `buffer`.
    fn view<Element: DeviceElement>(buffer: &Self::Buffer<Element>, range: Range<usize>) -> Self::View<'_, Element>;

    /// A writable window onto `range` of `buffer`.
    fn view_mut<Element: DeviceElement>(buffer: &mut Self::Buffer<Element>, range: Range<usize>) -> Self::ViewMut<'_, Element>;

    /// Elements in a writable window.
    fn view_mut_len<Element: DeviceElement>(view: &Self::ViewMut<'_, Element>) -> usize;

    /// A writable window onto `range` of another writable window.
    fn view_mut_range<'view, Element: DeviceElement>(view: &'view mut Self::ViewMut<'_, Element>, range: Range<usize>) -> Self::ViewMut<'view, Element>;

    /// All of `buffer` read as bytes, or `None` when the runtime cannot reinterpret it.
    fn bytes_mut<Element: DeviceElement>(buffer: &mut Self::Buffer<Element>) -> Option<Self::ViewMut<'_, u8>>;

    /// Records an event at the current end of the device queue.
    ///
    /// # Errors
    /// The runtime cannot record events; the caller then skips timing.
    fn record_event(&self) -> Result<Self::Event, Self::Error>;

    /// Time between two recorded events.
    ///
    /// # Errors
    /// An event has not completed or the runtime cannot time it.
    fn elapsed(start: &Self::Event, end: &Self::Event) -> Result<Duration, Self::Error>;

    /// Blocks until every queued device operation finished.
    ///
    /// # Errors
    /// A queued operation failed.
    fn synchronize(&self) -> Result<(), Self::Error>;

    /// The device's `(free_bytes, total_bytes)` right now.
    ///
    /// # Errors
    /// The device could not report.
    fn memory_info(&self) -> Result<(usize, usize), Self::Error>;

    /// A stable one-line name of the selected device, prefixed by the runtime.
    ///
    /// # Errors
    /// The device could not report.
    fn device_identity(&self) -> Result<String, Self::Error>;

    /// The kernel compile and cache counters this runtime keeps.
    fn compile_counters(&self) -> &CompileCounters;
}

/// The cudarc stream the CUDA executor always ran on, behind [`GpuRuntime`].
#[derive(Debug, Clone)]
pub struct CudarcRuntime {
    pub(crate) context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device: EmitDevice,
    // synchronous ffi callers share these short cache mutations; compilation and dispatch happen outside the guard.
    pub(crate) modules: Arc<Mutex<BTreeMap<String, Vec<CachedCudaModule>>>>,
    pub(crate) compile_counters: Arc<CompileCounters>,
}

impl CudarcRuntime {
    /// The cudarc stream, for the event recording only cudarc exposes.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    #[must_use]
    pub(crate) fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>, device: EmitDevice, compile_counters: Arc<CompileCounters>) -> Self {
        Self { context, stream, device, modules: Arc::new(Mutex::new(BTreeMap::new())), compile_counters }
    }

    /// The grant CUDA kernels bind under; the pure function [`GpuRuntime::executes`] returns.
    #[must_use]
    pub fn cuda_grant() -> Executes {
        cuda_executes(&Target::CUDA)
    }

    /// The target CUDA kernels render for; what [`GpuRuntime::emit_target`] returns.
    #[must_use]
    pub fn cuda_target() -> Target {
        Target::CUDA
    }
}

impl GpuRuntime for CudarcRuntime {
    type Error = DriverError;
    type BuildError = CudaDriverError;
    type Buffer<Element: DeviceElement> = CudaSlice<Element>;
    type View<'a, Element: DeviceElement> = CudaView<'a, Element>;
    type ViewMut<'a, Element: DeviceElement> = CudaViewMut<'a, Element>;
    type Event = CudaEvent;
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

    fn build_function(&self, kernel: &CudaKernel) -> Result<CudaFunction, CudaDriverError> {
        self.function_for(kernel)
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

    fn executes(&self) -> Executes {
        Self::cuda_grant()
    }

    fn emit_target(&self) -> Target {
        Self::cuda_target()
    }

    fn emit_device(&self) -> EmitDevice {
        self.device
    }

    fn len<Element: DeviceElement>(buffer: &CudaSlice<Element>) -> usize {
        buffer.len()
    }

    fn view<Element: DeviceElement>(buffer: &CudaSlice<Element>, range: Range<usize>) -> CudaView<'_, Element> {
        buffer.slice(range)
    }

    fn view_mut<Element: DeviceElement>(buffer: &mut CudaSlice<Element>, range: Range<usize>) -> CudaViewMut<'_, Element> {
        buffer.slice_mut(range)
    }

    fn view_mut_len<Element: DeviceElement>(view: &CudaViewMut<'_, Element>) -> usize {
        view.len()
    }

    fn view_mut_range<'view, Element: DeviceElement>(view: &'view mut CudaViewMut<'_, Element>, range: Range<usize>) -> CudaViewMut<'view, Element> {
        view.slice_mut(range)
    }

    fn bytes_mut<Element: DeviceElement>(buffer: &mut CudaSlice<Element>) -> Option<CudaViewMut<'_, u8>> {
        let bytes = buffer.len() * core::mem::size_of::<Element>();
        // the buffer's own allocation read as bytes: every byte pattern is a valid u8
        unsafe { buffer.transmute_mut::<u8>(bytes) }
    }

    fn record_event(&self) -> Result<CudaEvent, DriverError> {
        let event = self.context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        event.record(&self.stream)?;
        Ok(event)
    }

    fn elapsed(start: &CudaEvent, end: &CudaEvent) -> Result<Duration, DriverError> {
        crate::ffi::cuda::event_elapsed(start, end)
    }

    fn synchronize(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }

    fn memory_info(&self) -> Result<(usize, usize), DriverError> {
        self.context.mem_get_info()
    }

    fn device_identity(&self) -> Result<String, DriverError> {
        Ok(format!("cuda:{}:{}", self.context.ordinal(), self.context.name()?))
    }

    fn compile_counters(&self) -> &CompileCounters {
        &self.compile_counters
    }
}

impl<'a> LaunchArguments<'a, CudarcRuntime> for LaunchArgs<'a> {
    fn push<Element: DeviceElement>(&mut self, buffer: &'a CudaSlice<Element>) -> &mut Self {
        self.arg(buffer)
    }

    fn push_mut<Element: DeviceElement>(&mut self, buffer: &'a mut CudaSlice<Element>) -> &mut Self {
        self.arg(buffer)
    }

    fn push_view<Element: DeviceElement>(&mut self, view: &'a CudaView<'_, Element>) -> &mut Self {
        self.arg(view)
    }

    fn push_view_mut<Element: DeviceElement>(&mut self, view: &'a mut CudaViewMut<'_, Element>) -> &mut Self {
        self.arg(view)
    }

    fn push_value<Element: DeviceElement>(&mut self, value: &'a Element) -> &mut Self {
        self.arg(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cudarc_runtime_grants_what_the_cuda_executor_hard_coded() {
        assert_eq!(CudarcRuntime::cuda_grant(), cuda_executes(&Target::CUDA));
    }

    #[test]
    fn cudarc_runtime_emits_for_the_cuda_target() {
        assert_eq!(CudarcRuntime::cuda_target(), Target::CUDA);
    }
}
