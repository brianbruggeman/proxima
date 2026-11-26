use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::time::Duration;

use std::sync::Arc;

use crate::config::RuntimeConfig;
use crate::cuda::{CudaKernel, EmitDevice, HipKernel, HipWave};
use crate::cuda_driver::{
    CompileCounters, CudaDriver, DevicePlan, kernel_disk_store, stored_or_compiled,
};
use crate::ffi::hip::DevicePointer;
use crate::gpu_runtime::{DeviceElement, GpuRuntime, LaunchArguments, LaunchShape};
use crate::kernel_params::KernelParams;
use crate::lowering::Binding;

use super::buffer::{HipBuffer, HipView, offset_pointer};
use super::code_object::{FunctionCache, HIPRTC_OPTIONS, cached_by_source, code_object_key};
use super::device::{HipDevice, open_ordinal};
use super::error::HipDriverError;
use super::launch::HipLaunch;
use super::module::{HipFunction, HipImage, HipModule};
use super::shared::resolve_wave;
use super::window::{HipSlice, HipWindow, HipWindowMut, WindowRefusal, element_window};

/// The plan the same executor builds on the HIP runtime.
pub type HipPlan = DevicePlan<HipRuntime>;

/// Probe kernel: writes the device's `warpSize`, so the wave width the
/// emitter is built for comes from the hardware, never from a card table.
fn warp_size_kernel() -> Result<HipKernel, HipDriverError> {
    Ok(HipKernel {
        source: String::from(
            "extern \"C\" __global__ void proxima_warp_size(unsigned int *out, const unsigned int *u) {\n\
             \x20   if (threadIdx.x == 0) out[0] = warpSize;\n\
             }\n",
        ),
        entry: String::from("proxima_warp_size"),
        bindings: vec![
            Binding::Output(proxima_tensor::NodeId(0)),
            Binding::Uniforms,
        ],
        grid: crate::cuda::HipGridSpec::launch(
            proxima_tensor::NodeId(0),
            1,
            None,
            "warp_size_probe",
        )
        .map_err(|_| HipDriverError::IdleLaunch)?,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
        route_compaction_words: 0,
        reduce_path: "",
    })
}

/// The shared executor's runtime on a ROCm device: the [`GpuRuntime`] a `DevicePlan` runs on when
/// the lowering is CUDA's and the kernels render as HIP C++ at `wave`.
///
/// Teaching pointer: the cudarc runtime is the other [`GpuRuntime`]; both are built from the same
/// executor bodies. Launch parameters come from [`KernelParams`] built over borrowed arguments,
/// and the function cache is keyed by the kernel's entry and then its exact source text, as the
/// CUDA runtime's is, so two sources that share an entry name never share a function.
#[derive(Debug, Clone)]
pub struct HipRuntime {
    device: Arc<HipDevice>,
    wave: HipWave,
    functions: Arc<proxima_primitives::sync::blocking::Mutex<FunctionCache<HipFunction>>>,
    compile_counters: Arc<CompileCounters>,
}

impl HipRuntime {
    /// A runtime over an opened `device`, emitting for `wave`.
    #[must_use]
    pub fn new(device: HipDevice, wave: HipWave) -> Self {
        Self {
            device: Arc::new(device),
            wave,
            functions: Arc::new(proxima_primitives::sync::blocking::Mutex::new(
                FunctionCache::new(),
            )),
            compile_counters: Arc::new(CompileCounters::new(kernel_disk_store())),
        }
    }

    /// The device this runtime launches on.
    #[must_use]
    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    /// The module for `source`, its code object read from the kernel disk store when one built for
    /// this device's arch, this hipRTC version and these options is there. Without a known arch or
    /// compiler version no key can name the target, so the object is compiled and never stored.
    fn code_object_module(&self, source: &str) -> Result<HipModule, HipDriverError> {
        let compile = || -> Result<Vec<u8>, HipDriverError> {
            let code = self.device.compile_code_object(source, HIPRTC_OPTIONS)?;
            self.compile_counters
                .compilations
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(code)
        };
        let arch = self.device.arch().filter(|arch| !arch.is_empty());
        let code = match (arch, self.device.compiler_version()) {
            (Some(arch), Some(compiler)) => {
                let key = code_object_key(source, arch, compiler, HIPRTC_OPTIONS);
                stored_or_compiled(&self.compile_counters, &key, compile)?
            }
            _ => compile()?,
        };
        self.device.load_code_object(&code)
    }

    /// Opens `ordinal`, probes the width the device reports (the `warpSize` kernel launched through
    /// this runtime's own launch path), and emits for it. A `configured` width must equal the probe.
    ///
    /// # Errors
    /// The ROCm refusals of [`HipDevice::open`], [`HipDriverError::UnsupportedWaveWidth`]
    /// for a reported width other than 32 or 64, and [`HipDriverError::WaveMismatch`] when
    /// `configured` differs from the reported width.
    pub fn open(ordinal: i32, configured: Option<HipWave>) -> Result<Self, HipDriverError> {
        let probe = Self::new(HipDevice::open(ordinal)?, HipWave::Wave64);
        let lanes = probe.warp_size()?;
        let probed = HipWave::from_lanes(u64::from(lanes))
            .ok_or(HipDriverError::UnsupportedWaveWidth(lanes))?;
        let wave = resolve_wave(configured, probed)?;
        Ok(Self { wave, ..probe })
    }

    /// The device's wavefront width, read by launching a one-thread kernel that stores `warpSize`.
    ///
    /// # Errors
    /// Any build, launch or copy error.
    pub fn warp_size(&self) -> Result<u32, HipDriverError> {
        let function = self.build_function(&warp_size_kernel()?)?;
        let mut lanes = self.alloc_zeros::<u32>(1)?;
        let uniforms = self.upload::<u8>(&[0u8; 4])?;
        let mut arguments = self.launch_arguments(&function);
        arguments.push_mut(&mut lanes).push(&uniforms);
        let shape = LaunchShape {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the probe kernel's parameters are its output buffer then its uniforms, as pushed.
        unsafe { self.launch(arguments, shape) }?;
        self.synchronize()?;
        Ok(self.download(&lanes)?.first().copied().unwrap_or(0))
    }

    fn slice_from<Element: DeviceElement>(buffer: HipBuffer, len: usize) -> HipSlice<Element> {
        HipSlice {
            buffer: Arc::new(buffer),
            len,
            element: core::marker::PhantomData,
        }
    }

    fn byte_len<Element>(len: usize) -> Result<usize, HipDriverError> {
        len.checked_mul(core::mem::size_of::<Element>())
            .ok_or(HipDriverError::InputTooLarge(len))
    }

    fn check_range<Element>(
        buffer: &HipSlice<Element>,
        range: &core::ops::Range<usize>,
    ) -> Result<(), HipDriverError> {
        let size = core::mem::size_of::<Element>();
        (range.start <= range.end && range.end <= buffer.len)
            .then_some(())
            .ok_or(HipDriverError::ViewOutOfRange {
                offset: range.start * size,
                len: range.end.saturating_sub(range.start) * size,
                bytes: buffer.len * size,
            })
    }

    fn window_parts(
        window: Result<(usize, usize), WindowRefusal>,
    ) -> (usize, usize, Option<WindowRefusal>) {
        match window {
            Ok((offset, len)) => (offset, len, None),
            Err(refusal) => (0, 0, Some(refusal)),
        }
    }

    fn sub_view<'any>(pointer: DevicePointer, offset: usize, len: usize) -> HipView<'any> {
        HipView {
            pointer: offset_pointer(pointer, offset),
            len,
            owner: core::marker::PhantomData,
        }
    }

    fn element_bytes<Element: DeviceElement>(values: &[Element]) -> &[u8] {
        // SAFETY: the elements are plain device-representable values, read as their own bytes.
        unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        }
    }
}

impl CudaDriver<HipRuntime> {
    /// Opens the configured ROCm device and builds the shared executor's driver over it: the same
    /// [`CudaDriver`] a CUDA device gets, so `plan_with_config` lowers and runs HIP plans.
    ///
    /// # Errors
    /// The refusals of [`HipRuntime::open`].
    pub fn open(config: &RuntimeConfig, wave: Option<HipWave>) -> Result<Self, HipDriverError> {
        let runtime = HipRuntime::open(open_ordinal(config.hip_device)?, wave)?;
        let baseline_used_bytes = runtime
            .memory_info()
            .map_or(0, |(free, total)| total.saturating_sub(free) as u64);
        Ok(Self::over_runtime(runtime, baseline_used_bytes))
    }

    /// The device name this driver runs on.
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.runtime().device().name()
    }
}

impl GpuRuntime for HipRuntime {
    type Error = HipDriverError;
    type BuildError = HipDriverError;
    type Buffer<Element: DeviceElement> = HipSlice<Element>;
    type View<'a, Element: DeviceElement> = HipWindow<'a, Element>;
    type ViewMut<'a, Element: DeviceElement> = HipWindowMut<'a, Element>;
    type Event = core::convert::Infallible;
    type Function = HipFunction;
    type Module = HipModule;
    type Image = HipImage;
    type Launch<'a> = HipLaunch<'a>;

    fn alloc_zeros<Element: DeviceElement>(
        &self,
        len: usize,
    ) -> Result<HipSlice<Element>, HipDriverError> {
        let buffer = self
            .device
            .allocate_zeroed_bytes(Self::byte_len::<Element>(len)?)?;
        Ok(Self::slice_from(buffer, len))
    }

    fn upload<Element: DeviceElement>(
        &self,
        values: &[Element],
    ) -> Result<HipSlice<Element>, HipDriverError> {
        let buffer = self.device.upload_bytes(Self::element_bytes(values))?;
        Ok(Self::slice_from(buffer, values.len()))
    }

    fn upload_into<Element: DeviceElement>(
        &self,
        values: &[Element],
        buffer: &mut HipSlice<Element>,
    ) -> Result<(), HipDriverError> {
        self.upload_at(values, buffer, 0)
    }

    fn upload_at<Element: DeviceElement>(
        &self,
        values: &[Element],
        buffer: &mut HipSlice<Element>,
        offset: usize,
    ) -> Result<(), HipDriverError> {
        let range = offset..offset + values.len();
        Self::check_range(buffer, &range)?;
        self.device.bind()?;
        let size = core::mem::size_of::<Element>();
        let destination = offset_pointer(buffer.buffer.pointer, offset * size);
        // SAFETY: `check_range` proved the window lies inside the allocation `destination` starts in.
        unsafe {
            self.device
                .api
                .copy_to_device(destination, Self::element_bytes(values))
        }
    }

    fn download<Element: DeviceElement>(
        &self,
        buffer: &HipSlice<Element>,
    ) -> Result<Vec<Element>, HipDriverError> {
        self.download_range(buffer, 0..buffer.len)
    }

    fn download_range<Element: DeviceElement>(
        &self,
        buffer: &HipSlice<Element>,
        range: core::ops::Range<usize>,
    ) -> Result<Vec<Element>, HipDriverError> {
        Self::check_range(buffer, &range)?;
        self.device.bind()?;
        let size = core::mem::size_of::<Element>();
        let mut values = Vec::<Element>::with_capacity(range.len());
        let source = offset_pointer(buffer.buffer.pointer, range.start * size);
        // SAFETY: the capacity holds `range.len()` elements, and the byte view covers exactly them.
        let destination = unsafe {
            core::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                range.len() * size,
            )
        };
        // SAFETY: `check_range` proved the source window lies inside the allocation `source` starts in.
        unsafe { self.device.api.copy_from_device(destination, source) }?;
        // SAFETY: the copy initialized `range.len()` elements, and every bit pattern is a valid element.
        unsafe { values.set_len(range.len()) };
        Ok(values)
    }

    fn copy_range<Element: DeviceElement>(
        &self,
        source: &HipSlice<Element>,
        range: core::ops::Range<usize>,
        destination: &mut HipSlice<Element>,
        offset: usize,
    ) -> Result<(), HipDriverError> {
        Self::check_range(source, &range)?;
        Self::check_range(destination, &(offset..offset + range.len()))?;
        self.device.bind()?;
        let size = core::mem::size_of::<Element>();
        let from = offset_pointer(source.buffer.pointer, range.start * size);
        let to = offset_pointer(destination.buffer.pointer, offset * size);
        // SAFETY: `check_range` proved both windows lie inside their live allocations.
        unsafe {
            self.device
                .api
                .copy_within_device(to, from, range.len() * size)
        }
    }

    fn build_function(&self, kernel: &CudaKernel) -> Result<HipFunction, HipDriverError> {
        cached_by_source(&self.functions, &kernel.entry, &kernel.source, || {
            let module = Arc::new(self.code_object_module(&kernel.source)?);
            Ok(HipFunction {
                handle: module.function(&kernel.entry)?,
                _module: module,
            })
        })
    }

    fn image_from_source(source: &str) -> HipImage {
        HipImage(source.to_owned())
    }

    fn load_module(&self, image: HipImage) -> Result<Arc<HipModule>, HipDriverError> {
        Ok(Arc::new(self.code_object_module(&image.0)?))
    }

    fn load_function(
        &self,
        module: &Arc<HipModule>,
        entry: &str,
    ) -> Result<HipFunction, HipDriverError> {
        self.device.bind()?;
        Ok(HipFunction {
            handle: module.function(entry)?,
            _module: Arc::clone(module),
        })
    }

    fn launch_arguments<'a>(&'a self, function: &'a HipFunction) -> HipLaunch<'a> {
        HipLaunch {
            function,
            parameters: KernelParams::new(),
            refusal: None,
        }
    }

    unsafe fn launch(
        &self,
        mut arguments: HipLaunch<'_>,
        shape: LaunchShape,
    ) -> Result<(), HipDriverError> {
        if let Some(refusal) = arguments.refusal {
            return Err(refusal.error());
        }
        self.device.bind()?;
        // SAFETY: the caller matched the arguments to the kernel's parameter list; the params borrow
        // every argument until this call returns, and the launch is on the default stream.
        unsafe {
            self.device.api.launch(
                arguments.function.handle,
                shape.grid_dim,
                shape.block_dim,
                shape.shared_mem_bytes,
                arguments.parameters.as_mut_ptr(),
            )
        }
    }

    fn executes(&self) -> proxima_tensor::Executes {
        crate::cuda::executes(&self.emit_target())
    }

    fn emit_target(&self) -> crate::cuda::target::Target {
        crate::cuda::target::Target::hip_wave(self.wave.lanes())
    }

    fn emit_device(&self) -> EmitDevice {
        EmitDevice::default()
    }

    fn len<Element: DeviceElement>(buffer: &HipSlice<Element>) -> usize {
        buffer.len
    }

    fn view<Element: DeviceElement>(
        buffer: &HipSlice<Element>,
        range: core::ops::Range<usize>,
    ) -> HipWindow<'_, Element> {
        let (offset, len, refusal) = Self::window_parts(element_window(
            buffer.len,
            &range,
            core::mem::size_of::<Element>(),
        ));
        HipWindow {
            view: Self::sub_view(buffer.buffer.pointer, offset, len),
            refusal,
            element: core::marker::PhantomData,
        }
    }

    fn view_mut<Element: DeviceElement>(
        buffer: &mut HipSlice<Element>,
        range: core::ops::Range<usize>,
    ) -> HipWindowMut<'_, Element> {
        let (offset, len, refusal) = Self::window_parts(element_window(
            buffer.len,
            &range,
            core::mem::size_of::<Element>(),
        ));
        HipWindowMut {
            view: Self::sub_view(buffer.buffer.pointer, offset, len),
            refusal,
            exclusive: core::marker::PhantomData,
            element: core::marker::PhantomData,
        }
    }

    fn view_mut_len<Element: DeviceElement>(view: &HipWindowMut<'_, Element>) -> usize {
        view.view.len / core::mem::size_of::<Element>()
    }

    fn view_mut_range<'view, Element: DeviceElement>(
        view: &'view mut HipWindowMut<'_, Element>,
        range: core::ops::Range<usize>,
    ) -> HipWindowMut<'view, Element> {
        let (offset, len, refusal) = Self::window_parts(element_window(
            Self::view_mut_len(view),
            &range,
            core::mem::size_of::<Element>(),
        ));
        HipWindowMut {
            view: Self::sub_view(view.view.pointer, offset, len),
            refusal: view.refusal.or(refusal),
            exclusive: core::marker::PhantomData,
            element: core::marker::PhantomData,
        }
    }

    fn bytes_mut<Element: DeviceElement>(
        buffer: &mut HipSlice<Element>,
    ) -> Option<HipWindowMut<'_, u8>> {
        let bytes = buffer.len * core::mem::size_of::<Element>();
        Some(HipWindowMut {
            view: Self::sub_view(buffer.buffer.pointer, 0, bytes),
            refusal: None,
            exclusive: core::marker::PhantomData,
            element: core::marker::PhantomData,
        })
    }

    fn record_event(&self) -> Result<core::convert::Infallible, HipDriverError> {
        Err(HipDriverError::EventsUnsupported)
    }

    fn elapsed(
        _start: &core::convert::Infallible,
        _end: &core::convert::Infallible,
    ) -> Result<Duration, HipDriverError> {
        Err(HipDriverError::EventsUnsupported)
    }

    fn synchronize(&self) -> Result<(), HipDriverError> {
        self.device.synchronize()
    }

    fn memory_info(&self) -> Result<(usize, usize), HipDriverError> {
        self.device.bind()?;
        self.device.api.memory_info()
    }

    fn device_identity(&self) -> Result<String, HipDriverError> {
        Ok(format!("hip:{}:{}", self.device.ordinal, self.device.name))
    }

    fn compile_counters(&self) -> &CompileCounters {
        &self.compile_counters
    }
}
