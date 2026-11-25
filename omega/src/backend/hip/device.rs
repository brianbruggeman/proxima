use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use std::sync::Arc;

use proxima_numeric::NumericPolicy;

use crate::ffi::hip::HipApi;

use super::binding::bind_ordinal;
use super::buffer::{HipBuffer, HipView};
use super::error::HipDriverError;
use super::module::HipModule;

/// The options hipRTC receives: one `--offload-arch` naming the opened device
/// unless the caller already supplied an arch, so a code object never targets
/// a device other than the one the plan selected.
pub(super) fn compile_options(arch: Option<&str>, options: &[&str]) -> Vec<String> {
    let explicit = options.iter().any(|option| {
        option.starts_with("--offload-arch") || option.starts_with("--gpu-architecture")
    });
    let mut resolved = Vec::with_capacity(options.len() + 1);
    if !explicit && let Some(arch) = arch.filter(|arch| !arch.is_empty()) {
        resolved.push(format!("--offload-arch={arch}"));
    }
    resolved.extend(options.iter().map(|option| (*option).to_owned()));
    resolved
}

/// Converts the configured ordinal to HIP's integer device index before any
/// runtime call.
///
/// # Errors
/// [`HipDriverError::DeviceOrdinalTooLarge`] when `device` does not fit HIP's index.
pub fn open_ordinal(device: u32) -> Result<i32, HipDriverError> {
    i32::try_from(device).map_err(|_| HipDriverError::DeviceOrdinalTooLarge(device))
}

/// The HIP lane's numeric policy: the conflaguration/proxima-numeric policy
/// with the accumulator and flash narrowing every GPU lane applies.
#[must_use]
pub fn numeric_policy(numeric: NumericPolicy) -> NumericPolicy {
    crate::bind_plan::binding_policy(
        numeric,
        &crate::cuda::executes(&crate::cuda::target::Target::HIP),
    )
}

/// One ROCm device, resolved through the runtime API.
#[derive(Debug)]
pub struct HipDevice {
    pub(super) api: Arc<HipApi>,
    pub(super) ordinal: i32,
    pub(super) name: String,
    arch: Option<String>,
}

impl HipDevice {
    /// The devices this host exposes. An empty vector means the runtime loaded
    /// but sees no GPU.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when `hipInit` or `hipGetDeviceCount` fails.
    pub fn devices() -> Result<Vec<String>, HipDriverError> {
        let api = HipApi::load()?;
        api.initialize()?;
        let count = api.device_count()?;
        (0..count).map(|ordinal| api.device_name(ordinal)).collect()
    }

    /// Opens `ordinal`, making it the current device for every launch issued
    /// through this value.
    ///
    /// # Errors
    /// [`HipDriverError::RuntimeNotInstalled`] or
    /// [`HipDriverError::RtcNotInstalled`] when ROCm is absent,
    /// [`HipDriverError::DeviceOutOfRange`] past the device count, and
    /// [`HipDriverError::Runtime`] when the device cannot be selected.
    pub fn open(ordinal: i32) -> Result<Self, HipDriverError> {
        let api = HipApi::load()?;
        api.initialize()?;
        let count = api.device_count()?;
        if ordinal < 0 || ordinal >= count {
            return Err(HipDriverError::DeviceOutOfRange { ordinal, count });
        }
        bind_ordinal(ordinal, |ordinal| api.set_device(ordinal))?;
        let name = api.device_name(ordinal)?;
        let arch = api.device_arch(ordinal);
        Ok(Self {
            api,
            ordinal,
            name,
            arch,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The device's `gcnArchName` when the runtime exposes
    /// `hipGetDevicePropertiesR0600`; `None` lets hipRTC target the current
    /// device itself.
    #[must_use]
    pub fn arch(&self) -> Option<&str> {
        self.arch.as_deref()
    }

    /// The `(major, minor)` of the hipRTC that compiles for this device, or `None` when the
    /// library does not export `hiprtcVersion` or the call fails.
    #[must_use]
    pub fn compiler_version(&self) -> Option<(i32, i32)> {
        self.api.compiler_version()
    }

    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    #[must_use]
    pub fn api(&self) -> &Arc<HipApi> {
        &self.api
    }

    /// Makes this device the calling thread's current HIP device; a no-op when the thread already
    /// bound it. Every entry that allocates, copies, sets memory, compiles, loads, launches,
    /// synchronizes or queries memory starts here, whichever thread calls it.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when `hipSetDevice` fails.
    pub fn bind(&self) -> Result<(), HipDriverError> {
        bind_ordinal(self.ordinal, |ordinal| self.api.set_device(ordinal))
    }

    /// Compiles HIP C++ to a loadable code object.
    ///
    /// # Errors
    /// [`HipDriverError::CompileLog`] with the compiler's own log when the
    /// source is rejected, [`HipDriverError::Rtc`] when hipRTC itself fails,
    /// and [`HipDriverError::Runtime`] when the object cannot be loaded.
    pub fn compile(&self, source: &str, options: &[&str]) -> Result<HipModule, HipDriverError> {
        self.load_code_object(&self.compile_code_object(source, options)?)
    }

    /// Compiles HIP C++ to the code-object bytes [`Self::load_code_object`] loads, so a caller can
    /// keep the bytes between processes.
    ///
    /// # Errors
    /// [`HipDriverError::CompileLog`] with the compiler's own log when the source is rejected, and
    /// [`HipDriverError::Rtc`] when hipRTC itself fails.
    pub fn compile_code_object(
        &self,
        source: &str,
        options: &[&str],
    ) -> Result<Vec<u8>, HipDriverError> {
        self.bind()?;
        self.api
            .compile_code_object(source, &compile_options(self.arch(), options))
    }

    /// Loads a code object produced by [`Self::compile_code_object`] for this device's arch.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the object cannot be loaded.
    pub fn load_code_object(&self, code: &[u8]) -> Result<HipModule, HipDriverError> {
        self.bind()?;
        let handle = self.api.load_module(code)?;
        Ok(HipModule {
            api: Arc::clone(&self.api),
            handle,
            ordinal: self.ordinal,
        })
    }

    /// Uploads raw bytes (uniforms, packed weight blocks).
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the allocation or copy fails.
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<HipBuffer, HipDriverError> {
        self.bind()?;
        let buffer = self.allocate_bytes(bytes.len())?;
        // SAFETY: `buffer.pointer` holds `buffer.bytes >= bytes.len()` bytes.
        unsafe { self.api.copy_to_device(buffer.pointer, bytes) }?;
        Ok(buffer)
    }

    /// Allocates `bytes` of uninitialized device memory (at least one byte,
    /// so an empty request still owns a valid pointer).
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the allocation fails.
    pub fn allocate_bytes(&self, bytes: usize) -> Result<HipBuffer, HipDriverError> {
        self.bind()?;
        let requested = bytes.max(1);
        let pointer = self.api.allocate(requested)?;
        Ok(HipBuffer {
            api: Arc::clone(&self.api),
            pointer,
            bytes: requested,
            ordinal: self.ordinal,
        })
    }

    /// Allocates `bytes` of device memory and zeroes all of it on the device,
    /// with no host-side zero buffer: the device twin of `alloc_zeros`.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the allocation or the memset fails.
    pub fn allocate_zeroed_bytes(&self, bytes: usize) -> Result<HipBuffer, HipDriverError> {
        self.bind()?;
        let buffer = self.allocate_bytes(bytes)?;
        // SAFETY: `buffer.pointer` owns `buffer.bytes` bytes of device memory.
        unsafe { self.api.zero(buffer.pointer, buffer.bytes) }?;
        Ok(buffer)
    }

    /// Copies `source`'s bytes into the start of `destination` on the device.
    ///
    /// # Errors
    /// [`HipDriverError::ViewOutOfRange`] when `source` is longer than `destination`,
    /// [`HipDriverError::Runtime`] when the copy fails.
    pub fn copy_device(
        &self,
        destination: &HipView<'_>,
        source: &HipView<'_>,
    ) -> Result<(), HipDriverError> {
        if source.len > destination.len {
            return Err(HipDriverError::ViewOutOfRange {
                offset: 0,
                len: source.len,
                bytes: destination.len,
            });
        }
        if source.is_empty() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: both views were bounds-checked against live allocations
        // they borrow, and `source.len <= destination.len` by the check above.
        unsafe {
            self.api
                .copy_within_device(destination.pointer, source.pointer, source.len)
        }
    }

    /// Blocks until every queued launch on this device has completed.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when synchronization fails.
    pub fn synchronize(&self) -> Result<(), HipDriverError> {
        self.bind()?;
        self.api.synchronize()
    }
}
