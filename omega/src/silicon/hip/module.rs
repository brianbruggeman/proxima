use alloc::string::String;

use std::sync::Arc;

use crate::ffi::hip::{FunctionHandle, HipApi, ModuleHandle};

use super::binding::release_on_device;
use super::error::HipDriverError;

/// A loaded code object. Dropping it unloads the module, so a function handle
/// obtained from it must not outlive it.
pub struct HipModule {
    pub(super) api: Arc<HipApi>,
    pub(super) handle: ModuleHandle,
    pub(super) ordinal: i32,
}

impl core::fmt::Debug for HipModule {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("HipModule")
    }
}

impl HipModule {
    /// The entry point `name` inside this module.
    ///
    /// # Errors
    /// [`HipDriverError::Runtime`] when the module has no such symbol.
    pub fn function(&self, name: &str) -> Result<FunctionHandle, HipDriverError> {
        // SAFETY: `self.handle` is a live module for as long as `self` is.
        unsafe { self.api.module_function(self.handle, name) }
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        release_on_device(&self.api, self.ordinal, || {
            // SAFETY: the handle came from `hipModuleLoadData` on this ordinal's device, bound
            // first, and is unloaded once.
            unsafe { self.api.unload_module(self.handle) };
        });
    }
}

/// A kernel's source, awaiting [`GpuRuntime::load_module`](crate::gpu_runtime::GpuRuntime::load_module).
#[derive(Debug)]
pub struct HipImage(pub(super) String);

/// A compiled entry point; cloning shares the loaded module.
#[derive(Debug, Clone)]
pub struct HipFunction {
    // keeps the code object loaded while any handle to its function lives
    pub(super) _module: Arc<HipModule>,
    pub(super) handle: FunctionHandle,
}

// SAFETY: a HIP code object and function handle are process-wide runtime state addressed by pointer;
// the host never dereferences them, so moving or sharing the owning handle across threads cannot alias
// host memory. The function handle is only passed to the launch call, which binds the runtime's device
// on the launching thread first.
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}
unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}
