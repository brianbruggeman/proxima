//! The per-device Metal state shared by every thread: compiled pipelines and
//! rendered mixed-expert kernels.
//!
//! Composes `arc_swap::ArcSwap` (a lock-free snapshot read; a miss publishes
//! with `rcu`) over `objc2-metal`'s `MTLDevice` and `MTLComputePipelineState`,
//! which objc2-metal 0.3.2 marks `Send + Sync` (`MTLDevice.rs:664`,
//! `MTLComputePipeline.rs:340`), so sharing needs no `unsafe`. A compiled
//! pipeline is a function of kernel key and device, not of a thread or a
//! model, so the cache belongs to the device. Compose the maps directly when
//! you need a different key; use [`MetalDevice`] when you want the cache every
//! ragd thread already shares.
//!
//! [`MetalDevice::current`] is the device the active
//! [`MetalContext`](super::MetalContext) holds, else the one process-default
//! [`MetalDevice::system_default`] (the OS default device is itself a global).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use core::cell::RefCell;
use std::sync::OnceLock;

use arc_swap::ArcSwap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice};

use super::MetalError;
use crate::msl::Kernel;

type PipelineMap = BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>;

thread_local! {
    /// The active context's device, swapped in by `MetalContext::activate`.
    /// `None` means the process default applies.
    pub(super) static ACTIVE_DEVICE: RefCell<Option<MetalDevice>> = const { RefCell::new(None) };
}

static SYSTEM_DEFAULT: OnceLock<MetalDevice> = OnceLock::new();

struct DeviceShared {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipelines: ArcSwap<PipelineMap>,
    mixed_kernels: ArcSwap<BTreeMap<String, Kernel>>,
}

/// A cheap-to-clone handle on one Metal device and the caches compiled for it.
#[derive(Clone)]
pub struct MetalDevice(Arc<DeviceShared>);

impl MetalDevice {
    /// A fresh, empty cache set over `device`.
    #[must_use]
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self(Arc::new(DeviceShared {
            device,
            pipelines: ArcSwap::from_pointee(PipelineMap::new()),
            mixed_kernels: ArcSwap::from_pointee(BTreeMap::new()),
        }))
    }

    /// The process-wide handle on the OS default device, created on first use.
    pub fn system_default() -> Result<Self, MetalError> {
        if let Some(existing) = SYSTEM_DEFAULT.get() {
            return Ok(existing.clone());
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        Ok(SYSTEM_DEFAULT.get_or_init(|| Self::new(device)).clone())
    }

    /// The active context's device, else [`Self::system_default`].
    pub fn current() -> Result<Self, MetalError> {
        match ACTIVE_DEVICE.with(|slot| slot.borrow().clone()) {
            Some(active) => Ok(active),
            None => Self::system_default(),
        }
    }

    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.0.device
    }

    /// How many distinct pipelines are compiled on this device.
    #[must_use]
    pub fn pipeline_count(&self) -> usize {
        self.0.pipelines.load().len()
    }

    /// Every compiled pipeline's cache key.
    #[must_use]
    pub fn pipeline_keys(&self) -> alloc::vec::Vec<String> {
        self.0.pipelines.load().keys().cloned().collect()
    }

    /// A lock-free lookup of the pipeline compiled under `key`.
    #[must_use]
    pub fn pipeline(&self, key: &str) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.0.pipelines.load().get(key).cloned()
    }

    /// Publishes `pipeline` under `key` and returns the entry that won: when a
    /// racing thread published first, its pipeline is returned and `pipeline`
    /// (one wasted compile) is dropped, so every caller dispatches one object.
    pub fn insert_pipeline(
        &self,
        key: &str,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        self.0.pipelines.rcu(|current| {
            let mut next = PipelineMap::clone(current);
            next.entry(key.into()).or_insert_with(|| pipeline.clone());
            next
        });
        self.pipeline(key).unwrap_or(pipeline)
    }

    /// A lock-free lookup of the mixed-expert kernel rendered under `key`.
    #[must_use]
    pub fn mixed_kernel(&self, key: &str) -> Option<Kernel> {
        self.0.mixed_kernels.load().get(key).cloned()
    }

    pub fn insert_mixed_kernel(&self, key: &str, kernel: &Kernel) {
        self.0.mixed_kernels.rcu(|current| {
            let mut next = BTreeMap::clone(current);
            next.entry(key.into()).or_insert_with(|| kernel.clone());
            next
        });
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec::Vec;

    use super::*;
    use crate::metal::{MathMode, pipeline_for_kernel};
    use crate::msl::GridSpec;

    const PROBE_SOURCE: &str = "#include <metal_stdlib>\n\
        using namespace metal;\n\
        kernel void s5_probe(device float* output [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n\
            output[gid] = float(gid);\n\
        }\n";

    fn probe_kernel() -> Kernel {
        Kernel {
            source: PROBE_SOURCE.to_string(),
            entry: "s5_probe".to_string(),
            bindings: Vec::new(),
            grid: GridSpec { threads: 8, threadgroup_width: None, depth: 1, grid2d: None },
        }
    }

    fn private_device() -> Result<MetalDevice, MetalError> {
        Ok(MetalDevice::new(MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?))
    }

    // a spawned thread has no active context and would resolve the process default,
    // whose cache every other test mutates; `shared` scopes the cache to this test
    fn resolve_on_new_thread(
        shared: &MetalDevice,
        kernel: Kernel,
        key: &'static str,
    ) -> Result<usize, MetalError> {
        let shared = shared.clone();
        let handle = std::thread::spawn(move || {
            ACTIVE_DEVICE.with(|slot| *slot.borrow_mut() = Some(shared));
            let device = MetalDevice::current()?;
            let pipeline = pipeline_for_kernel(device.device(), &kernel, key, MathMode::Relaxed)?;
            Ok(Retained::as_ptr(&pipeline) as *const u8 as usize)
        });
        handle.join().unwrap_or(Err(MetalError::NoDevice))
    }

    #[test]
    fn two_threads_share_one_compiled_pipeline() -> Result<(), MetalError> {
        let device = private_device()?;
        assert_eq!(device.pipeline_count(), 0, "a fresh device starts with an empty cache");
        let first = resolve_on_new_thread(&device, probe_kernel(), "s5_two_threads_key")?;
        let second = resolve_on_new_thread(&device, probe_kernel(), "s5_two_threads_key")?;
        assert_eq!(device.pipeline_keys(), ["s5_two_threads_key"], "one key compiled once across two threads");
        assert_eq!(first, second, "the second thread must hit the first thread's pipeline");
        assert!(first != 0, "N==0 is red: no pipeline pointer came back");
        Ok(())
    }

    #[test]
    fn mixed_kernel_rendered_on_one_thread_is_found_on_another() -> Result<(), MetalError> {
        let device = MetalDevice::system_default()?;
        device.insert_mixed_kernel("s5_mixed_key", &probe_kernel());
        let found = std::thread::spawn(move || {
            MetalDevice::current().ok().and_then(|device| device.mixed_kernel("s5_mixed_key"))
        })
        .join()
        .unwrap_or(None);
        assert_eq!(found, Some(probe_kernel()));
        Ok(())
    }
}
