use core::marker::PhantomData;
use core::mem;
use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLResidencySet};

#[cfg(feature = "instrument")]
use super::arena_encode_dispatch_finish::{
    CAPTURE_DUMP_PENDING, CAPTURE_LIVE_PENDING, CaptureDumpEntry, CapturedDispatch,
};
use super::device_buffers_arena_plan::{ACTIVE_QUEUE, EXPERT_SOURCE_CACHE, StagedExpertSource};
use super::metal_device::{ACTIVE_DEVICE, MetalDevice};
use super::pipeline_buffers_upload::{CHECKPOINT_MAPPING, NOCOPY_BUFFERS};
use super::resident_nocopy_cache::{
    FAULT_BUFFER_POOL, RESIDENT_BUFFERS, UNIFORM_BUFFERS, UNIFORM_CACHE_CLOCK,
};
use super::shared_transient::{ARENA_TRANSIENT, ATTENTION_SCRATCH_TRANSIENT};
use super::warm_resident_buffers::{
    MAPPING_RESIDENCY, WARM_RESIDENCY_SET, release_all_mapping_residency, warm_active_context,
};
use super::{MetalBuffer, NodeId};

type ResidentBuffers = BTreeMap<String, (usize, usize, MetalBuffer)>;
type UniformBuffers = BTreeMap<Vec<u8>, (MetalBuffer, u64)>;
type FaultBuffers = BTreeMap<usize, Vec<MetalBuffer>>;
type MappingResidency = BTreeMap<String, Retained<ProtocolObject<dyn MTLResidencySet>>>;
type ExpertSources = BTreeMap<(usize, NodeId), (u64, StagedExpertSource)>;

/// independent model residency, retaining uploads between activations.
///
/// activate around model loading, evaluation, and destruction. destroy plans
/// before their model, while active, so no borrowed weight buffer outlives its
/// allocation. expert sidecar registration remains outside this context.
#[derive(Default)]
pub struct MetalContext {
    resident: ResidentBuffers,
    nocopy: ResidentBuffers,
    checkpoint: Option<(usize, usize)>,
    experts: ExpertSources,
    uniforms: UniformBuffers,
    uniform_clock: u64,
    fault_buffers: FaultBuffers,
    arena_transient: Weak<MetalBuffer>,
    attention_transient: Weak<MetalBuffer>,
    queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    device: Option<MetalDevice>,
    residency: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    mapping_residency: MappingResidency,
    #[cfg(feature = "instrument")]
    capture_dumps: Vec<CaptureDumpEntry>,
    #[cfg(feature = "instrument")]
    capture_live: Vec<CapturedDispatch>,
}

impl MetalContext {
    /// a context that owns the queue its model submits to and the residency
    /// set that keeps the model's wired weight pages resident. `None` for
    /// either means the thread's own queue, or no residency set, applies.
    #[must_use]
    pub fn new(
        queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        residency: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    ) -> Self {
        Self {
            queue,
            residency,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn queue(&self) -> Option<&ProtocolObject<dyn MTLCommandQueue>> {
        self.queue.as_deref()
    }

    #[must_use]
    pub fn residency(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.residency.as_deref()
    }

    /// weak handles to the pooled arena and attention-scratch buffers while this context is
    /// inactive. They hold no allocation alive: once every plan leasing a buffer and the
    /// context itself are gone, `upgrade` returns `None`, which is how a test reads that
    /// dropping a model released its Metal buffers.
    #[must_use]
    pub fn transient_handles(&self) -> (Weak<MetalBuffer>, Weak<MetalBuffer>) {
        (
            self.arena_transient.clone(),
            self.attention_transient.clone(),
        )
    }

    /// warms this context's resident buffers on its own queue: activates itself,
    /// gives itself a queue if it has none, and requests the residency set it
    /// keeps. A second call with a held set returns the held count. Composes
    /// the same body as [`super::warm_resident_buffers`], which stays the
    /// no-context entry on the thread's queue.
    pub fn warm(&mut self) -> Result<usize, super::MetalError> {
        let _guard = self.activate();
        warm_active_context()
    }

    /// restores the enclosing context when the guard drops, including unwind.
    pub fn activate(&mut self) -> MetalContextGuard<'_> {
        self.swap_active();
        MetalContextGuard {
            context: self,
            thread: PhantomData,
        }
    }

    #[cfg(not(feature = "instrument"))]
    fn swap_active(&mut self) {
        self.swap_residency();
    }

    #[cfg(feature = "instrument")]
    fn swap_active(&mut self) {
        CAPTURE_DUMP_PENDING.with(|dumps| {
            CAPTURE_LIVE_PENDING.with(|live| {
                let mut dumps = dumps.borrow_mut();
                let mut live = live.borrow_mut();
                self.swap_residency();
                mem::swap(&mut self.capture_dumps, &mut dumps);
                mem::swap(&mut self.capture_live, &mut live);
            });
        });
    }

    fn swap_residency(&mut self) {
        ACTIVE_QUEUE.with(|queue| mem::swap(&mut self.queue, &mut *queue.borrow_mut()));
        ACTIVE_DEVICE.with(|device| mem::swap(&mut self.device, &mut *device.borrow_mut()));
        WARM_RESIDENCY_SET.with(|set| mem::swap(&mut self.residency, &mut *set.borrow_mut()));
        MAPPING_RESIDENCY
            .with(|held| mem::swap(&mut self.mapping_residency, &mut *held.borrow_mut()));
        self.swap_caches();
        self.swap_scratch();
    }

    fn swap_scratch(&mut self) {
        self.swap_buffer_caches();
        self.swap_transient();
    }

    fn swap_buffer_caches(&mut self) {
        UNIFORM_BUFFERS.with(|uniforms| {
            UNIFORM_CACHE_CLOCK.with(|clock| {
                FAULT_BUFFER_POOL.with(|faults| {
                    let mut uniforms = uniforms.borrow_mut();
                    let mut clock = clock.borrow_mut();
                    let mut faults = faults.borrow_mut();
                    mem::swap(&mut self.uniforms, &mut uniforms);
                    mem::swap(&mut self.uniform_clock, &mut clock);
                    mem::swap(&mut self.fault_buffers, &mut faults);
                });
            });
        });
    }

    fn swap_transient(&mut self) {
        ARENA_TRANSIENT.with(|arena| arena.swap_live(&mut self.arena_transient));
        ATTENTION_SCRATCH_TRANSIENT
            .with(|attention| attention.swap_live(&mut self.attention_transient));
    }

    fn swap_caches(&mut self) {
        RESIDENT_BUFFERS.with(|resident| {
            NOCOPY_BUFFERS.with(|nocopy| {
                CHECKPOINT_MAPPING.with(|checkpoint| {
                    EXPERT_SOURCE_CACHE.with(|experts| {
                        // borrow every slot before changing any, so a failed borrow is atomic.
                        let mut resident = resident.borrow_mut();
                        let mut nocopy = nocopy.borrow_mut();
                        let mut checkpoint = checkpoint.borrow_mut();
                        let mut experts = experts.borrow_mut();
                        mem::swap(&mut self.resident, &mut resident);
                        mem::swap(&mut self.nocopy, &mut nocopy);
                        mem::swap(&mut self.checkpoint, &mut checkpoint);
                        mem::swap(&mut self.experts, &mut experts);
                    });
                });
            });
        });
    }
}

/// thread-bound activation; dropping it restores the enclosing model's caches.
#[must_use]
pub struct MetalContextGuard<'context> {
    context: &'context mut MetalContext,
    thread: PhantomData<Rc<()>>,
}

impl MetalContextGuard<'_> {
    /// releases active aliases after plans drop and before their backing model.
    pub fn clear(&mut self) {
        #[cfg(feature = "instrument")]
        {
            CAPTURE_DUMP_PENDING.with(|pending| pending.borrow_mut().clear());
            CAPTURE_LIVE_PENDING.with(|pending| pending.borrow_mut().clear());
        }
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        release_all_mapping_residency();
        NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().clear());
        RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().clear());
        CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    }
}

impl Drop for MetalContextGuard<'_> {
    fn drop(&mut self) {
        self.context.swap_active();
    }
}

#[cfg(test)]
#[path = "metal_context_tests.rs"]
mod metal_context_tests;
