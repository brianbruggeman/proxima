use core::marker::PhantomData;
use core::mem;
use std::collections::BTreeMap;
use std::rc::Rc;

#[cfg(feature = "instrument")]
use super::arena_encode_dispatch_finish::{
    CAPTURE_DUMP_PENDING, CAPTURE_LIVE_PENDING, CaptureDumpEntry, CapturedDispatch,
};
use super::device_buffers_arena_plan::{EXPERT_SOURCE_CACHE, StagedExpertSource};
use super::pipeline_buffers_upload::{CHECKPOINT_MAPPING, NOCOPY_BUFFERS};
use super::resident_nocopy_cache::RESIDENT_BUFFERS;
use super::{MetalBuffer, NodeId};

type ResidentBuffers = BTreeMap<String, (usize, usize, MetalBuffer)>;
type ExpertSources = BTreeMap<(usize, NodeId), (u64, StagedExpertSource)>;

/// independent model residency, retaining uploads between activations.
///
/// activate around model loading, evaluation, and destruction. destroy plans
/// before their model, while active, so no borrowed weight buffer outlives its
/// allocation. expert sidecar registration remains outside this context.
#[derive(Default)]
pub struct ResidentContext {
    resident: ResidentBuffers,
    nocopy: ResidentBuffers,
    checkpoint: Option<(usize, usize)>,
    experts: ExpertSources,
    #[cfg(feature = "instrument")]
    capture_dumps: Vec<CaptureDumpEntry>,
    #[cfg(feature = "instrument")]
    capture_live: Vec<CapturedDispatch>,
}

impl ResidentContext {
    /// restores the enclosing context when the guard drops, including unwind.
    pub fn activate(&mut self) -> ResidentContextGuard<'_> {
        self.swap_active();
        ResidentContextGuard {
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
pub struct ResidentContextGuard<'context> {
    context: &'context mut ResidentContext,
    thread: PhantomData<Rc<()>>,
}

impl ResidentContextGuard<'_> {
    /// releases active aliases after plans drop and before their backing model.
    pub fn clear(&mut self) {
        #[cfg(feature = "instrument")]
        {
            CAPTURE_DUMP_PENDING.with(|pending| pending.borrow_mut().clear());
            CAPTURE_LIVE_PENDING.with(|pending| pending.borrow_mut().clear());
        }
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().clear());
        RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().clear());
        CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    }
}

impl Drop for ResidentContextGuard<'_> {
    fn drop(&mut self) {
        self.context.swap_active();
    }
}

#[cfg(test)]
#[path = "resident_context_tests.rs"]
mod resident_context_tests;
