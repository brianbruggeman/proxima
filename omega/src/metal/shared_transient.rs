//! One device buffer shared by every plan on this thread whose contents never
//! outlive a single execute call.
//!
//! A plan's packed arena slot and its shared attention scratch are written and
//! read inside one call and dead when it returns, so two plans that never run
//! at once need one allocation between them, sized for the larger. llama
//! reserves one compute buffer for the worst case; this is the same shape,
//! built from the buffer a plan would have allocated anyway
//! ([`allocate_buffer_for`]) plus a [`Weak`] handle to it.
//!
//! # Why serial execution is guaranteed
//!
//! The pool is thread-local, like [`device_and_queue`]: every plan that leases
//! from it submits to this thread's one command queue, and
//! `execute_plan_with_placements` commits and `waitUntilCompleted`s its command
//! buffers before returning (`execute_and_hazards.rs`, "Cross-invocation
//! aliasing"). Two plans on one thread therefore never have the buffer in
//! flight together. A plan holds `Retained` buffers and `RefCell`s, so it is
//! not `Send` and cannot reach another thread's queue.
//!
//! Only transient contents may be leased. Outputs, resident nodes and
//! never-released nodes keep slots of their own (`lay_out_packed` puts exactly
//! the released, non-resident outputs in slot 0).
//!
//! # Lifetime
//!
//! The pool holds a [`Weak`], and each plan holds the [`Rc`] lease, so the
//! buffer lives while some plan uses it and is freed with the last one.

use super::*;
use alloc::rc::{Rc, Weak};

/// A buffer many plans lease; a plan keeps the [`Rc`] for as long as it uses it.
pub(super) type SharedBuffer = Rc<MetalBuffer>;

/// The most recent buffer leased for one purpose on this thread.
pub(super) struct TransientPool {
    live: RefCell<Weak<MetalBuffer>>,
}

impl TransientPool {
    const fn new() -> Self {
        Self {
            live: RefCell::new(Weak::new()),
        }
    }

    /// The pooled buffer when it is still alive and holds `element_count`
    /// elements of `dtype`; otherwise a fresh [`allocate_buffer_for`] buffer,
    /// which becomes the pooled one. A smaller plan joins the buffer a larger
    /// plan already holds; a larger plan replaces it, and plans that hold the
    /// smaller one keep it until they drop.
    pub(super) fn lease(
        &self,
        owner: &'static str,
        device: &ProtocolObject<dyn MTLDevice>,
        element_count: usize,
        dtype: DType,
    ) -> Result<SharedBuffer, MetalError> {
        let needed = allocation_capacity(element_count, dtype)?;
        if let Some(live) = self.live.borrow().upgrade()
            && live.length() >= needed
        {
            return Ok(live);
        }
        let fresh = Rc::new(allocate_buffer_for(
            owner,
            "plan",
            device,
            element_count,
            dtype,
        )?);
        *self.live.borrow_mut() = Rc::downgrade(&fresh);
        Ok(fresh)
    }
}

thread_local! {
    /// Slot 0 of every packed [`BufferArena`].
    pub(super) static ARENA_TRANSIENT: TransientPool = const { TransientPool::new() };
    /// The attention scratch that every sharing position of a plan reads and writes.
    pub(super) static ATTENTION_SCRATCH_TRANSIENT: TransientPool = const { TransientPool::new() };
}
