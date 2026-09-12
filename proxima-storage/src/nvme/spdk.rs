//! Callback ABI for driving the sans-IO NVMe engine from a local SPDK queue.
//!
//! Proxima intentionally does not link SPDK or take ownership of its reactor.
//! A small process-local C shim can pass an SPDK `struct spdk_nvme_qpair *` as
//! `context` and implement these callbacks with SPDK's qpair submission,
//! completion polling, and doorbell policy. The ring codec and async `Pipe`
//! remain Proxima-owned; this adapter is the boundary between the two runtimes.

use core::ffi::c_void;

use proxima_protocols::nvme::{command, completion};

use crate::nvme::backend::QueueBackend;

/// C-ABI operations supplied by the embedding SPDK shim.
#[derive(Clone, Copy)]
pub struct SpdkCallbacks {
    /// Opaque `spdk_nvme_qpair *` (or an adapter-owned wrapper).
    pub context: *mut c_void,
    /// Copy one encoded SQE into the SPDK submission path.
    pub write_submission: unsafe extern "C" fn(*mut c_void, u16, *const u8),
    /// Publish the submission tail, if the shim's queue requires it.
    pub ring_submit_doorbell: unsafe extern "C" fn(*mut c_void, u16),
    /// Copy one encoded CQE out of the SPDK completion path.
    pub read_completion: unsafe extern "C" fn(*mut c_void, u16, *mut u8),
    /// Publish the completion head, if the shim's queue requires it.
    pub ring_complete_doorbell: unsafe extern "C" fn(*mut c_void, u16),
}

/// A `QueueBackend` backed by callbacks into a process-local SPDK qpair.
///
/// The callback table is `Copy` and contains no Rust-owned allocation. The
/// caller must keep `context` and every callback target valid for this value's
/// lifetime, and must obey SPDK's qpair thread-affinity rules.
#[derive(Clone, Copy)]
pub struct SpdkQueueBackend {
    callbacks: SpdkCallbacks,
}

// SPDK qpair affinity is a caller contract, not something Rust can infer from
// the opaque pointer. Marking the adapter this way allows `QueuePair`'s
// `SendPipe` surface; the embedding shim must still keep one qpair on its
// owning SPDK thread.
unsafe impl Send for SpdkQueueBackend {}
unsafe impl Sync for SpdkQueueBackend {}

impl SpdkQueueBackend {
    /// Wrap an already-created SPDK qpair adapter.
    #[must_use]
    pub const fn new(callbacks: SpdkCallbacks) -> Self {
        Self { callbacks }
    }

    /// The callback table supplied at construction.
    #[must_use]
    pub const fn callbacks(&self) -> SpdkCallbacks {
        self.callbacks
    }
}

impl QueueBackend for SpdkQueueBackend {
    fn write_submission(&self, slot: u16, entry: &[u8; command::ENTRY_LEN]) {
        // SAFETY: the embedding shim owns the context and promises that the
        // callback copies exactly one NVMe SQE before returning.
        unsafe {
            (self.callbacks.write_submission)(self.callbacks.context, slot, entry.as_ptr());
        }
    }

    fn ring_submit_doorbell(&self, tail: u16) {
        // SAFETY: same lifetime/thread-affinity contract as `write_submission`.
        unsafe { (self.callbacks.ring_submit_doorbell)(self.callbacks.context, tail) }
    }

    fn read_completion(&self, slot: u16) -> [u8; completion::ENTRY_LEN] {
        let mut entry = [0u8; completion::ENTRY_LEN];
        // SAFETY: the embedding shim writes exactly one CQE into this valid
        // output buffer before returning.
        unsafe {
            (self.callbacks.read_completion)(self.callbacks.context, slot, entry.as_mut_ptr());
        }
        entry
    }

    fn ring_complete_doorbell(&self, head: u16) {
        // SAFETY: same lifetime/thread-affinity contract as `read_completion`.
        unsafe { (self.callbacks.ring_complete_doorbell)(self.callbacks.context, head) }
    }
}
