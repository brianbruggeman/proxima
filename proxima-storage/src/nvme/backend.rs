use proxima_protocols::nvme::{command, completion};

/// The I/O seam between the sans-IO codec and real queue memory. Everything
/// device-specific lives behind these four methods; the engine above is pure
/// codec + ring arithmetic and never names a syscall, an `mmap`, or a register.
///
/// All methods take `&self` — the backend owns its queue memory with interior
/// mutability (a hardware backend wraps mapped pointers + `write_volatile`
/// doorbells; the loopback test backend wraps a `Mutex`). Completions are
/// returned by value (a 16-byte copy) so the engine never holds a borrow into
/// backend memory across the reap.
pub trait QueueBackend {
    /// Write a 64-byte submission entry into submission-ring slot `slot`.
    fn write_submission(&self, slot: u16, entry: &[u8; command::ENTRY_LEN]);

    /// Publish the new submission-queue tail to the SQ doorbell — the controller
    /// may now consume up to (but not including) `tail`.
    fn ring_submit_doorbell(&self, tail: u16);

    /// Read the 16-byte completion entry currently in completion-ring slot
    /// `slot`. Returns owned bytes; the caller inspects the phase tag to decide
    /// whether it is fresh.
    fn read_completion(&self, slot: u16) -> [u8; completion::ENTRY_LEN];

    /// Publish the new completion-queue head to the CQ doorbell — the host has
    /// consumed completions up to (but not including) `head`.
    fn ring_complete_doorbell(&self, head: u16);
}
