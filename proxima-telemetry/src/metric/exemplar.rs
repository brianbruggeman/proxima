use core::sync::atomic::{AtomicU64, Ordering};

use crate::id::TraceId;

/// A best-effort exemplar for a duration histogram (C4): the trace id of the
/// slowest observation in the current window — "the p99 carries the trace that
/// was p99." Kept as a side cell so the size-pinned [`Histogram`] stays untouched.
///
/// Lock-free and deliberately racy: an exemplar is a representative pointer, not
/// an exact aggregate, so a torn `(max, trace)` pair across the brief CAS window
/// is acceptable (the OpenTelemetry exemplar contract). Three `AtomicU64` hold the
/// max duration and the 16-byte trace id split into two halves.
///
/// [`Histogram`]: crate::metric::histogram::Histogram
pub struct ExemplarCell {
    max_duration_ns: AtomicU64,
    trace_hi: AtomicU64,
    trace_lo: AtomicU64,
}

impl Default for ExemplarCell {
    fn default() -> Self {
        Self::new()
    }
}

impl ExemplarCell {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_duration_ns: AtomicU64::new(0),
            trace_hi: AtomicU64::new(0),
            trace_lo: AtomicU64::new(0),
        }
    }

    /// Offer an observation; it becomes the exemplar iff it is the slowest seen
    /// this window. The CAS on the max guards the swap; the trace store trails it
    /// (a reader between the two sees the new max with the prior trace — tolerable
    /// for an exemplar).
    pub fn observe(&self, duration_ns: u64, trace_id: TraceId) {
        let mut current = self.max_duration_ns.load(Ordering::Relaxed);
        while duration_ns > current {
            match self.max_duration_ns.compare_exchange_weak(
                current,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let (hi, lo) = split_trace(trace_id.to_bytes());
                    self.trace_hi.store(hi, Ordering::Relaxed);
                    self.trace_lo.store(lo, Ordering::Relaxed);
                    return;
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// The current exemplar (slowest duration + its trace), or `None` if nothing
    /// has landed. Does not reset — read alongside the histogram snapshot.
    #[must_use]
    pub fn peek(&self) -> Option<(u64, TraceId)> {
        self.read(self.max_duration_ns.load(Ordering::Relaxed))
    }

    /// Snapshot the exemplar and reset for the next window.
    #[must_use]
    pub fn snapshot_and_reset(&self) -> Option<(u64, TraceId)> {
        self.read(self.max_duration_ns.swap(0, Ordering::Relaxed))
    }

    fn read(&self, max: u64) -> Option<(u64, TraceId)> {
        if max == 0 {
            return None;
        }
        let hi = self.trace_hi.load(Ordering::Relaxed);
        let lo = self.trace_lo.load(Ordering::Relaxed);
        Some((max, TraceId::from_bytes(join_trace(hi, lo))))
    }
}

fn split_trace(bytes: [u8; 16]) -> (u64, u64) {
    let mut hi = [0u8; 8];
    let mut lo = [0u8; 8];
    hi.copy_from_slice(&bytes[..8]);
    lo.copy_from_slice(&bytes[8..]);
    (u64::from_le_bytes(hi), u64::from_le_bytes(lo))
}

fn join_trace(hi: u64, lo: u64) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&hi.to_le_bytes());
    bytes[8..].copy_from_slice(&lo.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::ExemplarCell;
    use crate::id::TraceId;

    fn trace(byte: u8) -> TraceId {
        TraceId::from_bytes([byte; 16])
    }

    // the slowest observation wins; a slower one displaces it, a faster one does not.
    #[test]
    fn slowest_observation_wins() {
        let cell = ExemplarCell::new();
        assert_eq!(cell.peek(), None);

        cell.observe(100, trace(0xa1));
        assert_eq!(cell.peek(), Some((100, trace(0xa1))));

        // a faster span does NOT displace the exemplar.
        cell.observe(50, trace(0xb2));
        assert_eq!(cell.peek(), Some((100, trace(0xa1))));

        // a slower span DOES.
        cell.observe(200, trace(0xc3));
        assert_eq!(cell.peek(), Some((200, trace(0xc3))));
    }

    // snapshot returns the exemplar and clears it for the next window.
    #[test]
    fn snapshot_resets() {
        let cell = ExemplarCell::new();
        cell.observe(75, trace(0xd4));
        assert_eq!(cell.snapshot_and_reset(), Some((75, trace(0xd4))));
        assert_eq!(cell.peek(), None, "exemplar clears after snapshot");
    }
}
