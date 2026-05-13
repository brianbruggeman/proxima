mod bounded;
mod lane;
mod mpsc;

#[cfg(feature = "alloc")]
pub use bounded::HeapBoundedQueue;
pub use bounded::{BoundedQueue, EnqueueOutcome, FailMode, RingStorage, StaticBoundedQueue};
pub use lane::LaneHandle;
pub use mpsc::StaticRing;
#[cfg(feature = "alloc")]
pub use mpsc::{Drainer, Ring};

/// The only failure constructing a [`Ring`](mpsc::Ring): a zero capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityError;

impl core::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ring capacity must be non-zero")
    }
}

impl core::error::Error for CapacityError {}
