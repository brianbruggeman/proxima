mod bounded;
mod mpsc;

#[cfg(feature = "alloc")]
pub use bounded::HeapBoundedQueue;
pub use bounded::{BoundedQueue, EnqueueOutcome, FailMode, RingStorage, StaticBoundedQueue};
pub use mpsc::StaticRing;
#[cfg(feature = "alloc")]
pub use mpsc::{Drainer, Ring};

/// The only failure constructing a [`Ring`]: a zero capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("ring capacity must be non-zero")]
pub struct CapacityError;
