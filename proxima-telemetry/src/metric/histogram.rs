//! The histogram primitive moved to `proxima-core::histogram` (a no_std, no-alloc
//! `[AtomicU64; 32]` measurement structure). Re-exported so
//! `crate::metric::histogram::{Histogram, MAX_BUCKETS, ...}` keep resolving.
pub use proxima_core::histogram::*;
