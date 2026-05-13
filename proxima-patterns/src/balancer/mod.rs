//! Selection strategies (failover, round-robin, weighted) over a set
//! of upstream Pipe refs. `UpstreamRef` carries the metric-instrumented
//! handle to a single upstream; `SelectionStrategy` picks which one to
//! dispatch a request through.
//!
//! Folded from the former `proxima-balancer` crate.
//!
//! Tier: tier marker. Selection wraps upstream pipes (std-bound — they
//! hit the network) and uses `std::time::Instant` for outlier-detection
//! metric windows. Under no_std + alloc the crate is a marker; under
//! std (default) the full surface.

#[cfg(feature = "std")]
pub mod selection;
#[cfg(feature = "std")]
pub mod upstream_ref;

#[cfg(feature = "std")]
pub use selection::{
    DispatchOutcome, DynSelection, Fallthrough, LeastConn, MissPolicy, MissReason, RoundRobin,
    Selection, SelectionHandle, ThreadLocalDynSelection, ThreadLocalSelection,
    ThreadLocalSelectionHandle, WeightedLeastConn, WeightedRoundRobin,
};
#[cfg(feature = "std")]
pub use upstream_ref::{
    CallTracker, OutlierPolicy, ThreadLocalUpstreamRef, UpstreamMetrics, UpstreamRef,
};
