//! open-loop arrival pacing + a bounded in-flight gate.
//!
//! the load-tester invariant lives here: arrivals are scheduled against
//! absolute time, never against when the target happened to answer, so a slow
//! target produces a catch-up burst rather than a silently slipped rate. and
//! in-flight is bounded without ever throttling the schedule — overflow is a
//! timeout, not backpressure. both are the coordinated-omission-correct shapes
//! the engine must own rather than inherit.

pub mod inflight;
pub mod pacer;
pub mod scheduler;

pub use inflight::{InFlight, Permit};
pub use pacer::{GridPacer, IntervalPacer, Pacer};
pub use scheduler::{RateSpec, Scheduler, SchedulerBuilder};
