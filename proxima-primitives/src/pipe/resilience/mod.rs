//! Resilience policy layer — pure sans-IO state machines for backoff, deadlines,
//! circuit breaking, and retry control, plus the `Retry` executor that drives
//! them over an injected [`Clock`](crate::pipe::capabilities::Clock). Core tier
//! (no_std, no-alloc) throughout — `Retry` holds its futures inline, nothing
//! boxed; the alloc-only `Fallback` Pipe combinator is gated behind `alloc`.

pub mod backoff;
pub mod circuit_breaker;
pub mod deadline;
pub mod retry;
pub mod retry_exec;

#[cfg(feature = "alloc")]
pub mod fallback;

pub use backoff::{Backoff, Jitter};
pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use deadline::Deadline;
pub use retry::{RetryAction, RetryController};
pub use retry_exec::Retry;

#[cfg(feature = "alloc")]
pub use fallback::Fallback;
