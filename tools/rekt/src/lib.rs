//! rekt as a library, so benches and future frontends can reach the engine
//! pieces. the binary in `main.rs` is a thin CLI shell over these.

pub mod error;
pub mod outcome;
pub mod report;
pub mod scenario;

#[cfg(feature = "scheduler")]
pub mod sched;

// the prime-backed engine and the mock engine are mutually exclusive: the
// `scheduler` flag swaps the real driver in for the mock one.
#[cfg(feature = "scheduler")]
pub mod engine;

// the throughput run as a first-class config (conflaguration) + fluent builder.
#[cfg(feature = "scheduler")]
pub mod plan;

// multiplexed HTTP/2 load — the h2 sibling of the engine's h1 drive.
#[cfg(feature = "scheduler")]
pub mod h2load;

// HTTP/3 load over proxima's native QUIC.
#[cfg(feature = "scheduler")]
pub mod h3load;

#[cfg(not(feature = "scheduler"))]
pub mod driver;
#[cfg(not(feature = "scheduler"))]
pub mod fsm;
