//! rekt as a library, so benches and future frontends can reach the engine
//! pieces. the binary in `main.rs` is a thin CLI shell over these.
//!
//! `scheduler` gates the prime-backed engine and everything that pulls proxima
//! proper. It used to also gate a second, mock engine (`driver` + `fsm` +
//! `MockTarget`) so a default build could run with "no proxima" — a firewall
//! that had already stopped holding: `proxima-telemetry` is an unconditional
//! dependency, so the default tree pulls it, dashmap and hdrhistogram included.
//! The mock existed to serve a property rekt no longer had.

pub mod error;
pub mod report;
pub mod scenario;

// the prime-backed engine.
#[cfg(feature = "scheduler")]
pub mod engine;

// multiplexed HTTP/2 load — the h2 sibling of the engine's h1 drive.
#[cfg(feature = "scheduler")]
pub mod h2load;

// HTTP/3 load over proxima's native QUIC.
#[cfg(feature = "scheduler")]
pub mod h3load;
