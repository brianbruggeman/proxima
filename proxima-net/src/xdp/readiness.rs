//! Read-readiness for the AF_XDP socket fd over the prime per-core reactor.
//!
//! The AF_XDP socket is a real fd: the kernel signals `POLLIN` on it whenever
//! the RX ring has entries. This is exactly the fd-generic wake-driver prime
//! already provides for any externally-owned fd — [`Readiness`]/[`ReadyState`]
//! live in [`prime::os::readiness`] and are re-exported here so existing xdp
//! call sites (`super::readiness::{Readiness, ReadyState}`) keep working
//! unchanged. See that module for the full behaviour (edge-triggered
//! `EPOLLET` re-drain contract, off-worker busy-poll fallback).

pub use prime::os::readiness::{Readiness, ReadyState};
