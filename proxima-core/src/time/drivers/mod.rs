//! First-party [`Driver`](super::Driver) implementations. Each is
//! behind its own `time-driver-*` feature; the active driver is bound at
//! link time via `BOUND_DRIVER` (emitted by `proxima-core/build.rs`).
//!
//! `unbound` is always compiled in as a panics-on-use fallback so the
//! crate links under any feature combination; that lets downstream
//! crates verify their alloc-only / thumbv7m builds without having to
//! pick a driver they aren't going to exercise.
//!
//! External drivers don't live here — a user HAL defines its own
//! `pub static DRIVER: ...` in its crate and references it via the
//! `timer = "<their_crate>::DRIVER"` profile entry.

pub mod unbound;

#[cfg(feature = "time-driver-std-thread")]
pub mod std_thread;

#[cfg(feature = "time-driver-mock")]
pub mod mock;

#[cfg(feature = "time-driver-wasm")]
pub mod wasm;

// compiled when the cargo feature is on OR a profile's timer axis selects the
// prime-wheel (link-injected) driver — build.rs emits `proxima_external_driver`
// for the latter so a profile alone suffices.
#[cfg(any(feature = "time-driver-prime-wheel", proxima_external_driver))]
pub mod external;
