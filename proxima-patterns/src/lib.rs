//! Pattern-set primitives for proxima: pure wiring of pipe primitives
//! (source/sink/transform/filter/gate/fan-in/out/observe/series) plus a
//! per-pattern domain data type. One feature per pattern so a consumer
//! pulls only what it needs.
//!
//! Folded from five formerly-standalone crates:
//!
//! - [`alert`] (was `proxima-notify`, renamed to resolve the collision with
//!   the `proxima_primitives::sync::Notify` primitive) — typed event protocol + std
//!   facade Pipes for alerting/guidance.
//! - [`balancer`] (was `proxima-balancer`) — load-balancing selection
//!   strategies over a set of upstream Pipe refs.
//! - [`middleware`] (was `proxima-middleware`) — Pipe-graph HTTP-leaf
//!   middleware: Auth, ClientAuth, ContextInject, WriteBack.
//! - [`control_plane`] (was `proxima-control-plane`) — control-plane trait
//!   surface: introspect / manage running Pipes at runtime.
//! - [`kv`] (was `proxima-kv`) — cache-entry primitives + write-back rules.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alert")]
pub mod alert;

#[cfg(feature = "balancer")]
pub mod balancer;

#[cfg(feature = "control_plane")]
pub mod control_plane;

#[cfg(feature = "kv")]
pub mod kv;

#[cfg(feature = "middleware")]
pub mod middleware;
