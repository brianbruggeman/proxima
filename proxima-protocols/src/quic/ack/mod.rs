//! ACK generation + scheduling per [RFC 9000 §13.2].
//!
//! [`scheduler::AckScheduler`] tracks the received-PN ranges + the
//! "should I emit an ACK now?" decision for ONE epoch. The connection
//! state machine carries one scheduler per epoch (Initial / Handshake /
//! Application).
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The scheduler composes
//! [`crate::quic::range_set::ArrayRangeSet`] (no-alloc) and pure POD timestamps.
//!
//! Per the [C13 paper proof] every transition the scheduler exposes
//! (`record_received` / `should_emit` / `on_emitted` / `next_deadline`
//! / `has_pending`) is named-and-mapped to a paragraph in the design
//! doc so the implementation drift is detectable.
//!
//! [RFC 9000 §13.2]: https://www.rfc-editor.org/rfc/rfc9000#section-13.2
//! [C13 paper proof]: ../../docs/proxima-quic/c13-ack-scheduler-design.md

pub mod scheduler;

pub use scheduler::AckScheduler;
