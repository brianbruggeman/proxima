//! Alerting + guidance for proxima — typed event protocol + std facade Pipes.
//!
//! Folded from the former `proxima-notify` crate (renamed `alert` to resolve
//! the collision with the `proxima_primitives::sync::Notify` primitive — the primitive
//! keeps the name, the pattern becomes `alert`).
//!
//! # Layers
//!
//! - [`event`] (tier-3-capable, behind `proto` feature) — AlertEvent +
//!   GuidanceQuestion + GuidanceAnswer + postcard codec. `#![no_std]`-clean
//!   when std is off, alloc-free with heapless containers; const-generic
//!   caps from `proxima-notify.toml` per principle 12.
//! - [`scheduled_trigger`] (tier-2, behind `scheduled-trigger` feature) —
//!   interval-driven producer Pipe.
//! - [`stdout_alert`] (tier-2, behind `stdout-alert` feature) — terminal
//!   sink that prints `AlertEvent` to stdout.
//! - [`stdio_guidance`] (tier-2, behind `stdio-guidance` feature) —
//!   duplex sync stdin/stdout for guidance round-trip.
//!
//! # Per-protocol integrations (Telegram, PagerDuty, ntfy, …) are CONFIG
//!
//! Integrations are TOML compositions of existing primitives:
//! `HttpUpstream`, `Transform` (with body-template DSL), `Validate`
//! (with proxima-config's schema module contracts), `Retry`, and `Isolate`. No
//! per-integration Rust required.

#![deny(missing_docs)]

#[cfg(all(feature = "alert", feature = "proto"))]
pub mod event;

#[cfg(all(feature = "alert", feature = "std"))]
mod std_facade {
    /// Method-byte constants for typed-payload dispatch (telemetry
    /// convention; see `proxima-telemetry/src/pipes.rs:322`).
    pub mod methods {
        use bytes::Bytes;
        use proxima_primitives::pipe::method::Method;

        /// Carries an `AlertEvent`; consumed by sink Pipes.
        pub const ALERT: &[u8] = b"ALERT";

        /// Carries an `AlertEvent` built from a scheduled-trigger fire.
        pub const SCHEDULED_TICK: &[u8] = b"SCHEDULED_TICK";

        /// Carries a `GuidanceQuestion`; response carries a `GuidanceAnswer`.
        pub const GUIDANCE_QUESTION: &[u8] = b"GUIDANCE_QUESTION";

        /// `Method`-wrapped `ALERT` (zero-copy static).
        #[must_use]
        pub fn alert_method() -> Method {
            Method::from_wire(Bytes::from_static(ALERT))
        }

        /// `Method`-wrapped `SCHEDULED_TICK` (zero-copy static).
        #[must_use]
        pub fn scheduled_tick_method() -> Method {
            Method::from_wire(Bytes::from_static(SCHEDULED_TICK))
        }

        /// `Method`-wrapped `GUIDANCE_QUESTION` (zero-copy static).
        #[must_use]
        pub fn guidance_question_method() -> Method {
            Method::from_wire(Bytes::from_static(GUIDANCE_QUESTION))
        }
    }
}

#[cfg(all(feature = "alert", feature = "std"))]
pub use std_facade::methods;

#[cfg(all(feature = "alert", feature = "std"))]
pub mod pipes;

#[cfg(all(feature = "alert", feature = "scheduled-trigger"))]
pub mod scheduled_trigger;

#[cfg(all(feature = "alert", feature = "stdout-alert"))]
pub mod stdout_alert;

#[cfg(all(feature = "alert", feature = "stdio-guidance"))]
pub mod stdio_guidance;
