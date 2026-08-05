//! Typed pipe handles for the alert pipeline.
//!
//! Alert and guidance handles are instantiations of the generic erased
//! form `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>` (see
//! `proxima-primitives/src/pipe/alloc_tier.rs`) — the scheduler and guidance
//! server hold typed inner pipes without knowing the concrete type at
//! compile time, with zero hand-rolled erasure machinery in this crate.
//!
//! Wrap a pipe with `proxima_primitives::pipe::alloc_tier::into_handle`; the
//! target alias picks the instantiation. There is deliberately no
//! `into_alert_handle` / `into_guidance_handle` pair — both were `pub use` of
//! that one function under two names, so neither constrained what it accepted
//! and `into_alert_handle` happily produced a `GuidancePipeHandle`.

use bytes::Bytes;
use proxima_primitives::pipe::alloc_tier;
use proxima_primitives::pipe::request::Response;

use crate::alert::event::{AlertEvent, GuidanceAnswer, GuidanceQuestion};

/// Typed request carrying an [`AlertEvent`] as payload.
pub type AlertRequest = proxima_primitives::pipe::request::Request<AlertEvent>;

/// Typed request carrying a [`GuidanceQuestion`] as payload.
pub type GuidanceRequest = proxima_primitives::pipe::request::Request<GuidanceQuestion>;

/// Typed response carrying a [`GuidanceAnswer`] as payload.
pub type GuidanceResponse = proxima_primitives::pipe::request::Response<GuidanceAnswer>;

/// Runtime-erased handle for alert-typed pipes.
pub type AlertPipeHandle = alloc_tier::PipeHandle<AlertRequest, Response<Bytes>>;

/// Runtime-erased handle for guidance-typed pipes.
pub type GuidancePipeHandle = alloc_tier::PipeHandle<GuidanceRequest, GuidanceResponse>;
