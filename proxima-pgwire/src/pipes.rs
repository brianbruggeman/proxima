//! Typed pipe handles for the pgwire pipeline.
//!
//! `PgRequest = Request<QueryRequest>` / `PgResponse = Response<PgReply>`:
//! the SQL engine pipe is fully typed — no downcast, no type erasure.
//! `PgPipeHandle` is an instantiation of the generic erased form
//! `proxima_primitives::pipe::PipeHandle<In, Out>`.

use proxima_primitives::pipe::alloc_tier;

use crate::pipe_contract::{PgReply, QueryRequest};

/// Typed request carrying a [`QueryRequest`] as payload.
pub type PgRequest = proxima_primitives::pipe::request::Request<QueryRequest>;

/// Typed response carrying a [`PgReply`] as payload.
pub type PgResponse = proxima_primitives::pipe::request::Response<PgReply>;

/// Runtime-erased handle for pgwire SQL engine pipes.
pub type PgPipeHandle = alloc_tier::PipeHandle<PgRequest, PgResponse>;

/// Wrap any pgwire-compatible pipe in a [`PgPipeHandle`].
pub use alloc_tier::into_handle as into_pg_handle;
