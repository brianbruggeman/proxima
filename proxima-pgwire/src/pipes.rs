//! Typed pipe handles for the pgwire pipeline.
//!
//! The business handler pipe carries [`QueryRequest`] straight to
//! [`PgReply`] — no `Request`/`Response` envelope cell (payload-no-cell: a
//! pipe is `P -> Q`, and `QueryRequest`/`PgReply` are already
//! self-describing). `PgPipeHandle` is an instantiation of the generic
//! erased form `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>`.
//! Mirrors `proxima_redis::pipes` / `proxima_kafka::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use crate::pipe_contract::{PgReply, QueryRequest};

/// Runtime-erased handle for pgwire SQL engine pipes.
pub type PgPipeHandle = alloc_tier::PipeHandle<QueryRequest, PgReply>;

/// Wrap any pgwire-compatible pipe in a [`PgPipeHandle`].
pub use alloc_tier::into_handle as into_pg_handle;
