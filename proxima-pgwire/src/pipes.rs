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

/// Runtime-erased handle for pgwire SQL engine pipes. Both mounts
/// ([`crate::listen::PgWireListenProtocol`], [`crate::any_protocol::PgWireAnyProtocol`])
/// and the connection pipe take one of these; [`into_pg_handle`] is how an
/// engine becomes one.
///
/// ```
/// use proxima_pgwire::codec::Oid;
/// use proxima_pgwire::{
///     ColumnDesc, PgPipeHandle, PgReply, ProximaError, QueryReply, QueryRequest, SendPipe,
///     SqlValue, Verb, into_pg_handle,
/// };
///
/// struct OneRowEngine;
///
/// impl SendPipe for OneRowEngine {
///     type In = QueryRequest;
///     type Out = PgReply;
///     type Err = ProximaError;
///
///     async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
///         Ok(match request.verb {
///             Verb::Query => PgReply::Query(QueryReply::rows(
///                 vec![ColumnDesc::new("n", Oid(23))],
///                 vec![vec![SqlValue::Int(1)]],
///             )),
///             _ => PgReply::Query(QueryReply::tag("OK")),
///         })
///     }
/// }
///
/// let engine: PgPipeHandle = into_pg_handle(OneRowEngine);
/// ```
pub type PgPipeHandle = alloc_tier::PipeHandle<QueryRequest, PgReply>;

/// Wrap any pgwire-compatible pipe in a [`PgPipeHandle`].
pub use alloc_tier::into_handle as into_pg_handle;
