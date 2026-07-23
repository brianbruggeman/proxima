//! Typed pipe handles for the memcached pipeline.
//!
//! The business handler pipe carries [`MemcachedRequest`] straight to
//! [`Reply`] — no `Request`/`Response` envelope cell (payload-no-cell: a
//! pipe is `P -> Q`, and `MemcachedRequest` is already self-describing —
//! nothing downstream reads a path, query, or metadata this crate would
//! otherwise have had to synthesize). `MemcachedPipeHandle` is an
//! instantiation of the generic erased form
//! `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>`. Mirrors
//! `proxima_redis::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use proxima_protocols::memcached::{MemcachedRequest, Reply};

/// Runtime-erased handle for memcached command-handler pipes.
pub type MemcachedPipeHandle = alloc_tier::PipeHandle<MemcachedRequest, Reply>;

/// Wrap any memcached-compatible pipe in a [`MemcachedPipeHandle`] — the
/// bridge between a business handler you write (`impl SendPipe<In =
/// MemcachedRequest, Out = Reply>`) and every seam that wants the
/// type-erased [`MemcachedPipeHandle`] ([`crate::MemcachedAnyProtocol::new`],
/// `proxima::ListenerProtocolExt::memcached`).
///
/// ```
/// use proxima_memcached::{MemcachedRequest, Reply, into_memcached_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct EchoStats;
/// impl SendPipe for EchoStats {
///     type In = MemcachedRequest;
///     type Out = Reply;
///     type Err = ProximaError;
///     async fn call(&self, _request: MemcachedRequest) -> Result<Reply, ProximaError> {
///         unreachable!("illustrative — no request is dispatched in this doctest")
///     }
/// }
///
/// let handle = into_memcached_handle(EchoStats);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_memcached_handle;
