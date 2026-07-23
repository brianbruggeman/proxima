//! Typed pipe handles for the memcached pipeline.
//!
//! `MemcachedPipeRequest = Request<MemcachedRequest>` / `MemcachedPipeReply
//! = Response<Reply>`: the business handler pipe is fully typed — no
//! downcast, no type erasure. `MemcachedPipeHandle` is an instantiation of
//! the generic erased form
//! `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>`. Mirrors
//! `proxima_redis::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use proxima_protocols::memcached::{MemcachedRequest, Reply};

/// Typed request carrying a [`MemcachedRequest`] as payload — the whole
/// parsed command, binary-safe.
pub type MemcachedPipeRequest = proxima_primitives::pipe::request::Request<MemcachedRequest>;

/// Typed response carrying a [`Reply`] as payload.
pub type MemcachedPipeReply = proxima_primitives::pipe::request::Response<Reply>;

/// Runtime-erased handle for memcached command-handler pipes.
pub type MemcachedPipeHandle = alloc_tier::PipeHandle<MemcachedPipeRequest, MemcachedPipeReply>;

/// Wrap any memcached-compatible pipe in a [`MemcachedPipeHandle`] — the
/// bridge between a business handler you write (`impl SendPipe<In =
/// MemcachedPipeRequest, Out = MemcachedPipeReply>`) and every seam that
/// wants the type-erased [`MemcachedPipeHandle`]
/// ([`crate::MemcachedAnyProtocol::new`], `proxima::ListenerProtocolExt::memcached`).
///
/// ```
/// use proxima_memcached::{MemcachedPipeRequest, MemcachedPipeReply, into_memcached_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct EchoStats;
/// impl SendPipe for EchoStats {
///     type In = MemcachedPipeRequest;
///     type Out = MemcachedPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, _request: MemcachedPipeRequest) -> Result<MemcachedPipeReply, ProximaError> {
///         unreachable!("illustrative — no request is dispatched in this doctest")
///     }
/// }
///
/// let handle = into_memcached_handle(EchoStats);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_memcached_handle;
