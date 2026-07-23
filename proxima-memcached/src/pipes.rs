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

/// Wrap any memcached-compatible pipe in a [`MemcachedPipeHandle`].
pub use alloc_tier::into_handle as into_memcached_handle;
