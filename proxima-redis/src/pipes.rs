//! Typed pipe handles for the redis pipeline.
//!
//! `RedisPipeRequest = Request<RedisRequest>` / `RedisPipeReply =
//! Response<RespValue>`: the business handler pipe is fully typed — no
//! downcast, no type erasure. `RedisPipeHandle` is an instantiation of the
//! generic erased form `proxima_primitives::pipe::alloc_tier::PipeHandle<In,
//! Out>`. Mirrors `proxima_pgwire::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use proxima_protocols::redis::{RedisRequest, RespValue};

/// Typed request carrying a [`RedisRequest`] as payload — the command's
/// argument list (everything after the verb), binary-safe.
pub type RedisPipeRequest = proxima_primitives::pipe::request::Request<RedisRequest>;

/// Typed response carrying a [`RespValue`] as payload.
pub type RedisPipeReply = proxima_primitives::pipe::request::Response<RespValue>;

/// Runtime-erased handle for redis command-handler pipes.
pub type RedisPipeHandle = alloc_tier::PipeHandle<RedisPipeRequest, RedisPipeReply>;

/// Wrap any redis-compatible pipe in a [`RedisPipeHandle`].
pub use alloc_tier::into_handle as into_redis_handle;
