//! Typed pipe handle for the redis pipeline.
//!
//! The business handler pipe carries [`RedisRequest`] straight to
//! [`RespValue`] — no `Request`/`Response` envelope cell (payload-no-cell: a
//! pipe is `P -> Q`, and `RedisRequest`/`RespValue` are already
//! self-describing). `RedisPipeHandle` is an instantiation of the generic
//! erased form `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>`.
//! Mirrors `proxima_kafka::pipes` / `proxima_pgwire::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use proxima_protocols::redis::{RedisRequest, RespValue};

/// Runtime-erased handle for redis command-handler pipes.
pub type RedisPipeHandle = alloc_tier::PipeHandle<RedisRequest, RespValue>;

/// Wrap any redis-compatible pipe in a [`RedisPipeHandle`].
pub use alloc_tier::into_handle as into_redis_handle;
