//! proxima's own Redis/Valkey client facade.
//!
//! The sans-IO RESP2/RESP3 codec ([`Frame`], [`ParseError`], [`RespValue`],
//! [`parse`], [`encode`], [`encode_command`]) and the RESP-over-`Pipe`
//! contract ([`pipe_contract`]) live in [`proxima_protocols::redis`] — see
//! its docs for the wire layer. This crate is the std client built on top:
//! the async [`client::RedisClientUpstream`] Pipe and the blocking
//! [`client::RedisClient`] driver, both driving the sans-IO
//! [`client::ClientSession`] over a pluggable transport (prime, tokio,
//! TLS-wrapped) — the same split pgwire uses between `proxima-pgwire-codec`
//! and `proxima-pgwire`.
//!
//! The `listen` feature (below) adds the server side: [`connection`]'s
//! sans-IO-over-any-`futures::io`-stream driver, [`pipe::RedisConnectionPipe`]
//! (the connection layer as a real `Pipe`), and
//! [`any_protocol::RedisAnyProtocol`] — the `AnyProtocol` candidate that
//! mounts redis into the open universal listener
//! (`Listener::builder().accept("redis")`). There is no standalone
//! `RedisListenProtocol` bind+accept loop: redis's listen-side surface has
//! always been an `AnyProtocol` candidate, driven by
//! `proxima_http::any_listener::AnyListenProtocol`'s ONE bind+accept loop
//! (real `ListenerCore`/`ConnAdmission` admission, graceful drain) —
//! mirroring `proxima-pgwire`'s own `listen` feature shape.

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod broker;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod connection;
#[cfg(feature = "listen")]
pub mod error;
#[cfg(feature = "listen")]
pub mod glob;
#[cfg(feature = "listen")]
pub mod pipe;
#[cfg(feature = "listen")]
pub mod pipes;
#[cfg(feature = "listen")]
pub mod wait_sources;

pub use proxima_protocols::redis::{
    Frame, ParseError, RespValue, encode, encode_command, parse, pipe_contract,
};
pub use proxima_protocols::redis::pipe_contract::{RedisRequest, verb};

#[cfg(feature = "client")]
pub use client::{
    Active, ClientError, ClientSession, PushStep, RedisClient, RedisClientConfig,
    RedisClientUpstream, RedisConfigError, RespProtocol, Step, Subscribed,
};

// the server-side surface a redis command handler builds against —
// re-exported so an engine author imports everything from proxima-redis
// and never reaches past it into proxima-primitives/proxima-protocols
// internals (teaching surface, workspace principle 2), mirroring
// proxima-pgwire's own top-level re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::RedisAnyProtocol;
#[cfg(feature = "listen")]
pub use broker::{PushSink, RedisBroker};
#[cfg(feature = "listen")]
pub use config::RedisServerConfig;
#[cfg(feature = "listen")]
pub use connection::serve_connection;
#[cfg(feature = "listen")]
pub use error::RedisServeError;
#[cfg(feature = "listen")]
pub use glob::GlobSet;
#[cfg(feature = "listen")]
pub use pipe::RedisConnectionPipe;
#[cfg(feature = "listen")]
pub use pipes::{RedisPipeHandle, into_redis_handle};
