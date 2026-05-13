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

#[cfg(feature = "client")]
pub mod client;

pub use proxima_protocols::redis::{
    Frame, ParseError, RespValue, encode, encode_command, parse, pipe_contract,
};
pub use proxima_protocols::redis::pipe_contract::{RedisRequest, verb};

#[cfg(feature = "client")]
pub use client::{
    ClientError, ClientSession, PushStep, RedisClient, RedisClientConfig, RedisClientUpstream,
    RedisConfigError, RespProtocol, Step,
};
