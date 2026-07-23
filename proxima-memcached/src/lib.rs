//! proxima's own memcached client facade.
//!
//! The sans-IO memcached text-protocol codec ([`Command`], [`ParseError`],
//! [`parse_command`], the [`Connection`] FSM, the [`Reply`] model, and the
//! memcached-over-`Pipe` contract [`MemcachedRequest`]) lives in
//! [`proxima_protocols::memcached`] — see its docs for the wire layer.
//! This crate is the std client built on top: the async
//! [`client::MemcachedClientUpstream`] Pipe and the blocking
//! [`client::MemcachedClient`] driver, both driving the sans-IO
//! [`client::ClientSession`] over a pluggable transport (prime, tokio,
//! TLS-wrapped) — the same split `proxima-redis` uses between
//! `proxima-protocols::redis` and its own client half.
//!
//! The `listen` feature (below) adds the server side: [`connection`]'s
//! sans-IO-over-any-`futures::io`-stream driver, [`pipe::MemcachedConnectionPipe`]
//! (the connection layer as a real `Pipe`), and
//! [`any_protocol::MemcachedAnyProtocol`] — the `AnyProtocol` candidate
//! that mounts memcached into the open universal listener
//! (`Listener::builder().accept("memcached")`). There is no standalone
//! `MemcachedListenProtocol` bind+accept loop: memcached's listen-side
//! surface has always been an `AnyProtocol` candidate, driven by
//! `proxima_http::any_listener::AnyListenProtocol`'s ONE bind+accept loop
//! (real `ListenerCore`/`ConnAdmission` admission, graceful drain) —
//! mirroring `proxima-redis`'s own `listen` feature shape.

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod connection;
#[cfg(feature = "listen")]
pub mod error;
#[cfg(feature = "listen")]
pub mod pipe;
#[cfg(feature = "listen")]
pub mod pipes;

pub use proxima_protocols::memcached::{
    Command, ParseError, Reply, ReplyHint, StoreMode, StoredValue, encode_reply, parse_command,
    parse_reply,
};
pub use proxima_protocols::memcached::pipe_contract::{MemcachedRequest, encode_request, verb};

#[cfg(feature = "client")]
pub use client::{
    ClientError, ClientSession, MemcachedClient, MemcachedClientConfig, MemcachedClientUpstream,
    MemcachedConfigError, Step,
};

// the server-side surface a memcached command handler builds against —
// re-exported so an engine author imports everything from proxima-memcached
// and never reaches past it into proxima-primitives/proxima-protocols
// internals (teaching surface, workspace principle 2), mirroring
// proxima-redis's own top-level re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::MemcachedAnyProtocol;
#[cfg(feature = "listen")]
pub use config::MemcachedServerConfig;
#[cfg(feature = "listen")]
pub use connection::serve_connection;
#[cfg(feature = "listen")]
pub use error::MemcachedServeError;
#[cfg(feature = "listen")]
pub use pipe::MemcachedConnectionPipe;
#[cfg(feature = "listen")]
pub use pipes::{MemcachedPipeHandle, MemcachedPipeReply, MemcachedPipeRequest, into_memcached_handle};
