//! proxima's own memcached facade — both halves of the connection.
//!
//! The sans-IO memcached text-protocol codec ([`Command`], [`ParseError`],
//! [`parse_command`], the `Connection` FSM, the [`Reply`] model, and the
//! memcached-over-`Pipe` contract `MemcachedRequest`) lives in
//! [`proxima_protocols::memcached`] — see its docs for the wire layer.
//! Everything here is built on top of it, so the two crates split the way
//! `proxima_protocols::redis` and `proxima-redis` do.
//!
//! Feature `client`: the async `client::MemcachedClientUpstream` Pipe and
//! the blocking `client::MemcachedClient` driver, both driving the sans-IO
//! `client::ClientSession` over a pluggable transport (prime, tokio,
//! TLS-wrapped).
//!
//! Feature `listen`: the server side — `framed_app::MemcachedFramedApp`
//! (the business-handler pipe wired as `proxima_listen::any::FramedAny`'s
//! `App`) and `any_protocol::MemcachedAnyProtocol`, the `AnyProtocol`
//! candidate that mounts memcached into the open universal listener
//! (`Listener::builder().accept("memcached")`) by building a `FramedAny`
//! internally. There is no standalone `MemcachedListenProtocol`
//! bind+accept loop: memcached's listen-side surface has always been an
//! `AnyProtocol` candidate, driven by
//! `proxima_http::any_listener::AnyListenProtocol`'s ONE bind+accept loop
//! (real `ListenerCore`/`ConnAdmission` admission, graceful drain).
//!
//! `FramedAny` is the ONE generic stateless `AnyProtocol` driver every
//! stateless request/reply wire shares; see `framed_app`'s module doc for
//! how memcached's `noreply`/`quit`/protocol-violation semantics map onto
//! its `AsFrame` seam. Its `Shed` closure carries the shed frame itself
//! (`Fn(ShedReason, &App::In) -> App::Out`), so `framed_app::shed_reply`
//! can keep admission per-command: a `noreply`-flagged command that gets
//! shed stays silent, and `quit` closes rather than answering a
//! `SERVER_ERROR`.
//!
//! `--no-default-features` drops `std` and leaves only the codec
//! re-exports, so the bare tier links on a bare-metal target
//! (`thumbv7m-none-eabi`); both `client` and `listen` imply `std`. The
//! feature-gated names above are deliberately unlinked — an intra-doc link
//! to them would break `cargo doc` on the bare tier, where they do not
//! exist.
//!
//! ## Scope
//!
//! **Text protocol only.** The binary protocol (opcode-framed, used by some
//! high-throughput clients) is not implemented — [`parse_command`] only
//! recognizes the ASCII command set (`get`/`gets`/`set`/`add`/`replace`/
//! `append`/`prepend`/`cas`/`delete`/`incr`/`decr`/`touch`/`flush_all`/
//! `stats`/`version`/`quit`). A client hard-coded to the binary protocol
//! (rare — most drivers default to text, or negotiate) will not interoperate
//! with this facade.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod framed_app;
#[cfg(feature = "listen")]
pub mod pipes;

pub use proxima_protocols::memcached::{
    Command, ParseError, Reply, ReplyHint, StoreMode, StoredValue, encode_reply, parse_command,
    parse_reply,
};
// `MemcachedRequest`'s fields are `Bytes`/`MultigetKeys` — needs
// proxima-protocols' `memcached-codec-trait` tier, which `client` and
// `listen` both now pull in (see this crate's Cargo.toml); the bare,
// zero-feature build never references it.
#[cfg(any(feature = "client", feature = "listen"))]
pub use proxima_protocols::memcached::pipe_contract::{
    MemcachedRequest, encode_request, iter_keys, verb,
};

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
pub use framed_app::{MemcachedFramedApp, MemcachedOutcome};
#[cfg(feature = "listen")]
pub use pipes::{MemcachedPipeHandle, into_memcached_handle};
