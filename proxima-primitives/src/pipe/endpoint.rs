//! Endpoint metadata value types destined for `proxima-pipe`.
//!
//! Lifted from `stream.rs` during Phase 1.5 of the decomposition (see
//! `docs/decomposition/discipline.md`). `BindAddr` and `PeerInfo` are
//! referenced in `StreamListener::local_addr()` and
//! `StreamConnection::peer()` trait signatures AND in `Request` /
//! `Response`'s `RequestContext`. Keeping them in `stream.rs` would
//! force `proxima-pipe` to depend on `proxima-stream` (cycle); putting
//! them in `proxima-net` would force `proxima-stream` to depend on
//! `proxima-net` (reverses planned DAG). Resolution: they live in
//! `proxima-pipe` as small std-only enums. Stream traits and pipe
//! Request both pull `proxima-pipe` for them.
//!
//! On Phase 2 (proxima-pipe extraction) this file's contents move to
//! the new `proxima-pipe` crate.

#![cfg(feature = "alloc")]

use alloc::string::String;

#[cfg(feature = "std")]
use std::net::SocketAddr;
#[cfg(feature = "std")]
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum BindAddr {
    #[cfg(feature = "std")]
    Tcp(SocketAddr),
    #[cfg(feature = "std")]
    Unix(PathBuf),
    /// Backends whose bind concept doesn't fit the common shapes
    /// (e.g. DPDK device + queue), or the fallback for no_std builds.
    Other(String),
}

#[derive(Debug, Clone)]
pub enum PeerInfo {
    #[cfg(feature = "std")]
    Tcp(SocketAddr),
    #[cfg(feature = "std")]
    Unix(Option<PathBuf>),
    /// `None` is valid for multiplexed transports (e.g. one QUIC
    /// connection carrying many streams), or the fallback for no_std builds.
    Other(String),
}
