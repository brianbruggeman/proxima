//! Sans-IO listener admission core — the accept-layer state machine.
//!
//! Given accept/release/drain *events*, it decides admit-or-shed (by
//! capacity and drain phase), routes an admitted connection to a core (by
//! dispatch policy), and drives the `Accepting -> Draining -> Closed`
//! lifecycle. No sockets, no futures, no spawn: the reactor adapter
//! (the rest of `proxima-listen`) owns `accept()` / `spawn()` and drives
//! this to decide *whether* and *where* a connection runs.
//!
//! This is the stream sibling of `proxima_protocols::quic::endpoint::EndpointDemux`
//! (which routes datagrams by connection id). That module's own docs say
//! "bounding the connection count against DoS is admission control and belongs
//! at the accept layer, NOT this routing table" — this module is that accept
//! layer.
//!
//! Folded from the former `proxima-listen-core` satellite crate (single
//! reverse-dependency: `proxima-listen` itself, plus `proxima-listeners-http`,
//! which already depends on `proxima-listen`) into `proxima-listen` as the
//! `admission` module.
//!
//! # Tier
//!
//! The `alloc` feature (on by default here, matching the former crate's
//! `default = ["std"]` -> `std = ["alloc"]`) routes through a growable
//! [`hashbrown::HashMap`] that scales with the live connection count; with
//! `alloc` disabled the bare `no_std + no_alloc` tier uses a fixed-cap
//! `heapless::FnvIndexMap` sized by [`sized::ADMISSION_TABLE_CAP`], so a
//! microcontroller's connection table is bounded at compile time. This
//! module's own no_std/no_alloc gates are unchanged from the former crate —
//! only the `alloc` feature now lives on `proxima-listen` (which itself
//! stays std, unlike the tier that this module can still compile under).

/// Build-time sizing constants generated from `proxima-listen-core.toml`.
/// These ARE the no_std+no_alloc tier's fixed-cap floor; the alloc tier
/// reads `PER_PEER_CAP_DEFAULT` as its default shed threshold and only
/// clamps its otherwise-unbounded table when a caller opts into
/// [`ListenerCore::with_capacity`]/[`ListenerCore::with_caps`].
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_listen_core_sized.rs"));
}

mod state;
pub use state::{
    Admission, ConnectionHandle, DispatchPolicy, DrainOutcome, ListenerCore, Phase, ReleaseOutcome,
    Route, ShedReason,
};

/// Request-level admission (the per-request twin of [`ListenerCore`]) —
/// std-tier only: every caller is an accept-loop reactor adapter already
/// gated on `std`.
#[cfg(feature = "std")]
mod request;
#[cfg(feature = "std")]
pub use request::{ConnAdmission, RequestAdmit};

/// Accept-edge DoS-blacklist — std-tier only, sibling of [`request`]: a
/// `DenySignature` (`crate::any::deny`) and the reject-hook composition in
/// `ListenerBuilder::serve` both need a `hashbrown`/`arc-swap`-backed table,
/// same as [`ConnAdmission`].
#[cfg(feature = "std")]
mod blacklist;
#[cfg(feature = "std")]
pub use blacklist::{BlacklistConfig, BlacklistLayerBuilder, BlacklistTable, Strike};
