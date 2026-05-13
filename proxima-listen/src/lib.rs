//! Listener registry + serve protocol surface for proxima.
//!
//! # Tier
//!
//! Base (no_std + no_alloc): the [`admission`] FSM core + [`preface`]
//! classifier — no sockets, no futures, no allocation. `alloc` grows the
//! admission core's connection table. `std` is strictly additive: the
//! listener registry, `ServeContext`, reuseport socket binding, the
//! conflaguration-backed tuning config, `Offload`, and `serve_pipe_upgrades`.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
pub mod handle;
#[cfg(feature = "std")]
pub use handle::{Listener, ListenerHandle, ListenerSpec, ShutdownPolicy};

#[cfg(feature = "std")]
mod config;
#[cfg(feature = "std")]
pub use config::{ListenTuningConfig, ListenTuningLayerBuilder};

/// Build-time sizing constants generated from `proxima-listen.toml`. At
/// no_std+no_alloc these consts ARE the config; at std they seed
/// [`ListenTuningConfig`]'s runtime defaults — never duplicated.
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_listen_sized.rs"));
}

#[cfg(feature = "std")]
mod offload;
#[cfg(feature = "std")]
pub use offload::Offload;

#[cfg(feature = "std")]
mod serve_pipe;
#[cfg(feature = "std")]
pub use serve_pipe::serve_pipe_upgrades;

/// The listener registry, `ServeContext`/`ListenProtocol` serve surface,
/// and the fluent `ServeBuilder` — the std-tier reactor adapter that drives
/// the [`admission`] core's decisions. Folded straight out of this crate
/// root into its own file as part of the no_std tiering: same types, same
/// behavior, just std-gated.
#[cfg(feature = "std")]
mod registry;
#[cfg(feature = "std")]
pub use registry::{
    HandlerDispatch, ListenProtocol, ListenProtocolFluent, ListenRegistry, ServeBuilder,
    ServeContext, ThreadLocalListenProtocol, ThreadLocalListenRegistry, dispatch_handler, peer_ip,
};

/// Sans-IO HTTP connection-preface classifier, folded in from the former
/// `proxima-preface-codec` crate (single consumer: `proxima-listeners-http`,
/// which depends on this crate already). no_std + no alloc.
pub mod preface;

/// Generic byte-stream `ListenProtocol` adapters, folded in from the
/// former `proxima-listeners-stream` crate.
#[cfg(feature = "stream")]
pub mod stream;

/// Sans-IO listener admission core, folded in from the former
/// `proxima-listen-core` crate.
pub mod admission;
pub use admission::{
    Admission, ConnectionHandle, DispatchPolicy, DrainOutcome, ListenerCore, Phase,
    ReleaseOutcome, Route, ShedReason,
};
