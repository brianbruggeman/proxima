//! Network primitives for proxima: UDP packet listeners + address helpers,
//! plus every platform network backend as a feature-gated module.
//!
//! Stream traits and BindAddr/PeerInfo live in `proxima-primitives::stream`.
//!
//! `packet` is std-gated: `PacketListener` returns `std::io::Result`, which
//! is genuinely std-bound. `stack` is pure sans-IO logic with no allocation,
//! so it is the crate's bare `no_std` no-alloc floor. `tcp_listener` and
//! `tcp_stack` are also sans-IO but their connection tables are genuinely
//! alloc-shaped (`BTreeMap`/`VecDeque`/`Vec`), so they are gated behind the
//! `alloc` feature.
//!
//! # Backends
//!
//! Each backend below is a former standalone crate (`proxima-net-prime`,
//! `proxima-net-tokio`, `proxima-net-wasm`, `proxima-net-dpdk`,
//! `proxima-net-xdp`), folded in as a feature-gated module so the base crate
//! stays no_std + alloc while a platform build opts into exactly the
//! backend(s) it links. All backend features are default-off and require
//! `std`; `dpdk`/`xdp` additionally require their host toolchain and stay
//! off a plain `cargo build --workspace`.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod packet;
pub mod stack;
#[cfg(feature = "alloc")]
pub mod tcp_listener;
#[cfg(feature = "alloc")]
pub mod tcp_stack;

#[cfg(feature = "dpdk")]
pub mod dpdk;
#[cfg(all(
    feature = "prime",
    any(target_os = "macos", target_os = "linux"),
    feature = "runtime-prime-inbox-alloc"
))]
pub mod prime;
#[cfg(feature = "tokio")]
pub mod tokio;
#[cfg(feature = "wasm")]
pub mod wasm;
#[cfg(feature = "xdp")]
pub mod xdp;

#[cfg(feature = "std")]
pub use packet::{Packet, PacketListener, PacketListenerExt};
pub use stack::{Action, build_arp_request, handle_frame, parse_arp_reply};
#[cfg(feature = "alloc")]
pub use tcp_listener::{EchoListener, Endpoint, Inbound, OutSegment, Response};
#[cfg(feature = "alloc")]
pub use tcp_stack::{ConnId, Outbound, TcpStack};
