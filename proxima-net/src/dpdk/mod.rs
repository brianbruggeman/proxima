//! DPDK EAL + poll-mode-driver shell for the proxima userspace network stack.
//!
//! This is the I/O floor that the sans-IO codecs (`proxima-inet-codec`,
//! `proxima-tcp`) run on top of: dpdk hands raw L2 frames up through [`Port`],
//! the codecs parse them, and replies go back down the same RX/TX rings.
//!
//! The dpdk seam ([`ffi`], [`eal`], [`port`]) is behind the default-off `dpdk`
//! feature and links `librte_*` — it builds only on a dpdk host. The sans-IO
//! L2/L3 responder and the backend-agnostic TCP driver live in `proxima_net`
//! (`stack`, `tcp_stack`, `tcp_listener`) — shared across backends and
//! unit-tested in mac CI there.

#[cfg(feature = "dpdk")]
mod ffi;

#[cfg(feature = "dpdk")]
pub mod eal;
#[cfg(feature = "dpdk")]
pub mod error;
#[cfg(feature = "dpdk")]
pub mod packet_listener;
#[cfg(feature = "dpdk")]
pub mod port;
#[cfg(feature = "dpdk")]
pub mod stream_listener;

#[cfg(feature = "dpdk")]
pub use eal::Eal;
#[cfg(feature = "dpdk")]
pub use error::DpdkError;
#[cfg(feature = "dpdk")]
pub use ffi::RteMbuf;
#[cfg(feature = "dpdk")]
pub use packet_listener::DpdkPacketListener;
#[cfg(feature = "dpdk")]
pub use port::{Mempool, Port, port_count};
#[cfg(feature = "dpdk")]
pub use stream_listener::{DpdkStreamConnection, DpdkStreamListener, DpdkStreamUpstream};

/// A raw, caller-owned dpdk packet buffer handle. Filled by [`Port::rx_burst`],
/// consumed by [`Port::tx_burst`] / [`port::free`].
#[cfg(feature = "dpdk")]
pub type RawMbuf = *mut RteMbuf;
