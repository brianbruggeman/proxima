//! DPDK EAL + poll-mode-driver shell for the proxima userspace network stack.
//!
//! This is the I/O floor that the sans-IO codecs (`proxima-inet-codec`,
//! `proxima-tcp`) run on top of: dpdk hands raw L2 frames up through [`Port`],
//! the codecs parse them, and replies go back down the same RX/TX rings.
//!
//! The dpdk seam ([`ffi`], [`eal`], [`port`]) links `librte_*` — it builds only
//! on a dpdk host. `crate::lib`'s `#[cfg(feature = "dpdk")] pub mod dpdk` is the
//! one gate on this whole module, so nothing inside repeats it (the `xdp`
//! sibling gates only on `target_os`, the condition its declaration does NOT
//! already carry). The sans-IO L2/L3 responder and the backend-agnostic TCP
//! driver live in `proxima_net` (`stack`, `tcp_stack`, `tcp_listener`) — shared
//! across backends and unit-tested in mac CI there.

mod ffi;

pub mod eal;
pub mod error;
pub mod packet_listener;
pub mod port;
pub mod stream_listener;

pub use eal::Eal;
pub use error::DpdkError;
pub use ffi::RteMbuf;
pub use packet_listener::DpdkPacketListener;
pub use port::{Mempool, Port, port_count};
pub use stream_listener::{DpdkStreamConnection, DpdkStreamListener, DpdkStreamUpstream};

/// A raw, caller-owned dpdk packet buffer handle. Filled by [`Port::rx_burst`],
/// consumed by [`Port::tx_burst`] / [`port::free`].
pub type RawMbuf = *mut RteMbuf;
