//! Sans-IO Ethernet/IPv4/TCP wire codec for a userspace network stack.
//!
//! Tier-3: compiles under `#![no_std]` with no allocator. Decode borrows
//! views over a caller-owned buffer; encode writes into caller-owned storage.
//! No I/O, no state held beyond the bytes — the codec parses and builds wire
//! frames, the connection state machine lives elsewhere.
//!
//! The DPDK packet path hands raw L2 frames up; this layer turns them into
//! typed views and back without copying.

pub mod checksum;
pub mod error;
pub mod ethernet;
pub mod ipv4;
pub mod tcp;
pub mod udp;

pub use checksum::Checksum;
pub use error::DecodeError;
pub use ethernet::{EtherType, EthernetFrame};
pub use ipv4::{Ipv4Header, Ipv4Protocol};
pub use tcp::{TcpFlags, TcpHeader};
pub use udp::UdpHeader;
