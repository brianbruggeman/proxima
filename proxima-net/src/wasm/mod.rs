//! Wasm host-backed [`PacketListener`](proxima_net::PacketListener).
//!
//! `wasm32-unknown-unknown` has no socket syscalls; the embedder owns
//! the datagram transport (browser WebTransport datagrams, a wasi UDP
//! shim, a test harness). This crate is the fourth net backend, a
//! sibling of `proxima-net-tokio` / `-prime` / `-dpdk`: it implements
//! the same sans-IO-friendly `poll_recv` / `poll_send` surface by
//! delegating to host imports, exactly as the `proxima-time` wasm
//! driver delegates the clock.
//!
//! # Host ABI
//!
//! Imports the wasm module expects:
//!
//! ```text
//! // try to receive one datagram on `handle`. on success writes a 19-byte
//! // packed peer address (see addr codec) into `addr_out` and the payload
//! // into `buf`, returning the payload length. -1 = would-block, -2 = error.
//! fn proxima_net_recv(handle: u32, addr_out: *mut u8, buf: *mut u8, buf_len: usize) -> i64;
//!
//! // send `buf` to the 19-byte packed peer address at `addr`. returns bytes
//! // sent, -1 = would-block, -2 = error.
//! fn proxima_net_send(handle: u32, addr: *const u8, buf: *const u8, buf_len: usize) -> i64;
//! ```
//!
//! Export the host calls when `handle` becomes ready (readable or
//! writable):
//!
//! ```text
//! fn proxima_net_wake(handle: u32);
//! ```
//!
//! Over-waking is safe: a spurious `proxima_net_wake` costs one extra
//! poll, which re-issues the host `recv`/`send` and re-parks on
//! would-block.
//!
//! `encode_addr`/`decode_addr`/`take_wakers` are pure logic and compile
//! under `no_std` + `alloc` (`core::net::SocketAddr` has been available
//! since Rust 1.77). The wasm32-gated [`backend`] module — the one piece
//! that actually calls `PacketListener`'s `std::io::Result` surface — is
//! std-bound by necessity, since it links `crate::PacketListener`. The
//! crate-level `no_std` + `alloc` gate lives on `proxima-net`'s own crate
//! root; this module inherits it.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Packed wire form of a peer address handed across the host boundary:
/// `[family, port_hi, port_lo, addr[0..16]]`. IPv4 occupies the first
/// four address bytes; IPv6 fills all sixteen. Fixed 19 bytes so the
/// host writes a constant-size scratch with no length negotiation.
pub const PACKED_ADDR_LEN: usize = 19;

const FAMILY_V4: u8 = 4;
const FAMILY_V6: u8 = 6;

/// Encode a [`SocketAddr`](core::net::SocketAddr) into the 19-byte host
/// wire form. Pure so the marshalling is testable without the host.
#[must_use]
pub fn encode_addr(addr: &core::net::SocketAddr) -> [u8; PACKED_ADDR_LEN] {
    let mut out = [0_u8; PACKED_ADDR_LEN];
    let port = addr.port().to_be_bytes();
    out[1] = port[0];
    out[2] = port[1];
    match addr {
        core::net::SocketAddr::V4(v4) => {
            out[0] = FAMILY_V4;
            out[3..7].copy_from_slice(&v4.ip().octets());
        }
        core::net::SocketAddr::V6(v6) => {
            out[0] = FAMILY_V6;
            out[3..19].copy_from_slice(&v6.ip().octets());
        }
    }
    out
}

/// Decode the 19-byte host wire form back into a `SocketAddr`. `None`
/// when the family byte is neither v4 nor v6 — a malformed host frame.
#[must_use]
pub fn decode_addr(bytes: &[u8; PACKED_ADDR_LEN]) -> Option<core::net::SocketAddr> {
    let port = u16::from_be_bytes([bytes[1], bytes[2]]);
    match bytes[0] {
        FAMILY_V4 => {
            let octets: [u8; 4] = [bytes[3], bytes[4], bytes[5], bytes[6]];
            Some(core::net::SocketAddr::from((octets, port)))
        }
        FAMILY_V6 => {
            let mut octets = [0_u8; 16];
            octets.copy_from_slice(&bytes[3..19]);
            Some(core::net::SocketAddr::from((octets, port)))
        }
        _ => None,
    }
}

/// Remove and return every waker registered for `handle`, leaving other
/// handles' waiters in place. Pure so the readiness fan-out is testable
/// without the host imports. Shared by the wasm backend's registry.
#[cfg(feature = "alloc")]
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code) // exercised by host unit tests + the wasm-gated backend
)]
fn take_wakers(entries: &mut Vec<(u32, core::task::Waker)>, handle: u32) -> Vec<core::task::Waker> {
    let mut ready = Vec::new();
    entries.retain(|(registered, waker)| {
        if *registered == handle {
            ready.push(waker.clone());
            false
        } else {
            true
        }
    });
    ready
}

#[cfg(all(target_arch = "wasm32", feature = "std"))]
mod backend;
#[cfg(all(target_arch = "wasm32", feature = "std"))]
pub use backend::WasmPacketListener;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    #[test]
    fn v4_round_trips_through_packed_form() {
        let addr: SocketAddr = "192.168.1.42:7769".parse().unwrap();

        let packed = encode_addr(&addr);

        assert_eq!(packed[0], FAMILY_V4);
        assert_eq!(decode_addr(&packed), Some(addr));
    }

    #[test]
    fn v6_round_trips_through_packed_form() {
        let addr: SocketAddr = "[2001:db8::1]:4887".parse().unwrap();

        let packed = encode_addr(&addr);

        assert_eq!(packed[0], FAMILY_V6);
        assert_eq!(decode_addr(&packed), Some(addr));
    }

    #[test]
    fn unknown_family_byte_is_rejected() {
        let mut packed = [0_u8; PACKED_ADDR_LEN];
        packed[0] = 9;

        assert_eq!(decode_addr(&packed), None);
    }

    fn noop_waker() -> std::task::Waker {
        use std::task::{RawWaker, RawWakerVTable, Waker};
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        fn noop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn take_wakers_returns_only_the_matching_handle() {
        let mut entries = vec![(7_u32, noop_waker()), (9, noop_waker()), (7, noop_waker())];

        let ready = take_wakers(&mut entries, 7);

        assert_eq!(ready.len(), 2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 9);
    }

    #[test]
    fn take_wakers_for_absent_handle_is_empty() {
        let mut entries = vec![(1_u32, noop_waker())];

        assert!(take_wakers(&mut entries, 42).is_empty());
        assert_eq!(entries.len(), 1);
    }
}
