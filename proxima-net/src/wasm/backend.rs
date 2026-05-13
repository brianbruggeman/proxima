//! Wasm32 host-backed [`PacketListener`] implementation. Gated to
//! `target_arch = "wasm32"` because it links the host import symbols;
//! the pure address + waker-registry logic it leans on lives in the
//! always-compiled crate root so native tests cover it.

use std::cell::UnsafeCell;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

use bytes::Bytes;

use crate::{Packet, PacketListener};
use super::{PACKED_ADDR_LEN, decode_addr, encode_addr, take_wakers};

const WOULD_BLOCK: i64 = -1;

unsafe extern "C" {
    fn proxima_net_recv(handle: u32, addr_out: *mut u8, buf: *mut u8, buf_len: usize) -> i64;
    fn proxima_net_send(handle: u32, addr: *const u8, buf: *const u8, buf_len: usize) -> i64;
}

/// `(handle, waker)` registry for sockets parked on would-block. Single
/// spinlock over an `alloc` vector; wasm is single-threaded so the lock
/// is uncontended, but the `Sync` bound a `&'static` registry needs
/// requires the synchronization to be sound on paper.
struct Registry {
    lock: AtomicBool,
    entries: UnsafeCell<Vec<(u32, Waker)>>,
}

// safety: every access holds `lock` for the `&mut` borrow; wasm32 has no
// second thread to race.
unsafe impl Sync for Registry {}

impl Registry {
    const fn new() -> Self {
        Self {
            lock: AtomicBool::new(false),
            entries: UnsafeCell::new(Vec::new()),
        }
    }

    fn with<R>(&self, body: impl FnOnce(&mut Vec<(u32, Waker)>) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        let entries = unsafe { &mut *self.entries.get() };
        let result = body(entries);
        self.lock.store(false, Ordering::Release);
        result
    }
}

static REGISTRY: Registry = Registry::new();

fn park(handle: u32, waker: &Waker) {
    REGISTRY.with(|entries| entries.push((handle, waker.clone())));
}

/// Host entry point: invoked when `handle` becomes ready. Wakes every
/// waiter parked on that handle. `#[no_mangle]` so the embedder calls it
/// by name.
#[unsafe(no_mangle)]
pub extern "C" fn proxima_net_wake(handle: u32) {
    let ready = REGISTRY.with(|entries| take_wakers(entries, handle));
    for waker in ready {
        waker.wake();
    }
}

/// A UDP datagram endpoint whose recv/send are serviced by the wasm
/// host. Construct from a host-assigned `handle` (the index the host
/// uses to identify this socket across the recv/send/wake ABI) and the
/// bound local address the host reports.
///
/// Sibling of `TokioUdpListener` / prime / dpdk packet listeners — same
/// [`PacketListener`] surface, host-driven readiness instead of an
/// epoll/kqueue reactor.
pub struct WasmPacketListener {
    handle: u32,
    local: Option<SocketAddr>,
}

impl WasmPacketListener {
    /// Bind to a host-assigned socket `handle` with the `local` address
    /// the host reports for it.
    #[must_use]
    pub fn from_handle(handle: u32, local: Option<SocketAddr>) -> Self {
        Self { handle, local }
    }
}

impl PacketListener for WasmPacketListener {
    fn poll_recv(&self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        let mut addr_scratch = [0_u8; PACKED_ADDR_LEN];
        let received = unsafe {
            proxima_net_recv(
                self.handle,
                addr_scratch.as_mut_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
            )
        };
        if received == WOULD_BLOCK {
            park(self.handle, cx.waker());
            return Poll::Pending;
        }
        if received < 0 {
            return Poll::Ready(Err(io::Error::other("proxima_net_recv host error")));
        }
        let Some(src) = decode_addr(&addr_scratch) else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "host returned a malformed peer address",
            )));
        };
        let length = received as usize;
        Poll::Ready(Ok(Packet {
            src,
            dst: self.local.unwrap_or(src),
            data: Bytes::copy_from_slice(&buf[..length]),
        }))
    }

    fn poll_send(&self, cx: &mut Context<'_>, packet: &Packet) -> Poll<io::Result<()>> {
        let addr = encode_addr(&packet.dst);
        let sent = unsafe {
            proxima_net_send(
                self.handle,
                addr.as_ptr(),
                packet.data.as_ptr(),
                packet.data.len(),
            )
        };
        if sent == WOULD_BLOCK {
            park(self.handle, cx.waker());
            return Poll::Pending;
        }
        if sent < 0 {
            return Poll::Ready(Err(io::Error::other("proxima_net_send host error")));
        }
        Poll::Ready(Ok(()))
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        self.local
    }
}
