//! Zero-allocation proof for the tiered datagram-I/O hot loop (P11).
//!
//! A counting global allocator wraps `System`. The steady-state serve cycle —
//! fill the [`RecvSlab`] to empty, read every datagram, stage a burst into the
//! [`SendBatch`], drive it to the socket, reset — is run N times after a warm-up
//! that grows the slab + arena to their working size.
//!
//! The proof is DIFFERENTIAL: a fixed amount of harness allocation is
//! unavoidable, so we run the cycle 1000 and 2000 times and assert the
//! allocation delta is IDENTICAL. A single allocating cycle would make the
//! 2000-iteration run allocate 1000 more times than the 1000-iteration run.
//! Equal deltas ⇒ per-cycle allocation is exactly zero — the slab/arena are
//! reused, the iovecs live on the stack.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use futures::task::noop_waker_ref;
use proxima_core::datagram_batch::{RecvSlab, SendBatch};
use proxima_primitives::stream::{DatagramSocket, DatagramSocketBatchExt};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// A 1-RTT short-header datagram (first byte 0x40-0x7f) — the steady-state shape
// once a connection is established.
const STEADY_DATAGRAM: &[u8] = &[
    0x40, 0x8a, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08, 0xde, 0xad, 0xbe, 0xef,
];
const DATAGRAMS_PER_CYCLE: usize = 48;

fn peer() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 4433)
}

/// Zero-alloc in-memory socket: serves `recv_budget` copies of the steady
/// datagram per refill then reports empty, and counts sent bytes without
/// retaining them (a `Vec` sink would itself allocate and mask the proof).
#[derive(Default)]
struct LoopbackSocket {
    recv_budget: usize,
    sent_bytes: usize,
}

impl DatagramSocket for LoopbackSocket {
    fn poll_recv_from(
        &mut self,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<(usize, SocketAddr)>> {
        if self.recv_budget == 0 {
            return Poll::Pending;
        }
        self.recv_budget -= 1;
        let len = STEADY_DATAGRAM.len().min(buf.len());
        buf[..len].copy_from_slice(&STEADY_DATAGRAM[..len]);
        Poll::Ready(Ok((len, peer())))
    }

    fn poll_send_to(
        &mut self,
        _cx: &mut Context<'_>,
        buf: &[u8],
        _peer: SocketAddr,
    ) -> Poll<std::io::Result<usize>> {
        self.sent_bytes += buf.len();
        Poll::Ready(Ok(buf.len()))
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        Ok(peer())
    }
}

fn run_cycles(
    socket: &mut LoopbackSocket,
    slab: &mut RecvSlab<2048, 32>,
    send: &mut SendBatch<65536, 256>,
    cycles: usize,
) {
    let mut cx = Context::from_waker(noop_waker_ref());
    let mut sink = 0u64;
    for _ in 0..cycles {
        socket.recv_budget = DATAGRAMS_PER_CYCLE;
        slab.clear();
        let _ = socket.poll_fill_recv_batch(&mut cx, slab);
        socket.drain_recv_to_empty(slab);
        for view in slab.filled_datagrams() {
            sink = sink.wrapping_add(view.bytes.len() as u64);
        }

        send.reset();
        for _ in 0..DATAGRAMS_PER_CYCLE {
            send.try_append(STEADY_DATAGRAM, peer())
                .expect("arena holds the burst");
        }
        let mut span_offset = 0;
        let _ = socket.poll_drive_send_batch(&mut cx, send, &mut span_offset);
    }
    std::hint::black_box(sink);
    std::hint::black_box(socket.sent_bytes);
}

fn alloc_delta_for(cycles: usize) -> usize {
    let mut socket = LoopbackSocket::default();
    let mut slab: RecvSlab<2048, 32> = RecvSlab::new();
    let mut send: SendBatch<65536, 256> = SendBatch::new();

    // warm-up grows the slab + arena to working size and primes the span Vec.
    run_cycles(&mut socket, &mut slab, &mut send, 64);

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    run_cycles(&mut socket, &mut slab, &mut send, cycles);
    ALLOC_COUNT.load(Ordering::Relaxed) - before
}

#[test]
fn datagram_batch_steady_state_is_zero_alloc() {
    let delta_1k = alloc_delta_for(1_000);
    let delta_2k = alloc_delta_for(2_000);

    assert_eq!(
        delta_1k, delta_2k,
        "per-cycle allocation must be zero: 1000-cycle delta={delta_1k}, \
         2000-cycle delta={delta_2k}; a nonzero difference scales with the cycle \
         count, meaning the recv-fill/send-stage/drive hot loop allocates"
    );
}
