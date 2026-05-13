//! Drive layer for the canonical datagram buffers in
//! [`proxima_core::datagram_batch`].
//!
//! [`DatagramSocketBatchExt`] is a blanket-implemented extension trait over
//! every [`DatagramSocket`]: it fills a [`RecvSlab`] from the socket and ships
//! a [`SendBatch`] to it, building the per-syscall iovec arrays on the STACK
//! (via [`arrayvec::ArrayVec`]) so the hot path allocates nothing. The buffers
//! themselves stay `no_std` in proxima-core; this is the std-tier I/O seam.
//!
//! Why an extension trait and not a method on the buffers or on the socket
//! trait: the buffers must stay `no_std` (can't name `DatagramSocket`), and
//! `DatagramSocket` must stay the minimal I/O boundary (no buffer-policy
//! knowledge). A blanket `impl<T: DatagramSocket>` gives every backend the
//! drive for free. A DPDK/AF_XDP ring source bypasses this trait and fills the
//! slab directly via [`RecvSlab::unfilled_slots_mut`] + [`RecvSlab::commit_filled`].

use std::io;

use core::net::SocketAddr;
use core::task::{Context, Poll};

use arrayvec::ArrayVec;
use futures::task::noop_waker_ref;
use proxima_core::datagram_batch::{RecvSlab, SendBatch};

use crate::stream::DatagramSocket;

/// Datagrams pulled per `recvmmsg` (the stack iovec array width). Matches
/// prime's `DATAGRAM_BATCH`.
const RECV_BATCH: usize = 32;
/// Datagrams shipped per `sendmmsg` (the stack iovec array width).
const SEND_CHUNK: usize = 64;

/// Fills and drains [`RecvSlab`]/[`SendBatch`] over any [`DatagramSocket`].
pub trait DatagramSocketBatchExt: DatagramSocket {
    /// Receive one batch into `slab`, registering the caller's waker. This is
    /// the AWAITED path (raced against a timer in the serve loop): on an empty
    /// socket it returns `Pending` with the real waker installed so the task
    /// parks. Returns the count of newly filled slots.
    ///
    /// Zero heap: the `&mut [u8]` iovec array and the `(len, peer)` scratch are
    /// stack `ArrayVec`/array of width [`RECV_BATCH`].
    fn poll_fill_recv_batch<const SLOT_BYTES: usize, const INITIAL_CAP: usize>(
        &mut self,
        cx: &mut Context<'_>,
        slab: &mut RecvSlab<SLOT_BYTES, INITIAL_CAP>,
    ) -> Poll<io::Result<usize>> {
        // Grow toward a full batch on the alloc tier; tolerate a partial tail
        // on the no-alloc tier (ensure_capacity errors there, ignored — the
        // remaining fixed slots are still filled).
        let _ = slab.ensure_capacity(slab.filled() + RECV_BATCH);

        let count = {
            // The receive writes bytes AND (len, peer) straight into the slab's
            // persistent storage — no staging array, no second copy. Only the
            // iovec pointer array is stack-local (recvmmsg needs `&mut [&mut [u8]]`).
            let (slots, meta) = slab.unfilled_mut();
            let batch = slots.len().min(meta.len()).min(RECV_BATCH);
            if batch == 0 {
                return Poll::Ready(Ok(0));
            }
            let mut bufs: ArrayVec<&mut [u8], RECV_BATCH> = slots
                .iter_mut()
                .take(batch)
                .map(|slot| slot.as_mut_slice())
                .collect();
            match self.poll_recv_batch(cx, bufs.as_mut_slice(), &mut meta[..batch]) {
                Poll::Ready(Ok(received)) => received,
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            }
        };

        slab.commit(count);
        Poll::Ready(Ok(count))
    }

    /// Drain the socket to empty into `slab` (non-blocking, noop waker), after
    /// an initial awaited [`poll_fill_recv_batch`](Self::poll_fill_recv_batch).
    /// Returns the additional slots drained. Decouples drain-rate from
    /// process-rate so the default kernel datagram buffer never overflows
    /// during a handshake burst (no dependence on a raised `rmem_max`).
    ///
    /// The noop waker is harmless: the real waker is reinstalled by the next
    /// iteration's awaited `poll_fill_recv_batch`.
    fn drain_recv_to_empty<const SLOT_BYTES: usize, const INITIAL_CAP: usize>(
        &mut self,
        slab: &mut RecvSlab<SLOT_BYTES, INITIAL_CAP>,
    ) -> usize {
        let mut drained = 0;
        loop {
            let mut cx = Context::from_waker(noop_waker_ref());
            match self.poll_fill_recv_batch(&mut cx, slab) {
                Poll::Ready(Ok(received)) if received > 0 => drained += received,
                _ => return drained,
            }
        }
    }

    /// Ship staged packets from `batch`, resuming at `span_offset`. Returns
    /// `Ready(Ok(()))` once `span_offset` reaches the staged count or the send
    /// buffer fills (partial progress recorded in `span_offset`); the caller
    /// loops until `span_offset == batch.len()`.
    ///
    /// Zero heap: each `sendmmsg` chunk is a stack `ArrayVec` of width
    /// [`SEND_CHUNK`].
    fn poll_drive_send_batch<const SEND_BYTES: usize, const SPAN_CAP: usize>(
        &mut self,
        cx: &mut Context<'_>,
        batch: &SendBatch<SEND_BYTES, SPAN_CAP>,
        span_offset: &mut usize,
    ) -> Poll<io::Result<()>> {
        let spans = batch.spans();
        while *span_offset < spans.len() {
            let chunk_end = (*span_offset + SEND_CHUNK).min(spans.len());
            let mut iov: ArrayVec<(&[u8], SocketAddr), SEND_CHUNK> = ArrayVec::new();
            for span in &spans[*span_offset..chunk_end] {
                iov.push((batch.slice_for(span), span.peer));
            }
            match self.poll_send_batch(cx, iov.as_slice()) {
                Poll::Ready(Ok(sent)) if sent > 0 => *span_offset += sent,
                // socket full or error mid-burst: stop, caller retries / drops.
                Poll::Ready(Ok(_)) | Poll::Ready(Err(_)) => return Poll::Ready(Ok(())),
                Poll::Pending if *span_offset > 0 => return Poll::Ready(Ok(())),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(()))
    }
}

impl<T: DatagramSocket + ?Sized> DatagramSocketBatchExt for T {}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::net::{IpAddr, Ipv4Addr};
    use std::task::Waker;

    use proxima_core::datagram_batch::{SendBatch, SendSpan};

    use super::*;

    // RFC 9001 §A.2 — the protected client Initial datagram, the exact bytes a
    // QUIC server reads off the wire on first contact (DCID 8394c8f03e515708,
    // version 0x00000001). Prefix is enough to prove byte-fidelity through the
    // drive layer (which never interprets the payload). https://www.rfc-editor.org/rfc/rfc9001#appendix-A.2
    const RFC9001_CLIENT_INITIAL_PREFIX: &[u8] = &[
        0xc0, 0x00, 0x00, 0x00, 0x01, 0x08, 0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08, 0x00,
        0x00, 0x44, 0x9e, 0x7b, 0x9a, 0xec, 0x34, 0xd1, 0xb1, 0xc9, 0x8d, 0xd7, 0x68, 0x9f, 0xb8,
        0xec, 0x11,
    ];
    // A 1-RTT short-header datagram (first byte 0x40-0x7f), the steady-state
    // shape after handshake — distinct from the Initial so order is checkable.
    const SHORT_HEADER_DATAGRAM: &[u8] = &[
        0x40, 0x8a, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08, 0xde, 0xad, 0xbe, 0xef,
    ];

    fn peer(last_octet: u8, port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, last_octet)), port)
    }

    /// In-memory `DatagramSocket`: a recv queue the drive layer drains and a
    /// sent log it appends to. `send_block_after` makes `poll_send_to` report
    /// the kernel buffer full (`Pending`) after N total sends, to exercise the
    /// partial-send resume path; raise it to unblock.
    #[derive(Default)]
    struct FakeSocket {
        inbound: VecDeque<(Vec<u8>, SocketAddr)>,
        sent: Vec<(Vec<u8>, SocketAddr)>,
        send_block_after: Option<usize>,
        last_recv_waker: Option<Waker>,
    }

    impl FakeSocket {
        fn with_inbound(datagrams: impl IntoIterator<Item = (Vec<u8>, SocketAddr)>) -> Self {
            Self {
                inbound: datagrams.into_iter().collect(),
                ..Self::default()
            }
        }
    }

    impl DatagramSocket for FakeSocket {
        fn poll_recv_from(
            &mut self,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<(usize, SocketAddr)>> {
            match self.inbound.pop_front() {
                Some((bytes, from)) => {
                    let len = bytes.len().min(buf.len());
                    buf[..len].copy_from_slice(&bytes[..len]);
                    Poll::Ready(Ok((len, from)))
                }
                None => {
                    self.last_recv_waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }

        fn poll_send_to(
            &mut self,
            _cx: &mut Context<'_>,
            buf: &[u8],
            peer: SocketAddr,
        ) -> Poll<io::Result<usize>> {
            if let Some(cap) = self.send_block_after
                && self.sent.len() >= cap
            {
                return Poll::Pending;
            }
            self.sent.push((buf.to_vec(), peer));
            Poll::Ready(Ok(buf.len()))
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(peer(1, 4433))
        }
    }

    fn waker_cx() -> Context<'static> {
        Context::from_waker(noop_waker_ref())
    }

    #[test]
    fn poll_fill_recv_batch_fills_slab_with_real_datagrams() {
        let inbound = vec![
            (RFC9001_CLIENT_INITIAL_PREFIX.to_vec(), peer(7, 4433)),
            (SHORT_HEADER_DATAGRAM.to_vec(), peer(9, 51000)),
        ];
        let mut socket = FakeSocket::with_inbound(inbound);
        let mut slab: RecvSlab<2048, 32> = RecvSlab::new();
        let mut cx = waker_cx();

        let filled = socket
            .poll_fill_recv_batch(&mut cx, &mut slab)
            .map(|res| res.expect("drive/fill ok"));

        assert_eq!(filled, Poll::Ready(2));
        let got: Vec<(Vec<u8>, SocketAddr)> = slab
            .filled_datagrams()
            .map(|view| (view.bytes.to_vec(), view.peer))
            .collect();
        assert_eq!(
            got[0],
            (RFC9001_CLIENT_INITIAL_PREFIX.to_vec(), peer(7, 4433))
        );
        assert_eq!(got[1], (SHORT_HEADER_DATAGRAM.to_vec(), peer(9, 51000)));
    }

    #[test]
    fn poll_fill_recv_batch_is_pending_on_empty_socket() {
        let mut socket = FakeSocket::default();
        let mut slab: RecvSlab<2048, 32> = RecvSlab::new();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        let outcome = socket.poll_fill_recv_batch(&mut cx, &mut slab);

        assert_eq!(
            outcome.map(|res| res.expect("drive/fill ok")),
            Poll::Pending
        );
        assert_eq!(slab.filled(), 0);
        assert!(
            socket.last_recv_waker.is_some(),
            "real waker installed for the park"
        );
    }

    #[test]
    fn drain_recv_to_empty_pulls_every_datagram_past_one_batch() {
        // More than RECV_BATCH so the growable slab path (ensure_capacity +
        // multiple drain rounds) is exercised, matching a handshake burst.
        let burst = RECV_BATCH + 5;
        let inbound: Vec<(Vec<u8>, SocketAddr)> = (0..burst)
            .map(|index| (SHORT_HEADER_DATAGRAM.to_vec(), peer(index as u8, 4433)))
            .collect();
        let mut socket = FakeSocket::with_inbound(inbound);
        let mut slab: RecvSlab<2048, 32> = RecvSlab::new();

        let drained = socket.drain_recv_to_empty(&mut slab);

        assert_eq!(drained, burst);
        assert_eq!(slab.filled(), burst);
        assert!(socket.inbound.is_empty());
    }

    #[test]
    fn poll_drive_send_batch_ships_all_spans_in_order() {
        let mut batch: SendBatch<65536, 256> = SendBatch::new();
        batch
            .try_append(RFC9001_CLIENT_INITIAL_PREFIX, peer(7, 4433))
            .expect("append fits the arena");
        batch
            .try_append(SHORT_HEADER_DATAGRAM, peer(9, 51000))
            .expect("append fits the arena");
        let mut socket = FakeSocket::default();
        let mut span_offset = 0;
        let mut cx = waker_cx();

        let outcome = socket.poll_drive_send_batch(&mut cx, &batch, &mut span_offset);

        assert_eq!(
            outcome.map(|res| res.expect("drive/fill ok")),
            Poll::Ready(())
        );
        assert_eq!(span_offset, 2);
        assert_eq!(
            socket.sent,
            vec![
                (RFC9001_CLIENT_INITIAL_PREFIX.to_vec(), peer(7, 4433)),
                (SHORT_HEADER_DATAGRAM.to_vec(), peer(9, 51000)),
            ]
        );
    }

    #[test]
    fn poll_drive_send_batch_resumes_after_partial_send() {
        let mut batch: SendBatch<65536, 256> = SendBatch::new();
        for index in 0..5u8 {
            batch
                .try_append(SHORT_HEADER_DATAGRAM, peer(index, 4433))
                .expect("append fits the arena");
        }
        // kernel buffer "fills" after 2 datagrams.
        let mut socket = FakeSocket {
            send_block_after: Some(2),
            ..FakeSocket::default()
        };
        let mut span_offset = 0;
        let mut cx = waker_cx();

        let first = socket.poll_drive_send_batch(&mut cx, &batch, &mut span_offset);
        assert_eq!(
            first.map(|res| res.expect("drive/fill ok")),
            Poll::Ready(()),
            "partial progress yields Ready"
        );
        assert_eq!(span_offset, 2, "resume cursor parked at the block point");

        socket.send_block_after = Some(usize::MAX);
        let second = socket.poll_drive_send_batch(&mut cx, &batch, &mut span_offset);
        assert_eq!(
            second.map(|res| res.expect("drive/fill ok")),
            Poll::Ready(())
        );
        assert_eq!(span_offset, 5, "remaining datagrams shipped on resume");
        assert_eq!(socket.sent.len(), 5);
        let peers: Vec<SocketAddr> = socket.sent.iter().map(|(_, addr)| *addr).collect();
        assert_eq!(
            peers,
            (0..5u8).map(|index| peer(index, 4433)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn poll_drive_send_batch_empty_is_immediately_ready() {
        let batch: SendBatch<65536, 256> = SendBatch::new();
        let mut socket = FakeSocket::default();
        let mut span_offset = 0;
        let mut cx = waker_cx();

        let outcome = socket.poll_drive_send_batch(&mut cx, &batch, &mut span_offset);

        assert_eq!(
            outcome.map(|res| res.expect("drive/fill ok")),
            Poll::Ready(())
        );
        assert_eq!(span_offset, 0);
        assert!(socket.sent.is_empty());
    }

    #[test]
    fn send_span_round_trips_through_slice_for() {
        // The drive layer trusts SendBatch::slice_for; assert the span maps back
        // to the exact appended bytes so a wrong offset can't silently ship garbage.
        let mut batch: SendBatch<65536, 256> = SendBatch::new();
        batch
            .try_append(RFC9001_CLIENT_INITIAL_PREFIX, peer(7, 4433))
            .expect("append fits the arena");
        let spans: &[SendSpan] = batch.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(batch.slice_for(&spans[0]), RFC9001_CLIENT_INITIAL_PREFIX);
        assert_eq!(spans[0].peer, peer(7, 4433));
    }
}
