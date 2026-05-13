//! Home-turf bench: the canonical tiered datagram-I/O buffers vs the hand-rolled
//! `Vec`-based recv-slab + send-arena they replace in `proxima-h3` listen.rs.
//!
//! The 80% case is the per-burst steady-state cycle: fill a batch of datagrams
//! into recv storage, read every datagram, stage an equal burst into send
//! storage, and build the `sendmmsg` iovecs. Both arms do identical work on
//! identical input — the only difference is the buffer representation. The
//! incumbent arm (`design-favors: incumbent`) is the exact hand-rolled shape
//! that shipped the proven 120k req/s/core number; the drive-layer arm
//! (`design-favors: proxima`) is the canonical primitive. A tie is the honest
//! target: the primitive must not regress the loop it canonicalizes.

use std::hint::black_box;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use arrayvec::ArrayVec;
use criterion::{Criterion, criterion_group, criterion_main};
use proxima_core::datagram_batch::{RecvSlab, SendBatch};

const STEADY_DATAGRAM: &[u8] = &[
    0x40, 0x8a, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08, 0xde, 0xad, 0xbe, 0xef,
];
const BATCH: usize = 48;

fn peer() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 4433)
}

/// The hand-rolled incumbent: `Vec<[u8; 2048]>` recv slots + `Vec<(len, peer)>`
/// meta + `Vec<u8>` send arena + `Vec<(offset, len, peer)>` spans, reused across
/// cycles exactly as listen.rs did before the migration.
struct HandRolled {
    recv_storage: Vec<[u8; 2048]>,
    recv_meta: Vec<(usize, SocketAddr)>,
    send_arena: Vec<u8>,
    send_spans: Vec<(usize, usize, SocketAddr)>,
}

impl HandRolled {
    fn new() -> Self {
        Self {
            recv_storage: vec![[0u8; 2048]; BATCH],
            recv_meta: vec![(0, peer()); BATCH],
            send_arena: Vec::with_capacity(BATCH * 2048),
            send_spans: Vec::new(),
        }
    }

    fn recv_cycle(&mut self) -> u64 {
        for index in 0..BATCH {
            let slot = &mut self.recv_storage[index];
            slot[..STEADY_DATAGRAM.len()].copy_from_slice(STEADY_DATAGRAM);
            self.recv_meta[index] = (STEADY_DATAGRAM.len(), peer());
        }
        let mut sink = 0u64;
        for index in 0..BATCH {
            let (len, addr) = self.recv_meta[index];
            let bytes = &self.recv_storage[index][..len];
            // production reads both the bytes and the peer (handle_inbound).
            sink = sink.wrapping_add(u64::from(bytes[0]));
            sink = sink.wrapping_add(u64::from(addr.port()));
        }
        sink
    }

    fn send_cycle(&mut self) -> u64 {
        self.send_arena.clear();
        self.send_spans.clear();
        for _ in 0..BATCH {
            let offset = self.send_arena.len();
            self.send_arena.extend_from_slice(STEADY_DATAGRAM);
            self.send_spans
                .push((offset, STEADY_DATAGRAM.len(), peer()));
        }
        let packets: Vec<(&[u8], SocketAddr)> = self
            .send_spans
            .iter()
            .map(|&(offset, len, addr)| (&self.send_arena[offset..offset + len], addr))
            .collect();
        packets.len() as u64
    }

    fn cycle(&mut self) -> u64 {
        self.recv_cycle().wrapping_add(self.send_cycle())
    }
}

/// The canonical primitive: `RecvSlab` filled via `commit_filled` + `SendBatch`
/// staged via `try_append`, the same operations the drive layer performs.
struct DriveLayer {
    slab: RecvSlab<2048, 32>,
    send: SendBatch<65536, 256>,
}

impl DriveLayer {
    fn new() -> Self {
        Self {
            slab: RecvSlab::new(),
            send: SendBatch::new(),
        }
    }

    fn recv_cycle(&mut self) -> u64 {
        self.slab.clear();
        let _ = self.slab.ensure_capacity(BATCH);
        {
            // mirror the real drive: the receive writes bytes + (len, peer)
            // into the slab's own storage in one pass, then commit advances.
            let (slots, meta) = self.slab.unfilled_mut();
            for (slot, descriptor) in slots.iter_mut().take(BATCH).zip(meta.iter_mut()) {
                slot[..STEADY_DATAGRAM.len()].copy_from_slice(STEADY_DATAGRAM);
                *descriptor = (STEADY_DATAGRAM.len(), peer());
            }
        }
        self.slab.commit(BATCH);
        let mut sink = 0u64;
        for view in self.slab.filled_datagrams() {
            sink = sink.wrapping_add(u64::from(view.bytes[0]));
            sink = sink.wrapping_add(u64::from(view.peer.port()));
        }
        sink
    }

    fn send_cycle(&mut self) -> u64 {
        self.send.reset();
        for _ in 0..BATCH {
            let _ = self.send.try_append(STEADY_DATAGRAM, peer());
        }
        // the real poll_drive_send_batch builds its sendmmsg iovecs on the STACK
        // (ArrayVec), NOT a heap Vec — unlike the incumbent's per-send `Vec`.
        let mut iov: ArrayVec<(&[u8], SocketAddr), 64> = ArrayVec::new();
        for span in self.send.spans() {
            iov.push((self.send.slice_for(span), span.peer));
        }
        iov.len() as u64
    }

    fn cycle(&mut self) -> u64 {
        self.recv_cycle().wrapping_add(self.send_cycle())
    }
}

fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("datagram_batch_cycle");

    let mut incumbent = HandRolled::new();
    incumbent.cycle();
    group.bench_function("handrolled_full", |bencher| {
        bencher.iter(|| black_box(incumbent.cycle()));
    });
    group.bench_function("handrolled_recv", |bencher| {
        bencher.iter(|| black_box(incumbent.recv_cycle()));
    });
    group.bench_function("handrolled_send", |bencher| {
        bencher.iter(|| black_box(incumbent.send_cycle()));
    });

    let mut drive = DriveLayer::new();
    drive.cycle();
    group.bench_function("drive_full", |bencher| {
        bencher.iter(|| black_box(drive.cycle()));
    });
    group.bench_function("drive_recv", |bencher| {
        bencher.iter(|| black_box(drive.recv_cycle()));
    });
    group.bench_function("drive_send", |bencher| {
        bencher.iter(|| black_box(drive.send_cycle()));
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
