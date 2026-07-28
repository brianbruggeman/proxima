//! End-to-end: a whole tunnel session, handshake through teardown.
//!
//! Gate point 7 asks for an e2e arm once something composes the component.
//! Nothing in the workspace consumes `proxima-centauri` yet — so this bench IS
//! the composition. It stands the two halves up together and drives a session
//! the way a relay would: negotiate, then push packets until the connection
//! closes. That is the shape the crate exists to serve, and it is the only arm
//! where the handshake's cost is weighed against the traffic it protects
//! rather than measured in isolation.
//!
//! The session-length sweep is the point. Setup is a fixed cost amortised over
//! however many packets follow it, so "is the handshake expensive?" has no
//! answer independent of session length — these arms give it one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_centauri::esp::{HEADER_LEN, OVERHEAD};
use proxima_centauri::{ChildSa, CounterDrbg, EspSpi, Handshake, IkeSpi, Progress, Role};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];
const INITIATOR_SPI: IkeSpi = IkeSpi::new(0x0102_0304_0506_0708);
const RESPONDER_SPI: IkeSpi = IkeSpi::new(0x1112_1314_1516_1718);

/// Payload of a QUIC-sized datagram — the packet a tunnel actually carries.
const DATAGRAM: usize = 1200;

/// A complete session: handshake, then `packets` datagrams each way, with both
/// peers driven from real entropy sources.
fn tunnel_session(packets: usize) -> usize {
    let now = Ticks::from_raw(1_000);
    let initiator_entropy = CounterDrbg::new([0x11; 32]);
    let responder_entropy = CounterDrbg::new([0x22; 32]);

    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

    let progress = initiator
        .step(&[], Some(initiator_entropy.draw().unwrap()), now)
        .unwrap();
    debug_assert_eq!(progress, Progress::Advanced);
    let mut init_message = [0u8; 92];
    init_message.copy_from_slice(initiator.outbound());

    let _ = responder
        .step(&init_message, Some(responder_entropy.draw().unwrap()), now)
        .unwrap();
    let mut response = [0u8; 92];
    response.copy_from_slice(responder.outbound());

    let _ = initiator.step(&response, None, now).unwrap();

    let mut client = ChildSa::from_session(
        initiator.keys().expect("established"),
        Role::Initiator,
        EspSpi::new(0xAAAA),
    );
    let mut server = ChildSa::from_session(
        responder.keys().expect("established"),
        Role::Responder,
        EspSpi::new(0xBBBB),
    );

    let mut buffer = [0u8; DATAGRAM + OVERHEAD];
    let mut delivered = 0usize;

    for _ in 0..packets {
        // client -> server
        buffer[HEADER_LEN..HEADER_LEN + DATAGRAM].fill(0x5A);
        let len = client.seal(&mut buffer, DATAGRAM).unwrap();
        delivered += server.open(&mut buffer[..len]).unwrap();

        // server -> client, so both directions and both replay windows run
        buffer[HEADER_LEN..HEADER_LEN + DATAGRAM].fill(0xA5);
        let len = server.seal(&mut buffer, DATAGRAM).unwrap();
        delivered += client.open(&mut buffer[..len]).unwrap();
    }

    delivered
}

fn sessions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("e2e_tunnel_session");

    // 0 isolates the handshake inside a session; the rest show it amortising.
    // A relay that churns connections lives at the low end; a long-lived
    // tunnel lives at the high end, and they are different verdicts.
    for packets in [0usize, 1, 10, 100, 1_000] {
        // bytes actually moved through the tunnel, both directions
        group.throughput(Throughput::Bytes((packets * DATAGRAM * 2) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(packets),
            &packets,
            |bencher, &packets| {
                bencher.iter(|| black_box(tunnel_session(black_box(packets))));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, sessions);
criterion_main!(benches);
