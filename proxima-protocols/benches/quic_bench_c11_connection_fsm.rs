//! C11 — connection state machine bench arms.
//!
//! Measures the per-call cost of the FSM dispatcher entry points
//! on a client connection driven by a `MockTlsProvider`:
//!
//! - `new_client` — fresh-connection setup including initial-keys
//!   derive + first ClientHello pump.
//! - `poll_transmit_initial` — building + protecting a 1200-byte
//!   Initial datagram (composes C2 header encode + C10 protect).
//! - `close_from_initial` — caller-initiated close from Initial state,
//!   exercising `core::mem::replace` based variant transition.
//! - `handle_timeout_continue` — single timeout tick with no deadline
//!   reached; the per-tick cost on the steady-state event loop.
//! - `handle_timeout_drained` — final tick that transitions Draining
//!   → Closed (the cleanup-path cost).
//!
//! No comparable incumbent benches: `quinn-proto::Connection` requires
//! a full TLS context + transport-params + actual cipher backend to
//! construct, putting it outside the apples-to-apples envelope for
//! an FSM-only micro-bench. The C11 incumbent comparison lands in
//! C11.5 / C29 when the full handshake + bulk-stream round-trip is
//! wireable.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_protocols::quic::connection::Connection;
use proxima_protocols::quic::time::{Duration, Instant};
use proxima_protocols::quic::tls::Epoch;
use proxima_protocols::quic::tls::mock::{MockStep, MockTlsProvider};

const RFC_9001_A1_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
const LOCAL_SCID: [u8; 8] = [0xc0, 0xff, 0xee, 0xba, 0xbe, 0x12, 0x34, 0x56];
const CLIENT_HELLO: [u8; 16] = [0xDE; 16];
const ORIGIN: Instant = Instant::from_micros(1_000_000);

fn script() -> alloc::vec::Vec<MockStep> {
    alloc::vec![MockStep::EmitHandshakeBytes {
        epoch: Epoch::Initial,
        bytes: CLIENT_HELLO.to_vec(),
    }]
}

extern crate alloc;

fn bench_new_client(criterion: &mut Criterion) {
    criterion.bench_function("c11_new_client", |bencher| {
        bencher.iter(|| {
            let config = MockTlsProvider::script_client(script());
            black_box(
                Connection::<MockTlsProvider>::new_client(
                    config,
                    b"",
                    black_box(&RFC_9001_A1_DCID),
                    black_box(&LOCAL_SCID),
                    ORIGIN,
                )
                .expect("new_client"),
            )
        });
    });
}

fn bench_poll_transmit_initial(criterion: &mut Criterion) {
    criterion.bench_function("c11_poll_transmit_initial_1200", |bencher| {
        let mut buf = [0u8; 1500];
        bencher.iter_batched(
            || {
                let config = MockTlsProvider::script_client(script());
                Connection::<MockTlsProvider>::new_client(
                    config,
                    b"",
                    &RFC_9001_A1_DCID,
                    &LOCAL_SCID,
                    ORIGIN,
                )
                .expect("new_client")
            },
            |mut connection| {
                let write = connection
                    .poll_transmit(ORIGIN, &mut buf)
                    .expect("poll_transmit ok")
                    .expect("first send");
                black_box(write);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_close_from_initial(criterion: &mut Criterion) {
    criterion.bench_function("c11_close_from_initial", |bencher| {
        bencher.iter_batched(
            || {
                let config = MockTlsProvider::script_client(script());
                Connection::<MockTlsProvider>::new_client(
                    config,
                    b"",
                    &RFC_9001_A1_DCID,
                    &LOCAL_SCID,
                    ORIGIN,
                )
                .expect("new_client")
            },
            |mut connection| {
                connection
                    .close(black_box(0), black_box(b"bye"))
                    .expect("close");
                black_box(connection);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_handle_timeout_continue(criterion: &mut Criterion) {
    criterion.bench_function("c11_handle_timeout_continue", |bencher| {
        let config = MockTlsProvider::script_client(script());
        let mut connection = Connection::<MockTlsProvider>::new_client(
            config,
            b"",
            &RFC_9001_A1_DCID,
            &LOCAL_SCID,
            ORIGIN,
        )
        .expect("new_client");
        let tick = ORIGIN + Duration::from_micros(100);
        bencher.iter(|| {
            // Drive forward in microsecond increments; each call should
            // return Continue (idle deadline is 30s out).
            let _ = black_box(connection.handle_timeout(tick).expect("ok"));
        });
    });
}

fn bench_handle_timeout_drained(criterion: &mut Criterion) {
    criterion.bench_function("c11_handle_timeout_close_then_drain", |bencher| {
        bencher.iter_batched(
            || {
                let config = MockTlsProvider::script_client(script());
                let mut connection = Connection::<MockTlsProvider>::new_client(
                    config,
                    b"",
                    &RFC_9001_A1_DCID,
                    &LOCAL_SCID,
                    ORIGIN,
                )
                .expect("new_client");
                connection.close(0, b"").expect("close");
                connection
                    .handle_timeout(Instant::from_micros(10_000_000))
                    .expect("→Draining");
                connection
            },
            |mut connection| {
                let outcome = connection
                    .handle_timeout(black_box(Instant::from_micros(100_000_000)))
                    .expect("→Closed");
                black_box(outcome);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_new_client,
    bench_poll_transmit_initial,
    bench_close_from_initial,
    bench_handle_timeout_continue,
    bench_handle_timeout_drained,
);
criterion_main!(benches);
