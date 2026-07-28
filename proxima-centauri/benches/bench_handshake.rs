//! Micro-benches for the SA_INIT handshake and the entropy sources.
//!
//! # What these arms are, and what they are not
//!
//! A handshake runs **once per connection**. It is the connection-setup path,
//! not the per-packet path. Under `/disciplined-component` point 13 the
//! home-turf arm must engage the incumbent's design point at realistic
//! frequency — for a security stack that is per-packet AEAD, which lives in
//! `ChildSa` and is not built here yet. So these numbers bound setup cost and
//! locate where it goes; **they do not support a "meet or beat" verdict for
//! the component**. That verdict needs the AEAD arm.
//!
//! Setup cost is still worth bounding: short-lived connections amortise it
//! over very little traffic, and a relay that churns connections pays it
//! constantly.
//!
//! The `with_entropy_source` arms draw through a real source rather than
//! handing over a pre-made value, because the incumbent generates its own
//! randomness inside `initiate`/`respond` — comparing an injected constant
//! against a `getrandom` call would flatter this side for free.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use proxima_centauri::{CounterDrbg, Entropy32, EntropyCell, Handshake, IkeSpi, hash};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];
const INITIATOR_SPI: IkeSpi = IkeSpi::new(0x0102_0304_0506_0708);
const RESPONDER_SPI: IkeSpi = IkeSpi::new(0x1112_1314_1516_1718);
const INITIATOR_SEED: [u8; 32] = [0x11; 32];
const RESPONDER_SEED: [u8; 32] = [0x22; 32];

fn now() -> Ticks {
    Ticks::from_raw(1_000)
}

/// One initiator INIT message, for arms that need a realistic input.
fn init_message() -> [u8; 92] {
    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
    let _ = initiator
        .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
        .expect("initiator sends");
    let mut message = [0u8; 92];
    message.copy_from_slice(initiator.outbound());
    message
}

/// The responder's reply, for the initiator-completes arm.
fn response_message() -> [u8; 92] {
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
    let _ = responder
        .step(&init_message(), Some(Entropy32::new(RESPONDER_SEED)), now())
        .expect("responder replies");
    let mut message = [0u8; 92];
    message.copy_from_slice(responder.outbound());
    message
}

fn handshake_steps(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("handshake_step");

    // one X25519 base-point mul + one message write
    group.bench_function("initiator_send_init", |bencher| {
        bencher.iter(|| {
            let mut initiator = Handshake::initiator(black_box(PSK), INITIATOR_SPI);
            let progress = initiator
                .step(&[], Some(Entropy32::new(black_box(INITIATOR_SEED))), now())
                .expect("sends");
            black_box((progress, initiator.outbound().len()))
        });
    });

    // the expensive side: parse + base-point mul + shared-secret mul + five
    // BLAKE3 derivations + message write
    let init = init_message();
    group.bench_function("responder_receive_init", |bencher| {
        bencher.iter(|| {
            let mut responder = Handshake::responder(black_box(PSK), RESPONDER_SPI);
            let progress = responder
                .step(
                    black_box(&init),
                    Some(Entropy32::new(RESPONDER_SEED)),
                    now(),
                )
                .expect("replies");
            black_box(progress)
        });
    });

    // parse + shared-secret mul + five derivations, no message written
    let response = response_message();
    group.bench_function("initiator_receive_response", |bencher| {
        bencher.iter(|| {
            let mut initiator = Handshake::initiator(black_box(PSK), INITIATOR_SPI);
            let _ = initiator
                .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
                .expect("sends");
            let progress = initiator
                .step(black_box(&response), None, now())
                .expect("completes");
            black_box(progress)
        });
    });

    group.finish();
}

fn full_handshake(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("handshake_full");

    // both sides, entropy handed in — the floor, excluding randomness cost
    group.bench_function("both_sides_injected_entropy", |bencher| {
        bencher.iter(|| {
            let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
            let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

            let _ = initiator
                .step(&[], Some(Entropy32::new(black_box(INITIATOR_SEED))), now())
                .expect("sends");
            let mut init = [0u8; 92];
            init.copy_from_slice(initiator.outbound());

            let _ = responder
                .step(
                    &init,
                    Some(Entropy32::new(black_box(RESPONDER_SEED))),
                    now(),
                )
                .expect("replies");
            let mut reply = [0u8; 92];
            reply.copy_from_slice(responder.outbound());

            let progress = initiator.step(&reply, None, now()).expect("completes");
            black_box(progress)
        });
    });

    // both sides drawing from a real DRBG — the honest comparison shape,
    // since the incumbent calls getrandom inside its own steps
    group.bench_function("both_sides_with_entropy_source", |bencher| {
        bencher.iter(|| {
            let initiator_source = CounterDrbg::new(black_box(INITIATOR_SEED));
            let responder_source = CounterDrbg::new(black_box(RESPONDER_SEED));

            let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
            let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

            let _ = initiator
                .step(&[], Some(initiator_source.draw().expect("draw")), now())
                .expect("sends");
            let mut init = [0u8; 92];
            init.copy_from_slice(initiator.outbound());

            let _ = responder
                .step(&init, Some(responder_source.draw().expect("draw")), now())
                .expect("replies");
            let mut reply = [0u8; 92];
            reply.copy_from_slice(responder.outbound());

            let progress = initiator.step(&reply, None, now()).expect("completes");
            black_box(progress)
        });
    });

    group.finish();
}

fn auth_exchange(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("auth");

    // both sides through SA_INIT, ready to authenticate
    let established = || {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI)
            .with_identity(b"peer-a")
            .unwrap();
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI)
            .with_identity(b"peer-b")
            .unwrap();
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let mut init = [0u8; 92];
        init.copy_from_slice(initiator.outbound());
        let _ = responder
            .step(&init, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();
        let mut reply = [0u8; 92];
        reply.copy_from_slice(responder.outbound());
        let _ = initiator.step(&reply, None, now()).unwrap();
        (initiator, responder)
    };

    // one keyed hash over header+identity, plus the message write
    group.bench_function("send_auth", |bencher| {
        bencher.iter_batched(
            &established,
            |(mut initiator, _)| black_box(initiator.send_auth().unwrap()),
            BatchSize::SmallInput,
        );
    });

    // parse, recompute the MAC, constant-time compare
    group.bench_function("verify_auth", |bencher| {
        bencher.iter_batched(
            || {
                let (mut initiator, responder) = established();
                let _ = initiator.send_auth().unwrap();
                let mut message = [0u8; 128];
                let len = initiator.outbound().len();
                message[..len].copy_from_slice(initiator.outbound());
                (responder, message, len)
            },
            |(mut responder, message, len)| {
                black_box(responder.step(&message[..len], None, now()).unwrap())
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn rekey_exchange(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rekey");

    let authenticated = || {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI)
            .with_identity(b"peer-a")
            .unwrap();
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI)
            .with_identity(b"peer-b")
            .unwrap();
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let mut init = [0u8; 92];
        init.copy_from_slice(initiator.outbound());
        let _ = responder
            .step(&init, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();
        let mut reply = [0u8; 92];
        reply.copy_from_slice(responder.outbound());
        let _ = initiator.step(&reply, None, now()).unwrap();

        let _ = initiator.send_auth().unwrap();
        let mut forward = [0u8; 128];
        let forward_len = initiator.outbound().len();
        forward[..forward_len].copy_from_slice(initiator.outbound());
        let _ = responder
            .step(&forward[..forward_len], None, now())
            .unwrap();
        let _ = responder.send_auth().unwrap();
        let mut back = [0u8; 128];
        let back_len = responder.outbound().len();
        back[..back_len].copy_from_slice(responder.outbound());
        let _ = initiator.step(&back[..back_len], None, now()).unwrap();

        (initiator, responder)
    };

    // one base-point mul plus the message write
    group.bench_function("send_rekey", |bencher| {
        bencher.iter_batched(
            &authenticated,
            |(mut initiator, _)| {
                black_box(
                    initiator
                        .send_rekey(Some(Entropy32::new([0x31; 32])), now())
                        .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });

    // the whole exchange: two curve muls a side, six derivations, both messages
    group.bench_function("full_rekey", |bencher| {
        bencher.iter_batched(
            &authenticated,
            |(mut initiator, mut responder)| {
                let _ = initiator
                    .send_rekey(Some(Entropy32::new([0x31; 32])), now())
                    .unwrap();
                let mut request = [0u8; 92];
                request.copy_from_slice(initiator.outbound());
                let _ = responder
                    .step(&request, Some(Entropy32::new([0x32; 32])), now())
                    .unwrap();
                let mut reply = [0u8; 92];
                reply.copy_from_slice(responder.outbound());
                black_box(initiator.step(&reply, None, now()).unwrap())
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn entropy_sources(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("entropy_draw");

    let drbg = CounterDrbg::new([0x33; 32]);
    group.bench_function("counter_drbg", |bencher| {
        bencher.iter(|| black_box(drbg.draw().expect("within counter range")));
    });

    // fill + claim, since a take-once cell cannot be drawn twice
    let cell = EntropyCell::new();
    group.bench_function("cell_set_then_draw", |bencher| {
        bencher.iter(|| {
            cell.set(black_box([0x44; 32])).expect("cell is empty");
            black_box(cell.draw().expect("cell is full"))
        });
    });

    group.finish();
}

fn primitives(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("primitive");

    // the KDF is called six times per handshake side; locate its share
    let material = [0x55u8; 128];
    group.bench_function("blake3_derive_key_128b", |bencher| {
        bencher.iter(|| {
            black_box(hash::derive_key(
                "proxima-centauri-bench",
                black_box(&material),
            ))
        });
    });

    let key = [0x66u8; 32];
    group.bench_function("blake3_keyed_hash_32b", |bencher| {
        bencher.iter(|| black_box(hash::keyed_hash(&key, black_box(&[0x77u8; 32]))));
    });

    group.finish();
}

criterion_group!(
    benches,
    handshake_steps,
    full_handshake,
    auth_exchange,
    rekey_exchange,
    entropy_sources,
    primitives
);
criterion_main!(benches);
