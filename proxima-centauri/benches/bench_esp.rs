//! The home-turf arm: per-packet AEAD.
//!
//! This is the 80% case — one seal or open per packet, against one handshake
//! per connection. `csr-security::ChildSa` is the named incumbent and its
//! mirrored arms live in `csr/crates/csr-security/benches/bench_esp.rs`.
//!
//! Sizes cover the range principle 11 requires plus the shape that actually
//! matters for a tunnel: 16 B (minimum), 1200 B (QUIC datagram, MTU-safe),
//! 8 KB, 64 KB. Throughput is reported so the ≥55 MB/s per-connection
//! invariant can be read straight off the output.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_centauri::esp::{HEADER_LEN, OVERHEAD};
use proxima_centauri::{ChildSa, Entropy32, EspSpi, Handshake, IkeSpi, Role};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];
const SIZES: [usize; 4] = [16, 1200, 8 * 1024, 64 * 1024];

/// Two SAs that agreed through a real handshake.
fn agreed_pair() -> (ChildSa, ChildSa) {
    let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1));
    let mut responder = Handshake::responder(PSK, IkeSpi::new(2));
    let now = Ticks::from_raw(1);

    let _ = initiator
        .step(&[], Some(Entropy32::new([0x11; 32])), now)
        .unwrap();
    let mut init = [0u8; 92];
    init.copy_from_slice(initiator.outbound());

    let _ = responder
        .step(&init, Some(Entropy32::new([0x22; 32])), now)
        .unwrap();
    let mut reply = [0u8; 92];
    reply.copy_from_slice(responder.outbound());

    let _ = initiator.step(&reply, None, now).unwrap();

    (
        ChildSa::from_session(
            initiator.keys().unwrap(),
            Role::Initiator,
            EspSpi::new(0xAAAA),
        ),
        ChildSa::from_session(
            responder.keys().unwrap(),
            Role::Responder,
            EspSpi::new(0xBBBB),
        ),
    )
}

/// A pair speaking a named suite.
#[cfg(feature = "aead-aes-gcm")]
fn agreed_pair_with(suite: &proxima_centauri::AeadSuite) -> (ChildSa, ChildSa) {
    let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1));
    let mut responder = Handshake::responder(PSK, IkeSpi::new(2));
    let now = Ticks::from_raw(1);

    let _ = initiator
        .step(&[], Some(Entropy32::new([0x11; 32])), now)
        .unwrap();
    let mut init = [0u8; 92];
    init.copy_from_slice(initiator.outbound());
    let _ = responder
        .step(&init, Some(Entropy32::new([0x22; 32])), now)
        .unwrap();
    let mut reply = [0u8; 92];
    reply.copy_from_slice(responder.outbound());
    let _ = initiator.step(&reply, None, now).unwrap();

    (
        ChildSa::from_session_with(
            initiator.keys().unwrap(),
            Role::Initiator,
            EspSpi::new(0xAAAA),
            suite,
        ),
        ChildSa::from_session_with(
            responder.keys().unwrap(),
            Role::Responder,
            EspSpi::new(0xBBBB),
            suite,
        ),
    )
}

/// Which AEAD suite to build with is a per-target decision, and this is the
/// arm that decides it. Both suites are compiled here so the comparison is
/// one binary on one host, not two builds compared across runs.
#[cfg(feature = "aead-aes-gcm")]
fn aead_suites(criterion: &mut Criterion) {
    use proxima_centauri::AeadSuite;

    let mut group = criterion.benchmark_group("esp_suite_seal");

    for (label, suite) in [
        ("chacha20poly1305", AeadSuite::ChaCha20Poly1305),
        ("aes256gcm", AeadSuite::Aes256Gcm),
    ] {
        for size in [1200usize, 8 * 1024] {
            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new(label, size),
                &(suite, size),
                |bencher, &(suite, size)| {
                    let (mut sender, _) = agreed_pair_with(&suite);
                    let mut buffer = vec![0u8; size + OVERHEAD];
                    bencher.iter(|| black_box(sender.seal(black_box(&mut buffer), size).unwrap()));
                },
            );
        }
    }

    group.finish();
}

fn seal(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("esp_seal");

    for size in SIZES {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                let (mut sender, _) = agreed_pair();
                let mut buffer = vec![0u8; size + OVERHEAD];
                bencher.iter(|| {
                    let written = sender
                        .seal(black_box(&mut buffer), black_box(size))
                        .unwrap();
                    black_box(written)
                });
            },
        );
    }

    group.finish();
}

fn open(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("esp_open");

    for size in SIZES {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                // sealing happens in untimed setup: opening advances the replay
                // window, so each iteration needs a fresh sequence number, and
                // sealing inside the timed loop would measure seal+open.
                let (mut sender, mut receiver) = agreed_pair();
                let mut scratch = vec![0u8; size + OVERHEAD];

                bencher.iter_batched(
                    || {
                        let len = sender.seal(&mut scratch, size).expect("seals");
                        scratch[..len].to_vec()
                    },
                    |mut packet| {
                        let opened = receiver.open(black_box(&mut packet)).unwrap();
                        black_box(opened)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn replay_reject(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("esp_replay_reject");

    // the cheap path: a replayed sequence is refused on a bitmap lookup,
    // before any AEAD work. Frequency is low in honest traffic and high under
    // attack, which is exactly why it must not cost a decrypt.
    group.bench_function("rejected_before_decrypt", |bencher| {
        let (mut sender, mut receiver) = agreed_pair();
        let mut buffer = vec![0u8; 1200 + OVERHEAD];
        let len = sender.seal(&mut buffer, 1200).unwrap();
        let _ = receiver.open(&mut buffer[..len]).unwrap();

        let sealed = buffer[..len].to_vec();
        bencher.iter_batched(
            || sealed.clone(),
            |mut replayed| black_box(receiver.open(&mut replayed).is_err()),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("esp_round_trip");

    // seal + open at the QUIC datagram size — the per-packet cost a tunnel
    // actually pays on both ends
    let size = 1200usize;
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_function("1200b", |bencher| {
        let (mut sender, mut receiver) = agreed_pair();
        let mut buffer = vec![0u8; size + OVERHEAD];
        buffer[HEADER_LEN..HEADER_LEN + size].fill(0x5A);

        bencher.iter(|| {
            let len = sender.seal(&mut buffer, size).unwrap();
            let opened = receiver.open(&mut buffer[..len]).unwrap();
            black_box(opened)
        });
    });

    group.finish();
}

#[cfg(feature = "aead-aes-gcm")]
criterion_group!(benches, seal, open, replay_reject, round_trip, aead_suites);

#[cfg(not(feature = "aead-aes-gcm"))]
criterion_group!(benches, seal, open, replay_reject, round_trip);
criterion_main!(benches);
