// bench fixtures legitimately fail-fast on encoder errors.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C5 — HKDF-Expand-Label + initial-secret derivation bench arms.
//!
//! **Note on home-turf comparison**: `quinn-proto`'s initial-secret
//! derivation goes through `quinn-proto::crypto::ring::initial_keys`
//! (or the `aws-lc-rs` variant) — both are crate-private and use C-backed
//! crypto (ring / aws-lc-rs). Direct head-to-head requires re-exporting
//! their internals or shimming through their public Cipher trait, neither
//! of which is clean for a bench. The C5 bench measures proxima's
//! standalone derivation cost; the C10 (TLS handshake) bench will land
//! the apples-to-apples comparison.
//!
//! Arms:
//!
//! - **derive_initial_key_pair** — full RFC 9001 §5.2 derivation
//!   (HKDF-Extract + 8 HKDF-Expand-Label calls).
//! - **expand_label** — a single HKDF-Expand-Label call (the hot path
//!   inside key updates and per-direction key derivation).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::crypto::{
    expand_label::{SHA256_OUTPUT_LEN, expand_label_sha256},
    initial_keys::{self, QUIC_KEY_LEN},
};

const RFC_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

fn bench_derive_initial_key_pair(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_derive_initial_key_pair");
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let pair = initial_keys::derive(std::hint::black_box(&RFC_DCID)).expect("derive");
            std::hint::black_box(pair);
        });
    });
    group.finish();
}

fn bench_expand_label(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_expand_label_single");
    group.throughput(Throughput::Bytes(QUIC_KEY_LEN as u64));
    let secret = [0xab; SHA256_OUTPUT_LEN];
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut output = [0u8; QUIC_KEY_LEN];
        bencher.iter(|| {
            expand_label_sha256(
                std::hint::black_box(&secret),
                std::hint::black_box(b"quic key"),
                std::hint::black_box(b""),
                std::hint::black_box(&mut output),
            )
            .expect("expand");
            std::hint::black_box(output);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_derive_initial_key_pair, bench_expand_label,);
criterion_main!(benches);
