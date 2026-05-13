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

//! C7 — header-protection bench arms.
//!
//! Both AEAD families (AES-128 and ChaCha20-Poly1305) plus the
//! `apply_mask` step. Direct head-to-head vs `quinn-proto::crypto::HeaderKey`
//! deferred to C10 (quinn's HeaderKey trait lives in the std-tier facade).

use aws_lc_rs::cipher::{AES_128, EncryptingKey, UnboundCipherKey};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::crypto::header_protection::{self, MASK_LEN, SAMPLE_LEN};

// Compare against aws_lc_rs's AES-128-ECB primitive — this is what
// quinn-proto's HeaderKey uses under the hood via rustls + aws-lc-rs.
// QUIC header protection (RFC 9001 §5.4.3) is "encrypt the 16-byte
// sample with the HP key in ECB mode, take the first 5 bytes as mask"
// — i.e. one AES-128 block encryption per packet. The compare exposes
// our `aes_gcm`/RustCrypto-based wrapper vs aws-lc-rs's C/ASM
// (AES-NI on x86, ARM Crypto Extensions on aarch64).

fn bench_aes_128_mask(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_aes_128_mask");
    let hp_key = [0xabu8; 16];
    let sample = [0xcdu8; SAMPLE_LEN];
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let mask = header_protection::aes_128_mask(
                std::hint::black_box(&hp_key),
                std::hint::black_box(&sample),
            );
            std::hint::black_box(mask);
        });
    });
    // aws-lc-rs compare: AES-128-ECB encryption of the 16-byte sample.
    // EncryptingKey is constructed once and reused across iterations
    // (the realistic hot-path shape — keys are per-connection, not
    // per-packet). The encrypt call is what runs once per packet.
    group.bench_function("aws_lc_rs", |bencher| {
        let unbound = UnboundCipherKey::new(&AES_128, &hp_key).expect("key");
        let key = EncryptingKey::ecb(unbound).expect("ecb");
        let mut block = [0u8; SAMPLE_LEN];
        bencher.iter(|| {
            block.copy_from_slice(&sample);
            let _ctx = key
                .encrypt(std::hint::black_box(&mut block))
                .expect("encrypt");
            // First 5 bytes are the mask per RFC 9001 §5.4.3
            std::hint::black_box(&block[..MASK_LEN]);
        });
    });
    group.finish();
}

fn bench_chacha20_mask(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_chacha20_mask");
    let hp_key = [0x12u8; 32];
    let sample = [0x34u8; SAMPLE_LEN];
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let mask = header_protection::chacha20_mask(
                std::hint::black_box(&hp_key),
                std::hint::black_box(&sample),
            );
            std::hint::black_box(mask);
        });
    });
    group.finish();
}

fn bench_apply_mask(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_apply_mask");
    let mask = [0x42u8; MASK_LEN];
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut first = 0u8;
        let mut pn = [0u8; 4];
        bencher.iter(|| {
            header_protection::apply_mask(
                std::hint::black_box(&mut first),
                std::hint::black_box(&mut pn),
                std::hint::black_box(&mask),
                true,
            )
            .expect("apply");
            std::hint::black_box(&first);
            std::hint::black_box(&pn);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_aes_128_mask,
    bench_chacha20_mask,
    bench_apply_mask,
);
criterion_main!(benches);
