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

//! C6 — AEAD packet protection bench arms.
//!
//! Compares both AEAD MUST-implement algorithms at the QUIC datagram
//! shapes that drive perf (1200-byte initial MTU + 1452-byte common MTU).
//! Direct head-to-head vs `quinn-proto::crypto::ring::aead` deferred to
//! C10 (the crypto::Cipher trait is the natural comparison surface and
//! it lives in quinn's std-tier facade).

use aws_lc_rs::aead::{AES_128_GCM, CHACHA20_POLY1305, LessSafeKey, Nonce as AwsNonce, UnboundKey};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::crypto::aead::{
    self, AES_128_GCM_KEY_LEN, CHACHA20_POLY1305_KEY_LEN, NONCE_LEN, TAG_LEN,
};

const PAYLOAD_SIZES: &[usize] = &[16, 1200, 1452, 8192];

// Compare against `aws_lc_rs::aead` — this is what quinn-proto uses
// under the hood via rustls's `aws_lc_rs` crypto provider. Our
// `proxima_protocols::quic::crypto::aead` wraps the RustCrypto `aes_gcm`
// crate (pure Rust). The real per-packet hot-path comparison is
// pure-Rust vs C/ASM with AES-NI (x86) or ARM Crypto Extensions
// (aarch64) — at the same QUIC datagram sizes that drive throughput.

fn bench_aes_128_gcm_encrypt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_aes_128_gcm_encrypt");
    let key = [0xab; AES_128_GCM_KEY_LEN];
    let nonce = [0xcd; NONCE_LEN];
    let aad = [0xef; 32];
    for &size in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", size),
            &size,
            |bencher, &size| {
                let mut buffer = vec![0u8; size];
                let mut tag = [0u8; TAG_LEN];
                bencher.iter(|| {
                    aead::aes_128_gcm_encrypt(
                        std::hint::black_box(&key),
                        std::hint::black_box(&nonce),
                        std::hint::black_box(&aad),
                        std::hint::black_box(&mut buffer),
                        std::hint::black_box(&mut tag),
                    )
                    .expect("encrypt");
                    std::hint::black_box(&tag);
                });
            },
        );
        // aws_lc_rs compare arm: same key/nonce/aad, same workload.
        // Using LessSafeKey to permit nonce reuse across iterations
        // (this is a bench, not a security boundary).
        group.bench_with_input(
            BenchmarkId::new("aws_lc_rs", size),
            &size,
            |bencher, &size| {
                let unbound = UnboundKey::new(&AES_128_GCM, &key).expect("key");
                let lsk = LessSafeKey::new(unbound);
                let mut buffer = vec![0u8; size + AES_128_GCM.tag_len()];
                bencher.iter(|| {
                    buffer.truncate(size);
                    let nonce_obj = AwsNonce::assume_unique_for_key(nonce);
                    lsk.seal_in_place_append_tag(
                        nonce_obj,
                        aws_lc_rs::aead::Aad::from(std::hint::black_box(&aad)),
                        std::hint::black_box(&mut buffer),
                    )
                    .expect("encrypt");
                    std::hint::black_box(&buffer);
                });
            },
        );
    }
    group.finish();
}

fn bench_aes_128_gcm_decrypt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_aes_128_gcm_decrypt");
    let key = [0xab; AES_128_GCM_KEY_LEN];
    let nonce = [0xcd; NONCE_LEN];
    let aad = [0xef; 32];
    for &size in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(size as u64));
        // pre-encrypt so the decrypt path is the only thing measured
        let mut buffer = vec![0u8; size];
        let mut tag = [0u8; TAG_LEN];
        aead::aes_128_gcm_encrypt(&key, &nonce, &aad, &mut buffer, &mut tag).unwrap();
        let ciphertext = buffer.clone();
        let saved_tag = tag;
        // build aws-lc-rs ciphertext with the SAME parameters so the
        // decrypt benches each library's own ciphertext (both verify
        // the tag matches; both must succeed).
        let aws_unbound_enc = UnboundKey::new(&AES_128_GCM, &key).expect("key");
        let aws_lsk_enc = LessSafeKey::new(aws_unbound_enc);
        let mut aws_ct = vec![0u8; size];
        let aws_nonce_enc = AwsNonce::assume_unique_for_key(nonce);
        aws_lsk_enc
            .seal_in_place_append_tag(aws_nonce_enc, aws_lc_rs::aead::Aad::from(aad), &mut aws_ct)
            .expect("aws seal");
        let aws_ct_with_tag = aws_ct.clone();
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", size),
            &size,
            |bencher, _size| {
                bencher.iter(|| {
                    let mut work = ciphertext.clone();
                    aead::aes_128_gcm_decrypt(
                        std::hint::black_box(&key),
                        std::hint::black_box(&nonce),
                        std::hint::black_box(&aad),
                        std::hint::black_box(&mut work),
                        std::hint::black_box(&saved_tag),
                    )
                    .expect("decrypt");
                    std::hint::black_box(work);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("aws_lc_rs", size),
            &size,
            |bencher, _size| {
                let unbound = UnboundKey::new(&AES_128_GCM, &key).expect("key");
                let lsk = LessSafeKey::new(unbound);
                bencher.iter(|| {
                    let mut work = aws_ct_with_tag.clone();
                    let nonce_obj = AwsNonce::assume_unique_for_key(nonce);
                    let _ = lsk
                        .open_in_place(
                            nonce_obj,
                            aws_lc_rs::aead::Aad::from(std::hint::black_box(&aad)),
                            std::hint::black_box(&mut work),
                        )
                        .expect("decrypt");
                    std::hint::black_box(work);
                });
            },
        );
    }
    group.finish();
}

fn bench_chacha20_poly1305_encrypt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_chacha20_poly1305_encrypt");
    let key = [0x12; CHACHA20_POLY1305_KEY_LEN];
    let nonce = [0x34; NONCE_LEN];
    let aad = [0x56; 32];
    for &size in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", size),
            &size,
            |bencher, &size| {
                let mut buffer = vec![0u8; size];
                let mut tag = [0u8; TAG_LEN];
                bencher.iter(|| {
                    aead::chacha20_poly1305_encrypt(
                        std::hint::black_box(&key),
                        std::hint::black_box(&nonce),
                        std::hint::black_box(&aad),
                        std::hint::black_box(&mut buffer),
                        std::hint::black_box(&mut tag),
                    )
                    .expect("encrypt");
                    std::hint::black_box(&tag);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("aws_lc_rs", size),
            &size,
            |bencher, &size| {
                let unbound = UnboundKey::new(&CHACHA20_POLY1305, &key).expect("key");
                let lsk = LessSafeKey::new(unbound);
                let mut buffer = vec![0u8; size + CHACHA20_POLY1305.tag_len()];
                bencher.iter(|| {
                    buffer.truncate(size);
                    let nonce_obj = AwsNonce::assume_unique_for_key(nonce);
                    lsk.seal_in_place_append_tag(
                        nonce_obj,
                        aws_lc_rs::aead::Aad::from(std::hint::black_box(&aad)),
                        std::hint::black_box(&mut buffer),
                    )
                    .expect("encrypt");
                    std::hint::black_box(&buffer);
                });
            },
        );
    }
    group.finish();
}

fn bench_build_nonce(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_build_nonce");
    let iv = [0xab; NONCE_LEN];
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut pn = 0u64;
        bencher.iter(|| {
            let nonce = aead::build_nonce(std::hint::black_box(&iv), std::hint::black_box(pn));
            std::hint::black_box(nonce);
            pn = pn.wrapping_add(1);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_build_nonce,
    bench_aes_128_gcm_encrypt,
    bench_aes_128_gcm_decrypt,
    bench_chacha20_poly1305_encrypt,
);
criterion_main!(benches);
