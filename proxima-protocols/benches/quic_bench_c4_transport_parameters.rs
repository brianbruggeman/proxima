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

//! C4 — transport parameters codec bench arms.
//!
//! **Note on home-turf comparison**: `quinn-proto::TransportParameters`
//! is `pub` but its decoder API requires a `&mut B: Buf` and is intertwined
//! with crypto setup (`TransportParameters::read(&mut buf, side)`). The
//! C4 bench measures proxima's standalone parse/encode of a typical
//! full transport-parameters extension blob; the comparison vs quinn
//! is left to the C10 (TLS handshake) bench when the full handshake
//! path lands.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::transport_parameters::{
    self, PREFERRED_ADDRESS_IPV4_LEN, PREFERRED_ADDRESS_IPV6_LEN, PreferredAddress,
    STATELESS_RESET_TOKEN_LEN, TransportParameters,
};

fn build_full_params() -> (Vec<u8>, TransportParameters<'static>) {
    static TOKEN: [u8; STATELESS_RESET_TOKEN_LEN] = [0x42; STATELESS_RESET_TOKEN_LEN];
    static DCID: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    static SCID: [u8; 4] = [9, 10, 11, 12];
    let original = TransportParameters {
        original_destination_connection_id: Some(&DCID),
        max_idle_timeout_ms: Some(30_000),
        stateless_reset_token: Some(&TOKEN),
        max_udp_payload_size: Some(1452),
        initial_max_data: Some(16_777_216),
        initial_max_stream_data_bidi_local: Some(1_048_576),
        initial_max_stream_data_bidi_remote: Some(1_048_576),
        initial_max_stream_data_uni: Some(1_048_576),
        initial_max_streams_bidi: Some(100),
        initial_max_streams_uni: Some(100),
        ack_delay_exponent: Some(3),
        max_ack_delay_ms: Some(25),
        disable_active_migration: true,
        preferred_address: None,
        active_connection_id_limit: Some(4),
        initial_source_connection_id: Some(&SCID),
        retry_source_connection_id: None,
        max_datagram_frame_size: Some(1200),
        initial_max_path_id: Some(4),
    };
    let mut buffer = vec![0u8; 256];
    let written = original.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    (buffer, original)
}

fn build_params_with_preferred_address() -> Vec<u8> {
    static TOKEN: [u8; STATELESS_RESET_TOKEN_LEN] = [0x99; STATELESS_RESET_TOKEN_LEN];
    static CID: [u8; 4] = [0xa, 0xb, 0xc, 0xd];
    let ipv4: [u8; PREFERRED_ADDRESS_IPV4_LEN] = [10, 0, 0, 1, 0x01, 0xbb];
    let ipv6: [u8; PREFERRED_ADDRESS_IPV6_LEN] = [
        0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x01, 0xbb,
    ];
    let preferred = PreferredAddress {
        ipv4,
        ipv6,
        connection_id: &CID,
        stateless_reset_token: &TOKEN,
    };
    let params = TransportParameters {
        preferred_address: Some(preferred),
        initial_max_data: Some(1_000_000),
        ..Default::default()
    };
    let mut buffer = vec![0u8; 128];
    let written = params.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    buffer
}

fn bench_parse_full(criterion: &mut Criterion) {
    let (bytes, _) = build_full_params();
    let mut group = criterion.benchmark_group("c4_parse_full_params");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let parsed = transport_parameters::parse(std::hint::black_box(&bytes)).expect("parse");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

fn bench_parse_with_preferred_address(criterion: &mut Criterion) {
    let bytes = build_params_with_preferred_address();
    let mut group = criterion.benchmark_group("c4_parse_preferred_address");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let parsed = transport_parameters::parse(std::hint::black_box(&bytes)).expect("parse");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

fn bench_encode_full(criterion: &mut Criterion) {
    let (_, params) = build_full_params();
    let mut group = criterion.benchmark_group("c4_encode_full_params");
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut buffer = vec![0u8; 256];
        bencher.iter(|| {
            let written = params
                .encode(std::hint::black_box(&mut buffer))
                .expect("encode");
            std::hint::black_box(written);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_full,
    bench_parse_with_preferred_address,
    bench_encode_full,
);
criterion_main!(benches);
