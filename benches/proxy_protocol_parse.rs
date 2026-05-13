#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! PROXY protocol sans-IO parser microbench. Establishes the per-
//! connection cost we're committing every accept to when a listener
//! has `proxy_protocol = true`.
//!
//! Six cases cover the wire shapes the parser actually sees in the
//! wild plus the rejection paths:
//!
//! - v1 TCP4 — the most common LB output (NLB, haproxy)
//! - v1 TCP6 — IPv6 client through the LB
//! - v1 UNKNOWN — LB couldn't determine, asks upstream to fall back
//! - v2 IPv4 — newer LB stacks default to v2 binary
//! - v2 IPv6 — same, IPv6 case
//! - v2 LOCAL — LB health-check probe (no address translation)
//!
//! Plus a `non_proxy_prefix_rejection` case that measures the
//! "wait, this isn't PROXY at all" decision — relevant if a fall-
//! through-to-application-bytes mode is added later.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::proxy_protocol::{ParseError, ProxyHeader, parse};

const V2_SIGNATURE: &[u8; 12] = b"\r\n\r\n\x00\r\nQUIT\n";

fn v1_tcp4() -> Vec<u8> {
    b"PROXY TCP4 192.168.0.1 192.168.0.11 56324 443\r\n".to_vec()
}

fn v1_tcp6() -> Vec<u8> {
    b"PROXY TCP6 2001:db8::1 2001:db8::2 12345 443\r\n".to_vec()
}

fn v1_unknown() -> Vec<u8> {
    b"PROXY UNKNOWN\r\n".to_vec()
}

fn v2_ipv4() -> Vec<u8> {
    let mut buf = V2_SIGNATURE.to_vec();
    buf.push(0x21); // v2 | PROXY
    buf.push(0x11); // AF_INET | TCP
    buf.extend_from_slice(&12u16.to_be_bytes());
    buf.extend_from_slice(&[1, 2, 3, 4]);
    buf.extend_from_slice(&[5, 6, 7, 8]);
    buf.extend_from_slice(&8080u16.to_be_bytes());
    buf.extend_from_slice(&443u16.to_be_bytes());
    buf
}

fn v2_ipv6() -> Vec<u8> {
    let mut buf = V2_SIGNATURE.to_vec();
    buf.push(0x21); // v2 | PROXY
    buf.push(0x21); // AF_INET6 | TCP
    buf.extend_from_slice(&36u16.to_be_bytes());
    buf.extend_from_slice(&[0u8; 16]); // src ipv6
    buf.extend_from_slice(&[0u8; 16]); // dst ipv6
    buf.extend_from_slice(&8080u16.to_be_bytes());
    buf.extend_from_slice(&443u16.to_be_bytes());
    buf
}

fn v2_local() -> Vec<u8> {
    let mut buf = V2_SIGNATURE.to_vec();
    buf.push(0x20); // v2 | LOCAL
    buf.push(0x00);
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf
}

fn non_proxy_prefix() -> Vec<u8> {
    b"GET / HTTP/1.1\r\n".to_vec()
}

fn happy_path(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxy_protocol_parse");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let cases: &[(&str, Vec<u8>)] = &[
        ("v1_tcp4", v1_tcp4()),
        ("v1_tcp6", v1_tcp6()),
        ("v1_unknown", v1_unknown()),
        ("v2_ipv4", v2_ipv4()),
        ("v2_ipv6", v2_ipv6()),
        ("v2_local", v2_local()),
    ];
    for (label, payload) in cases {
        group.bench_function(*label, |bencher| {
            bencher.iter(|| {
                let result = parse(std::hint::black_box(payload));
                let (header, consumed) = result.expect("parse ok");
                std::hint::black_box(consumed);
                std::hint::black_box(matches!(header, ProxyHeader::Tcp { .. }));
            });
        });
    }
    group.finish();
}

fn non_proxy_rejection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxy_protocol_parse_reject");
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(1));
    let payload = non_proxy_prefix();
    group.bench_function("not_proxy", |bencher| {
        bencher.iter(|| {
            let result = parse(std::hint::black_box(&payload));
            std::hint::black_box(matches!(result, Err(ParseError::NotProxyProtocol)));
        });
    });
    group.finish();
}

criterion_group!(benches, happy_path, non_proxy_rejection);
criterion_main!(benches);
