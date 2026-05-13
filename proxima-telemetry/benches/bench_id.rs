#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::id::parse_traceparent;

const REF_TRACEPARENT: &[u8] = b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
const TRACE_HEX_16: &[u8] = b"0af7651916cd43dd8448eb211c80319c";
const SPAN_HEX_8: &[u8] = b"b7ad6b7169203331";

fn bench_proxima_parse_traceparent(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_id");
    group.bench_function("proxima_parse_traceparent", |bencher| {
        bencher.iter(|| black_box(parse_traceparent(black_box(REF_TRACEPARENT))));
    });
    group.finish();
}

fn bench_proxima_hex_decode_16(criterion: &mut Criterion) {
    // inner hot path: decode the 16-byte trace_id hex only
    let mut group = criterion.benchmark_group("c2_id");
    group.bench_function("proxima_hex_decode_16", |bencher| {
        bencher.iter(|| {
            // parse a minimal-overhead traceparent that exercises decode_hex_16 in isolation
            let input = black_box(TRACE_HEX_16);
            // wrap it so parse_traceparent is not invoked; call the scalar decode via parse on a
            // crafted full traceparent containing the same 32 hex chars as trace_id
            let mut buf = [0u8; 55];
            buf[..2].copy_from_slice(b"00");
            buf[2] = b'-';
            buf[3..35].copy_from_slice(input);
            buf[35] = b'-';
            buf[36..52].copy_from_slice(SPAN_HEX_8);
            buf[52] = b'-';
            buf[53..55].copy_from_slice(b"01");
            black_box(parse_traceparent(&buf))
        });
    });
    group.finish();
}

fn bench_faster_hex_decode_16(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_id");
    group.bench_function("faster_hex_decode_16", |bencher| {
        let mut out = [0u8; 16];
        bencher.iter(|| black_box(faster_hex::hex_decode(black_box(TRACE_HEX_16), &mut out)));
    });
    group.finish();
}

fn bench_hex_decode_16(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_id");
    group.bench_function("hex_decode_16", |bencher| {
        bencher.iter(|| black_box(hex::decode(black_box(TRACE_HEX_16))));
    });
    group.finish();
}

fn bench_opentelemetry_parse_traceparent(criterion: &mut Criterion) {
    // opentelemetry TraceId::from_hex + SpanId::from_hex is the OTel scalar parse path;
    // there is no single parse_traceparent fn in the API crate — this exercises the same work
    let ref_str = std::str::from_utf8(REF_TRACEPARENT).unwrap();
    let mut group = criterion.benchmark_group("c2_id");
    group.bench_function("opentelemetry_parse_traceparent", |bencher| {
        bencher.iter(|| {
            let input = black_box(ref_str);
            // manually mirror what otel sdk does: split on '-', parse fields
            let parts: Vec<&str> = input.splitn(4, '-').collect();
            if parts.len() == 4 {
                let trace_id = opentelemetry::trace::TraceId::from_hex(parts[1]).ok();
                let span_id = opentelemetry::trace::SpanId::from_hex(parts[2]).ok();
                black_box((trace_id, span_id))
            } else {
                black_box((
                    None::<opentelemetry::trace::TraceId>,
                    None::<opentelemetry::trace::SpanId>,
                ))
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_parse_traceparent,
    bench_proxima_hex_decode_16,
    bench_faster_hex_decode_16,
    bench_hex_decode_16,
    bench_opentelemetry_parse_traceparent,
);
criterion_main!(benches);
