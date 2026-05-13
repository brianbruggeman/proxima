//! Outbound HTTP construction micro-bench — the per-send build cost a load
//! generator (rekt, proxima::Client) pays modeling closed-set HTTP values as
//! heap byte-strings. Arms isolate the two acceptance targets: request build
//! (~67 ns today) and response + content-length build (the ~146 ns synth arm).
//!
//! Outbound only (no wire buffer to borrow from), so every `&str` field today
//! costs a `copy_from_slice` and content-length costs a `to_string`. These are
//! the allocations the typed `Method` / `HeaderName` / numeric `HeaderValue`
//! work removes. Re-bench each arm after each type lands.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use proxima_primitives::pipe::{Request, Response};

fn bench_http_construction(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("http_construction");

    // outbound request build: `GET /` — the load-gen per-send build. method +
    // path each `copy_from_slice` a static str; build() materializes a fresh
    // RequestContext::default().
    group.bench_function("request_build_get", |bencher| {
        bencher.iter(|| {
            let request = Request::builder()
                .method(black_box("GET"))
                .path(black_box("/"))
                .build()
                .expect("build request");
            black_box(request)
        });
    });

    // build once, clone per send (a load generator firing the same request) —
    // clone is refcount bumps, no path copy, no RequestContext::default().
    let template = Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("build template");
    group.bench_function("request_clone_get", |bencher| {
        bencher.iter(|| black_box(black_box(&template).clone()));
    });

    // response in the synth shape: status + body + STRINGIFIED content-length.
    // `length.to_string()` allocates a String, "content-length" copies, and the
    // insert does a linear case-insensitive scan.
    let body = Bytes::from_static(b"hello, world");
    group.bench_function("response_synth_shape", |bencher| {
        bencher.iter(|| {
            let length = black_box(body.len());
            let response = Response::new(black_box(200))
                .with_body(Bytes::clone(&body))
                .with_header("content-length", length.to_string());
            black_box(response)
        });
    });

    // same response but the NAME is a known HeaderName → Bytes::from_static, no
    // copy. isolates the header-name allocation (the value still stringifies;
    // that's the storage-typed HeaderValue work, separate).
    group.bench_function("response_typed_name", |bencher| {
        bencher.iter(|| {
            let length = black_box(body.len());
            let response = Response::new(black_box(200))
                .with_body(Bytes::clone(&body))
                .with_header(proxima_primitives::pipe::HeaderName::ContentLength, length.to_string());
            black_box(response)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_http_construction);
criterion_main!(benches);
