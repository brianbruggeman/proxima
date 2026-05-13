#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! HPACK static table reverse-lookup microbench (RFC 7541 §A).
//!
//! Encoder hot path: given a `(name, value)` from the user's
//! request, find the predefined index (1..=61) or `None`. Apples-
//! to-apples vs h2's `index_static`, vendored verbatim with the
//! `HeaderName` wrapper stripped so both impls take `&[u8]` slices.
//!
//! Workloads:
//!  - `method_get`           — pseudo-header full hit (index 2, value matched)
//!  - `method_delete`        — pseudo-header name hit, value miss (idx 2, false)
//!  - `status_200`           — pseudo-header status full hit (idx 8)
//!  - `accept_encoding_gz`   — value-checked common header full hit
//!  - `content_type`         — common header name hit, value not in table
//!  - `user_agent`           — long-ish name, common header
//!  - `www_authenticate`     — last-entry path (length-61 match late)
//!  - `unknown_short`        — `x-foo` miss (length not in table)
//!  - `unknown_long`         — `x-request-id` miss

#[path = "vendored_h2/mod.rs"]
mod h2_vendored;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_protocols::hpack::static_lookup;

#[derive(Clone, Copy)]
struct Workload {
    label: &'static str,
    name: &'static [u8],
    value: &'static [u8],
}

const WORKLOADS: &[Workload] = &[
    Workload {
        label: "method_get",
        name: b":method",
        value: b"GET",
    },
    Workload {
        label: "method_delete",
        name: b":method",
        value: b"DELETE",
    },
    Workload {
        label: "status_200",
        name: b":status",
        value: b"200",
    },
    Workload {
        label: "accept_encoding_gz",
        name: b"accept-encoding",
        value: b"gzip, deflate",
    },
    Workload {
        label: "content_type",
        name: b"content-type",
        value: b"application/json",
    },
    Workload {
        label: "user_agent",
        name: b"user-agent",
        value: b"Mozilla/5.0",
    },
    Workload {
        label: "www_authenticate",
        name: b"www-authenticate",
        value: b"Basic",
    },
    Workload {
        label: "unknown_short",
        name: b"x-foo",
        value: b"bar",
    },
    Workload {
        label: "unknown_long",
        name: b"x-request-id",
        value: b"abc-123",
    },
];

fn lookup_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_static_lookup");
    group.measurement_time(Duration::from_secs(2));
    for workload in WORKLOADS {
        group.bench_function(format!("proxima_native/{}", workload.label), |bencher| {
            bencher.iter(|| {
                let result = static_lookup(
                    std::hint::black_box(workload.name),
                    std::hint::black_box(workload.value),
                );
                std::hint::black_box(result);
            });
        });
        group.bench_function(format!("h2_crate/{}", workload.label), |bencher| {
            bencher.iter(|| {
                let result = h2_vendored::static_lookup::index_static(
                    std::hint::black_box(workload.name),
                    std::hint::black_box(workload.value),
                );
                std::hint::black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, lookup_compare);
criterion_main!(benches);
