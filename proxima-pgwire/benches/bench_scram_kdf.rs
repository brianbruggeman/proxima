//! Micro-bench for the SCRAM per-auth key-derivation cost.
//!
//! `ScramServer::new` runs PBKDF2-HMAC-SHA256 over 4096 iterations (the
//! RFC 7677 / PostgreSQL default). On the inline path this whole cost is
//! spent on the per-core async reactor, stalling it for every SCRAM auth.
//! This bench measures that wall-time directly: the reported ns/auth IS
//! the reactor-occupancy the `spawn_background_blocking` offload moves
//! off-core. (It does not bench the offload itself — a concurrent
//! auth-storm p99 needs a prime harness, gate G7 territory.)
//!
//! required-features: scram.

#![cfg(feature = "scram")]
#![allow(clippy::expect_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_pgwire::scram::ScramServer;

fn scram_kdf(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scram_kdf");
    group.bench_function("server_new_pbkdf2_4096", |bencher| {
        bencher.iter(|| {
            let server = ScramServer::new(black_box("hunter2")).expect("kdf must succeed");
            black_box(server)
        });
    });
    group.finish();
}

criterion_group!(benches, scram_kdf);
criterion_main!(benches);
