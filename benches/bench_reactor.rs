//! micro-bench for the proxima Reactor vs tokio I/O baseline.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - tokio 1.x — mio-abstracted, scheduler-integrated I/O readiness; design
//!     point is AsyncWriteExt + readable()/writable() driven by the runtime
//!
//! groups (and design-favors per workload):
//!   - reactor_wake_latency      design-favors: incumbent
//!     (tokio_unix arm IS tokio on home turf: tokio::net::UnixStream +
//!     AsyncWriteExt::write_all + readable().await, driven by a real
//!     current-thread tokio runtime. proxima's raw kqueue/epoll path has
//!     no mio/scheduler layer to traverse, so a win is structural.)
//!   - reactor_turn_n_ready_16   design-favors: prime
//!     (proxima-only; tokio has no "turn fires N" single-call API)
//!
//! requires-features: runtime-prime-reactor, runtime-tokio.

#![cfg(all(
    feature = "runtime-prime-reactor",
    feature = "runtime-tokio",
    any(target_os = "macos", target_os = "linux")
))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::prime::os::reactor::{Interest, Reactor};

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn noop_waker() -> std::task::Waker {
    std::task::Waker::noop().clone()
}

// design-favors: incumbent — tokio's `AsyncWriteExt + readable()` IS the
// canonical async I/O readiness pattern this crate was built for. Proxima's
// `register + write + turn` is the same operation without the mio/scheduler
// indirection. The ~2× win here engages tokio's actual machinery.
fn bench_wake_latency(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reactor_wake_latency");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(1));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut reactor = Reactor::new().expect("reactor");
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (left, mut right) = UnixStream::pair().unwrap();
                left.set_nonblocking(true).unwrap();
                let key = reactor
                    .register(left.as_raw_fd(), Interest::Read)
                    .expect("register");
                reactor.set_read_waker(key, noop_waker());
                let started = Instant::now();
                right.write_all(b"x").unwrap();
                let fired = reactor.turn(Some(Duration::from_secs(1))).unwrap();
                total += started.elapsed();
                assert!(fired >= 1);
                reactor.deregister(key).expect("deregister");
            }
            total
        });
    });

    group.bench_function("tokio_unix", |bencher| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let (left, right) = UnixStream::pair().unwrap();
                    left.set_nonblocking(true).unwrap();
                    right.set_nonblocking(true).unwrap();
                    let tokio_left = tokio::net::UnixStream::from_std(left).unwrap();
                    let mut tokio_right = tokio::net::UnixStream::from_std(right).unwrap();
                    let started = Instant::now();
                    tokio::io::AsyncWriteExt::write_all(&mut tokio_right, b"x")
                        .await
                        .unwrap();
                    tokio_left.readable().await.unwrap();
                    total += started.elapsed();
                    drop(tokio_left);
                    drop(tokio_right);
                }
            });
            total
        });
    });

    group.finish();
}

// design-favors: prime — proxima-only. tokio has no "drain N ready sources in
// one syscall" API; its readable()/writable() futures fire individually. This
// arm measures proxima's batched-drain primitive, not a comparison.
fn bench_turn_n_ready(criterion: &mut Criterion) {
    const N: usize = 16;
    let mut group = criterion.benchmark_group(format!("reactor_turn_n_ready_{N}"));
    configure_group(&mut group);
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut reactor = Reactor::new().expect("reactor");
                let mut pairs = Vec::with_capacity(N);
                let mut keys = Vec::with_capacity(N);
                for _ in 0..N {
                    let (left, right) = UnixStream::pair().unwrap();
                    left.set_nonblocking(true).unwrap();
                    let key = reactor
                        .register(left.as_raw_fd(), Interest::Read)
                        .expect("register");
                    reactor.set_read_waker(key, noop_waker());
                    pairs.push((left, right));
                    keys.push(key);
                }
                let started = Instant::now();
                for (_, right) in &mut pairs {
                    right.write_all(b"x").unwrap();
                }
                let mut total_fired = 0;
                while total_fired < N {
                    total_fired += reactor.turn(Some(Duration::from_secs(1))).unwrap();
                }
                total += started.elapsed();
                for key in keys {
                    reactor.deregister(key).expect("deregister");
                }
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_wake_latency, bench_turn_n_ready);
criterion_main!(benches);
