//! compare-bench: prime TCP connect vs tokio TCP connect (LOOPBACK).
//!
//! this is a loopback micro — RTT≈0, so it does NOT measure the
//! handshake-RTT-bound regime the consumer (the consumer relay network dial)
//! actually runs in. what it isolates instead is the runtime's wakeup
//! latency: "fd writable → reactor delivers event → task re-polls → done".
//! result (see `docs/runtime-prime/discipline-net-connect.md`): prime ≈ 7×
//! slower than tokio here, because tokio's mio reactor wakes inline on the
//! current thread while prime's per-core shard wakes from a `kevent(NULL)`
//! park. on a network dial (ms RTT ≫ the ~180µs wake gap) the two are at
//! parity — that parity is proven by the relay E2E (gate point 7), NOT here.
//! this micro exists to (a) satisfy the compare-bench gate honestly and
//! (b) pin the wakeup-latency finding with a saved baseline.
//!
//! the bracket measures ONLY `connect().await` on both arms; the prime arm's
//! cross-thread dispatch + rendezvous-spin are outside it, so the gap is not
//! a harness artifact.
//!
//! arms:
//!   - tokio — `tokio::net::TcpStream::connect` on a current-thread runtime.
//!   - prime — `prime::os::net::TcpStream::connect` on a prime core shard.
//!
//! listener: one OS accept thread per arm. accepted sockets are closed with
//! SO_LINGER=0 (RST, no FIN) so the kernel skips TIME_WAIT on the server
//! side. the connecting socket also sets SO_LINGER=0 before drop. together
//! these prevent ephemeral port exhaustion during criterion's warm-up burst.

#![cfg(any(target_os = "macos", target_os = "linux"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use prime::os::core_shard;
use proxima_runtime::CoreId;
use socket2::{Domain, Protocol, Socket, Type};

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
}

fn make_listener() -> SocketAddr {
    let sock = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect("socket");
    sock.set_reuse_address(true).expect("SO_REUSEADDR");
    sock.bind(&"127.0.0.1:0".parse::<SocketAddr>().unwrap().into())
        .expect("bind");
    sock.listen(1024).expect("listen");
    let addr = sock.local_addr().expect("local_addr").as_socket().unwrap();

    std::thread::Builder::new()
        .name("bench-accept".into())
        .spawn(move || {
            loop {
                if let Ok((conn, _)) = sock.accept() {
                    drop(conn);
                }
            }
        })
        .expect("spawn accept thread");

    addr
}

fn bench_tcp_connect(criterion: &mut Criterion) {
    let tokio_addr = make_listener();
    let prime_addr = make_listener();

    let mut group = criterion.benchmark_group("tcp_connect");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(1));

    group.bench_function("tokio", |bencher| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .expect("tokio runtime");

        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let socket = tokio::net::TcpSocket::new_v4().expect("tokio socket");
                    let start = Instant::now();
                    let stream = socket.connect(tokio_addr).await.expect("tokio connect");
                    total += start.elapsed();
                    let std_stream = stream.into_std().expect("into_std");
                    let sock2 = Socket::from(std_stream);
                    sock2.set_linger(Some(Duration::ZERO)).ok();
                    drop(sock2);
                }
            });
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let handle =
            core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("prime shard launch");

        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                let result_for_task = result_slot.clone();
                let addr = prime_addr;

                handle
                    .dispatch_send(Box::pin(async move {
                        let start = Instant::now();
                        let stream = prime::os::net::TcpStream::connect(addr)
                            .await
                            .expect("prime connect");
                        let elapsed = start.elapsed();
                        drop(stream);
                        *result_for_task.lock().unwrap() = Some(elapsed);
                    }))
                    .expect("dispatch_send");

                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    if result_slot.lock().unwrap().is_some() {
                        break;
                    }
                    assert!(Instant::now() < deadline, "prime connect timed out");
                    std::thread::sleep(Duration::from_millis(1));
                }

                total += result_slot.lock().unwrap().expect("elapsed not set");
            }

            total
        });

        handle.shutdown_and_join().expect("prime shard shutdown");
    });

    group.finish();
}

criterion_group!(benches, bench_tcp_connect);
criterion_main!(benches);
