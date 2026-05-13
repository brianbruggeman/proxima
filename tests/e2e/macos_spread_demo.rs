#![allow(clippy::expect_used, clippy::let_and_return)]
//! End-to-end demonstration that the consolidated HTTP serve (a) WORKS on
//! macOS through the real `PrimeServeExt::serve_http` shim — which on Darwin
//! auto-selects one accept lane + `HandlerDispatch::SpreadToPeers` inside
//! `run_with_runtime`, NOT forced here — and (b) is USEFUL: spreading handlers
//! across peer cores keeps a burst of blocking requests from serializing.
//!
//! The proof is a wall-clock contrast of the SAME server + load at two core
//! counts. Multi-core spreads the blocking handlers across peer cores (overlap);
//! single-core has no peers, so they serialize. The ratio is the usefulness.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::let_and_return)]
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    feature = "http1",
    feature = "rayon"
))]

use std::future::Future;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use bytes::Bytes;

use proxima::error::ProximaError;
use proxima::pipe::{into_handle};
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;
use proxima_primitives::pipe::SendPipe;

struct BlockingHandler {
    block_ms: u64,
}

impl SendPipe for BlockingHandler {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let block_ms = self.block_ms;
        async move {
            // a synchronous block, the worst case for head-of-line blocking:
            // it pins the whole core for block_ms.
            std::thread::sleep(Duration::from_millis(block_ms));
            Ok(Response::ok("ok"))
        }
    }
}


fn reserve_port() -> SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn connect_with_retry(addr: SocketAddr) -> std::net::TcpStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(50)) {
            Ok(stream) => return stream,
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Err(err) => panic!("connect: {err}"),
        }
    }
}

/// Fire `concurrency` simultaneous GETs (released together by a barrier) and
/// return (wall_ms, successes).
fn concurrent_burst(addr: SocketAddr, concurrency: usize) -> (u128, usize) {
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let workers: Vec<_> = (0..concurrency)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || -> bool {
                let mut stream = connect_with_retry(addr);
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("timeout");
                barrier.wait();
                stream
                    .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                    .expect("write");
                stream.flush().expect("flush");
                let mut response = Vec::with_capacity(256);
                stream.read_to_end(&mut response).expect("read");
                response.starts_with(b"HTTP/1.1 200")
            })
        })
        .collect();

    barrier.wait();
    let started = Instant::now();
    let successes = workers
        .into_iter()
        .map(|w| w.join().expect("client panicked"))
        .filter(|&ok| ok)
        .count();
    (started.elapsed().as_millis(), successes)
}

fn measure(cores: usize, concurrency: usize, block_ms: u64) -> (u128, usize) {
    // default (rayon) background pool, now sized to `cores` — `background_inline()`
    // spawns one unbounded std::thread per Offloaded call regardless of `cores`,
    // which would parallelize the burst identically at cores=1 and defeat the
    // whole point of this measurement.
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(cores)
            .build()
            .expect("prime runtime"),
    );
    let addr = reserve_port();
    // the REAL serve path — on macOS this auto-selects 1 accept lane + spread.
    let _handle = runtime
        .serve_http(addr, into_handle(BlockingHandler { block_ms }))
        .expect("serve_http");
    let result = concurrent_burst(addr, concurrency);
    result
}

#[test]
fn macos_spread_keeps_blocking_burst_parallel() {
    let avail = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(2);
    let multi = avail.max(4);
    let block_ms: u64 = 300;
    let concurrency: usize = 8;

    let (multi_wall, multi_ok) = measure(multi, concurrency, block_ms);
    let (single_wall, single_ok) = measure(1, concurrency, block_ms);

    let serial_floor = block_ms as u128 * concurrency as u128;
    eprintln!("\nmacOS handler-spread usefulness ({} cores avail)", avail);
    eprintln!("workload: {concurrency} concurrent requests, each blocks {block_ms}ms");
    eprintln!(
        "  multi-core ({multi} cores, auto-spread): wall = {multi_wall}ms  success={multi_ok}/{concurrency}"
    );
    eprintln!(
        "  single-core (1 core, no peers):          wall = {single_wall}ms  success={single_ok}/{concurrency}"
    );
    eprintln!("  serialized floor (1 core ideal):         {serial_floor}ms");
    eprintln!(
        "  speedup from spread: {:.1}x\n",
        single_wall as f64 / multi_wall.max(1) as f64
    );

    // WORKS: every request answered 200 on both configs, on macOS, via serve_http.
    assert_eq!(
        multi_ok, concurrency,
        "multi-core: not all requests succeeded"
    );
    assert_eq!(
        single_ok, concurrency,
        "single-core: not all requests succeeded"
    );
    // USEFUL: spreading across peer cores beats single-core serialization
    // decisively (single-core must be near the serial floor; multi-core well under).
    assert!(
        (multi_wall as f64) < (single_wall as f64) * 0.6,
        "spread did not parallelize: multi={multi_wall}ms single={single_wall}ms"
    );
}
