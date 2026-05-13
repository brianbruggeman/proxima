//! Composite runtime: prime and tokio serve CONCURRENTLY in one process,
//! both dispatching into the same sans-IO pipe with shared state across the
//! runtime boundary.
//!
//! tokio and glommio and monoio are process-singletons — one runtime per
//! process, full stop. proxima's `Runtime` trait is just an interface: any
//! number of implementations can live in the same process side by side. This
//! example proves it with the smallest shape that can't be faked: two real
//! TCP listeners, two independent executors, one `Arc<AtomicU64>` neither
//! runtime owns exclusively.
//!
//! ```sh
//! cargo run --example multi_runtime --features "runtime-tokio tokio"
//! ```
//!
//! See `examples/multi_runtime.README.md` for the full writeup.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use proxima::prime::PrimeRuntime;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, ListenerSpec, PipeHandle, ProximaError, Request, Response, Runtime, SendPipe,
    TokioPerCoreRuntime, into_handle,
};

const PRIME_BIND: &str = "127.0.0.1:8081";
const TOKIO_BIND: &str = "127.0.0.1:8082";
const REQUESTS_PER_LISTENER: u64 = 3;

/// Runtime-neutral: this same instance is mounted on both listeners below,
/// so the increment below runs regardless of which runtime dispatched it.
struct SharedCounterPipe {
    total: Arc<AtomicU64>,
}

impl SendPipe for SharedCounterPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let total = self.total.clone();
        async move {
            let observed = total.fetch_add(1, Ordering::AcqRel) + 1;
            Ok(Response::ok(format!("shared_total={observed}\n")))
        }
    }
}


// `#[proxima::main(cores = 1)]` supplies the one throwaway core
// `App::builder()` needs before `.with_runtime(...)` below overrides each
// app's runtime with the real prime/tokio one — no env var, no reaching
// for a global.
#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    let shared_total = Arc::new(AtomicU64::new(0));
    let pipe: PipeHandle = into_handle(SharedCounterPipe {
        total: shared_total.clone(),
    });

    let prime_bind: SocketAddr = PRIME_BIND.parse().expect("valid socket addr");
    let tokio_bind: SocketAddr = TOKIO_BIND.parse().expect("valid socket addr");

    // prime: N worker threads it owns, its own reactor, its own executor.
    let prime_runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(2)?);
    let prime_app = App::builder()
        .with_defaults()?
        .build()?
        .with_runtime(prime_runtime.clone())
        .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
    prime_app.mount("/", pipe.clone())?;

    // tokio: a completely separate set of worker threads, a separate
    // executor, wired to the SAME pipe (same Arc<AtomicU64> inside it).
    let tokio_runtime: Arc<dyn Runtime> = Arc::new(TokioPerCoreRuntime::new(2)?);
    let tokio_app = App::builder()
        .with_defaults()?
        .build()?
        .with_runtime(tokio_runtime.clone())
        .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
    tokio_app.mount("/", pipe.clone())?;

    // both listeners bind and start accepting now — two live runtimes, one
    // process, at the same time. Each `build_listener` blocks until its own
    // accept lane has acked ready — no polling, no sleeping.
    let prime_listener = prime_app.build_listener(ListenerSpec::http(prime_bind))?;
    let tokio_listener = tokio_app.build_listener(ListenerSpec::http(tokio_bind))?;
    println!(
        "prime listener on {prime_bind} (prime runtime, {} cores)",
        prime_runtime.num_cores()
    );
    println!(
        "tokio listener on {tokio_bind} (tokio runtime, {} cores)",
        tokio_runtime.num_cores()
    );

    // hit both listeners from separate OS threads so the requests actually
    // race — the proof is that a shared counter survives concurrent access
    // from two independently-scheduled runtimes, not just sequential access.
    let prime_client = std::thread::spawn(move || {
        (0..REQUESTS_PER_LISTENER)
            .map(|_| blocking_get(prime_bind))
            .collect::<Vec<_>>()
    });
    let tokio_client = std::thread::spawn(move || {
        (0..REQUESTS_PER_LISTENER)
            .map(|_| blocking_get(tokio_bind))
            .collect::<Vec<_>>()
    });
    let prime_responses = prime_client.join().expect("prime client thread");
    let tokio_responses = tokio_client.join().expect("tokio client thread");

    let prime_totals: Vec<u64> = prime_responses
        .iter()
        .map(|response| extract_shared_total(response))
        .collect();
    let tokio_totals: Vec<u64> = tokio_responses
        .iter()
        .map(|response| extract_shared_total(response))
        .collect();
    for total in &prime_totals {
        println!("GET http://{prime_bind}/ (prime) -> shared_total={total}");
    }
    for total in &tokio_totals {
        println!("GET http://{tokio_bind}/ (tokio) -> shared_total={total}");
    }

    let mut totals: Vec<u64> = prime_totals.into_iter().chain(tokio_totals).collect();
    totals.sort_unstable();
    let expected: Vec<u64> = (1..=REQUESTS_PER_LISTENER * 2).collect();
    println!("observed totals across both listeners (sorted): {totals:?}");
    assert_eq!(
        totals, expected,
        "requests split across two runtimes must land on one contiguous, lock-free \
         shared counter — no lost updates, no double counts"
    );

    prime_listener.shutdown();
    tokio_listener.shutdown();
    let prime_report = ShutdownBarrier::new(prime_runtime).broadcast_drop().await;
    let tokio_report = ShutdownBarrier::new(tokio_runtime).broadcast_drop().await;
    println!(
        "prime drained: cores_acked={} hooks_drained={}",
        prime_report.cores_acked, prime_report.hooks_drained
    );
    println!(
        "tokio drained: cores_acked={} hooks_drained={}",
        tokio_report.cores_acked, tokio_report.hooks_drained
    );
    println!(
        "both runtimes shut down cleanly; final shared total = {}",
        shared_total.load(Ordering::Acquire)
    );

    Ok(())
}

/// One-shot GET over a plain blocking `TcpStream` — deliberately not another
/// async runtime. `Connection: close` lets us read to EOF instead of framing
/// the body ourselves.
fn blocking_get(addr: SocketAddr) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    String::from_utf8_lossy(&raw).into_owned()
}

/// Pulls `shared_total=<digits>` out of a raw HTTP response regardless of
/// framing (content-length or chunked) — the digits are written in one
/// contiguous write, so a substring search is sufficient without a full
/// dechunking pass.
fn extract_shared_total(response_text: &str) -> u64 {
    let marker = "shared_total=";
    let start = response_text
        .find(marker)
        .map(|position| position + marker.len())
        .unwrap_or_else(|| panic!("{marker} not found in response: {response_text:?}"));
    let digits: String = response_text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|error| panic!("shared_total digits {digits:?} did not parse: {error}"))
}
