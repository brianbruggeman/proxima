//! The same `Pipe`, served first on prime, then again on tokio — proof that
//! the pipe never knows or cares which `Runtime` dispatched it.
//!
//! `hello` wires one pipe to one runtime and stops there. This example takes
//! that exact pipe shape and runs it TWICE, back to back, once per runtime,
//! reusing the identical `select_pipe` instance both times. Only the
//! `Runtime` + `AcceptorFactory` pair passed to `App::with_runtime` /
//! `with_acceptor_factory` changes.
//!
//! `multi_runtime` is the next rung: prime and tokio serving CONCURRENTLY,
//! sharing state across the runtime boundary. This example is the simpler,
//! sequential half of that proof — same pipe, one runtime at a time.
//!
//! ```sh
//! cargo run --example runtime_select --features "runtime-tokio tokio"
//! ```
//!
//! See `examples/runtime_select/README.md` for the full writeup.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

use bytes::Bytes;
use proxima::prime::PrimeRuntime;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, ListenerSpec, PipeHandle, ProximaError, Request, Response, Runtime,
    TokioPerCoreRuntime, into_handle,
};
use proxima_primitives::stream::AcceptorFactory;

const PRIME_BIND: &str = "127.0.0.1:8083";
const TOKIO_BIND: &str = "127.0.0.1:8084";

/// Runtime-neutral, same as `hello`'s pipe: no socket, no runtime handle,
/// just `Request -> Response`. That's what makes swapping the runtime under
/// it a config change, not a rewrite. Stateless, so `#[proxima::piped]` writes
/// the `SendPipe` impl — exactly `hello`'s own idiom.
#[proxima::piped(send)]
async fn select_pipe(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello from whichever runtime is listening\n"))
}


// `#[proxima::main(cores = 1)]` supplies the one throwaway core
// `App::builder()` needs before each pass's `.with_runtime(...)` overrides
// it with the real prime/tokio one.
#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    let pipe: PipeHandle = into_handle(select_pipe);

    println!("--- pass 1: the SAME pipe served on prime ---");
    serve_and_check(
        pipe.clone(),
        PRIME_BIND.parse().expect("valid socket addr"),
        Arc::new(PrimeRuntime::new(1)?),
        Arc::new(proxima_net::prime::PrimeAcceptorFactory),
        "prime",
    )
    .await?;

    println!("\n--- pass 2: the SAME pipe served on tokio ---");
    serve_and_check(
        pipe,
        TOKIO_BIND.parse().expect("valid socket addr"),
        Arc::new(TokioPerCoreRuntime::new(1)?),
        Arc::new(proxima_net::tokio::TokioAcceptorFactory),
        "tokio",
    )
    .await?;

    println!("\nsame Pipe, two runtimes, identical response both times.");
    Ok(())
}

async fn serve_and_check(
    pipe: PipeHandle,
    bind: SocketAddr,
    runtime: Arc<dyn Runtime>,
    acceptor_factory: Arc<dyn AcceptorFactory>,
    runtime_name: &str,
) -> Result<(), ProximaError> {
    let app = App::builder()
        .with_defaults()?
        .build()?
        .with_runtime(runtime.clone())
        .with_acceptor_factory(acceptor_factory);
    app.mount("/", pipe)?;

    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let listener = app.build_listener(ListenerSpec::http(bind))?;
    println!(
        "listening on {bind} ({runtime_name} runtime, {} core)",
        runtime.num_cores()
    );

    let response = blocking_get(bind);
    println!("GET http://{bind}/ ({runtime_name}) ->\n{response}");
    assert!(
        response.contains("hello from whichever runtime is listening"),
        "the {runtime_name} pass must serve the same pipe body as every other runtime"
    );

    listener.shutdown();
    let report = ShutdownBarrier::new(runtime).broadcast_drop().await;
    println!(
        "{runtime_name} drained: cores_acked={} hooks_drained={}",
        report.cores_acked, report.hooks_drained
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
