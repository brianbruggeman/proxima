//! Tokio-freedom proof for the h1 listener: serves `ListenerSpec::http`
//! (via `PrimeServeExt::serve_http`, which registers `HttpListenProtocol`
//! and injects `proxima_net::prime::PrimeAcceptorFactory`) over the prime
//! runtime, sends one real HTTP/1.1 request with a plain blocking
//! `std::net::TcpStream`, and asserts the round trip — all under a build
//! with `tokio` nowhere in the dependency graph.
//!
//! Verify the build has no tokio:
//!
//!   cargo tree --no-default-features \
//!     --features "http1-native,serve-prime,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,macros" \
//!     -e normal -i tokio
//!   # -> "warning: nothing to print" (empty result)
//!
//! Run it:
//!
//!   cargo run --example h1_native_prime_round_trip --no-default-features \
//!     --features "http1-native,serve-prime,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,macros"

use std::error::Error;
use std::future::Future;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use bytes::Bytes;
use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;

const GREETING: &str = "hello from the tokio-free h1 listener";

struct Greeter;

impl SendPipe for Greeter {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(GREETING)) }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let runtime = Arc::new(PrimeRuntime::builder().cores(1).background_inline().build()?);
    let pipe: PipeHandle = into_handle(Greeter);

    // Port 0 -> OS-assigned; `serve_http` blocks (via the listener's
    // `ready_signal`) until the bind completes, so `bind_addr()` is
    // populated by the time it returns.
    let bind = "127.0.0.1:0".parse()?;
    let handle = runtime.serve_http(bind, pipe)?;
    let addr = handle
        .bind_addr()
        .ok_or("listener did not report a bound address")?;

    let mut stream = TcpStream::connect(addr)?;
    stream.write_all(b"GET /hello HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;

    let response_text = String::from_utf8_lossy(&response);
    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "expected a 200 response, got: {response_text}"
    );
    assert!(
        response_text.contains(GREETING),
        "expected the greeting in the response body: {response_text}"
    );

    println!("h1_native_prime_round_trip: OK — served {addr} with zero tokio in the build");
    handle.shutdown();
    Ok(())
}
