//! The bench target, but BOTH sides are the substrate now: a proxima HTTP/1.1
//! server. Same contract as the std `bench_target` (fixed `200` / `"ok"` /
//! keep-alive), served through proxima's prime h1 stack — a trivial handler
//! Pipe (`OkHandler`) attached to `PrimeRuntime::serve_http`, so the response
//! goes out the same `Connection` state machine + body framing the daemon uses.
//!
//!   cargo run --release --example bench_server -- 127.0.0.1:8080 [cores]

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use proxima::ProximaError;
use proxima::SendPipe;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::runtime::{PrimeRuntime, PrimeServeExt};

/// 200 OK with a 2-byte body — the same fixed response `bench_target` writes,
/// but produced by a real proxima Pipe the listener dispatches per request.
struct OkHandler;

impl SendPipe for OkHandler {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    // the trait method is `-> impl Future + Send`; the explicit form carries the
    // Send bound the RPITIT requires, so the async-block shape is intentional.
    #[allow(clippy::manual_async_fn)]
    fn call(&self, _request: Request<Bytes>) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async { Ok(Response::new(200).with_body(Bytes::from_static(b"ok"))) }
    }
}


fn main() -> Result<(), ProximaError> {
    let mut args = std::env::args().skip(1);
    let addr_text = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let addr = addr_text
        .parse()
        .map_err(|err| ProximaError::Config(format!("bind addr `{addr_text}`: {err}")))?;

    // workers float by default (prime default); the OS schedules them.
    let runtime = Arc::new(PrimeRuntime::new(cores)?);
    // hold the handle for the process lifetime — the prime worker threads serve
    // autonomously; the main thread only has to stay alive so they live.
    let _server = runtime.serve_http(addr, into_handle(OkHandler))?;
    println!("bench_server: proxima h1 on {addr} ({cores} core(s))");
    loop {
        std::thread::park();
    }
}
