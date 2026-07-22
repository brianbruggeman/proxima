//! The h2 sibling of `bench_server`: proxima's HTTP/2 (h2c, prior-knowledge)
//! server on PRIME — same per-core SO_REUSEPORT serving path as the h1
//! `bench_server`, just the single-candidate `AnyListenProtocol` "h2" instead
//! of `HttpListenProtocol` (`H2ListenProtocol` is retired onto
//! `AnyListenProtocol`'s single bind+accept loop — see
//! `proxima_http::http2`'s module doc). No tokio: the listener runs on
//! prime's per-core executor via `PrimeAcceptorFactory` (this is exactly
//! what `PrimeServeExt::serve_http` does internally).
//!
//!   cargo run --release --features scheduler --example bench_server_h2 -- 127.0.0.1:8090 [cores]

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use proxima::listeners::{AnyListenProtocol, H2PriorKnowledgeAnyProtocol};
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::runtime::{PrimeRuntime, Runtime};
use proxima::{ListenRegistry, ListenerSpec, NoopTelemetry, ProximaError, SendPipe};
use proxima_net::prime::PrimeAcceptorFactory;

/// 200 OK + 2-byte body — identical contract to `bench_server`'s h1 handler.
struct OkHandler;

impl SendPipe for OkHandler {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    #[allow(clippy::manual_async_fn)]
    fn call(&self, _request: Request<Bytes>) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async { Ok(Response::new(200).with_body(Bytes::from_static(b"ok"))) }
    }
}


fn main() -> Result<(), ProximaError> {
    let mut args = std::env::args().skip(1);
    let addr_text = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8090".to_string());
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let addr = addr_text
        .parse()
        .map_err(|err| ProximaError::Config(format!("bind addr `{addr_text}`: {err}")))?;

    let runtime = Arc::new(PrimeRuntime::new(cores)?);

    // mirror PrimeServeExt::serve_http, but register the h2c listener and select
    // it by protocol name. The acceptor factory keeps the accept loop on prime.
    let registry = ListenRegistry::new();
    registry.register(Arc::new(AnyListenProtocol::single_candidate(
        "h2",
        Arc::new(H2PriorKnowledgeAnyProtocol::new()),
    )))?;
    let runtime_dyn: Arc<dyn Runtime> = runtime.clone();
    let acceptor = Arc::new(PrimeAcceptorFactory);
    let mut spec = ListenerSpec::http(addr);
    spec.protocol_name = "h2".into();
    let _server = spec
        .attach(into_handle(OkHandler))
        .run_with_runtime(
            &registry,
            NoopTelemetry::handle(),
            Some(runtime_dyn),
            Some(acceptor),
            None, // datagram factory - unused for TCP h2c
        )?;

    println!("bench_server_h2: proxima h2c (prime) on {addr} ({cores} core(s))");
    loop {
        std::thread::park();
    }
}
