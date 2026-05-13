//! HTTP/3 (native proxima QUIC, h3 over UDP) bench server: `H3NativeListenProtocol`
//! on prime with a dev self-signed cert. The h3 sibling of `bench_server` /
//! `bench_server_h3` — same `OkHandler`. Native single-task per-connection driver
//! (server fan-out is a documented substrate follow-on).
//!
//!   cargo run --release --features scheduler --example bench_server_h3 -- 127.0.0.1:8094 [cores]

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use proxima::h3::native::H3NativeListenProtocol;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::runtime::{PrimeRuntime, Runtime};
use proxima::{ListenRegistry, ListenerSpec, NoopTelemetry, ProximaError, SendPipe};
use proxima_net::prime::PrimeDatagramFactory;
use serde_json::json;

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
    // one-liner: level-routed console logging (info/debug→stdout, warn/error→stderr).
    let _telemetry = proxima_telemetry::export::install_console_logging();
    let mut args = std::env::args().skip(1);
    let addr_text = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8094".to_string());
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let addr = addr_text
        .parse()
        .map_err(|err| ProximaError::Config(format!("bind addr `{addr_text}`: {err}")))?;

    let runtime = Arc::new(PrimeRuntime::new(cores)?);
    let registry = ListenRegistry::new();
    registry.register(Arc::new(H3NativeListenProtocol::new()))?;
    let runtime_dyn: Arc<dyn Runtime> = runtime.clone();
    let mut spec = ListenerSpec::http(addr);
    spec.protocol_name = "h3-native".into();
    let spec = spec.with_spec(json!({ "dev_self_signed": true }));
    let _server = spec
        .attach(into_handle(OkHandler))
        .run_with_runtime(
            &registry,
            NoopTelemetry::handle(),
            Some(runtime_dyn),
            None,                                 // TCP acceptor — unused for h3c
            Some(Arc::new(PrimeDatagramFactory)), // runtime-agnostic UDP socket source
        )?;

    println!("bench_server_h3: proxima native h3 (prime, dev cert) on {addr} ({cores} core(s))");
    loop {
        std::thread::park();
    }
}
