//! A reverse proxy is a pipe whose "transform" is "forward this request to
//! an upstream and return its response". No new machinery: `proxima::Client`
//! is itself a `SendPipe<In = Request<Bytes>, Out = Response<Bytes>>`, so a
//! proxy pipe's entire body is handing the inbound request to a `Client` and
//! returning what comes back.
//!
//! ```sh
//! cargo run --example proxy
//! ```
//!
//! See `examples/proxy/README.md` for the full writeup. `distributed_trace`
//! is the closest precedent — two proxima instances, one forwarding to the
//! other — but forwards over a hand-rolled `TcpStream` to prove trace
//! propagation without a client stack. This example forwards through
//! `proxima::Client`, the primitive that exists for exactly this job.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

use bytes::Bytes;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, Client, ListenerSpec, PipeHandle, ProximaError, Request, Response, SendPipe, into_handle,
};
use proxima_macros::piped;

const ORIGIN_BIND: &str = "127.0.0.1:8081";
const PROXY_BIND: &str = "127.0.0.1:8080";

/// The upstream. A trivial pipe returning a known status, header, and body —
/// deliberately distinct from a plain 200 so the proxy's forwarded response
/// can be checked field-by-field against something a passthrough couldn't
/// fake by accident. Stateless, so `#[proxima::piped]` writes the `SendPipe`
/// impl.
#[proxima::piped(send)]
async fn origin_pipe(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::new(201)
        .with_header("x-origin", "proxima-origin")
        .with_body("origin response body\n"))
}

/// The proxy. `client` is bound to the origin's base URL; forwarding is
/// handing the inbound request straight to `Client`'s own `SendPipe` impl —
/// the same dispatch seam `RequestBuilder::send` uses — and returning
/// whatever it returns. No field copying, no rewriting: the request that
/// reaches the origin and the response that reaches the caller are the same
/// values a hand-built HTTP round trip would produce, because `Client` IS
/// the HTTP round trip.
struct ProxyPipe {
    client: Client,
}

#[piped(send)]
impl ProxyPipe {
    async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let client = self.client.clone();
        SendPipe::call(&client, request).await
    }
}

// each app below builds its own independent runtime (no ambient one is
// installed here — `runtime = "tokio"` just gives `main` an async context
// to `.await` on), so `runtime = "tokio"` rather than `worker_threads`.
#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), ProximaError> {
    let origin_bind: SocketAddr = ORIGIN_BIND.parse().expect("valid socket addr");
    let proxy_bind: SocketAddr = PROXY_BIND.parse().expect("valid socket addr");

    // one core per app is enough for one listener answering one request —
    // set explicitly via the builder, no env var, no build-and-discard.
    let origin_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    origin_app.mount("/", origin_pipe)?;

    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let origin_listener = origin_app.build_listener(ListenerSpec::http(origin_bind))?;
    println!("origin listening on {origin_bind}");

    // the proxy's whole config is "where's the upstream" — Client::http
    // resolves the same prime HTTP backend a hand-built client would use.
    let client = Client::http(format!("http://{origin_bind}"))?;
    let proxy_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    let proxy_pipe: PipeHandle = into_handle(ProxyPipe { client });
    proxy_app.mount("/", proxy_pipe)?;

    let proxy_listener = proxy_app.build_listener(ListenerSpec::http(proxy_bind))?;
    println!("proxy  listening on {proxy_bind}, forwards to {origin_bind}");

    let raw_response = blocking_get(proxy_bind);
    println!("\nclient -> proxy raw response:\n{raw_response}");

    assert!(
        raw_response.starts_with("HTTP/1.1 201"),
        "the proxy must return the origin's exact status, not invent its own: {raw_response:?}"
    );
    assert!(
        raw_response
            .to_ascii_lowercase()
            .contains("x-origin: proxima-origin"),
        "the origin's response header must survive the forward unchanged: {raw_response:?}"
    );
    assert!(
        raw_response.contains("origin response body"),
        "the origin's response body must survive the forward unchanged: {raw_response:?}"
    );
    println!(
        "\nPASS: forward-to-upstream is composition — the proxy pipe added no bytes, dropped none."
    );

    proxy_listener.shutdown();
    origin_listener.shutdown();
    let proxy_runtime = proxy_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("proxy app has no runtime installed".into()))?;
    let origin_runtime = origin_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("origin app has no runtime installed".into()))?;
    let proxy_report = ShutdownBarrier::new(proxy_runtime).broadcast_drop().await;
    let origin_report = ShutdownBarrier::new(origin_runtime).broadcast_drop().await;
    println!(
        "proxy  drained: cores_acked={} hooks_drained={}",
        proxy_report.cores_acked, proxy_report.hooks_drained
    );
    println!(
        "origin drained: cores_acked={} hooks_drained={}",
        origin_report.cores_acked, origin_report.hooks_drained
    );

    Ok(())
}

/// One-shot GET over a plain blocking `TcpStream` — the client hitting the
/// proxy, deliberately not another proxima pipe or runtime. `Connection:
/// close` lets us read to EOF instead of framing the body ourselves.
fn blocking_get(addr: SocketAddr) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    String::from_utf8_lossy(&raw).into_owned()
}
