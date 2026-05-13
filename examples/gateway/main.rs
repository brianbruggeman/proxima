//! A gateway is the `proxy` example's forward pipe with a policy chain
//! composed in front of it. No new primitive family: `Auth<Inner>` and
//! `RoutingPipe<Handle>` are ordinary `Pipe`s (the same vocabulary `filter`
//! and `gate` taught), and `RateLimit<Inner, Extractor, Clk>` is the token-
//! bucket gate from `rate_limit` — a request rejected by one policy never
//! reaches the next.
//!
//! Chain, outside-in: `Auth` (401 on a missing/wrong bearer token) wraps
//! `RoutingPipe` (picks an upstream by path prefix) which routes to one of
//! two `RateLimit<ForwardPipe>` stacks (429 once a per-upstream budget is
//! exhausted), each wrapping a `ForwardPipe` — the one-line
//! `SendPipe::call(&client, request)` forward from `proxy`.
//!
//! ```sh
//! cargo run --example gateway
//! ```
//!
//! See `examples/gateway/README.md` for the full writeup.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, Auth, Client, KeyExtractor, ListenerHandle, ListenerSpec, PipeHandle, ProximaError,
    RateLimit, RateLimitCaps, Request, Response, RoutingPipe, SendPipe, TokenBucketConfig,
    into_handle,
};
use proxima_primitives::pipe::capabilities::Clock;

const ORIGIN_API_BIND: &str = "127.0.0.1:8091";
const ORIGIN_WEB_BIND: &str = "127.0.0.1:8092";
const GATEWAY_BIND: &str = "127.0.0.1:8090";

const VALID_TOKEN: &str = "let-me-in";
const RATE_LIMIT_CAPACITY: u64 = 2;

// each app below builds its own independent runtime (no ambient one is
// installed here — `runtime = "tokio"` just gives `main` an async context
// to `.await` on), so `runtime = "tokio"` rather than `worker_threads`.
#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), ProximaError> {
    let origin_api_bind: SocketAddr = ORIGIN_API_BIND.parse().expect("valid socket addr");
    let origin_web_bind: SocketAddr = ORIGIN_WEB_BIND.parse().expect("valid socket addr");
    let gateway_bind: SocketAddr = GATEWAY_BIND.parse().expect("valid socket addr");

    let api_calls = Arc::new(AtomicUsize::new(0));
    let web_calls = Arc::new(AtomicUsize::new(0));

    let (origin_api_app, origin_api_listener) =
        spawn_origin(origin_api_bind, "api", Arc::clone(&api_calls))?;
    let (origin_web_app, origin_web_listener) =
        spawn_origin(origin_web_bind, "web", Arc::clone(&web_calls))?;
    println!("origin (api) listening on {origin_api_bind}");
    println!("origin (web) listening on {origin_web_bind}");

    // deterministic budget: capacity=2, refill_per_sec=0 means the boundary
    // is reached purely by call count, never by wall-clock timing. `FakeClock`
    // is still injected (never real time) — see `rate_limit`'s section 3 for
    // the same seam driven with an advancing clock instead.
    let clock = FakeClock::default();

    let api_client = Client::http(format!("http://{origin_api_bind}"))?;
    let api_forward: PipeHandle = into_handle(RateLimit::with_clock(
        ForwardPipe { client: api_client },
        TokenBucketConfig {
            capacity: RATE_LIMIT_CAPACITY,
            refill_per_sec: 0,
        },
        KeyExtractor::ConstantKey("api-upstream".into()),
        RateLimitCaps::default(),
        clock.clone(),
    ));

    let web_client = Client::http(format!("http://{origin_web_bind}"))?;
    let web_forward: PipeHandle = into_handle(RateLimit::with_clock(
        ForwardPipe { client: web_client },
        TokenBucketConfig {
            capacity: RATE_LIMIT_CAPACITY,
            refill_per_sec: 0,
        },
        KeyExtractor::ConstantKey("web-upstream".into()),
        RateLimitCaps::default(),
        clock,
    ));

    // "/api/..." goes to the api upstream; everything else falls through to
    // web — the routing decision itself, not a network detail.
    let routed = RoutingPipe::new("gateway-router")
        .route("/api/{*rest}", api_forward)
        .fallback(web_forward);

    // wraps the router, so a rejected request never reaches routing or the
    // rate limit at all — the same short-circuit `filter` demonstrated.
    let gateway_pipe = Auth {
        inner: into_handle(routed),
        header: "authorization".to_string(),
        allow: BTreeSet::from([VALID_TOKEN.to_string()]),
        realm: Arc::from(b"gateway".as_slice()),
        on_unauthorized_status: 401,
        strip_prefix: Some("Bearer ".to_string()),
    };

    // one core per app is enough for one listener answering one request —
    // set explicitly via the builder, no env var, no build-and-discard.
    let gateway_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    // wildcard mount: the App's own top-level router pattern-matches paths
    // exactly (see `path_pattern`), so a catch-all is needed to let
    // "/api/..." and "/web/..." both reach the gateway pipe.
    gateway_app.mount("/{*rest}", into_handle(gateway_pipe))?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let gateway_listener = gateway_app.build_listener(ListenerSpec::http(gateway_bind))?;
    println!("gateway listening on {gateway_bind}\n");

    run_scenarios(gateway_bind, &api_calls, &web_calls);

    gateway_listener.shutdown();
    origin_api_listener.shutdown();
    origin_web_listener.shutdown();
    let gateway_runtime = gateway_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("gateway app has no runtime installed".into()))?;
    let origin_api_runtime = origin_api_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("origin api app has no runtime installed".into()))?;
    let origin_web_runtime = origin_web_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("origin web app has no runtime installed".into()))?;
    let gateway_report = ShutdownBarrier::new(gateway_runtime).broadcast_drop().await;
    let origin_api_report = ShutdownBarrier::new(origin_api_runtime).broadcast_drop().await;
    let origin_web_report = ShutdownBarrier::new(origin_web_runtime).broadcast_drop().await;
    println!(
        "gateway    drained: cores_acked={} hooks_drained={}",
        gateway_report.cores_acked, gateway_report.hooks_drained
    );
    println!(
        "origin api drained: cores_acked={} hooks_drained={}",
        origin_api_report.cores_acked, origin_api_report.hooks_drained
    );
    println!(
        "origin web drained: cores_acked={} hooks_drained={}",
        origin_web_report.cores_acked, origin_web_report.hooks_drained
    );

    Ok(())
}

fn run_scenarios(
    gateway_bind: SocketAddr,
    api_calls: &Arc<AtomicUsize>,
    web_calls: &Arc<AtomicUsize>,
) {
    println!("auth: missing token never reaches route or the origin");
    let missing = blocking_request(gateway_bind, "/api/orders", &[]);
    println!("{missing}");
    assert!(
        missing.starts_with("HTTP/1.1 401"),
        "missing token: {missing:?}"
    );
    assert_eq!(
        api_calls.load(Ordering::Relaxed),
        0,
        "auth rejects before the origin is ever hit"
    );

    println!("auth: wrong token also rejects (401), origin still untouched");
    let wrong = blocking_request(
        gateway_bind,
        "/api/orders",
        &[("authorization", "Bearer nope")],
    );
    println!("{wrong}");
    assert!(wrong.starts_with("HTTP/1.1 401"), "wrong token: {wrong:?}");
    assert_eq!(
        api_calls.load(Ordering::Relaxed),
        0,
        "a wrong token is still a rejection, not a forward"
    );

    let auth_header = ("authorization", "Bearer let-me-in");

    println!("route: authorized \"/api/...\" forwards to the api upstream");
    let api_response = blocking_request(gateway_bind, "/api/orders", &[auth_header]);
    println!("{api_response}");
    assert!(
        api_response.starts_with("HTTP/1.1 200"),
        "api forward: {api_response:?}"
    );
    assert!(
        api_response
            .to_ascii_lowercase()
            .contains("x-upstream: api"),
        "must reach the api origin, not web: {api_response:?}"
    );
    assert_eq!(
        api_calls.load(Ordering::Relaxed),
        1,
        "one request reached the api origin"
    );

    println!("route: authorized \"/web/...\" falls through to the web upstream");
    let web_response = blocking_request(gateway_bind, "/web/home", &[auth_header]);
    println!("{web_response}");
    assert!(
        web_response.starts_with("HTTP/1.1 200"),
        "web forward: {web_response:?}"
    );
    assert!(
        web_response
            .to_ascii_lowercase()
            .contains("x-upstream: web"),
        "must reach the web origin, not api: {web_response:?}"
    );
    assert_eq!(
        web_calls.load(Ordering::Relaxed),
        1,
        "one request reached the web origin"
    );

    println!("rate-limit: the api budget (capacity=2) admits its second call");
    let second_api = blocking_request(gateway_bind, "/api/orders", &[auth_header]);
    println!("{second_api}");
    assert!(
        second_api.starts_with("HTTP/1.1 200"),
        "second api call: {second_api:?}"
    );
    assert_eq!(
        api_calls.load(Ordering::Relaxed),
        2,
        "the api bucket had exactly one token left"
    );

    println!("rate-limit: a third call exceeds the budget (429), origin never hit");
    let throttled = blocking_request(gateway_bind, "/api/orders", &[auth_header]);
    println!("{throttled}");
    assert!(
        throttled.starts_with("HTTP/1.1 429"),
        "throttled call: {throttled:?}"
    );
    assert!(
        throttled.to_ascii_lowercase().contains("retry-after: 1"),
        "a 429 carries a retry-after hint: {throttled:?}"
    );
    assert_eq!(
        api_calls.load(Ordering::Relaxed),
        2,
        "the throttled call never reached the origin"
    );

    println!(
        "\nPASS: auth rejects before route, route sends each prefix to its own upstream, \
         rate-limit throttles per upstream before the forward — three composed policies, \
         no bytes copied by hand."
    );
}

// ORIGIN: a plain answering pipe, distinct per upstream so a forwarded
// response is provably from the origin the router chose, not a passthrough
// default

struct OriginPipe {
    label: &'static str,
    calls: Arc<AtomicUsize>,
}

impl SendPipe for OriginPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let label = self.label;
        async move {
            Ok(Response::new(200)
                .with_header("x-upstream", label)
                .with_body(format!("{label} origin response\n")))
        }
    }
}


fn spawn_origin(
    bind: SocketAddr,
    label: &'static str,
    calls: Arc<AtomicUsize>,
) -> Result<(App, ListenerHandle), ProximaError> {
    let app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    app.mount(
        "/{*rest}",
        into_handle(OriginPipe { label, calls }),
    )?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let listener = app.build_listener(ListenerSpec::http(bind))?;
    Ok((app, listener))
}

// FORWARD: the proxy example's whole pipe, reused as RateLimit's inner —
// forwarding IS `Client`'s own `SendPipe` impl, nothing added

#[derive(Clone)]
struct ForwardPipe {
    client: Client,
}

impl SendPipe for ForwardPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let client = self.client.clone();
        async move { SendPipe::call(&client, request).await }
    }
}


// deterministic clock: real time never advances on its own, so the
// rate-limit boundary in scenario 5/6 is reached by call count alone

#[derive(Clone, Default)]
struct FakeClock {
    now_nanos: Arc<AtomicU64>,
}

impl Clock for FakeClock {
    type Delay = core::future::Ready<()>;

    fn now_nanos(&self) -> u64 {
        self.now_nanos.load(Ordering::Relaxed)
    }

    fn delay(&self, _duration: Duration) -> Self::Delay {
        core::future::ready(())
    }
}

// test driver: plain blocking TCP, deliberately not another proxima
// runtime or client — proves the wire, not the harness

fn blocking_request(addr: SocketAddr, path: &str, headers: &[(&str, &str)]) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    let mut request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    String::from_utf8_lossy(&raw).into_owned()
}
