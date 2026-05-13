//! A load balancer is `proxy`'s forward composed with a selection policy
//! over a pool: instead of one `Client` bound to one upstream, `N` clients
//! bound to `N` backends, and the pipe's `call` picks which one to forward
//! to before doing exactly what `ProxyPipe` does. `fan-in` already taught
//! "many sources, one merged stream, pull only the ready" — this is the
//! mirror shape: one request in, one backend picked out of many, skipping
//! whichever aren't ready (here: not healthy).
//!
//! ```sh
//! cargo run --example load-balance
//! ```
//!
//! See `examples/load-balance/README.md` for the full writeup.

use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use bytes::Bytes;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    App, Client, ListenerHandle, ListenerSpec, PipeHandle, ProximaError, Request, Response,
    SendPipe, into_handle,
};

const ORIGIN_A_BIND: &str = "127.0.0.1:8091";
const ORIGIN_B_BIND: &str = "127.0.0.1:8092";
const ORIGIN_C_BIND: &str = "127.0.0.1:8093";
const LOAD_BALANCER_BIND: &str = "127.0.0.1:8090";
const REQUEST_COUNT: u32 = 12;

/// One origin backend: answers every request with its own id stamped into a
/// header, and counts how many requests it actually served. That counter,
/// not the load balancer's own bookkeeping, is the ground truth the
/// distribution assertions check against.
struct OriginPipe {
    backend_id: &'static str,
    hits: Arc<AtomicU32>,
}

impl SendPipe for OriginPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let backend_id = self.backend_id;
        let hits = self.hits.clone();
        async move {
            hits.fetch_add(1, Ordering::SeqCst);
            Ok(Response::ok(format!("answered by {backend_id}\n"))
                .with_header("x-backend-id", backend_id))
        }
    }
}


/// A slot in the load balancer's pool: the client bound to that backend,
/// plus the health flag the selection policy checks before routing to it.
struct Backend {
    label: &'static str,
    healthy: bool,
    client: Client,
}

/// Round-robin over the healthy backends. The transform IS `proxy`'s
/// forward, aimed at whichever `Client` the selection policy picks this
/// call. `cursor` walks the full pool every call rather than a
/// healthy-only subset, so a backend that later recovers resumes its place
/// in rotation with no separate re-registration step.
struct LoadBalancerPipe {
    backends: Vec<Backend>,
    cursor: AtomicUsize,
}

impl LoadBalancerPipe {
    fn new(backends: Vec<Backend>) -> Self {
        Self {
            backends,
            cursor: AtomicUsize::new(0),
        }
    }

    // one full rotation past the last-served index, returning the first
    // healthy backend found; none healthy means the pool is down, not a
    // case to paper over with a fallback.
    fn select_backend(&self) -> Option<Client> {
        let backend_count = self.backends.len();
        for _ in 0..backend_count {
            let index = self.cursor.fetch_add(1, Ordering::SeqCst) % backend_count;
            let backend = &self.backends[index];
            if backend.healthy {
                return Some(backend.client.clone());
            }
        }
        None
    }
}

impl SendPipe for LoadBalancerPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let selected = self.select_backend();
        async move {
            match selected {
                Some(client) => SendPipe::call(&client, request).await,
                None => Err(ProximaError::Io(std::io::Error::other(
                    "load balancer: no healthy backend available",
                ))),
            }
        }
    }
}


// each app below builds its own independent runtime (no ambient one is
// installed here — `runtime = "tokio"` just gives `main` an async context
// to `.await` on), so `runtime = "tokio"` rather than `worker_threads`.
#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), ProximaError> {
    let origin_a_bind = parse_addr(ORIGIN_A_BIND)?;
    let origin_b_bind = parse_addr(ORIGIN_B_BIND)?;
    let origin_c_bind = parse_addr(ORIGIN_C_BIND)?;
    let load_balancer_bind = parse_addr(LOAD_BALANCER_BIND)?;

    let (origin_a_app, origin_a_listener, origin_a_hits) =
        spin_up_origin(origin_a_bind, "origin-a")?;
    let (origin_b_app, origin_b_listener, origin_b_hits) =
        spin_up_origin(origin_b_bind, "origin-b")?;
    let (origin_c_app, origin_c_listener, origin_c_hits) =
        spin_up_origin(origin_c_bind, "origin-c")?;
    println!("origin-a listening on {origin_a_bind} (healthy)");
    println!("origin-b listening on {origin_b_bind} (marked unhealthy)");
    println!("origin-c listening on {origin_c_bind} (healthy)");

    let backends = vec![
        build_backend(origin_a_bind, "origin-a", true)?,
        build_backend(origin_b_bind, "origin-b", false)?,
        build_backend(origin_c_bind, "origin-c", true)?,
    ];
    let pool_description = describe_pool(&backends);
    let (load_balancer_app, load_balancer_listener) =
        spin_up_load_balancer(load_balancer_bind, backends)?;
    println!("load balancer listening on {load_balancer_bind}, pool: {pool_description}\n");

    drive_requests(load_balancer_bind, REQUEST_COUNT)?;

    let origin_a_count = origin_a_hits.load(Ordering::SeqCst);
    let origin_b_count = origin_b_hits.load(Ordering::SeqCst);
    let origin_c_count = origin_c_hits.load(Ordering::SeqCst);
    println!(
        "\nper-backend counts: origin-a={origin_a_count} origin-b={origin_b_count} origin-c={origin_c_count}"
    );
    assert_distribution(
        origin_a_count,
        origin_b_count,
        origin_c_count,
        REQUEST_COUNT,
    );
    println!(
        "PASS: distributed across healthy backends only, unhealthy backend saw zero requests."
    );

    drain_and_report("load balancer", &load_balancer_app, load_balancer_listener).await?;
    drain_and_report("origin-a", &origin_a_app, origin_a_listener).await?;
    drain_and_report("origin-b", &origin_b_app, origin_b_listener).await?;
    drain_and_report("origin-c", &origin_c_app, origin_c_listener).await?;

    Ok(())
}

fn parse_addr(raw: &str) -> Result<SocketAddr, ProximaError> {
    raw.parse().map_err(|parse_error| {
        ProximaError::Config(format!("invalid socket address {raw}: {parse_error}"))
    })
}

// stands up one origin listener and returns its shutdown handle plus the
// live hit counter its `OriginPipe` increments per request.
fn spin_up_origin(
    bind: SocketAddr,
    backend_id: &'static str,
) -> Result<(App, ListenerHandle, Arc<AtomicU32>), ProximaError> {
    let hits = Arc::new(AtomicU32::new(0));
    // one core per app is enough for one listener answering one request at
    // a time — set explicitly via the builder, no env var, no
    // build-and-discard.
    let app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    let pipe: PipeHandle = into_handle(OriginPipe {
        backend_id,
        hits: hits.clone(),
    });
    app.mount("/", pipe)?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let listener = app.build_listener(ListenerSpec::http(bind))?;
    Ok((app, listener, hits))
}

fn build_backend(
    bind: SocketAddr,
    label: &'static str,
    healthy: bool,
) -> Result<Backend, ProximaError> {
    let client = Client::http(format!("http://{bind}"))?;
    Ok(Backend {
        label,
        healthy,
        client,
    })
}

// renders each backend's label with its health so the pool print reflects
// the actual `Backend` values passed to the load balancer, not a guess.
fn describe_pool(backends: &[Backend]) -> String {
    backends
        .iter()
        .map(|backend| {
            let health = if backend.healthy {
                "healthy"
            } else {
                "unhealthy"
            };
            format!("{}({health})", backend.label)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn spin_up_load_balancer(
    bind: SocketAddr,
    backends: Vec<Backend>,
) -> Result<(App, ListenerHandle), ProximaError> {
    let app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    let pipe: PipeHandle = into_handle(LoadBalancerPipe::new(backends));
    app.mount("/", pipe)?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let listener = app.build_listener(ListenerSpec::http(bind))?;
    Ok((app, listener))
}

// drives `request_count` real HTTP requests through the load balancer,
// printing which backend answered each one (parsed from the response the
// origin stamped, not from the load balancer's internal state).
fn drive_requests(bind: SocketAddr, request_count: u32) -> Result<(), ProximaError> {
    for request_number in 1..=request_count {
        let raw_response = blocking_get(bind)?;
        let served_by = extract_backend_id(&raw_response);
        println!("request {request_number:>2}: served by {served_by}");
    }
    Ok(())
}

fn assert_distribution(
    origin_a_count: u32,
    origin_b_count: u32,
    origin_c_count: u32,
    request_count: u32,
) {
    assert_eq!(
        origin_b_count, 0,
        "unhealthy backend must never be selected"
    );
    assert!(
        origin_a_count > 0,
        "every healthy backend must serve at least one request"
    );
    assert!(
        origin_c_count > 0,
        "every healthy backend must serve at least one request"
    );
    assert_eq!(
        origin_a_count + origin_c_count,
        request_count,
        "every request lands on exactly one healthy backend"
    );
    assert_eq!(
        origin_a_count, origin_c_count,
        "round-robin over 2 healthy backends splits an even request count exactly in half"
    );
}

async fn drain_and_report(
    label: &str,
    app: &App,
    listener: ListenerHandle,
) -> Result<(), ProximaError> {
    listener.shutdown();
    let runtime = app
        .runtime()
        .ok_or_else(|| ProximaError::Config(format!("{label} app has no runtime installed")))?;
    let report = ShutdownBarrier::new(runtime).broadcast_drop().await;
    println!(
        "{label} drained: cores_acked={} hooks_drained={}",
        report.cores_acked, report.hooks_drained
    );
    Ok(())
}

/// One-shot GET over a plain blocking `TcpStream` — the client hitting the
/// load balancer, deliberately not another proxima pipe or runtime.
/// `Connection: close` lets us read to EOF instead of framing the body
/// ourselves.
fn blocking_get(addr: SocketAddr) -> Result<String, ProximaError> {
    let mut stream = TcpStream::connect(addr)?;
    stream.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

// pulls the `x-backend-id` header value back out of a raw HTTP response —
// proof of which origin actually answered, read from the wire, not assumed.
fn extract_backend_id(raw_response: &str) -> &str {
    for line in raw_response.lines() {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("x-backend-id")
        {
            return value.trim();
        }
    }
    "unknown"
}
