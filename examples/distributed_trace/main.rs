//! Distributed trace propagation across TWO proxima instances in one process.
//!
//! A client hits instance A ("front"); A forwards to instance B ("origin") over
//! a real TCP/HTTP hop. The question this example answers empirically: do A's
//! span and B's span land in the SAME trace (one `trace_id`, B parented under
//! A), or do they come out as two disconnected traces?
//!
//! ```sh
//! cargo run --example distributed_trace
//! ```
//!
//! See `examples/distributed_trace.README.md` for the full writeup, including
//! the `#[instrument(parent = ...)]` seam that connects the span layer to the
//! `RequestContext` each hop's H1 listener already extracted at ingress.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

use bytes::Bytes;
use proxima::shutdown::ShutdownBarrier;
use proxima::telemetry::id::parse_traceparent;
use proxima::telemetry::pipes::{FormatterPipe, InMemoryPipe, LogFormat, TelemetryRequest};
use proxima::telemetry::recorder::Recorder;
use proxima::{
    App, HeaderList, ListenerSpec, PipeHandle, ProximaError, Request, Response,
    SendPipe, into_handle,
};

const FRONT_BIND: &str = "127.0.0.1:8091";
const ORIGIN_BIND: &str = "127.0.0.1:8092";

/// Instance A. Forwards every request to `origin_addr`, injecting its own
/// `RequestContext`'s W3C trace context onto the outbound request first.
struct FrontPipe {
    origin_addr: SocketAddr,
}

impl SendPipe for FrontPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let origin_addr = self.origin_addr;
        async move {
            let front_traceparent = request
                .context
                .trace_id
                .as_deref()
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .unwrap_or_default();

            // `request.context.traceparent()` is the boundary seam: A's H1
            // listener already called `establish_trace_context` + `adopt_trace_context`
            // on ingress, so
            // this is A's own restamped span context, ready to hand straight
            // to `parent =` — no ambient lookup, just the value already
            // sitting on the request.
            front_hop(request.context.traceparent());

            // egress-inject: stamp THIS request's trace context onto the
            // outbound headers. This is the fix — no forwarding path anywhere
            // in proxima calls `inject_propagation` today (every existing
            // caller is a unit test), so without this line B never sees a
            // `traceparent` header and mints an unrelated trace of its own.
            let mut outbound_headers = HeaderList::new();
            request.context.inject_propagation(&mut outbound_headers);

            let raw_response = blocking_forward(origin_addr, &outbound_headers);
            let origin_traceparent = extract_marker(&raw_response, "origin_traceparent=");

            let body = format!(
                "front_traceparent={front_traceparent}\norigin_traceparent={origin_traceparent}\n"
            );
            Ok(Response::ok(body))
        }
    }
}


/// Instance B. Its own H1 listener already calls `establish_trace_context` +
/// `adopt_trace_context` on ingress (see `proxima-http/src/http1/serve.rs`), so
/// `request.context.trace_id` reflects
/// whatever the inbound `traceparent` carried, restamped with B's own span id.
/// A bare `async fn` mounts directly, no attribute needed. No `#[proxima::instrument]`
/// here on purpose: `origin_hop` below already opens the one span this example
/// measures (`name == "origin"`, parent-linked by hand); an auto-span on the
/// handler itself would add an untested, un-parented span next to it and
/// change the exact `spans().len()` this example's proof depends on.
async fn origin_pipe(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    let origin_traceparent = request
        .context
        .trace_id
        .as_deref()
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default();

    // same seam as the front hop: B's H1 listener already extracted
    // the inbound `traceparent` A injected, so B's own span joins the
    // same trace instead of minting an unrelated one.
    origin_hop(request.context.traceparent());

    Ok(Response::ok(format!(
        "origin_traceparent={origin_traceparent}\n"
    )))
}


// one #[instrument] span per instance, each handed its own instance's
// `RequestContext::traceparent()` via the explicit `parent =` seam: `Some`
// continues the caller's trace (`span_from_traceparent` under the hood,
// inheriting trace_id + recording parent_span_id); `None` would fall back to
// a fresh root, same as no `parent` arg at all.
#[proxima::telemetry::instrument(name = "front", parent = parent)]
fn front_hop(parent: Option<&[u8]>) {}

#[proxima::telemetry::instrument(name = "origin", parent = parent)]
fn origin_hop(parent: Option<&[u8]>) {}

/// Fans every telemetry record out to a real console sink (`FormatterPipe`,
/// what `Exporter::stdout()` builds internally) AND an in-memory capture
/// buffer, so the exported spans are both visible on stdout and inspectable
/// after the run. `Exporter::export()` is single-sink today (fan-out over
/// multiple exporters is noted in `proxima-telemetry/src/export.rs` as a
/// future OTLP-slice stage), so this composes the two underlying pipes by
/// hand — still real, already-built primitives, no new plumbing.
struct DualSinkPipe {
    console: Arc<FormatterPipe<std::io::Stdout>>,
    memory: InMemoryPipe,
}

impl SendPipe for DualSinkPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let console = Arc::clone(&self.console);
        let memory = self.memory.clone();
        async move {
            memory.call(request.clone()).await?;
            console.call(request).await
        }
    }
}

// both apps below build their own independent runtime (no ambient one is
// installed here — `runtime = "tokio"` just gives `main` an async context
// to `.await` on), so `runtime = "tokio"` rather than `worker_threads`.
#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), ProximaError> {
    let memory = InMemoryPipe::new();
    let dual_sink = DualSinkPipe {
        console: Arc::new(FormatterPipe::new(std::io::stdout(), LogFormat::Human)),
        memory: memory.clone(),
    };
    let recorder = Recorder::builder()
        .pipe(dual_sink)
        .install()
        .expect("recorder install");

    let front_bind: SocketAddr = FRONT_BIND.parse().expect("valid socket addr");
    let origin_bind: SocketAddr = ORIGIN_BIND.parse().expect("valid socket addr");

    let origin_app = App::builder()
        .with_runtime_cores(2)
        .with_defaults()?
        .build()?;
    origin_app.mount("/", origin_pipe)?;

    let front_app = App::builder()
        .with_runtime_cores(2)
        .with_defaults()?
        .build()?;
    let front_pipe: PipeHandle = into_handle(FrontPipe {
        origin_addr: origin_bind,
    });
    front_app.mount("/", front_pipe)?;

    // each blocks until its own accept lane has acked ready — no polling,
    // no sleeping.
    let origin_listener = origin_app.build_listener(ListenerSpec::http(origin_bind))?;
    let front_listener = front_app.build_listener(ListenerSpec::http(front_bind))?;

    println!("origin (B) listening on {origin_bind}");
    println!("front  (A) listening on {front_bind}, forwards to {origin_bind}");

    let raw_response = blocking_get(front_bind);
    println!("\nclient -> front raw response:\n{raw_response}");

    let front_traceparent = extract_marker(&raw_response, "front_traceparent=");
    let origin_traceparent = extract_marker(&raw_response, "origin_traceparent=");
    let front_header_parsed = parse_traceparent(front_traceparent.as_bytes());
    let origin_header_parsed = parse_traceparent(origin_traceparent.as_bytes());
    let front_header_trace = front_header_parsed.map(|(trace_id, ..)| trace_id);
    let origin_header_trace = origin_header_parsed.map(|(trace_id, ..)| trace_id);
    // The span-id component of front's OWN restamped traceparent — the value
    // `establish_trace_context` now preserves (instead of
    // discarding) when origin re-extracts it from the inbound `traceparent`
    // header. This is the literal cross-hop parent link, proven below.
    let front_header_span = front_header_parsed.map(|(_, span_id, _)| span_id);

    let drained = recorder.drain();
    let spans = memory.spans();
    println!(
        "drained {drained} telemetry records, {} spans captured:",
        spans.len()
    );
    for span in &spans {
        let parent = span
            .parent_span_id
            .map_or_else(|| "-".to_string(), |id| id.to_string());
        println!(
            "  name={:<8} trace_id={} span_id={} parent_span_id={parent}",
            span.name, span.trace_id, span.span_id
        );
    }
    let front_span = spans
        .iter()
        .find(|span| span.name == "front")
        .expect("front span recorded");
    let origin_span = spans
        .iter()
        .find(|span| span.name == "origin")
        .expect("origin span recorded");

    let header_connected =
        matches!((front_header_trace, origin_header_trace), (Some(a), Some(b)) if a == b);
    let span_connected = front_span.trace_id == origin_span.trace_id;

    println!("\n--- validation ---");
    println!(
        "W3C header layer (RequestContext.trace_id via inject_propagation/establish_trace_context):"
    );
    println!("  front  traceparent = {front_traceparent}");
    println!("  origin traceparent = {origin_traceparent}");
    println!(
        "  -> {}",
        if header_connected {
            "CONNECTED: same trace_id crossed the A -> B hop"
        } else {
            "SPLIT"
        }
    );
    assert!(
        header_connected,
        "the egress-inject fix (context.inject_propagation) must make the header-level \
         trace_id match end to end — if this fails, the wire-level propagation itself is broken"
    );

    println!(
        "\ntelemetry span layer (#[proxima::telemetry::instrument(parent = ...)] on each pipe):"
    );
    println!("  front  span trace_id = {}", front_span.trace_id);
    println!("  origin span trace_id = {}", origin_span.trace_id);
    println!(
        "  -> {}",
        if span_connected {
            "CONNECTED: one trace, two spans"
        } else {
            "SPLIT: two independent traces"
        }
    );
    assert!(
        span_connected,
        "front_hop/origin_hop each pass `parent = request.context.traceparent()`, so the span \
         layer must land in the SAME trace as the header layer above — if this fails, either the \
         `parent =` macro arg or the `RequestContext::traceparent()` boundary helper regressed"
    );
    // The literal cross-hop link: `establish_trace_context` now
    // PRESERVES the inbound span-id (see `proxima-telemetry/src/propagation.rs`)
    // instead of discarding it and minting its own — so origin's context
    // carries front's own span-id verbatim, and `origin_hop`'s
    // `parent = request.context.traceparent()` records exactly that as its
    // `parent_span_id`. Not just "one shared trace_id" (proven above) but a
    // real single-tree reconstruction: origin's recorded parent IS the
    // span-id front put on the wire.
    println!(
        "\nliteral parent_span_id chain (establish_trace_context preserves the inbound span-id):"
    );
    println!("  front header span-id  = {front_header_span:?}");
    println!(
        "  origin span parent_span_id = {:?}",
        origin_span.parent_span_id
    );
    assert_eq!(
        origin_span.parent_span_id, front_header_span,
        "origin's span must record front's own wire-level span-id as its parent -- if this \
         fails, `establish_trace_context` regressed back to discarding the inbound \
         span-id"
    );

    println!("\nPASS: distributed tracing across two proxima instances lands in ONE trace.");
    println!(
        "      Both layers agree: the header layer via inject_propagation/establish_trace_context,"
    );
    println!(
        "      the span layer via #[instrument(parent = request.context.traceparent())] routing"
    );
    println!("      to `Recorder::span_from_traceparent` instead of a fresh root.");
    println!(
        "      The literal parent_span_id chain crosses the wire hop too: establish_trace_context"
    );
    println!("      preserves the inbound span-id instead of discarding it.");

    origin_listener.shutdown();
    front_listener.shutdown();
    let origin_runtime = origin_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("origin app has no runtime installed".into()))?;
    let front_runtime = front_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("front app has no runtime installed".into()))?;
    let origin_report = ShutdownBarrier::new(origin_runtime).broadcast_drop().await;
    let front_report = ShutdownBarrier::new(front_runtime).broadcast_drop().await;
    println!(
        "\norigin drained: cores_acked={} hooks_drained={}",
        origin_report.cores_acked, origin_report.hooks_drained
    );
    println!(
        "front  drained: cores_acked={} hooks_drained={}",
        front_report.cores_acked, front_report.hooks_drained
    );

    Ok(())
}

fn blocking_get(addr: SocketAddr) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    String::from_utf8_lossy(&raw).into_owned()
}

/// Blocking on purpose: A's forward to B is one localhost round trip per demo
/// request, driven synchronously inside the async `FrontPipe::call` body —
/// proving trace propagation doesn't need a second HTTP client stack.
fn blocking_forward(addr: SocketAddr, headers: &HeaderList) -> String {
    let mut request_text =
        String::from("GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    for (name, value) in headers.iter() {
        request_text.push_str(&String::from_utf8_lossy(name));
        request_text.push_str(": ");
        request_text.push_str(&String::from_utf8_lossy(value));
        request_text.push_str("\r\n");
    }
    request_text.push_str("\r\n");

    let mut stream = TcpStream::connect(addr).expect("connect to origin");
    stream
        .write_all(request_text.as_bytes())
        .expect("write forwarded request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read origin response");
    String::from_utf8_lossy(&raw).into_owned()
}

/// Pull a `key=value` marker out of a raw HTTP response body, stopping at the
/// next newline — same technique as `multi_runtime.rs`'s `extract_shared_total`.
fn extract_marker(text: &str, marker: &str) -> String {
    let start = text
        .find(marker)
        .map(|position| position + marker.len())
        .unwrap_or_else(|| panic!("{marker} not found in response: {text:?}"));
    let rest = &text[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    rest[..end].trim().to_string()
}
