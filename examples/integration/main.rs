//! Integrating a third-party API is `proxy` + `record` + `replay` composed:
//! front the vendor (proxy), tee everything it actually returns onto a
//! cassette (record), then swap the fronted call for the capture (replay)
//! so tests and local dev never touch the vendor again.
//!
//! Phase 1 (LIVE) mounts `RecordUpstream<Client>` as the edge's own pipe —
//! the exact "forward via `Client`" shape `proxy` uses, with a cassette tee
//! bolted on. A real client hits the edge, the edge really calls the
//! vendor, and the interaction lands on disk.
//!
//! Phase 2 (REPLAY) kills the vendor for real — its `App` is drained — then
//! rebuilds the edge from `ReplayUpstream` alone. Same bind address, same
//! client code, no upstream call: the response comes straight off the
//! cassette `record` wrote, byte-identical to what was captured.
//!
//! Run: `cargo run --example integration`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use proxima::prime::PrimeRuntime;
use proxima::shutdown::ShutdownBarrier;
use proxima::{
    AccumulatingSink, App, Client, DynRecordingSink, FormatKind, HttpEvent, JsonlSource,
    LazyFanOut, ListenerSpec, ProtocolEvent, ProximaError, RecordUpstream, RecordingEvent,
    RecordingSource, ReplayUpstream, Request, Response, Runtime, SendPipe, SinkSpec,
    TerminalSignal, deferred_runtime, into_handle,
};

const ORIGIN_BIND: &str = "127.0.0.1:8095";
const EDGE_BIND: &str = "127.0.0.1:8096";
const VENDOR_HEADER: &str = "x-vendor";
const VENDOR_ID: &str = "acme-quotes-api";
const VENDOR_BODY: &str = "{\"symbol\":\"ACME\",\"price\":42.17}\n";

/// Stand-in for a real third-party API: a fixed status, header, and body so
/// the capture-then-replay proof has a known payload to check against.
/// Stateless, so `#[proxima::piped]` writes the `SendPipe` impl.
#[proxima::piped(send)]
async fn third_party_pipe(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::new(200)
        .with_header(VENDOR_HEADER, VENDOR_ID)
        .with_body(VENDOR_BODY))
}


// each app below builds its own independent runtime (no ambient one is
// installed here — `runtime = "tokio"` just gives `main` an async context
// to `.await` on), so `runtime = "tokio"` rather than `worker_threads`.
#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), ProximaError> {
    let origin_bind: SocketAddr = ORIGIN_BIND.parse().expect("valid socket addr");
    let edge_bind: SocketAddr = EDGE_BIND.parse().expect("valid socket addr");
    let cassette_dir = tempfile::tempdir()?;
    let cassette_path = cassette_dir.path().join("vendor.jsonl");

    println!("phase 1: LIVE — front the vendor, record every response\n");

    // one core per app is enough for one listener answering one request —
    // set explicitly via the builder, no env var, no build-and-discard.
    let origin_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    origin_app.mount("/", third_party_pipe)?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let origin_listener = origin_app.build_listener(ListenerSpec::http(origin_bind))?;
    println!("vendor (third-party) listening on {origin_bind}");

    // the edge App is built first so its runtime already exists; one spigot
    // arms both the durable sink and the recorder's own drainer, sharing
    // that SAME runtime — no separate, independently-built runtime to keep
    // in sync with the App's.
    let edge_live_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    let edge_runtime = edge_live_app.runtime().expect("builder installs a runtime");
    let spigot = deferred_runtime();
    spigot.set(Arc::clone(&edge_runtime)).ok();
    let durable = Arc::new(LazyFanOut::new(
        vec![SinkSpec::new(
            cassette_path.to_string_lossy(),
            FormatKind::Json,
        )],
        spigot.clone(),
    ));
    let accumulating: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
    let terminal = Arc::new(TerminalSignal::new(
        accumulating,
        |event: &RecordingEvent| {
            matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. }))
        },
    ));
    let sink: DynRecordingSink = terminal.clone();

    let client = Client::http(format!("http://{origin_bind}"))?;
    let recorder =
        RecordUpstream::new("live-front", client, sink, "third-party").with_runtime(spigot);

    edge_live_app.mount("/", into_handle(recorder))?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let edge_live_listener = edge_live_app.build_listener(ListenerSpec::http(edge_bind))?;
    println!("edge (live front) listening on {edge_bind}, forwards to {origin_bind}\n");

    let live_response = blocking_get(edge_bind);
    println!("client -> edge -> vendor:\n{live_response}");
    assert!(
        live_response.contains(VENDOR_BODY.trim_end()),
        "the edge must forward the vendor's exact body: {live_response:?}"
    );
    assert!(
        live_response.to_ascii_lowercase().contains(&format!(
            "{VENDOR_HEADER}: {}",
            VENDOR_ID.to_ascii_lowercase()
        )),
        "the edge must forward the vendor's exact header: {live_response:?}"
    );

    println!("awaiting terminal.drained() before the cassette is read back...");
    terminal.drained().await;

    edge_live_listener.shutdown();
    origin_listener.shutdown();
    let edge_live_report = ShutdownBarrier::new(edge_runtime).broadcast_drop().await;
    let origin_runtime = origin_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("origin app has no runtime installed".into()))?;
    let origin_report = ShutdownBarrier::new(origin_runtime).broadcast_drop().await;
    println!(
        "edge (live) drained: cores_acked={} hooks_drained={}",
        edge_live_report.cores_acked, edge_live_report.hooks_drained
    );
    println!(
        "vendor drained: cores_acked={} hooks_drained={} -- the vendor is now GONE\n",
        origin_report.cores_acked, origin_report.hooks_drained
    );

    println!("phase 2: REPLAY — serve the capture, no vendor required\n");

    let cassette_runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1)?);
    let recorded_body = recorded_response_body(&cassette_path, cassette_runtime).await?;

    // the fake edge App is built first (one core, set explicitly) so
    // `ReplayUpstream` shares its SAME runtime instead of a
    // separately-built one only reused after the fact.
    let edge_fake_app = App::builder()
        .with_runtime_cores(1)
        .with_defaults()?
        .build()?;
    let replay_runtime = edge_fake_app.runtime().expect("builder installs a runtime");
    let replay = ReplayUpstream::from_jsonl(&cassette_path, "fake-front", replay_runtime).await?;
    println!(
        "cassette loaded, known match keys: {:?}",
        replay.known_keys()
    );

    let probe = Request::builder().method("GET").path("/").build()?;
    let replayed = SendPipe::call(&replay, probe).await?;
    let replayed_status = replayed.status;
    let replayed_body = replayed.collect_body().await?;
    assert_eq!(
        replayed_status, 200,
        "replay must preserve the recorded status"
    );
    assert_eq!(
        &replayed_body[..],
        &recorded_body[..],
        "replay must serve exactly what was recorded, byte for byte"
    );
    println!(
        "in-process proof: {} bytes recorded == {} bytes replayed, no vendor call made",
        recorded_body.len(),
        replayed_body.len()
    );

    edge_fake_app.mount("/", into_handle(replay))?;
    // blocks until the accept lane has acked ready — no polling, no sleeping.
    let edge_fake_listener = edge_fake_app.build_listener(ListenerSpec::http(edge_bind))?;

    let fake_response = blocking_get(edge_bind);
    println!("\nsame client, same address {edge_bind}, vendor is dead:\n{fake_response}");
    assert!(
        fake_response.contains(VENDOR_BODY.trim_end()),
        "the fake must serve the vendor's exact body from the cassette: {fake_response:?}"
    );
    assert!(
        fake_response.to_ascii_lowercase().contains(&format!(
            "{VENDOR_HEADER}: {}",
            VENDOR_ID.to_ascii_lowercase()
        )),
        "the fake must serve the vendor's exact header from the cassette: {fake_response:?}"
    );

    edge_fake_listener.shutdown();
    let edge_fake_runtime = edge_fake_app
        .runtime()
        .ok_or_else(|| ProximaError::Config("edge fake app has no runtime installed".into()))?;
    let edge_fake_report = ShutdownBarrier::new(edge_fake_runtime).broadcast_drop().await;
    println!(
        "edge (fake) drained: cores_acked={} hooks_drained={}",
        edge_fake_report.cores_acked, edge_fake_report.hooks_drained
    );

    println!(
        "\nPASS: {VENDOR_ID} was fronted live, recorded, and replayed byte-identical with the \
         vendor removed."
    );

    Ok(())
}

/// The cassette's recorded response, reconstructed by concatenating its
/// `ResponseChunk` events in order — the ground truth `replay` is checked
/// against, read straight off disk the same way `record`'s own proof does.
async fn recorded_response_body(
    path: &Path,
    runtime: Arc<dyn Runtime>,
) -> Result<Bytes, ProximaError> {
    let source = JsonlSource::new(path, runtime);
    let mut events = source.events();
    let mut body = BytesMut::new();
    while let Some(event) = events.next().await {
        if let ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) = event?.event {
            body.extend_from_slice(&data);
        }
    }
    Ok(body.freeze())
}

/// One-shot GET over a plain blocking `TcpStream` — the client hitting the
/// edge, deliberately not another proxima pipe or runtime. `Connection:
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
