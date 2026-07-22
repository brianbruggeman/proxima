#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The `.any()` listener, grown up: tracing + metrics, an accept/deny
//! allowlist with a DoS blacklist, request-level admission that renders a
//! real `ShedReason` on the wire, and the same-port-vs-separate-port
//! decision — one process, one narrative, each section building on the last.
//!
//! Run: `cargo run --example any_listener_production --features http1-native`
//!
//! # 1. Telemetry: console + file sinks, a real counter
//!
//! `Recorder::builder().export(..).export(..).install()` — house rule:
//! telemetry export is never OTLP-only, console AND file are wired side by
//! side (`~/.claude/rules/rust.md`). `.counter(name)` (`proxima-telemetry/src/
//! recorder/mod.rs:1541`) is the direct-instrument fast path: one
//! `AtomicU64::fetch_add` per call, no ring, no allocation — this is what
//! "metrics on" means concretely, not a separate on/off flag.
//!
//! # 2/3. Accept + deny + blacklist: the simple form
//!
//! `.accept("h1")` selects exactly the legit candidate this service speaks;
//! `.deny(name, literal)` registers a scanner signature ALONGSIDE it (never
//! instead of it — `src/listener/handle.rs`'s own doc on `.deny`); a match
//! records a strike and drops the connection, no handler dispatch, ever
//! (`proxima-listen/src/any/deny.rs`). `.blacklist(config)` turns the strike
//! table's thresholds into a real ban: one `Strike::Deny` bans by default
//! (`deny_strike_threshold = 1`, deliberately low — a signature match is not
//! ambiguous noise) for `ban_duration_ms` (default 300s). This whole wiring
//! IS the simple form — nothing here is elided for the "production" framing;
//! a real service looks exactly like this.
//!
//! # 4. Request-level admission: a real `ShedReason` on the wire
//!
//! `max_in_flight_requests` is a raw spec key
//! (`.spec("max_in_flight_requests", json!(1))`), not a typed builder method
//! — there is no `.max_in_flight_requests(n)` on `ListenerBuilder` today (see
//! this file's own report). `proxima-http/src/any_listener.rs:574-583` reads
//! it and builds a listener-wide `ConnAdmission::new(n)`
//! (`proxima-listen/src/admission/request.rs`). h2 is the candidate that
//! actually enforces it: `proxima-http/src/http2/server.rs:307-320` calls
//! `admission.request_admit()` at its own per-stream boundary and renders a
//! real in-band 503 + `retry-after: 1` on `RequestAdmit::Shed`. (h1's own
//! request loop does NOT call `request_admit()` — see this file's report for
//! why that matters and which candidate to reach for when you need the cap
//! actually enforced.)
//!
//! # 7. Same port vs. separate port
//!
//! `.any()` binds ONE socket and classifies every connection against every
//! registered candidate — use this when every candidate should be reachable
//! at one address (a proxy in front of it, a single firewall rule, one DNS
//! name). `.accept(name)` on N separate `Listener::builder()...serve()`
//! calls, each with its OWN bind address, gives you N sockets, each pinned to
//! exactly one wire — use this when candidates need independent lifecycle
//! (bind/drain/restart one without touching the other), independent
//! firewalling, or when a candidate's own wire needs a dedicated port by
//! convention (h2 on 8443, a metrics-only h1 port on 9090). Both are shown
//! below, back to back, on the SAME running process.

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr, TcpStream as StdTcpStream};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use proxima::h2::H2ClientUpstream;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::telemetry::emit::EnvFilter;
use proxima::telemetry::emit::global::install as install_emit_filter;
use proxima::telemetry::info;
use proxima::telemetry::pipes::{FormatterPipe, LogFormat, fan_exporters, into_telemetry_handle};
use proxima::telemetry::recorder::Recorder;
use proxima::time::sleep;
use proxima::{Listener, ListenerBuilderEntry, PrimeTcpUpstream, ProximaError};
use proxima_listen::admission::BlacklistConfig;
use proxima_primitives::pipe::SendPipe;

const SCANNER_LITERAL: &[u8] = b"XSCANPROBE\r\n";

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

/// The one business handler every section below dispatches to — counts
/// every call on a real metrics counter (§1).
struct CountingOk {
    requests_total: Arc<proxima::telemetry::metric::Counter>,
}

impl SendPipe for CountingOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        self.requests_total.add(1, &[]);
        async move { Ok(Response::new(200).with_body(Bytes::from_static(b"legit-ok"))) }
    }
}

/// The handler behind §4's admission demo: sleeps long enough that two
/// concurrent requests are guaranteed to overlap, so a `max_in_flight_requests
/// = 1` cap has something real to shed.
struct SlowOk;

impl SendPipe for SlowOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            sleep(Duration::from_millis(200)).await;
            Ok(Response::new(200).with_body(Bytes::from_static(b"slow-ok")))
        }
    }
}

fn dial_and_collect(addr: SocketAddr, payload: &[u8]) -> Vec<u8> {
    use std::io::{Read, Write};
    let mut collected = Vec::new();
    if let Ok(mut stream) = StdTcpStream::connect(addr)
        && stream.write_all(payload).is_ok()
    {
        let _ = stream.flush();
        let _ = stream.read_to_end(&mut collected);
    }
    collected
}

fn legit_h1_request(addr: SocketAddr) -> String {
    let response = dial_and_collect(
        addr,
        b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    );
    String::from_utf8_lossy(&response).into_owned()
}

/// Bodyless GET — deliberately no request body. A body-carrying request
/// that gets shed by `ConnAdmission::request_admit()` on the native h2
/// listener drops its half-built `Request` (and the body-stream receiver
/// bundled into it) before the client's own DATA frame arrives; the
/// connection's `BodyData` handler then finds the channel closed and resets
/// the stream (`INTERNAL_ERROR`) instead of delivering the 503 it already
/// queued (`proxima-http/src/http2/server.rs`'s `BodyData` arm, `Some(Err(_))
/// => send_rst`). Real, reproducible defect — filed in this task's report,
/// not fixed here (out of scope for a docs task). A bodyless request never
/// opens that channel, so §4's admission demo below is unaffected by it.
fn constant_ok_request() -> Result<Request<Bytes>, ProximaError> {
    Request::builder().method("GET").path("/").build()
}

async fn h2_call(addr: SocketAddr, label: &str) -> Result<Response<Bytes>, ProximaError> {
    let client =
        H2ClientUpstream::new(PrimeTcpUpstream::new(addr), format!("{addr}"), false, label);
    client.call(constant_ok_request()?).await
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    // ── §1: telemetry — console AND file, plus a real counter ──────────────
    // `RecorderBuilder::export` composes exactly ONE `Exporter`
    // (`proxima-telemetry/src/export.rs:277-282`'s own doc: "fan-out over
    // multiple exporters lands with the OTLP slice's FanOut stage" — not
    // built yet). Two real sinks side by side today means building each as
    // its own `FormatterPipe`, then `fan_exporters` — the identical
    // combinator `examples/logs/main.rs`'s `run_fanout_sinks` uses.
    // the emit filter is read lazily on the first emit and cached per
    // callsite (examples/logs/main.rs's own comment) — install it before
    // this process's first info!/debug! call or the default floor filters
    // it out.
    install_emit_filter(EnvFilter::parse("debug"));
    let log_dir = tempfile::tempdir().expect("tempdir for the file sink");
    let log_path = log_dir.path().join("any-listener-production.log");
    let stdout_handle =
        into_telemetry_handle(FormatterPipe::new(std::io::stdout(), LogFormat::Human));
    let file_handle = into_telemetry_handle(FormatterPipe::new(
        std::fs::File::create(&log_path).expect("create log file"),
        LogFormat::Human,
    ));
    let fanned = fan_exporters(vec![stdout_handle, file_handle]);
    let recorder = Recorder::builder()
        .pipe(fanned)
        .core_count(1)
        .install()
        .expect("recorder installs as the process default");
    let requests_total = recorder.counter("proxima.any_listener.requests_total");
    info!(sink = "console+file", "telemetry installed");
    println!(
        "telemetry: console + file ({}) sinks wired",
        log_path.display()
    );

    // ── §2/3: accept + deny + blacklist — the whole simple form ─────────────
    let service_bind = free_loopback_addr()?;
    let service = Listener::builder()
        .bind(service_bind)
        .accept("h1")
        .deny("scanner", SCANNER_LITERAL.to_vec())
        .blacklist(
            BlacklistConfig::layered()
                .with_deny_strike_threshold(1)
                .build(),
        )
        .handle(into_handle(CountingOk {
            requests_total: requests_total.clone(),
        }))
        .serve()
        .await?;

    let legit_text = legit_h1_request(service_bind);
    assert!(
        legit_text.starts_with("HTTP/1.1 200"),
        "legit h1 traffic must route normally alongside the deny signature: {legit_text:?}"
    );
    println!(
        "§2/3: legit h1 request served, counter now at {}",
        requests_total.get()
    );

    let scanner_response = dial_and_collect(service_bind, SCANNER_LITERAL);
    assert!(
        String::from_utf8_lossy(&scanner_response).is_empty()
            || !String::from_utf8_lossy(&scanner_response).starts_with("HTTP/"),
        "a deny-signature match must never dispatch to the handler"
    );
    println!("§2/3: scanner literal dropped, no HTTP response, peer now banned");

    let banned_text = legit_h1_request(service_bind);
    assert!(
        !banned_text.starts_with("HTTP/"),
        "the SAME peer's next connection — even a legit payload — must be dropped pre-classify \
         while banned: {banned_text:?}"
    );
    println!("§2/3: same peer's next connection (legit payload!) dropped — banned pre-classify");

    service.stop();

    // ── §4: request-level admission — a real ShedReason on the wire ────────
    let admission_bind = free_loopback_addr()?;
    let admission_service = Listener::builder()
        .bind(admission_bind)
        .accept("h2")
        .spec("max_in_flight_requests", json!(1))
        .handle(into_handle(SlowOk))
        .serve()
        .await?;

    let (first, second) = futures::join!(
        h2_call(admission_bind, "admission-a"),
        h2_call(admission_bind, "admission-b"),
    );
    let statuses: Vec<u16> = [&first, &second]
        .iter()
        .map(|outcome| {
            outcome
                .as_ref()
                .expect("h2 call itself must not error")
                .status
        })
        .collect();
    println!("§4: two concurrent h2 requests against max_in_flight_requests=1 -> {statuses:?}");
    assert!(
        statuses.contains(&200) && statuses.contains(&503),
        "with a cap of 1 and two overlapping requests, exactly one must be admitted (200) and \
         the other shed (503); got {statuses:?}"
    );
    let shed_body = if first.as_ref().unwrap().status == 503 {
        first.as_ref().unwrap().payload.clone()
    } else {
        second.as_ref().unwrap().payload.clone()
    };
    assert_eq!(
        shed_body.as_ref(),
        b"service unavailable",
        "the shed response's body is the listener's own rendering of ShedReason::AtCapacity \
         (proxima-http/src/http2/server.rs's 503 arm), not a generic error"
    );
    println!("§4: the shed request's body is the listener's real 503 rendering, not a stub");

    admission_service.stop();

    // ── §7: same port vs. separate port, side by side ───────────────────────
    let same_port_bind = free_loopback_addr()?;
    let same_port_service = Listener::builder()
        .bind(same_port_bind)
        .any()
        .handle(into_handle(CountingOk {
            requests_total: requests_total.clone(),
        }))
        .serve()
        .await?;
    let same_port_h2 = h2_call(same_port_bind, "same-port").await?;
    assert_eq!(same_port_h2.status, 200);
    println!("§7: SAME port {same_port_bind} answers both h1 and h2 (.any())");
    same_port_service.stop();

    let h1_only_bind = free_loopback_addr()?;
    let h2_only_bind = free_loopback_addr()?;
    let h1_only_service = Listener::builder()
        .bind(h1_only_bind)
        .accept("h1")
        .handle(into_handle(CountingOk {
            requests_total: requests_total.clone(),
        }))
        .serve()
        .await?;
    let h2_only_service = Listener::builder()
        .bind(h2_only_bind)
        .accept("h2")
        .handle(into_handle(CountingOk {
            requests_total: requests_total.clone(),
        }))
        .serve()
        .await?;
    let h1_only_text = legit_h1_request(h1_only_bind);
    assert!(h1_only_text.starts_with("HTTP/1.1 200"));
    let h2_only_response = h2_call(h2_only_bind, "h2-only-port").await?;
    assert_eq!(h2_only_response.status, 200);
    println!(
        "§7: SEPARATE ports — h1 only on {h1_only_bind}, h2 only on {h2_only_bind} — each with \
         its own bind, its own lifecycle"
    );
    h1_only_service.stop();
    h2_only_service.stop();

    // `Counter::get` reads live; the drainer's own export snapshot calls
    // `snapshot_and_reset` (`proxima-telemetry/src/metric/counter.rs:59`), so
    // read the total BEFORE `drain()` or it reports 0.
    let total_requests = requests_total.get();
    let exported = recorder.drain();
    let file_contents = std::fs::read_to_string(&log_path).expect("read the file sink");
    assert!(
        file_contents.contains("telemetry installed"),
        "the file sink must have actually received the real info! event: {file_contents:?}"
    );
    println!(
        "\nany_listener_production: telemetry ({exported} records flushed to {}) + \
         accept/deny/blacklist + admission-shed + same-port/separate-port all OK, {total_requests} \
         total requests counted",
        log_path.display(),
    );
    Ok(())
}
