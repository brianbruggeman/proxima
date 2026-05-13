//! OTLP over the PRIME wire via the fluent face — `OtlpClient::http()
//! .endpoint(..).build()` lowers to a `OtlpHttpCodec -> prime H1 client` chain
//! that POSTs to a prime `TcpListener` collector. This proves the sss-tier
//! surface: transport is a verb, the endpoint is the only required input, the
//! composition falls out of it, and the auth / resilience verbs
//! (`.header`/`.retry`/`.timeout`) are real composed stages (the auth header is
//! asserted on the wire). Everything is reached through `proxima::<thing>`.

#![cfg(all(
    feature = "otlp-http",
    feature = "http-prime",
    any(target_os = "linux", target_os = "macos")
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use futures::io::{AsyncReadExt as _, AsyncWriteExt as _};
use prost::Message as _;
use proxima::SendPipe;
use proxima::otlp::OtlpClient;
use proxima::runtime::prime::os::net::TcpListener;
use proxima::telemetry::capture::capture;
use proxima::telemetry::out::otlp_http::proto::{
    ExportMetricsServiceRequest, ExportTraceServiceRequest,
};
use proxima::telemetry::pipes::span_batch_request;
use proxima::telemetry::recorder::Recorder;

fn content_length(head: &[u8]) -> usize {
    let text = String::from_utf8_lossy(head).to_ascii_lowercase();
    for line in text.split("\r\n") {
        if let Some(rest) = line.strip_prefix("content-length:") {
            return rest.trim().parse().unwrap_or(0);
        }
    }
    0
}

// accept one POST, read head+body, reply 200. returns the (lowercased) request
// head (so tests can assert headers) and the raw protobuf body (so tests decode
// the OTLP signal they expect). The HTTP framing is signal-agnostic.
async fn read_one_http_post(listener: &mut TcpListener) -> (String, Vec<u8>) {
    let (mut stream, _peer) = listener.accept().await.expect("accept");
    let mut buffer = Vec::new();
    let mut scratch = [0_u8; 4096];
    let head_end = loop {
        let read = stream.read(&mut scratch).await.expect("read head");
        buffer.extend_from_slice(&scratch[..read]);
        if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).to_ascii_lowercase();
    let body_len = content_length(&buffer[..head_end]);
    while buffer.len() < head_end + body_len {
        let read = stream.read(&mut scratch).await.expect("read body");
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&scratch[..read]);
    }
    let body = buffer[head_end..head_end + body_len].to_vec();
    stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        .await
        .expect("write 200");
    (head, body)
}

// accept one OTLP/traces POST and return the decoded span count + request head.
async fn collect_one_otlp_post(listener: &mut TcpListener) -> (usize, String) {
    let (head, body) = read_one_http_post(listener).await;
    let request = ExportTraceServiceRequest::decode(&body[..]).expect("decode OTLP traces");
    let count: usize = request
        .resource_spans
        .iter()
        .flat_map(|resource| resource.scope_spans.iter())
        .map(|scope| scope.spans.len())
        .sum();
    (count, head)
}

// accept one OTLP/metrics POST and return the decoded metric (instrument) count.
async fn collect_one_otlp_metrics_post(listener: &mut TcpListener) -> usize {
    let (_head, body) = read_one_http_post(listener).await;
    let request = ExportMetricsServiceRequest::decode(&body[..]).expect("decode OTLP metrics");
    request
        .resource_metrics
        .iter()
        .flat_map(|resource| resource.scope_metrics.iter())
        .map(|scope| scope.metrics.len())
        .sum()
}

fn seven_spans() -> Vec<proxima::telemetry::trace::SpanRecord> {
    let spans = capture(|recorder| {
        for _ in 0..7 {
            recorder
                .span("process")
                .tag("route", "/v1")
                .tag("status", 200u64)
                .start();
        }
    })
    .spans();
    assert_eq!(spans.len(), 7);
    spans
}

#[proxima::test]
async fn otlp_send_over_prime_wire_via_facade() {
    let spans = seven_spans();
    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    let server = async move { collect_one_otlp_post(&mut listener).await };
    let client = async move {
        // the fluent face: transport is a verb, endpoint is the only input, the
        // codec -> prime client chain falls out of build().
        let exporter = OtlpClient::http()
            .endpoint(format!("http://{bound}"))
            .build()
            .await
            .expect("build otlp client");
        let response = exporter
            .call(span_batch_request(spans))
            .await
            .expect("send call");
        assert_eq!(response.status, 200, "collector accepted the OTLP POST");
    };

    let (count, _head) = futures::future::join(server, client).await.0;
    assert_eq!(
        count, 7,
        "prime collector prost-decoded exactly the 7 exported spans"
    );
}

#[proxima::test]
async fn otlp_client_composes_auth_and_resilience_axes() {
    let spans = seven_spans();
    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    let server = async move { collect_one_otlp_post(&mut listener).await };
    let client = async move {
        // all three axes enabled: header (auth) must reach the wire; retry +
        // timeout are inserted stages that stay transparent on the happy path.
        let exporter = OtlpClient::http()
            .endpoint(format!("http://{bound}"))
            .header("authorization", "Bearer secret-token")
            .retry(3)
            .timeout(Duration::from_secs(5))
            .build()
            .await
            .expect("build otlp client");
        let response = exporter
            .call(span_batch_request(spans))
            .await
            .expect("send call");
        assert_eq!(response.status, 200, "collector accepted the OTLP POST");
    };

    let (count, head) = futures::future::join(server, client).await.0;
    assert_eq!(
        count, 7,
        "all 7 spans delivered through codec -> retry -> timeout -> header -> transport"
    );
    assert!(
        head.contains("authorization: bearer secret-token"),
        "the .header() axis is a real Transform stage that reached the wire; head was:\n{head}"
    );
}

// The full pipeline, prime-first: recorder.span().start() -> per-core ring ->
// recorder.drain_async() (awaits the terminal on the prime reactor, no block_on)
// -> OtlpClient codec -> prime H1 client -> prime collector. This is the
// drain-runtime keystone: the export I/O is awaited, not blocked.
#[proxima::test]
async fn recorder_drain_async_sends_over_prime_wire() {
    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    let exporter = OtlpClient::http()
        .endpoint(format!("http://{bound}"))
        .build()
        .await
        .expect("build otlp client");
    let recorder = Recorder::builder()
        .pipe_handle(exporter.handle())
        .core_count(1)
        .start()
        .expect("start recorder");

    for _ in 0..7 {
        recorder
            .span("process")
            .tag("route", "/v1")
            .tag("status", 200u64)
            .start();
    }

    let server = async move { collect_one_otlp_post(&mut listener).await };
    let client = async move {
        let drained = recorder.drain_async().await;
        assert_eq!(
            drained, 7,
            "drain_async exported all 7 ring spans by awaiting the terminal"
        );
    };

    let (count, _head) = futures::future::join(server, client).await.0;
    assert_eq!(
        count, 7,
        "collector received 7 spans via recorder -> drain_async -> prime wire"
    );
}

// The AUTOMATIC pipeline (the managed-pump keystone): a recorder built purely
// from config (`OtlpHttp` endpoint) plus a prime drain pump exports emitted
// spans over the prime wire with NO manual drain anywhere. `spawn_prime_pump`
// runs a detached prime task that drains on a prime timer; the test only emits
// and asserts the collector received them — the pump moves them to the wire.
#[proxima::test]
async fn prime_pump_auto_exports_over_prime_wire() {
    use std::sync::Arc;

    use proxima::otlp::{recorder_from_config, spawn_prime_pump};
    use proxima::telemetry::config::{ExporterChoice, TelemetryConfig};

    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    // config alone wires the transport (OtlpHttp endpoint -> prime H1 client via
    // proxima::Client); the pump drives the drain. flush 500us keeps the test fast.
    let cfg = TelemetryConfig::builder()
        .exporter(ExporterChoice::OtlpHttp {
            endpoint: format!("http://{bound}"),
        })
        .core_count(1)
        .flush_interval_micros(500)
        .build();
    let recorder = Arc::new(
        recorder_from_config(&cfg)
            .await
            .expect("recorder from config")
            .start()
            .expect("start"),
    );

    // emit synchronously — all 7 land in the ring before any await — THEN start the
    // pump, so its first drain carries exactly one 7-span batch. No sleep, no race:
    // the pump auto-exports; the test never calls drain.
    for _ in 0..7 {
        recorder
            .span("process")
            .tag("route", "/v1")
            .tag("status", 200u64)
            .start();
    }
    let pump = spawn_prime_pump(Arc::clone(&recorder), Duration::from_micros(500));

    let (count, _head) = collect_one_otlp_post(&mut listener).await;
    assert_eq!(
        count, 7,
        "prime pump auto-exported the 7 spans over the prime wire with no manual drain"
    );

    pump.stop().await;
}

// The metrics counterpart: a recorder counter drains via the registry async path
// (drain_instruments_async), NOT the rings, so this proves the pump exports
// counters/histograms over the async network terminal too — the path the
// in-memory unit test can't exercise (its pipe is always Ready). No manual drain.
#[proxima::test]
async fn prime_pump_auto_exports_metrics_over_prime_wire() {
    use std::sync::Arc;

    use proxima::otlp::{recorder_from_config, spawn_prime_pump};
    use proxima::telemetry::config::{ExporterChoice, TelemetryConfig};

    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    let cfg = TelemetryConfig::builder()
        .exporter(ExporterChoice::OtlpHttp {
            endpoint: format!("http://{bound}"),
        })
        .core_count(1)
        .flush_interval_micros(500)
        .build();
    let recorder = Arc::new(
        recorder_from_config(&cfg)
            .await
            .expect("recorder from config")
            .start()
            .expect("start"),
    );

    recorder.counter("db.queries").add(3, &[]);
    let pump = spawn_prime_pump(Arc::clone(&recorder), Duration::from_micros(500));

    let metric_count = collect_one_otlp_metrics_post(&mut listener).await;
    assert!(
        metric_count >= 1,
        "prime pump auto-exported the counter over /v1/metrics with no manual drain"
    );

    pump.stop().await;
}
