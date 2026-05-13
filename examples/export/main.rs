#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Export is not bespoke machinery bolted onto telemetry. It is `transform`'s
//! degenerate `sink` form (`Pipe<In = TelemetryRecord, Out = ()>` — consumes,
//! produces nothing) fanned out to N destinations via
//! `proxima_telemetry::pipes::fan_exporters`, the same primitive `logs.rs`
//! already used for console + file. This example widens the lens: export
//! ALWAYS means multiple sinks, never one — the "never OTLP-only" rule.
//!
//! Wiring OTLP as the sole sink is a real production failure mode: the
//! collector is down, the network partitions, or the endpoint is
//! misconfigured, and every log/span/metric silently vanishes with it.
//! `Exporter` composes `stdout()` / `stderr()` / `std()` (severity-split) /
//! `file(path)` / `writer(w)` / `pipe(handle)` — console and file are always
//! free, in-process sinks; OTLP (when wired) is ADDED to that set, never
//! substituted for it.
//!
//! Run: `cargo run --example export`
//! With the OTLP arm: `cargo run --example export --features otlp-http`

use std::fs::File;
#[cfg(feature = "otlp-http")]
use std::sync::Arc;

use proxima_telemetry::export::Exporter;
use proxima_telemetry::level::Level;
use proxima_telemetry::pipes::{
    FormatterPipe, LogFormat, TelemetryPipeHandle, fan_exporters, into_telemetry_handle,
};
use proxima_telemetry::recorder::Recorder;

#[cfg(feature = "otlp-http")]
use prost::Message as _;
#[cfg(feature = "otlp-http")]
use proxima_telemetry::out::otlp_http::proto::{
    ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest,
};
#[cfg(feature = "otlp-http")]
use proxima_telemetry::pipes::OtlpHttpPipe;

fn main() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let file_path = temp_dir.path().join("proxima-export-demo.log");

    let console_handle =
        into_telemetry_handle(FormatterPipe::new(std::io::stdout(), LogFormat::Human));
    let file_handle = into_telemetry_handle(FormatterPipe::new(
        File::create(&file_path).expect("create file sink"),
        LogFormat::Human,
    ));
    #[cfg_attr(not(feature = "otlp-http"), allow(unused_mut))]
    let mut sinks: Vec<TelemetryPipeHandle> = vec![console_handle, file_handle];

    // OTLP is encode-only in proxima-telemetry (proto encode + gRPC/HTTP
    // framing); it never opens a socket on its own — a real deployment hangs
    // an HTTP client off `OtlpHttpCodec`'s downstream slot (see
    // `proxima::otlp::OtlpClient`). Kept here as an Arc so we can call it, the
    // record-consuming half, AND read its buffered protobuf back afterward —
    // proof the fanned record actually reached an OTLP-shaped sink with no
    // live collector required.
    #[cfg(feature = "otlp-http")]
    let otlp_pipe = Arc::new(OtlpHttpPipe::new("http://otel-collector.internal:4318"));
    #[cfg(feature = "otlp-http")]
    sinks.push(Arc::clone(&otlp_pipe) as TelemetryPipeHandle);
    #[cfg(not(feature = "otlp-http"))]
    println!(
        "otlp arm skipped: build with `--features otlp-http` to add a third sink \
         (encode-verified below; the network POST itself needs a real collector \
         — see examples/export.README.md)."
    );

    let sink_count = sinks.len();
    let fanned = fan_exporters(sinks);

    println!(
        "export = a sink (Pipe<In = TelemetryRecord, Out = ()>) fanned to {sink_count} destinations"
    );

    let recorder = Recorder::builder()
        .export(Exporter::pipe(fanned))
        .expect("fanned exporter composes")
        .core_count(1)
        .start()
        .expect("recorder starts");

    // one span, two logs, a counter and a gauge — one of each record shape the
    // fan-out has to carry, all through the SAME handle.
    let span = recorder
        .span("checkout")
        .tag("route", "/v1/checkout")
        .start();
    drop(span);

    recorder
        .log()
        .level(Level::INFO)
        .message("request served")
        .tag("route", "checkout")
        .emit();
    recorder
        .log()
        .level(Level::WARN)
        .message("latency budget exceeded")
        .tag("elapsed_ms", 812u64)
        .emit();

    // recorder-scoped instruments: `recorder.counter(name)` registers into
    // THIS recorder's InstrumentRegistry, which `drain()` snapshots and routes
    // through the same fanned pipe as the logs and span above. This is a
    // different registry than the ambient `counter!`/`gauge!`/`histogram!`
    // macros' static instruments (see examples/metrics.rs) — those back onto
    // the global registry stub in `proxima-telemetry/src/metric/registry.rs`
    // ("v1 stub; C9 will wire a global recorder here"), which has no sink at
    // all. Recorder-scoped instruments are the real, exportable metrics path.
    recorder.counter("requests_total").add(3, &[]);
    recorder.gauge("queue_depth").set_u64(7, &[]);

    let exported = recorder.drain();
    println!("drained {exported} records (span + 2 logs + counter + gauge) to {sink_count} sinks");

    let file_contents = std::fs::read_to_string(&file_path).expect("read file sink");
    println!(
        "--- file sink ({}) ---\n{file_contents}",
        file_path.display()
    );

    for needle in [
        "request served",
        "latency budget exceeded",
        "checkout: duration_ns=",
        "COUNTER value=U64(3)",
        "GAUGE value=U64(7)",
    ] {
        assert!(
            file_contents.contains(needle),
            "file sink is missing {needle:?}; fan-out dropped a record: {file_contents}"
        );
    }
    println!("file sink read back: span, both logs, the counter AND the gauge all landed");

    #[cfg(feature = "otlp-http")]
    {
        let logs = otlp_pipe.flush_logs();
        let spans = otlp_pipe.flush_spans();
        let metrics = otlp_pipe.flush_metrics();
        assert!(!logs.is_empty(), "otlp sink buffered no log bytes");
        assert!(!spans.is_empty(), "otlp sink buffered no span bytes");
        assert!(!metrics.is_empty(), "otlp sink buffered no metric bytes");

        let decoded_logs =
            ExportLogsServiceRequest::decode(logs.as_ref()).expect("decode OTLP logs");
        let decoded_spans =
            ExportTraceServiceRequest::decode(spans.as_ref()).expect("decode OTLP spans");
        let decoded_metrics =
            ExportMetricsServiceRequest::decode(metrics.as_ref()).expect("decode OTLP metrics");

        let log_count: usize = decoded_logs
            .resource_logs
            .iter()
            .flat_map(|resource| resource.scope_logs.iter())
            .map(|scope| scope.log_records.len())
            .sum();
        let span_count: usize = decoded_spans
            .resource_spans
            .iter()
            .flat_map(|resource| resource.scope_spans.iter())
            .map(|scope| scope.spans.len())
            .sum();
        let metric_count: usize = decoded_metrics
            .resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics.iter())
            .map(|scope| scope.metrics.len())
            .sum();

        assert_eq!(log_count, 2, "otlp sink received both fanned logs");
        assert_eq!(span_count, 1, "otlp sink received the fanned span");
        assert_eq!(
            metric_count, 2,
            "otlp sink received both fanned metric snapshots (counter + gauge)"
        );
        println!(
            "otlp sink decoded back: {log_count} logs, {span_count} span, {metric_count} metric points \
             (real OTLP protobuf, no live collector — a network POST would need one)"
        );
    }

    println!(
        "\nthe never-otlp-only rule, made concrete: console and file received every record \
         above with zero network dependency; had OTLP been the ONLY sink and the collector been \
         unreachable, every one of them would have been lost with it."
    );
}
