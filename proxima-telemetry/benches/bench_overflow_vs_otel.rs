//! Lossless-axis comparison under overload: proxima park-for-slot vs OpenTelemetry
//! `BatchSpanProcessor`, same offered load + same bounded buffer + same slow
//! per-span sink. The incumbent's queue is bounded and DROPS on full by design
//! (no backpressure to the caller); proxima's `Block`+pump PARKS the producer and
//! drops nothing. This reports the drop count/% for each — the P14 incumbent-
//! relative number on the dimension that matters for a lossless substrate.
//!
//! Run: `cargo bench -p proxima-telemetry --bench bench_overflow_vs_otel`
//!
//! Harness, not a unit test — it prints a report; nothing asserts.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider, SpanData, SpanExporter,
};

use bytes::Bytes;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_telemetry::config::OverflowPolicy;
use proxima_telemetry::pipes::{TelemetryRecord, TelemetryRequest};
use proxima_telemetry::recorder::Recorder;

// busy-spin `spin_ns` per span exported (per-span sink cost), counting receipts.
fn spin_for(spin_ns: u64) {
    if spin_ns == 0 {
        return;
    }
    let started = Instant::now();
    while (started.elapsed().as_nanos() as u64) < spin_ns {
        core::hint::spin_loop();
    }
}

// ---- OTel arm: BatchSpanProcessor + a slow, counting exporter ---------------

struct SlowCountingExporter {
    received: Arc<AtomicU64>,
    // ONE round-trip per export batch — the realistic OTLP-over-HTTP shape (a
    // batch is one POST), not a per-span cost.
    round_trip_ns: u64,
}

impl fmt::Debug for SlowCountingExporter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SlowCountingExporter")
    }
}

impl SpanExporter for SlowCountingExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.received
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        spin_for(self.round_trip_ns);
        Ok(())
    }
}

struct Outcome {
    exported: u64,
    dropped: u64,
    emit_wall: std::time::Duration,
    total_wall: std::time::Duration,
}

fn otel_run(emit: u64, queue: usize, round_trip_ns: u64) -> Outcome {
    let received = Arc::new(AtomicU64::new(0));
    let exporter = SlowCountingExporter {
        received: Arc::clone(&received),
        round_trip_ns,
    };
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(queue)
                .build(),
        )
        .build();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .build();
    let tracer = provider.tracer("bench");
    let started = Instant::now();
    for _ in 0..emit {
        let mut span = tracer.start(black_box("process"));
        span.set_attribute(KeyValue::new("route", black_box("/v1")));
        span.end();
    }
    let emit_wall = started.elapsed();
    // give the incumbent every chance to flush so `dropped` is true enqueue-time
    // shedding, not a shutdown-timeout cutoff.
    let _ = provider.force_flush();
    let _ = provider.shutdown();
    let total_wall = started.elapsed();
    let exported = received.load(Ordering::Relaxed);
    Outcome {
        exported,
        dropped: emit.saturating_sub(exported),
        emit_wall,
        total_wall,
    }
}

// ---- proxima arm: park-for-slot + a slow, counting sink --------------------

// proxima's terminal pipe receives a WHOLE drained batch in one call (the codec
// would encode it into one OTLP POST), so it pays ONE round-trip per batch — the
// same batched-network shape as OTel's per-export-batch cost. counts batch size.
struct BatchSpinPipe {
    received: Arc<AtomicU64>,
    round_trip_ns: u64,
}

impl SendPipe for BatchSpinPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl core::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let count = match &request.payload {
            TelemetryRecord::SpanBatch(records) => records.len() as u64,
            TelemetryRecord::SpanBatchArc(records) => records.len() as u64,
            _ => 0,
        };
        self.received.fetch_add(count, Ordering::Relaxed);
        spin_for(self.round_trip_ns);
        async move { Ok(Response::ok(bytes::Bytes::new())) }
    }
}

fn proxima_run(emit: u64, ring_cap: usize, round_trip_ns: u64) -> Outcome {
    let received = Arc::new(AtomicU64::new(0));
    let pipe = BatchSpinPipe {
        received: Arc::clone(&received),
        round_trip_ns,
    };
    let recorder = Recorder::builder()
        .pipe(pipe)
        .core_count(1)
        .ring_capacity(ring_cap)
        .overflow(OverflowPolicy::Block)
        .managed_drainer(true)
        .start()
        .expect("recorder");
    let started = Instant::now();
    for _ in 0..emit {
        drop(
            recorder
                .span(black_box("process"))
                .tag("route", black_box("/v1"))
                .start(),
        );
    }
    let emit_wall = started.elapsed(); // includes the parking — this is the throttle
    drop(recorder); // stop pump + lossless shutdown flush
    let total_wall = started.elapsed();
    let exported = received.load(Ordering::Relaxed);
    Outcome {
        exported,
        dropped: emit.saturating_sub(exported),
        emit_wall,
        total_wall,
    }
}

fn main() {
    // matched buffer (OTel max_queue_size == proxima ring_cap) and matched per-span
    // sink cost. 'none' is the control: a free sink must drop ~0 for BOTH, proving
    // any drop is overload-driven, not a harness artifact. 'network' scaled to
    // 50 µs/span to keep the run short.
    let buffer = 2048usize;
    let emit = 50_000u64;
    // per-EXPORT-BATCH round-trip cost (one POST per batch), the realistic OTLP
    // shape — NOT per span. both stacks batch (~512/batch), so 50k spans ≈ 98 POSTs.
    let sinks: [(&str, u64); 3] = [
        ("none(ctl)", 0),
        ("local 50us", 50_000),
        ("network 1ms", 1_000_000),
    ];

    println!("# Lossless vs lossy under overload: proxima park vs OTel BatchSpanProcessor\n");
    println!("offered load {emit} spans, bounded buffer {buffer} (OTel max_queue_size ==");
    println!("proxima ring_cap), matched ONE-round-trip-per-export-batch sink (realistic OTLP:");
    println!("a batch is one POST). THIS IS A TRADEOFF: read drop% AND emit-wall together.");
    println!("'none(ctl)' is the artifact check.\n");
    println!(
        "  {:<10} {:<26} {:>9} {:>8} {:>10} {:>10}",
        "sink", "stack", "exported", "drop%", "emit-wall", "total-wall"
    );
    for (label, spin_ns) in sinks {
        let otel = otel_run(emit, buffer, spin_ns);
        let prox = proxima_run(emit, buffer, spin_ns);
        let pct = |dropped: u64| 100.0 * dropped as f64 / emit as f64;
        println!(
            "  {label:<10} {:<26} {:>9} {:>7.1}% {:>10} {:>10}",
            "OTel BatchSpanProcessor",
            otel.exported,
            pct(otel.dropped),
            fmt_dur(otel.emit_wall),
            fmt_dur(otel.total_wall)
        );
        println!(
            "  {label:<10} {:<26} {:>9} {:>7.1}% {:>10} {:>10}",
            "proxima park (Block+pump)",
            prox.exported,
            pct(prox.dropped),
            fmt_dur(prox.emit_wall),
            fmt_dur(prox.total_wall)
        );
    }
    println!("\n  -> read it honestly, both directions:");
    println!("     - control 'none(ctl)': both ~0 drop => the slow-sink drops below are real");
    println!("       overload behavior, not a harness bug.");
    println!("     - under a slow sink the two make OPPOSITE choices. OTel has no backpressure:");
    println!("       emit stays fast (emit-wall ~ms) and the bounded queue SHEDS the excess.");
    println!("       proxima's Block+pump PARKS the producer (emit-wall stretches to sink rate)");
    println!("       and keeps every span. neither is 'broken': OTel optimizes app speed and");
    println!("       treats spans as droppable; proxima treats them as data you cannot lose and");
    println!("       pays for it in producer throughput. proxima's choice is the one a memory/");
    println!("       audit substrate needs — but the COST (throttled producers) is the emit-wall");
    println!("       column, stated plainly, not hidden.");
}

fn fmt_dur(dur: std::time::Duration) -> String {
    let ns = dur.as_nanos() as u64;
    if ns >= 1_000_000_000 {
        std::format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        std::format!("{:.1}ms", ns as f64 / 1e6)
    } else {
        std::format!("{:.1}us", ns as f64 / 1e3)
    }
}
