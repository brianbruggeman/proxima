#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
// pedagogy: everything is a Pipe, even telemetry.
// records flow through the same Pipe envelope as HTTP requests.
// CountingPipe surfaces how many of each type landed.

use std::sync::atomic::Ordering;

use proxima_telemetry::pipes::CountingPipe;
use proxima_telemetry::recorder::Recorder;

fn main() {
    // build a recorder with a counting terminal Pipe so records are observable.
    let (pipe, spans, _events, logs, metrics, _links) = CountingPipe::new();

    let recorder = Recorder::builder()
        .pipe(pipe)
        .core_count(1)
        .start()
        .expect("recorder build failed");

    // emit different telemetry shapes through the same Pipe envelope.
    recorder.log().message("hello world").emit();
    recorder.log().message("starting request").emit();

    let counter = recorder.counter("requests");
    counter.add(1, &[]);
    counter.add(2, &[]);

    let span = recorder.span("process_request").start();
    drop(span);

    let span2 = recorder.span("db_query").tag("table", "users").start();
    drop(span2);

    // drain dispatches every queued record through the terminal Pipe.
    let exported = recorder.drain();
    println!("exported {exported} records via Pipe envelope");

    // CountingPipe surfaces per-record-type counts after drain.
    println!("logs:    {}", logs.load(Ordering::Relaxed));
    println!("spans:   {}", spans.load(Ordering::Relaxed));
    println!("metrics: {}", metrics.load(Ordering::Relaxed));

    // install TracingLayer so tracing::info! also lands in the ring.
    #[cfg(feature = "tracing-init")]
    {
        use proxima_telemetry::tracing_bridge::TracingLayer;
        use std::sync::Arc;
        use tracing_subscriber::EnvFilter;
        use tracing_subscriber::layer::SubscriberExt;

        let recorder2 = Arc::new(
            Recorder::builder()
                .pipe(CountingPipe::new().0)
                .core_count(1)
                .start()
                .expect("recorder2 build failed"),
        );
        let layer = TracingLayer::new(Arc::clone(&recorder2));
        let filter = EnvFilter::new("info");
        let subscriber = tracing_subscriber::registry().with(filter).with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(user = "alice", "user logged in");
        let bridge_drained = recorder2.drain();
        println!("tracing bridge exported {bridge_drained} records");
    }
}
