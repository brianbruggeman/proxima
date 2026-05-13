// Local model of the debug-under-load recovery, the death-spiral condition: a
// sink whose every write costs `WRITE_LATENCY` (a backed-up terminal/pipe/loaded
// host). The fix made the terminal do ONE write per drain batch instead of one
// per record. Under a slow sink that is the whole game: the slow write runs on
// the thread that drains (the serve thread, when a producer self-assists under
// Block), so N writes vs 1 write is how long that thread is stalled.
//
//   batched    — emit N, drain once  -> one slow write   (the fixed path)
//   per_record — N x (emit 1, drain)  -> N slow writes    (the pre-fix cost:
//                                        N writes for N records)
//
// Same N records, same total bytes; only the write COUNT differs — which is
// exactly what the batching changed. Run:
//   CARGO_TARGET_DIR=<scratch> cargo run --example slow_sink_recovery

#![allow(clippy::unwrap_used)]

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proxima_telemetry::pipes::{FormatterPipe, LogFormat};
use proxima_telemetry::recorder::Recorder;

const WRITE_LATENCY: Duration = Duration::from_micros(50);
const N: usize = 256;

// every write pays the sink latency once — a stand-in for a write() that blocks.
struct SlowSink;
impl Write for SlowSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        std::thread::sleep(WRITE_LATENCY);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn build_recorder() -> Arc<Recorder> {
    Arc::new(
        Recorder::builder()
            .pipe(FormatterPipe::new(SlowSink, LogFormat::Human))
            .core_count(1)
            .ring_capacity(8192)
            .start()
            .unwrap(),
    )
}

fn emit_one(recorder: &Arc<Recorder>) {
    let guard = recorder.span("serve").tag("route", "/v1").start();
    drop(guard);
}

fn main() {
    // batched: N records buffered, drained in one pass -> one slow write.
    let recorder = build_recorder();
    let start = Instant::now();
    for _ in 0..N {
        emit_one(&recorder);
    }
    recorder.drain();
    let batched = start.elapsed();

    // per-record: one slow write per record -> the pre-fix cost shape.
    let recorder = build_recorder();
    let start = Instant::now();
    for _ in 0..N {
        emit_one(&recorder);
        recorder.drain();
    }
    let per_record = start.elapsed();

    let ratio = per_record.as_secs_f64() / batched.as_secs_f64();
    println!("slow sink: {WRITE_LATENCY:?} per write, N={N} records");
    println!("  batched    (1 write/drain): {batched:?}");
    println!("  per_record (N writes):      {per_record:?}");
    println!("  serve thread freed {ratio:.0}x sooner under the slow sink");
}
