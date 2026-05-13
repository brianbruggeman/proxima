//! `transform` named `observe` as `Pipe<In = T, Out = T>` — a call kept for
//! its side effect, the value passed through unchanged. A metric is that
//! degenerate form pushed one step further: instead of returning the value
//! at all, the call folds it into a running aggregate (`Counter`, `Gauge`,
//! `Histogram`) and the only way to see it again is to read the instrument
//! back, not the call's return.
//!
//! Run: `cargo run --example metrics`

use proxima_telemetry::metric::{Counter, Gauge, Histogram};
use proxima_telemetry::{counter, gauge, histogram};

static REQUESTS_TOTAL: Counter = Counter::new("requests_total");
static QUEUE_DEPTH: Gauge = Gauge::new("queue_depth");
static REQUEST_LATENCY_MS: Histogram<f64> = Histogram::new("request_latency_ms");

/// One simulated request on the hot path. Latency is a deterministic
/// function of `payload_size` so the read-back below has an exact expected
/// value, not a timing-dependent one.
fn handle_request(payload_size: u64, queue_depth_after: u64) -> f64 {
    counter!(REQUESTS_TOTAL, 1);
    gauge!(QUEUE_DEPTH, queue_depth_after);

    let latency_ms = payload_size as f64 * 0.5;
    histogram!(REQUEST_LATENCY_MS, latency_ms);

    println!(
        "  request: payload={payload_size} -> latency={latency_ms}ms, queue_depth={queue_depth_after}"
    );
    latency_ms
}

fn main() {
    println!("--- hot path: five requests, three instruments watching ---");

    let payload_sizes = [2u64, 4, 6, 8, 10];
    let queue_depths_after = [1u64, 2, 3, 2, 0];

    let mut expected_latency_sum = 0.0;
    for (payload_size, queue_depth_after) in payload_sizes.iter().zip(queue_depths_after.iter()) {
        expected_latency_sum += handle_request(*payload_size, *queue_depth_after);
    }

    println!("--- read back: the aggregate, not any one call's return value ---");

    let total_requests = REQUESTS_TOTAL.get();
    println!("counter   requests_total:     {total_requests}");
    assert_eq!(
        total_requests,
        payload_sizes.len() as u64,
        "counter adds one per call, regardless of payload size"
    );

    let final_queue_depth = QUEUE_DEPTH.get_u64();
    println!("gauge     queue_depth:        {final_queue_depth}");
    assert_eq!(
        final_queue_depth,
        queue_depths_after[queue_depths_after.len() - 1],
        "gauge holds only the last observation, not a running total"
    );

    let observed_latencies = REQUEST_LATENCY_MS.count();
    let summed_latency_ms = REQUEST_LATENCY_MS.sum();
    println!("histogram request_latency_ms: count={observed_latencies} sum={summed_latency_ms}ms");
    assert_eq!(observed_latencies, payload_sizes.len() as u64);
    assert!(
        (summed_latency_ms - expected_latency_sum).abs() < f64::EPSILON,
        "histogram sums every observation, unlike the gauge"
    );

    println!("all three instruments proved their own aggregation semantics");
}
