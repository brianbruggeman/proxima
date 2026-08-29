// measurement: drives the concrete H1ClientUpstream::call path (the generic pipe
// surface) so its throughput can be compared head-to-head with the hand-rolled
// send_raw path (rekt_load). validates the keep-alive fix (errors should be 0)
// and quantifies the residual envelope cost per call.
//
//   cargo run --release --features scheduler --example rekt_call -- \
//     http://127.0.0.1:8080/ <connections_per_core> <duration_secs> <cores>

use std::time::Duration;

use rekt::engine::drive_h1_call;

fn main() {
    let mut args = std::env::args().skip(1);
    let url = args.next().unwrap_or_else(|| "http://127.0.0.1:8080/".to_string());
    let connections: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(8);
    let seconds: u64 = args.next().and_then(|value| value.parse().ok()).unwrap_or(5);
    let cores: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(1);

    match drive_h1_call(&url, connections, cores, Duration::from_secs(seconds)) {
        Ok(throughput) => {
            println!("rekt-call (H1ClientUpstream::call): {} cores x {} conns", throughput.cores, connections);
            println!("rekt-call: {} completed, {} errors", throughput.completed, throughput.errors);
            println!("rekt-call: Requests/sec: {:.2}", throughput.per_sec());
        }
        Err(error) => {
            eprintln!("rekt_call: {error}");
            std::process::exit(1);
        }
    }
}
