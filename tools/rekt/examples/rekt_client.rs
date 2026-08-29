// measurement: drives the GENERIC proxima::Client::call path (the type-erased
// handle that boxes a future per request) so it can sit in the client matrix
// beside send_raw, H1ClientUpstream::call, and wrk. the gap to rekt_call is what
// the erasure box + dyn dispatch costs.
//
//   cargo run --release --features scheduler --example rekt_client -- \
//     http://127.0.0.1:8080/ <connections_per_core> <duration_secs> <cores>

use std::time::Duration;

use rekt::engine::drive_h1_client;

fn main() {
    let mut args = std::env::args().skip(1);
    let url = args.next().unwrap_or_else(|| "http://127.0.0.1:8080/".to_string());
    let connections: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(8);
    let seconds: u64 = args.next().and_then(|value| value.parse().ok()).unwrap_or(5);
    let cores: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(1);

    match drive_h1_client(&url, connections, cores, Duration::from_secs(seconds)) {
        Ok(throughput) => {
            println!("rekt-client (proxima::Client::call): {} cores x {} conns", throughput.cores, connections);
            println!("rekt-client: {} completed, {} errors", throughput.completed, throughput.errors);
            println!("rekt-client: Requests/sec: {:.2}", throughput.per_sec());
        }
        Err(error) => {
            eprintln!("rekt_client: {error}");
            std::process::exit(1);
        }
    }
}
