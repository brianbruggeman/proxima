// closed-loop throughput driver for the rekt-vs-wrk harness. drives rekt's
// concurrent prime loader against a real HTTP target and prints requests/sec.
// `cores` spreads the load across that many distinct prime cores (one
// PrimeRuntime, one worker thread per core), each driving `connections`
// keep-alive clients — the analog of `wrk -t<cores> -c<cores*connections>`.
//
//   cargo run --release --features scheduler --example rekt_load -- \
//     http://127.0.0.1:8080/ <connections_per_core> <duration_secs> <cores>

use std::time::Duration;

use rekt::engine::drive_throughput;

fn main() {
    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8080/".to_string());
    let connections: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let seconds: u64 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let cores: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);

    match drive_throughput(&url, connections, cores, Duration::from_secs(seconds)) {
        Ok(throughput) => {
            println!("rekt: {} cores x {} conns = {} connections, {}s", throughput.cores, connections, throughput.connections, seconds);
            println!("rekt: {} completed, {} errors", throughput.completed, throughput.errors);
            println!("rekt: Requests/sec: {:.2}", throughput.per_sec());
        }
        Err(error) => {
            eprintln!("rekt_load: {error}");
            std::process::exit(1);
        }
    }
}
