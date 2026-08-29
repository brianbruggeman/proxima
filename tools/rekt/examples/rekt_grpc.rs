//! gRPC load driver over proxima's multiplexed h2. The grpc sibling of
//! `rekt_h2`: each connection keeps `streams_per_conn` unary calls to `path` in
//! flight (POST + application/grpc + a length-prefixed message), refilled the
//! instant the response trailer closes a stream.
//!
//!   rekt_grpc <url> <connections_per_core> <streams_per_conn> <cores> <secs> [path]
//!   rekt_grpc http://127.0.0.1:8100/ 4 64 1 5 /helloworld.Greeter/SayHello

use std::time::Duration;

use rekt::grpcload::drive_grpc;

fn main() {
    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8100/".to_string());
    let connections: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(4);
    let streams: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(64);
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let secs: u64 = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(5);
    let path = args
        .next()
        .unwrap_or_else(|| "/helloworld.Greeter/SayHello".to_string());

    match drive_grpc(&url, &path, connections, streams, cores, Duration::from_secs(secs)) {
        Ok(throughput) => {
            println!("rekt: {} conns x {} streams, {}s", throughput.connections, streams, secs);
            println!("rekt: {} completed, {} errors", throughput.completed, throughput.errors);
            println!("rekt: Requests/sec: {:.2}", throughput.per_sec());
        }
        Err(error) => {
            eprintln!("rekt_grpc: {error}");
            std::process::exit(1);
        }
    }
}
