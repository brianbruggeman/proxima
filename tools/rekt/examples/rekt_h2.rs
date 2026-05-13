//! Multiplexed HTTP/2 load driver — the h2 sibling of `rekt_plan`. Opens
//! persistent h2 connections and keeps N concurrent streams in flight per
//! connection (real h2 multiplexing), measuring proxima's h2 server throughput.
//!
//!   rekt_h2 <url> <connections_per_core> <streams_per_conn> <cores> <secs>
//!   rekt_h2 http://127.0.0.1:8090/ 1 64 1 5

use std::time::Duration;

use rekt::error::Error;
use rekt::h2load::drive_h2;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8090/".to_string());
    let connections: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
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

    let throughput = drive_h2(&url, connections, streams, cores, Duration::from_secs(secs))?;
    println!(
        "rekt h2: {} completed, {} errors ({} conn x {} streams x {} cores)",
        throughput.completed, throughput.errors, connections, streams, cores
    );
    println!("rekt h2: Requests/sec: {:.2}", throughput.per_sec());
    Ok(())
}
