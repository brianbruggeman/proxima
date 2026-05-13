//! HTTP/3 load driver over proxima's native QUIC. The h3 sibling of `rekt_h2`.
//! Multiplexed: each connection keeps `streams_per_conn` concurrent GETs in
//! flight, refilled the instant one finishes — h3's whole point.
//!
//!   rekt_h3 <host:port> <connections_per_core> <streams_per_conn> <cores> <secs> [server_name]
//!   rekt_h3 127.0.0.1:8094 8 100 1 5

use std::time::Duration;

use rekt::error::Error;
use rekt::h3load::drive_h3;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let addr_text = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8094".to_string());
    let connections: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(8);
    let streams: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(100);
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let secs: u64 = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(5);
    let server_name = args
        .next()
        .unwrap_or_else(|| "localhost".to_string());

    let addr = addr_text
        .parse()
        .map_err(|err| Error::Engine(format!("h3 addr `{addr_text}`: {err}")))?;

    let throughput = drive_h3(addr, &server_name, connections, cores, Duration::from_secs(secs), streams)?;
    println!(
        "rekt h3: {} completed, {} errors ({} conns x {} streams x {} cores, native QUIC)",
        throughput.completed, throughput.errors, connections, streams, cores
    );
    println!("rekt h3: Requests/sec: {:.2}", throughput.per_sec());
    Ok(())
}
