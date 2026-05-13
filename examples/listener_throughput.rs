#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Multi-core listener throughput harness.
//!
//! Spins up an `App` serving a constant `"ok"` response over HTTP/1
//! with SO_REUSEPORT per-core lanes. Then concurrently fires N
//! client connections (each running K keep-alive requests in
//! sequence) and measures aggregate requests/second.
//!
//! Run:
//!     cargo run --release --example listener_throughput
//!     cargo run --release --example listener_throughput -- --cores 4 \
//!         --connections 64 --requests 10000
//!
//! ## macOS caveat
//!
//! macOS's `SO_REUSEPORT` accepts multiple binds to the same port
//! but does NOT kernel-side load-balance across them. Effectively
//! one of the bound sockets receives most of the traffic. So you'll
//! see flat numbers regardless of `--cores` on Mac. Linux's
//! `SO_REUSEPORT` distributes by 5-tuple hash; the same harness
//! there scales near-linearly to core count up to where the kernel
//! / NIC / DRAM bandwidth caps out.
//!
//! ## Methodology
//!
//! The client tasks run inside the same tokio runtime as the
//! load-test process — so for high core counts the client side can
//! become the bottleneck. For production-grade benchmarks, run
//! `wrk`/`bombardier`/`h2load` against a separately-bound proxima
//! daemon and measure from outside.

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use proxima::{App, Spec, TokioPerCoreRuntime};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Debug)]
struct Settings {
    cores: usize,
    connections: usize,
    requests_per_conn: usize,
    warmup_secs: u64,
}

impl Settings {
    fn from_args() -> Self {
        let mut settings = Self {
            cores: num_cpus::get(),
            connections: 64,
            requests_per_conn: 5_000,
            warmup_secs: 1,
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--cores" => {
                    settings.cores = args.next().unwrap_or_default().parse().unwrap_or(1);
                }
                "--connections" => {
                    settings.connections = args.next().unwrap_or_default().parse().unwrap_or(64);
                }
                "--requests" => {
                    settings.requests_per_conn =
                        args.next().unwrap_or_default().parse().unwrap_or(5_000);
                }
                "--warmup" => {
                    settings.warmup_secs = args.next().unwrap_or_default().parse().unwrap_or(1);
                }
                _ => {
                    eprintln!("unknown arg: {arg}");
                }
            }
        }
        settings
    }
}

const KEEPALIVE_REQUEST: &[u8] = b"GET / HTTP/1.1\r\nHost: bench\r\nConnection: keep-alive\r\n\r\n";

async fn drive_connection(addr: std::net::SocketAddr, requests: usize, counter: Arc<AtomicU64>) {
    let mut stream = match TcpStream::connect(addr).await {
        Ok(stream) => stream,
        Err(_) => return,
    };
    let _ = stream.set_nodelay(true);
    let mut read_buf = vec![0_u8; 8 * 1024];
    let mut first = std::env::var("PROXIMA_BENCH_DEBUG").is_ok();
    for _ in 0..requests {
        if stream.write_all(KEEPALIVE_REQUEST).await.is_err() {
            return;
        }
        // Read until we've seen one full response. The server returns
        // a status-line + headers + 2-byte body ("ok"). Track bytes
        // consumed since the headers-end position to know when the
        // body is fully received.
        let mut header_done = false;
        let mut body_remaining: usize = 2;
        let mut header_buffer: Vec<u8> = Vec::with_capacity(256);
        let mut completed = false;
        let mut debug_capture: Vec<u8> = if first {
            Vec::with_capacity(512)
        } else {
            Vec::new()
        };
        while !completed {
            let read = match stream.read(&mut read_buf).await {
                Ok(0) => return,
                Ok(n) => n,
                Err(_) => return,
            };
            if first {
                debug_capture.extend_from_slice(&read_buf[..read]);
                if debug_capture.len() >= 256 || header_done {
                    eprintln!(
                        "first response bytes ({} captured): {:?}",
                        debug_capture.len(),
                        String::from_utf8_lossy(&debug_capture)
                    );
                    first = false;
                }
            }
            if !header_done {
                header_buffer.extend_from_slice(&read_buf[..read]);
                if let Some(pos) = find_double_crlf(&header_buffer) {
                    header_done = true;
                    let body_in_this_read = header_buffer.len().saturating_sub(pos);
                    if body_in_this_read >= body_remaining {
                        completed = true;
                    } else {
                        body_remaining -= body_in_this_read;
                    }
                }
            } else if read >= body_remaining {
                completed = true;
            } else {
                body_remaining -= read;
            }
        }
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

fn find_double_crlf(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() {
    let settings = Settings::from_args();
    println!(
        "config: cores={} connections={} requests_per_conn={} warmup_secs={}",
        settings.cores, settings.connections, settings.requests_per_conn, settings.warmup_secs
    );

    // Single App + single per-core runtime. load_full triggers the
    // SO_REUSEPORT-per-core path (run_until_signal is single-listener).
    let runtime =
        Arc::new(TokioPerCoreRuntime::new(settings.cores).expect("build per-core runtime"));
    let mut app = App::new().expect("app").with_runtime(runtime);
    let listener_config = json!({
        "pipe": [
            { "name": "echo", "synth": { "status": 200, "body": "ok" } }
        ],
        "listen": [
            { "type": "http", "bind": "127.0.0.1:0", "pipe": "echo" }
        ]
    });
    let listener_handles = app
        .load_full(Spec::Inline(listener_config))
        .await
        .expect("load full");
    let listener_addr = listener_handles
        .first()
        .and_then(|handle| handle.bind_addr())
        .expect("bind_addr");
    println!("listener bound at {listener_addr}");

    // Wait for the listener to actually be accepting.
    for _ in 0..100 {
        if TcpStream::connect(listener_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let total_requests = (settings.connections * settings.requests_per_conn) as u64;
    let counter = Arc::new(AtomicU64::new(0));

    // Optional warmup: drive a small number of requests to ensure the
    // per-core listeners are all hot.
    if settings.warmup_secs > 0 {
        println!("warmup ({} s)...", settings.warmup_secs);
        let warmup_counter = Arc::new(AtomicU64::new(0));
        let warmup_handles: Vec<_> = (0..settings.connections.min(16))
            .map(|_| {
                let counter = warmup_counter.clone();
                tokio::spawn(async move {
                    drive_connection(listener_addr, 500, counter).await;
                })
            })
            .collect();
        for handle in warmup_handles {
            let _ = handle.await;
        }
    }

    // Real load.
    println!("running load: {total_requests} total requests");
    let started = Instant::now();
    let mut handles = Vec::with_capacity(settings.connections);
    for _ in 0..settings.connections {
        let counter = counter.clone();
        let requests = settings.requests_per_conn;
        handles.push(tokio::spawn(async move {
            drive_connection(listener_addr, requests, counter).await;
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
    let elapsed = started.elapsed();
    let completed = counter.load(Ordering::Relaxed);
    let rps = completed as f64 / elapsed.as_secs_f64();
    let per_core = rps / settings.cores as f64;

    println!();
    println!("results:");
    println!("  completed:   {completed} / {total_requests} requests");
    println!("  elapsed:     {:.3} s", elapsed.as_secs_f64());
    println!("  rps:         {rps:.0} req/s aggregate");
    println!("  per-core:    {per_core:.0} req/s/core");

    // ListenerHandle drop fires the shutdown signal automatically.
    drop(listener_handles);
    drop(app);
}
