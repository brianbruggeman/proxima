//! The incumbent baseline for the h1 matrix: axum (hyper/tokio) serving the
//! same contract as `bench_server` — fixed `200` / `"ok"` / keep-alive — so
//! the only variable across the two servers is the substrate underneath.
//!
//!   cargo run --release --example bench_server_axum -- 127.0.0.1:8080 [cores]

use std::error::Error;
use std::net::SocketAddr;

use axum::Router;
use axum::routing::get;

fn main() {
    if let Err(error) = run() {
        eprintln!("bench_server_axum: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let addr_text = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());
    let cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1);
    let addr: SocketAddr = addr_text.parse()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cores.max(1))
        .enable_all()
        .build()?;

    runtime.block_on(async {
        let app = Router::new().route("/", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("bench_server_axum: axum h1 on {addr} ({cores} core(s))");
        axum::serve(listener, app).await?;
        Ok(())
    })
}
