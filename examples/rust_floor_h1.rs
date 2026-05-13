//! rust_floor_h1 — the apples-to-apples FLOOR for proxima's h1 server: a minimal
//! HTTP/1.1 static-`200 ok` responder on RAW prime (same runtime as proxima),
//! with NO Pipe, NO ListenProtocol, NO h1 codec — just per-core SO_REUSEPORT
//! accept loops and a fixed response write. The gap between this and proxima's
//! full h1 stack is proxima's composable overhead with the runtime held
//! constant; the gap between this and nginx's `return 200 "ok"` is the raw
//! C-static-path ceiling. prime's TcpListener::bind sets SO_REUSEPORT
//! unconditionally, so N per-core listeners on one port let the kernel
//! load-balance accepts across cores.
//!
//!   cargo run --release --example rust_floor_h1 -- 127.0.0.1:8070 [cores]

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use prime::os::core_shard::spawn_on_current_core;
use prime::os::net::{TcpListener, TcpStream};
use proxima::runtime::{CoreId, PrimeRuntime, Runtime};

// same 2-byte "ok" body as nginx `return 200 "ok"`, keep-alive by default (1.1).
const RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _telemetry = proxima_telemetry::export::install_console_logging();
    let mut args = std::env::args().skip(1);
    let addr: SocketAddr = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:8070".to_string())
        .parse()?;
    let cores: usize = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(1);

    let runtime = Arc::new(PrimeRuntime::new(cores)?);
    for core in 0..cores {
        let factory: Box<
            dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
        > = Box::new(move || {
            Box::pin(async move {
                let mut listener = match TcpListener::bind(addr) {
                    Ok(listener) => listener,
                    Err(err) => {
                        eprintln!("[floor] core {core} bind failed: {err}");
                        return;
                    }
                };
                eprintln!("[floor] core {core} listening on {addr}");
                loop {
                    match listener.accept().await {
                        Ok((stream, _peer)) => spawn_on_current_core(Box::pin(serve(stream))),
                        Err(err) => {
                            eprintln!("[floor] core {core} accept error: {err}");
                            break;
                        }
                    }
                }
            })
        });
        runtime.spawn_factory_on_core(CoreId(core), factory)?;
    }

    eprintln!("rust_floor_h1: raw-prime static-200 on {addr} ({cores} core(s))");
    loop {
        std::thread::park();
    }
}

// one bounded read then a fixed response, looped for keep-alive. No request
// parsing — the floor does the least a 200-ok server can.
async fn serve(mut stream: TcpStream) {
    let mut buf = [0u8; 1024];
    loop {
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if stream.write_all(RESPONSE).await.is_err() {
                    break;
                }
            }
        }
    }
}
