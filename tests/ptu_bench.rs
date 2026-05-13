//! P-TU quick latency measurement (NOT the formal criterion gate): prime-default
//! dial vs `"wire":"tokio"` dial against a persistent loopback, off a bare thread
//! so each path crosses its runtime hop. P-TU is a compatibility capability, not a
//! speedup — the expected result is parity (the wire selector + sidecar hop add no
//! meaningful per-dial overhead over the prime path).
//!
//!   cargo test --test ptu_bench --features "http-hyper,tokio-runtime,runtime-tokio" -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::time::Instant;

use proxima::Client;
use serde_json::json;

fn persistent_loopback() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut socket) = stream else { continue };
            std::thread::spawn(move || {
                let mut scratch = [0_u8; 1024];
                let mut buffer = Vec::new();
                loop {
                    let Ok(read) = socket.read(&mut scratch) else {
                        return;
                    };
                    if read == 0 {
                        return;
                    }
                    buffer.extend_from_slice(&scratch[..read]);
                    if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi");
                let _ = socket.flush();
            });
        }
    });
    port
}

#[test]
fn measure_prime_vs_wire_tokio_dial_latency() {
    let port = persistent_loopback();
    let url = format!("http://127.0.0.1:{port}");
    const N: u32 = 200;

    let dial = |client: Client| {
        futures::executor::block_on(async move {
            let _ = client.call("GET", "/").send().await; // warmup
            let start = Instant::now();
            for _ in 0..N {
                let response = client.call("GET", "/").send().await.expect("send");
                assert_eq!(response.status(), 200);
                let _ = response.bytes().await.expect("bytes");
            }
            start.elapsed()
        })
    };

    let prime = dial(Client::http(&url).expect("prime client"));
    let tokio_wire = dial(
        Client::builder()
            .spec("http", json!(url))
            .spec("wire", json!("tokio"))
            .build()
            .expect("tokio-wire client"),
    );

    println!(
        "PTU_BENCH N={N}  prime_wire={:?}/dial  wire_tokio={:?}/dial",
        prime / N,
        tokio_wire / N
    );
}
