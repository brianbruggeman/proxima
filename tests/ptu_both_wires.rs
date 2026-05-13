//! P-TU slice 2 — both wires in one build, runtime-selected.
//!
//! A prime-default proxima build that ALSO carries the tokio transport. A
//! `proxima::Client` whose upstream spec sets `"wire": "tokio"` resolves to the
//! hyper (tokio) backend and, dialed off a tokio reactor, hops onto a shared
//! tokio sidecar to construct the stream — while every other upstream stays on
//! the prime wire by default. This is the compatibility path for systems that
//! stay on tokio: a prime process can still dial a tokio-only upstream.
//!
//! Both-wires build (default = prime + http-prime; plus the tokio wire):
//!   cargo test --test ptu_both_wires --features "http-hyper,tokio-runtime,runtime-tokio"
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read as _, Write as _};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::mpsc;

use proxima::Client;
use serde_json::json;

fn spawn_loopback(reply: &'static [u8]) -> u16 {
    let (port_tx, port_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
        port_tx
            .send(listener.local_addr().expect("addr").port())
            .expect("send port");
        let (mut socket, _) = listener.accept().expect("accept");
        let mut buffer = Vec::new();
        let mut scratch = [0_u8; 1024];
        loop {
            let read = socket.read(&mut scratch).expect("read");
            buffer.extend_from_slice(&scratch[..read]);
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        socket.write_all(reply).expect("write");
        socket.flush().expect("flush");
    });
    port_rx.recv().expect("port")
}

// the both-wires proof: `"wire":"tokio"` routes to the hyper backend and hops
// onto the shared tokio sidecar, completing a real dial — in a build whose
// default wire is prime.
#[test]
fn wire_tokio_upstream_dials_via_the_tokio_sidecar() {
    let port = spawn_loopback(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi");

    let body = futures::executor::block_on(async {
        let client = Client::builder()
            .spec("http", json!(format!("http://127.0.0.1:{port}")))
            .spec("wire", json!("tokio"))
            .build()
            .expect("build client");
        let response = client
            .call("GET", "/")
            .send()
            .await
            .expect("send wire:tokio dial");
        assert_eq!(response.status(), 200);
        response.bytes().await.expect("bytes")
    });

    assert_eq!(&body[..], b"hi");
}

// the default wire stays prime: an upstream with NO `"wire"` field resolves to
// the prime backend and dials over it (same loopback server, prime transport).
#[test]
fn default_upstream_stays_on_the_prime_wire() {
    let port = spawn_loopback(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nprime");

    let body = futures::executor::block_on(async {
        let client = Client::http(format!("http://127.0.0.1:{port}")).expect("build client");
        let response = client
            .call("GET", "/")
            .send()
            .await
            .expect("send default-prime dial");
        assert_eq!(response.status(), 200);
        response.bytes().await.expect("bytes")
    });

    assert_eq!(&body[..], b"prime");
}
