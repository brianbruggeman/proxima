//! P-TU slice 1B — tokio-hosts-proxima.
//!
//! A `proxima::Client` handed the host's tokio runtime via
//! `TokioPerCoreRuntime::from_handle`, called from a BARE thread (no tokio
//! reactor on it), hops onto the host runtime to construct the transport stream
//! and dials a real loopback server. This is the "any thread can use the client
//! over tokio transport" property — the tokio mirror of the prime off-worker
//! auto-dispatch — and it proves the injected runtime is honored on the tokio
//! path (which it was not before slice 1B: the off-worker hop was
//! `#[cfg(feature = "runtime-prime")]` and the injected runtime ignored).
//!
//! Pure-tokio build:
//!   cargo test --test ptu_tokio_host --no-default-features \
//!     --features "tokio-runtime,runtime-tokio,http-hyper,http1,http2"
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read as _, Write as _};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::mpsc;

use proxima::{Client, Runtime, TokioPerCoreRuntime};
use serde_json::json;

#[test]
fn tokio_host_client_dials_off_a_bare_thread_via_injected_runtime() {
    // a one-shot loopback HTTP/1 server on a std thread.
    let (port_tx, port_rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
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
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
            .expect("write");
        socket.flush().expect("flush");
    });
    let port = port_rx.recv().expect("port");

    // the host tokio runtime the application already owns; the client rides IT.
    let host = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("host runtime");
    let injected: Arc<dyn Runtime> =
        Arc::new(TokioPerCoreRuntime::from_handle(host.handle().clone()));

    // this libtest thread is NOT on a tokio reactor; futures' executor drives
    // send(), so dispatch must hop onto the injected host runtime to construct
    // the hyper stream. without slice 1B this fails for want of a reactor.
    let body = futures::executor::block_on(async {
        let client = Client::builder()
            .spec("http", json!(format!("http://127.0.0.1:{port}")))
            .runtime(injected)
            .build()
            .expect("build client");
        let response = client
            .call("GET", "/")
            .send()
            .await
            .expect("send off the host thread");
        assert_eq!(response.status(), 200);
        response.bytes().await.expect("bytes")
    });

    assert_eq!(&body[..], b"hi");
    server.join().expect("server join");
}
