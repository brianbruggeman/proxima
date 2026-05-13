//! P-TU test #1 — does the `"wire":"tokio"` path dial `https://`?
//!
//! GOTCHA proven here: `http-hyper` alone is PLAINTEXT — the hyper client only
//! gets an `HttpsConnector` (webpki roots) when proxima-h1's `tls` feature is on.
//! So a both-wires build that needs https tokio upstreams must enable `tls`.
//!
//! This stands up a real self-signed TLS server and dials it via `wire:tokio`.
//! The hyper client trusts only webpki roots, so the self-signed cert is rejected
//! — and that CERT-VALIDATION error is the proof: the `wire:tokio` path routed the
//! https URL through hyper+TLS on the tokio sidecar and ran the handshake up to
//! cert validation. A reactor/unsupported/pre-TLS-connect error would mean https
//! is not actually wired through the tokio wire.
//!
//!   cargo test --test ptu_https_wire \
//!     --features "http-hyper,tokio-runtime,runtime-tokio,tls" -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]
// the internal gate must match the full combo in the doc comment above, not
// just `tls`: without http-hyper the "http-tokio" wire alias this test
// requests is never registered (default_pipe_factory_registry), so the
// client silently fails before ever dialing the server instead of cleanly
// skipping — the "5s timeout waiting for an accept" symptom.
#![cfg(all(
    feature = "tls",
    feature = "http-hyper",
    feature = "tokio-runtime",
    feature = "runtime-tokio"
))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use proxima::Client;
use proxima::tls::{TlsConfig, build_acceptor};
use serde_json::json;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;

// a real self-signed HTTPS loopback, served on its own current-thread tokio
// runtime (the TLS acceptor is tokio-rustls). `accepts` counts TCP connections so
// the test can tell "client reached us over TCP then failed at TLS" (cert
// rejection) from "connection refused / server down".
fn https_loopback(accepts: Arc<AtomicUsize>) -> (u16, mpsc::Receiver<()>) {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let acceptor = build_acceptor(&TlsConfig::self_signed()).expect("build acceptor");
    let (port_tx, port_rx) = mpsc::channel();
    // the accept signal is the happens-before edge the test waits on instead of
    // sleeping: the server fires it AFTER bumping `accepts`, so a recv proves the
    // TCP connection landed without racing a timer.
    let (accept_tx, accept_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tls server runtime");
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            port_tx
                .send(listener.local_addr().expect("addr").port())
                .expect("send port");
            loop {
                let Ok((socket, _peer)) = listener.accept().await else {
                    continue;
                };
                accepts.fetch_add(1, Ordering::Release);
                let _ = accept_tx.send(());
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let Ok(mut tls) = acceptor.accept(socket).await else {
                        return;
                    };
                    let mut scratch = [0_u8; 1024];
                    let _ = tls.read(&mut scratch).await;
                    let _ = tls
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
                        .await;
                    let _ = tls.shutdown().await;
                });
            }
        });
    });
    (port_rx.recv().expect("port"), accept_rx)
}

#[test]
fn wire_tokio_dials_https_through_hyper_tls() {
    let accepts = Arc::new(AtomicUsize::new(0));
    let (port, accept_rx) = https_loopback(accepts.clone());
    let url = format!("https://127.0.0.1:{port}");

    let outcome = futures::executor::block_on(async {
        let client = Client::builder()
            .spec("http", json!(url))
            .spec("wire", json!("tokio"))
            .build()
            .expect("build client");
        client.call("GET", "/").send().await
    });

    match outcome {
        // a trusted-roots client correctly rejects the self-signed cert; a success
        // would only happen if trust were bypassed (it isn't).
        Ok(response) => panic!(
            "unexpected success: self-signed cert should not be trusted (status {})",
            response.status()
        ),
        Err(err) => {
            // ProximaError flattens hyper's inner TLS cause to "client error
            // (Connect)", so disambiguate via the server: the client reached us
            // over TCP (≥1 accept) and then failed — i.e. the failure is at the TLS
            // layer (webpki correctly rejecting the self-signed cert), NOT a
            // connection refusal. that proves the wire:tokio path routed https
            // through hyper+TLS on the sidecar and ran the handshake.
            accept_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("server never accepted a TCP connection within 5s");
            let tcp_accepts = accepts.load(Ordering::Acquire);
            println!("PTU_HTTPS_TLS wire:tokio https err={err}  server_tcp_accepts={tcp_accepts}");
            assert!(
                tcp_accepts >= 1,
                "client never reached the TLS server over TCP — https not wired through the tokio wire (err: {err})"
            );
        }
    }
}
