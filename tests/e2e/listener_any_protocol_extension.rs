//! Worked example: `.protocol(impl AnyProtocol)` — the listener-side mirror
//! of `Client::builder().protocol(impl ClientProtocol)`
//! (`src/client/handle.rs:543`) — proving the open universal listener is
//! extensible from OUTSIDE this crate with zero core edits.
//!
//! `MiniProtocol` is authored right here, as a downstream crate would author
//! it: it never imports `proxima_listen` directly, only `proxima::prelude`
//! (for the `AnyProtocol` trait + `ProbeVerdict`) and `proxima::{listen,
//! stream, error}` (for the trait's own signature types — `ConnAdmission`,
//! `AnyHandler`, `PeerInfo`, `StreamConnection`, `ProximaError` — all already
//! reachable through the umbrella crate, so a real third party never needs
//! `proxima-listen` as a direct Cargo dependency either). Registered via
//! `Listener::builder().any().protocol(MiniProtocol)`, it is classified and
//! driven ALONGSIDE the built-in h1 candidate on the SAME listener — a legit
//! h1 request routes through `.handle(pipe)` as always, and a connection
//! opening with the `MINI/1.0\r\n` literal routes to `MiniProtocol::drive`,
//! which this test authored, not `proxima`.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1",
    any(
        feature = "runtime-tokio",
        all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        )
    )
))]

use std::future::Future;
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use proxima::error::ProximaError;
use proxima::listen::admission::ConnAdmission;
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::stream::{PeerInfo, StreamConnection};
use proxima::SendPipe;

/// The literal a real `MINI/1.0` client would open a connection with — a
/// fixed positive-match prefix, the same shape `DenySignature`'s own probe
/// compares against, distinct from any h1 request line or h2 preface.
const MINI_LITERAL: &[u8] = b"MINI/1.0\r\n";

/// The fixed reply `MiniProtocol::drive` writes once it wins classification
/// — proof the connection was actually handed to THIS candidate's own drive,
/// not silently dropped or misrouted to the h1 fallback handler.
const MINI_REPLY: &[u8] = b"MINI/1.0 200 OK\r\n\r\nhello-from-third-party-protocol";

/// A third-party `AnyProtocol` candidate, authored entirely in this test —
/// standing in for a kafka/mqtt/private-wire crate that would `impl
/// AnyProtocol` against `proxima::prelude::AnyProtocol` with no dependency on
/// `proxima-listen`.
struct MiniProtocol;

impl AnyProtocol for MiniProtocol {
    fn name(&self) -> &str {
        "mini"
    }

    fn max_prefix_bytes(&self) -> usize {
        MINI_LITERAL.len()
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(MINI_LITERAL.len());
        if prefix[..compare_len] != MINI_LITERAL[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < MINI_LITERAL.len() {
            return ProbeVerdict::NeedMore {
                at_least: MINI_LITERAL.len(),
            };
        }
        ProbeVerdict::Match {
            consumed: MINI_LITERAL.len(),
        }
    }

    fn drive<'a>(
        &'a self,
        mut stream: Box<dyn StreamConnection>,
        _handler: proxima::listen::any::AnyHandler,
        _spec: &'a Value,
        _peer: Option<PeerInfo>,
        _admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            use futures::AsyncWriteExt as _;
            stream.write_all(MINI_REPLY).await?;
            stream.close().await?;
            Ok(())
        })
    }
}

struct LegitOk;

impl SendPipe for LegitOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok("legit-ok")) }
    }
}

fn free_loopback_addr() -> SocketAddr {
    let probe = StdTcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..200 {
        if StdTcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("listener at {addr} never came up");
}

async fn dial_and_collect(conn: &mut TcpStream, payload: &[u8]) -> Vec<u8> {
    let mut collected = Vec::new();
    if conn.write_all(payload).await.is_ok() {
        let _ = conn.flush().await;
        let _ = conn.read_to_end(&mut collected).await;
    }
    collected
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_any_protocol_registered_via_dot_protocol_is_classified_and_driven() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(LegitOk))
        .any()
        .protocol(MiniProtocol)
        .serve()
        .await
        .expect("listener builder serves");

    wait_until_listening(bind);

    // The built-in h1 candidate still routes normally — registering an
    // external candidate must not shadow or narrow away the first-party set.
    let mut legit_conn = TcpStream::connect(bind).await.expect("legit connect");
    legit_conn
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("legit write");
    legit_conn.flush().await.expect("legit flush");
    let mut legit_response = Vec::with_capacity(256);
    legit_conn
        .read_to_end(&mut legit_response)
        .await
        .expect("legit read");
    let legit_text = String::from_utf8_lossy(&legit_response);
    assert!(
        legit_text.starts_with("HTTP/1.1 200"),
        "legit h1 traffic must still route normally alongside the external \
         candidate; got: {legit_text:?}"
    );
    assert!(legit_text.contains("legit-ok"), "got: {legit_text:?}");

    // A connection opening with the third-party protocol's own wire literal
    // is classified against `MiniProtocol` and driven by ITS OWN `drive` —
    // proof the open-registration seam reaches the same accept loop the
    // built-in candidates use, with zero edits to this crate.
    let mut mini_conn = TcpStream::connect(bind).await.expect("mini connect");
    let mini_response = dial_and_collect(&mut mini_conn, MINI_LITERAL).await;
    let mini_text = String::from_utf8_lossy(&mini_response);
    assert!(
        mini_text.starts_with("MINI/1.0 200 OK"),
        "the externally-registered candidate must drive its own reply; got: {mini_text:?}"
    );
    assert!(
        mini_text.contains("hello-from-third-party-protocol"),
        "got: {mini_text:?}"
    );

    server.stop();
}
