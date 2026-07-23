#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Add your own protocol — `Listener::builder().any().protocol(impl
//! AnyProtocol)`, the open seam a downstream crate uses to plug a wire
//! proxima doesn't ship into the SAME accept loop as the built-in h1/h2
//! candidates, with ZERO edits to this crate.
//!
//! `PingPongProtocol` below is authored right here, exactly as a real
//! kafka/mqtt/private-wire crate would author it: it never imports
//! `proxima_listen` directly, only `proxima::prelude` (for the `AnyProtocol`
//! trait + `ProbeVerdict`) and `proxima::{listen, stream, error}` (for the
//! trait's own signature types) — all reachable through the umbrella crate,
//! so a real third party never needs `proxima-listen` as a direct Cargo
//! dependency either.
//!
//! Two things this file proves:
//! 1. `PingPongProtocol` is classified and driven ALONGSIDE the built-in h1
//!    candidate on the SAME bind — registering an external candidate does
//!    not shadow or narrow the first-party set.
//! 2. A one-line ext trait (`PingPongExt::ping_pong`) makes the seam read
//!    exactly like a first-party axis (`.kafka(handler)`, `.mqtt(handler)`)
//!    — because it delegates to the SAME public `.protocol()` method those
//!    use internally. First-party protocols are not special; this IS the
//!    mechanism, not a simplified stand-in for it.
//!
//! Grounded in `tests/e2e/listener_any_protocol_extension.rs` (an equivalent
//! `MiniProtocol`, proven end to end as a `#[proxima::test]`) — this file is
//! the `cargo run`-able version, tokio-free.
//!
//! Run: `cargo run --example extend_protocol --features http1-native`
//! `cargo tree --example extend_protocol --features http1-native -e normal -i tokio`
//! is empty — tokio-free end to end, same as `any_listener.rs`.

use std::future::Future;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream as StdTcpStream};
use std::pin::Pin;

use bytes::Bytes;
use serde_json::Value;

use proxima::listen::admission::ConnAdmission;
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::stream::{PeerInfo, StreamConnection};
use proxima::{ProximaError, SendPipe};

/// The literal a real `PINGPONG/1` client would open a connection with — a
/// fixed prefix distinct from any h1 request line or h2 preface, so the
/// classifier can tell it apart from the built-in candidates by byte zero.
const PING: &[u8] = b"PINGPONG/1 PING\r\n";
const PONG: &[u8] = b"PINGPONG/1 PONG\r\n";

/// A third-party `AnyProtocol` candidate — standing in for a kafka/mqtt/
/// private-wire crate that would `impl AnyProtocol` against
/// `proxima::prelude::AnyProtocol` with no dependency on `proxima-listen`.
struct PingPongProtocol;

impl AnyProtocol for PingPongProtocol {
    fn name(&self) -> &str {
        "pingpong"
    }

    fn max_prefix_bytes(&self) -> usize {
        PING.len()
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(PING.len());
        if prefix[..compare_len] != PING[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < PING.len() {
            return ProbeVerdict::NeedMore { at_least: PING.len() };
        }
        ProbeVerdict::Match { consumed: PING.len() }
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
            stream.write_all(PONG).await?;
            stream.close().await?;
            Ok(())
        })
    }
}

/// The one-line ext trait a downstream crate mints on top of the open
/// `.protocol()` seam — WHY this is legal: `PingPongExt` is a trait THIS
/// crate (the caller) defines, implemented for the FOREIGN `ListenerBuilder`
/// type. Rust's orphan rule allows `impl ForeignTrait for ForeignType` only
/// when either the trait or the type is local; here the TRAIT is local, so
/// implementing it for `ListenerBuilder` (defined in the `proxima` crate) is
/// legal even though neither this file nor `ListenerBuilder` know about each
/// other. `.kafka(handler)`/`.mqtt(handler)`/`.amqp(handler)` are the
/// first-party version of EXACTLY this pattern — see `src/listener/protocol.rs`.
trait PingPongExt: Sized {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self;
}

impl PingPongExt for ListenerBuilder {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self {
        self.protocol(protocol)
    }
}

struct LegitOk;

impl SendPipe for LegitOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Ok(Response::ok("legit-h1-still-works"))
    }
}

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn h1_round_trip(addr: SocketAddr) -> Result<String, ProximaError> {
    let mut stream = StdTcpStream::connect(addr)?;
    stream.write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(String::from_utf8_lossy(&response).into_owned())
}

fn pingpong_round_trip(addr: SocketAddr) -> Result<Vec<u8>, ProximaError> {
    let mut stream = StdTcpStream::connect(addr)?;
    stream.write_all(PING)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;

    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(LegitOk))
        .any()
        .ping_pong(PingPongProtocol)
        .serve()
        .await?;

    // wait for the accept loop, the same bounded poll every on-ramp example uses
    for _ in 0..200 {
        if StdTcpStream::connect(bind).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let h1_text = h1_round_trip(bind)?;
    assert!(
        h1_text.starts_with("HTTP/1.1 200"),
        "the built-in h1 candidate must still route normally alongside the external one: {h1_text:?}"
    );
    assert!(h1_text.contains("legit-h1-still-works"), "got: {h1_text:?}");
    println!(".any().ping_pong(PingPongProtocol) still routes legit h1 traffic on {bind}");

    let pong = pingpong_round_trip(bind)?;
    assert_eq!(&pong[..], PONG, "the externally-registered candidate must drive its own reply");
    println!(
        "a PINGPONG/1 connection on the SAME {bind} is classified and driven by PingPongProtocol's \
         own drive() -> {:?}",
        String::from_utf8_lossy(&pong)
    );

    server.stop();
    println!("\nextend_protocol: a third-party AnyProtocol candidate works with zero core edits");
    Ok(())
}
