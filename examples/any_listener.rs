#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `.any()` through `Listener::builder()`: one bind, no protocol pick — the
//! open universal listener classifies each connection's own leading bytes
//! (`proxima_listen::any::Classifier`, `proxima-listen/src/any/classifier.rs`)
//! against every registered `AnyProtocol` candidate (`H1AnyProtocol`,
//! `H2PriorKnowledgeAnyProtocol`) and dispatches to whichever one matches. You
//! never tell it "this port speaks h1" or "this port speaks h2" — it sniffs.
//!
//! Two listeners, side by side, proving the two ends of the same axis:
//!
//! 1. SAME PORT — `.any()` on one bind, one handler. An h1 client and an h2
//!    client both connect to the identical address and both get routed
//!    correctly.
//! 2. SEPARATE PORT — `.accept("h2")` hard-pins a SECOND bind to exactly one
//!    candidate. The same h2 client succeeds against it; an h1 client gets no
//!    response at all (the classifier only knows about "h2" on this port).
//!
//! `.accept(name)` is sugar over `.accepts(&[name])` with one entry
//! (`src/listener/handle.rs`'s `ListenerBuilder::accept`); `.accepts(&[...])`
//! restricts the open listener to a named subset instead of every registered
//! candidate. Both narrow the SAME classifier, they just shrink the candidate
//! set it is allowed to pick from.
//!
//! Run: `cargo run --example any_listener --features http1-native`
//!
//! `cargo tree --example any_listener --features http1-native -e normal -i tokio`
//! is empty — tokio-free end to end, same as `h2_native_server`.

use std::future::Future;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream as StdTcpStream};
use std::time::Duration;

use bytes::Bytes;
use proxima::h2::H2ClientUpstream;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::time::sleep;
use proxima::{Listener, ListenerBuilderEntry, PrimeTcpUpstream, ProximaError};
use proxima_primitives::pipe::SendPipe;

/// The one handler both listeners in this file dispatch to — same shape
/// regardless of which wire the classifier picked.
struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::new(200).with_body(Bytes::from_static(b"ok"))) }
    }
}

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn constant_ok_request() -> Result<Request<Bytes>, ProximaError> {
    Request::builder()
        .method("POST")
        .path("/")
        .body(Bytes::from_static(b"hello"))
        .build()
}

/// A plain h1/1.1 request over a blocking `std::net::TcpStream` — the
/// tokio-free client idiom `h1_native_prime_round_trip.rs` uses, reused here
/// so the h1 half of this file needs no async client dependency at all.
fn h1_round_trip(addr: SocketAddr) -> Result<String, ProximaError> {
    let mut stream = StdTcpStream::connect(addr)?;
    stream.write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(String::from_utf8_lossy(&response).into_owned())
}

async fn h2_round_trip(addr: SocketAddr, label: &str) -> Result<Response<Bytes>, ProximaError> {
    let client =
        H2ClientUpstream::new(PrimeTcpUpstream::new(addr), format!("{addr}"), false, label);
    let mut attempts_left = 100;
    loop {
        match client.call(constant_ok_request()?).await {
            Ok(response) => return Ok(response),
            Err(error) if attempts_left > 0 => {
                attempts_left -= 1;
                sleep(Duration::from_millis(20)).await;
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
}

/// A connection this port refuses to classify returns nothing over the wire
/// — no status line, no framing, just a closed/reset socket once the
/// classifier gives up. `dial_and_collect` proves "got no HTTP response" the
/// same way `tests/e2e/listener_deny_blacklist.rs` does: tolerate a clean EOF
/// or a reset, either is a legitimate way for a dropped connection to end.
fn dial_and_collect(addr: SocketAddr, payload: &[u8]) -> Vec<u8> {
    let mut collected = Vec::new();
    if let Ok(mut stream) = StdTcpStream::connect(addr)
        && stream.write_all(payload).is_ok()
    {
        let _ = stream.flush();
        let _ = stream.read_to_end(&mut collected);
    }
    collected
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    // ── 1. SAME PORT: `.any()` sniffs h1 vs h2 on one bind ──────────────────
    let any_bind = free_loopback_addr()?;
    let any_server = Listener::builder()
        .bind(any_bind)
        .any()
        .handle(into_handle(ConstantOk))
        .serve()
        .await?;

    let h1_text = h1_round_trip(any_bind)?;
    assert!(
        h1_text.starts_with("HTTP/1.1 200"),
        "an h1 client dialing the .any() listener must be classified as h1: {h1_text:?}"
    );
    println!(".any() classified a plain h1 client correctly on {any_bind}");

    let h2_response = h2_round_trip(any_bind, "any-h1-and-h2").await?;
    assert_eq!(
        h2_response.status, 200,
        "an h2 client dialing the SAME .any() listener must be classified as h2"
    );
    assert_eq!(h2_response.payload.as_ref(), b"ok");
    println!(".any() classified a native h2 client correctly on the SAME port {any_bind}");

    any_server.stop();

    // ── 2. SEPARATE PORT: `.accept("h2")` hard-pins one candidate ──────────
    let pinned_bind = free_loopback_addr()?;
    let pinned_server = Listener::builder()
        .bind(pinned_bind)
        .accept("h2")
        .handle(into_handle(ConstantOk))
        .serve()
        .await?;

    let pinned_h2_response = h2_round_trip(pinned_bind, "pinned-h2-only").await?;
    assert_eq!(pinned_h2_response.status, 200);
    println!(".accept(\"h2\") still serves a real h2 client on its own port {pinned_bind}");

    let rejected = dial_and_collect(
        pinned_bind,
        b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    );
    let rejected_text = String::from_utf8_lossy(&rejected);
    assert!(
        !rejected_text.starts_with("HTTP/"),
        "an h1 client dialing a port pinned to .accept(\"h2\") must get no HTTP response, got: {rejected_text:?}"
    );
    println!(
        ".accept(\"h2\") never classifies an h1 client on {pinned_bind} — {} bytes came back, no status line",
        rejected.len()
    );

    pinned_server.stop();

    println!("any_listener: same-port auto-detect and separate-port hard-pin both OK");
    Ok(())
}
