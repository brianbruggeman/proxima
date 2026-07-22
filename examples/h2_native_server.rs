//! `.h2()` through `Listener::builder()`: binds the native, tokio-free h2c
//! listener ([`H2ListenProtocol`](proxima::listeners::H2ListenProtocol)) and
//! proves it with a real h2 client round trip — the native
//! [`H2ClientUpstream`](proxima::h2::H2ClientUpstream) over
//! [`PrimeTcpUpstream`](proxima::PrimeTcpUpstream), no external `h2` crate,
//! no tokio anywhere in the request path.
//!
//! ```sh
//! cargo run --example h2_native_server
//! ```
//!
//! `cargo tree --example h2_native_server -e normal -i tokio` is empty:
//! `.h2()` needs neither `tokio` nor `http1` (the ALPN h1/h2 combiner is a
//! different listener — see `examples/hello`, which DOES need `tokio`).

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use bytes::Bytes;
use proxima::h2::H2ClientUpstream;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::time::sleep;
use proxima::{Listener, ListenerBuilderEntry, PrimeTcpUpstream, ProximaError};
use proxima_primitives::pipe::SendPipe;

/// The handler mounted behind `.h2()` — same one-line pipe shape every
/// proxima listener dispatches to, regardless of wire version. Fixed
/// response (not an echo): `h2_client_prime_e2e.rs`'s `PongPipe` is the
/// established pattern for "prove the round trip", separate from body
/// fidelity, which the h2 codec's own tests already cover.
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

/// Grab a free port with a plain std socket (closes synchronously on drop,
/// unlike an async listener whose driver can linger), then hand the vacated
/// port to `.bind()` — the same probe-then-reuse trick
/// `tests/e2e/listener_h3_native.rs` uses, needed because `Server` (unlike
/// `ListenerHandle`) does not expose `bind_addr()` to discover an ephemeral
/// port after the fact.
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
        .body(Bytes::from_static(b"hello, h2c"))
        .build()
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;

    let server = Listener::builder()
        .bind(bind)
        .h2()
        .handle(into_handle(ConstantOk))
        .serve()
        .await?;

    // `.serve()` resolves once the listener lane is SPAWNED, not once it is
    // actually accepting (`ListenerBuilder::serve`'s readiness-race doc) —
    // bounded retry-connect closes that gap.
    let client = H2ClientUpstream::new(
        PrimeTcpUpstream::new(bind),
        format!("{bind}"),
        false,
        "h2-native-example",
    );
    let mut attempts_left = 100;
    let response = loop {
        match client.call(constant_ok_request()?).await {
            Ok(response) => break response,
            Err(error) if attempts_left > 0 => {
                attempts_left -= 1;
                sleep(Duration::from_millis(20)).await;
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    };

    assert_eq!(response.status, 200, "h2c round trip must return 200");
    assert_eq!(
        response.payload.as_ref(),
        b"ok",
        "h2c round trip must read the server's response body"
    );
    println!("h2_native_server: h2c round trip through Listener::builder().h2() OK on {bind}");

    server.stop();
    Ok(())
}
