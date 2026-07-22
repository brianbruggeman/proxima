//! The production shape of a proxima REST service: write a handler, mount it,
//! serve it until a signal. Three lines of wiring — the rest of any service is
//! more pipes composed in front of this one; the shape below never changes.
//!
//! ```sh
//! cargo run --example hello --features http1-native
//! # in another shell:
//! curl http://127.0.0.1:8080/     # -> hello, proxima
//! # ctrl-c the server: it stops accepting, drains in-flight requests, exits.
//! ```
//!
//! `http1-native` is required, not default: it registers the h1+h2
//! ALPN-multiplexed listener `RunConfig::http` names, over the tokio-free
//! sans-IO h1 driver (`http1-native`'s `serve_connection`/`serve_h1_connection`
//! in `proxima-http`, generic over `futures::io::AsyncRead`/`AsyncWrite`).
//! This example is tokio-free end to end — verify with
//! `cargo tree --no-default-features --features
//! runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http1-native
//! -e normal -i tokio` (empty). `http1` layers the legacy hyper/tokio h1
//! client+listener stack on top of `http1-native`, for callers that need it —
//! see the `tokio` feature's doc comment in `Cargo.toml`.
//!
//! See `examples/hello/README.md` for the writeup. `runtime-select` reuses this
//! exact pipe and swaps only the runtime underneath it.

use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use proxima::{App, ProximaError, Request, Response, RunConfig};

/// A handler is just an `async fn`: typed request in, typed response out, nothing
/// more. It never touches a socket — the listener owns that; the handler answers.
///
/// No attribute is needed to mount it: `App::mount` takes a bare
/// `async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>` directly.
/// `#[proxima::instrument]` wraps it in a span so every call is traced — one
/// attribute yields trace + metric + log. Reach for `#[proxima::piped]` only when
/// you want a *named*, reusable pipe type instead of a one-off handler.
#[proxima::instrument]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 8080));

    let app = App::new()?;
    app.mount("/", hello)?;

    // `serve` spawns the listener and returns once it is actually accepting —
    // no polling, no sleeping, no discovering ECONNREFUSED the hard way.
    let server = app.serve(RunConfig::http(bind)).await?;
    println!("listening on http://{bind}");

    // serve until SIGINT/SIGTERM, then stop accepting and let in-flight
    // requests finish. This is the whole shutdown story — no ceremony.
    server.run_until_signal().await;
    Ok(())
}
