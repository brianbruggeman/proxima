//! The production shape of a proxima REST service: write a pipe, mount it,
//! serve it until a signal. Three lines of wiring — the rest of any service is
//! more pipes composed in front of this one; the shape below never changes.
//!
//! ```sh
//! cargo run --example hello
//! # in another shell:
//! curl http://127.0.0.1:8080/     # -> hello, proxima
//! # ctrl-c the server: it stops accepting, drains in-flight requests, exits.
//! ```
//!
//! See `examples/hello/README.md` for the writeup. `runtime-select` reuses this
//! exact pipe and swaps only the runtime underneath it.

use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use proxima::{App, ProximaError, Request, Response, RunConfig};

/// The service is a pipe: typed request in, typed response out, nothing more.
/// It never touches a socket — the listener owns that; the pipe only answers.
///
/// `#[proxima::piped(send)]` makes this function the pipe: `In`/`Out`/`Err` come
/// from the signature, and `send` is the rung — named, because crossing a core
/// is a cost you opt into. The name you mount below is this function.
#[proxima::piped(send)]
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
