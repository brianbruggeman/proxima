# Listener on-ramp, part 1: hello

**Prerequisites:** you're comfortable with Rust and `async`/`.await`; you've
built an HTTP server before (axum, actix, tokio's own hyper wrapper — any of
them). You have never seen proxima. This page and its two follow-ons
(part 2: [the universal listener](./05-listener-universal.md), part 3:
[growing it into production](./06-listener-production.md)) are a fast,
standalone on-ramp to proxima's serve side — they do not require reading
[Foundations](./00-foundations.md) first. If you later want the full pipe
algebra (composition, retries, fan-out, the works), Foundations is where
that lives; this on-ramp only teaches enough to stand up and grow a real
listener.

**You will:** write the smallest complete proxima service that exists —
one handler, one listener, one real HTTP round trip — and understand every
line of it.

**New concepts (in order):** handler · `App` · `App::mount` · `RunConfig` ·
`app.serve` · graceful shutdown.

Every code block below is copied verbatim from a real, runnable file in
this repository, or is a command you can run yourself. Nothing is invented.

## 1. What a listener is

A client *dials out*: "I need to talk to something." A listener does the
opposite — it *waits* for something else to dial in. If you've written a
server in any framework, you already know the shape: bind an address,
register something to answer requests, start accepting connections. proxima
calls the thing that answers a **handler**.

## 2. A handler is just an `async fn`

No special trait, no macro required to get started. A handler is exactly
this shape — typed request in, typed response out
(`examples/hello/main.rs:40-42`):

```rust
use bytes::Bytes;
use proxima::{ProximaError, Request, Response};

async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}
```

`Request<Bytes>` and `Response<Bytes>` are proxima's request/response types —
`Bytes` here is the payload type (the body), same idea as `axum`'s
`Bytes`/`String` extractors, just spelled as a generic parameter. The
handler never touches a socket. It never parses HTTP. It answers one
question — "given this request, what's the response?" — and nothing else.
Everything socket-shaped is the listener's job, not the handler's.

`Response::ok(payload)` (`proxima-primitives/src/pipe/request.rs:612`) is
the one-line "200 OK with this body" constructor you'll use constantly.

## 3. Wiring it up: `App`

`App` is the piece that turns a handler into a running server: it owns the
router (which paths go to which handler) and the actual bind/accept/serve
loop. Three lines, the whole wiring (`examples/hello/main.rs:48-53`):

```rust
let app = App::new()?;
app.mount("/", hello)?;
let server = app.serve(RunConfig::http(bind)).await?;
```

`App::new()` builds (or adopts — more on this in
[Foundations part 3](./03-native-runtime.md) if you're curious later) the
runtime that will drive your handler's futures. `app.mount("/", hello)`
registers `hello` at the root path — mount takes a bare `async fn` directly,
no wrapping required. `app.serve(RunConfig::http(bind))` binds the address
and starts accepting; it returns once the listener is actually
accepting connections, not just spawned — no polling, no sleeping, no
discovering a connection-refused error the hard way.

## 4. Shutdown, in one line

```rust
server.run_until_signal().await;
```

This blocks until SIGINT/SIGTERM (ctrl-c), then stops accepting new
connections and lets in-flight requests finish before the process exits.
That's the entire graceful-shutdown story for this shape — no separate
drain timer to configure, no signal handler to register by hand.

## 5. The whole file

This is `examples/hello/main.rs` in full (trimmed of its own doc comments —
see the file itself for those):

```rust
use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use proxima::{App, ProximaError, Request, Response, RunConfig};

async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 8080));

    let app = App::new()?;
    app.mount("/", hello)?;

    let server = app.serve(RunConfig::http(bind)).await?;
    println!("listening on http://{bind}");

    server.run_until_signal().await;
    Ok(())
}
```

`#[proxima::main]` is the one attribute every proxima binary's `main`
carries — it boots the runtime that drives everything below it (prime,
proxima's own tokio-free runtime, by default).

## 6. Run it, for real

```sh
cargo run --example hello --features http1-native
```

`http1-native` is required — it is not a default feature — because it
registers the tokio-free HTTP/1 driver `RunConfig::http` names
(`proxima-http`'s sans-IO h1 codec, generic over any async socket). In
another shell, a real HTTP client, not an in-process call:

```sh
curl -i http://127.0.0.1:8080/
```

This is the actual output from running exactly that, on this machine, in
this repository:

```
HTTP/1.1 200 OK
traceparent: 00-45b8637f2d7660f884d14e7a4646d943-f545e54f8cacc8ad-01
content-length: 15

hello, proxima
```

The `traceparent` header is not something this tutorial asked for — it
shows up because every request is traced by default (proxima's
observability is on from the first line, not bolted on later; you'll see
more of this in part 3). Press ctrl-c in the server's shell: it stops
accepting, drains, exits cleanly.

That's it. Bind, one handler, serve, one real round trip. Every proxima
service — no matter how large — starts here; nothing about this shape
changes as you grow it. Part 2 keeps the "one handler, one listener" shape
but removes the one thing you still had to decide: which wire protocol you
were speaking.

## Where to go next

- [Part 2: the universal listener](./05-listener-universal.md) — stop
  picking a protocol; let the listener sniff it.
- [Foundations: the Pipe](./00-foundations.md) — once this on-ramp feels
  solid, this is where the deeper composition story (retries, fan-out,
  filters, the whole pipe algebra) lives.
