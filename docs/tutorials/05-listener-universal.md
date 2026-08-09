# Listener on-ramp, part 2: the universal listener

**Prerequisites:** [part 1: hello](./04-listener-hello.md) — you should be
comfortable with "handler = `async fn`, mount it, serve it."

**You will:** bind ONE port that answers both HTTP/1.1 and HTTP/2 clients,
without ever telling it which one to expect — then narrow it back down to
exactly one wire, in one line, when you want that instead.

**New concepts (in order):** `Listener::builder()` · `.any()` · `.accept(name)`
/ `.accepts([...])`.

Every code block below is copied verbatim from a real, runnable file in this
repository, a command you can run yourself, or a real excerpt wrapped in just
enough real-signature scaffolding to type-check on its own — flagged inline,
every time, right where it happens. The full, runnable file behind sections 3
and 4 is `examples/any_listener.rs` (`cargo run --example any_listener
--features http1-native`; verified tokio-free with `cargo tree --features
http1-native -e normal -i tokio`, empty result).

## 1. The problem this solves

Part 1's `RunConfig::http(bind)` already answered both a plain h1 client and
a native h2 client on the same *plaintext* port — no TLS anywhere in that
demo. That is not TLS's ALPN extension (ALPN only exists once TLS is on the
wire, and part 1 never turned TLS on): `HttpListenProtocol`, what
`RunConfig::http`/`Listener::http` resolve to, is already built on the exact
byte-prefix classifier this page is about, just with the candidate set welded
shut to exactly `{h1, h2}` — its own doc says so directly: "this combiner's
... plumbing collapsed onto the SAME machinery `.any()` drives"
(`proxima-http/src/listener/mod.rs:294-330`). Each candidate answers one
question against the connection's own leading bytes — an h1 request line
(`GET / HTTP/1.1\r\n...`) looks nothing like h2's fixed 24-byte connection
preface, so telling them apart never needs a negotiation, only a look.
(Layering `.tls(cfg)` on top adds one shortcut on top of the SAME fallback:
ALPN gets first look, and a candidate it names wins outright without ever
reaching the classifier — but an absent or unmatched ALPN result still falls
through to the identical byte-sniff, `proxima-http/src/any_listener.rs:1369-1415`.
Out of scope here — this page stays plaintext throughout.)

`.any()` is that identical classifier, generalized: instead of a hardcoded
`{h1, h2}` pair welded into one fixed `ListenProtocol`, it accepts EVERY
candidate currently registered on your `App` — kafka, redis, dns, pgwire, a
scanner deny-list, or a protocol you define yourself (part 6) — and it is
reachable directly from `Listener::builder()` instead of being buried inside
one hardcoded combination.

## 2. Meet `Listener::builder()`

Part 1 used `App` directly. `Listener::builder()` is a second front door onto
the exact same serve machinery — a fluent chain that ends in `.serve()`,
which composes `App::new` + `App::mount` + `App::serve` under the hood, the
identical `into_handle(pipe) -> App::new()? -> app.mount(...)? ->
app.serve(...)` idiom part 1 taught, just automated behind the builder — no
second serve loop was invented for this (`src/listener/handle.rs:427-433`,
its own doc says so directly). The shape is `.bind(addr).any().handle(pipe).serve()`
— one fluent chain, ending in an `.await`.

`.handle(pipe)` is the one thing every `Listener::builder()` chain needs that
`App::mount` also needed — where to dispatch (`src/listener/handle.rs:193-198`).
It asks for one thing MORE than `App::mount` did in part 1, though.
`app.mount("/", hello)` accepted `hello` as a bare `async fn` only because
`App::mount` carries its own private adapter for exactly that shape
(`FnHandler`, reachable solely through `App::mount`'s own `IntoMountTarget<ViaFn>`
arm, never exported — `src/app.rs:1474-1506`). `Listener::builder().handle()`
has no such adapter: it wants an actual `Handler`-shaped VALUE — anything
implementing `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err =
ProximaError>` (`proxima-primitives/src/pipe/handler.rs:97-105`) — which
`into_handle` then erases into the `PipeHandle` `.handle()` stores
(`proxima-primitives/src/pipe/handler.rs:113-118`). The one-line fix for a
bare handler fn like part 1's `hello` is the same attribute
[Foundations §7](./00-foundations.md) and [ergonomics §6](./01-ergonomics.md)
teach in full — `#[proxima::piped(send)]` turns a plain fn into a fieldless
struct that already qualifies:

```rust
use std::net::SocketAddr;

use bytes::Bytes;
use proxima::pipe::into_handle;
use proxima::{Listener, ListenerBuilderEntry, ProximaError, Request, Response};

#[proxima::piped(send)]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

async fn start(bind: SocketAddr) -> Result<(), ProximaError> {
    let server = Listener::builder()
        .bind(bind)
        .any()
        .handle(into_handle(hello))
        .serve()
        .await?;
    server.stop();
    Ok(())
}
```

## 3. `.any()`: stop picking

`.any()` (`src/listener/handle.rs:230-235`) accepts every protocol candidate
currently registered on this `App` — by default, that's h1 and h2
prior-knowledge (`src/app.rs:302-322`, `new_any_registry`, the function
`App::new` calls to populate its `AnyRegistry`; h1 registers unconditionally,
h2 prior-knowledge when the `http2` feature is compiled in). Each accepted
connection is classified from its own leading bytes and routed to whichever
candidate matches.

`examples/any_listener.rs` proves it directly with a real handler
(`SendPipe`-implemented by hand this time, not macro-generated — the same
`Handler` shape section 2 built, just spelled out — `:45-58`) and a real
listener (`:123-128`). Same imports as section 2's snippet, plus the two new
names this one needs — `Future` (the trait method's own return type) and
`SendPipe` (the trait `ConstantOk` implements directly):

```rust
use std::future::Future;

use proxima::SendPipe;

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

async fn start_any(any_bind: SocketAddr) -> Result<(), ProximaError> {
    let any_server = Listener::builder()
        .bind(any_bind)
        .any()
        .handle(into_handle(ConstantOk))
        .serve()
        .await?;
    any_server.stop();
    Ok(())
}
```

(Wrapped in a `bind: SocketAddr` parameter here instead of the real file's
own `free_loopback_addr()?` helper — everything else is unedited.) A plain h1
client (`std::net::TcpStream`) and a native h2 client (`H2ClientUpstream`)
both dial `any_bind` — same address, same handler — and both get routed
correctly. Running the real file produces exactly this:

```text
.any() classified a plain h1 client correctly on 127.0.0.1:59021
.any() classified a native h2 client correctly on the SAME port 127.0.0.1:59021
```

### How the sniff actually works (one paragraph, not a deep dive)

Each candidate protocol answers one question against the bytes seen so far:
"is this mine, not yet enough to tell, or definitely not mine?"
(`ProbeVerdict::{Match, NeedMore, No}`, `proxima-listen/src/any/probe.rs:94-119`).
Candidates are checked in priority order (default `100` for both h1 and h2,
`proxima-listen/src/any/probe.rs:249-257`; ties are fine), and a low-priority
match is held back as long as a higher-priority candidate could still win —
so nothing is misrouted while classification is still ambiguous
(`proxima-listen/src/any/classifier.rs:1-31`, the "priority-ordered-wait"
rule; two candidates tied at the SAME winning priority report
`ClassifyOutcome::AmbiguousMatch` rather than silently picking one). You
don't need to reimplement any of this — it's exactly what `.any()` already
does; the paragraph above is here so the word "classifier" doesn't feel like
magic later.

## 4. `.accept(name)` / `.accepts([...])`: narrow it back down

Sometimes you want the opposite of `.any()` — pin a port to exactly one wire.
`.accept(name)` (`src/listener/handle.rs:250-254`) is sugar over
`.accepts(&[name])` with one entry; `.accepts(&[...])`
(`src/listener/handle.rs:239-246`) restricts the SAME classifier to a named
subset instead of every registered candidate. This is a SEPARATE bind from
`.any()`'s — a full, side-by-side comparison of when to use which is part
3's closing section. For now, the real excerpt (`examples/any_listener.rs:148-154`,
reusing the same `ConstantOk` handler section 3 defined):

```rust
async fn start_pinned(pinned_bind: SocketAddr) -> Result<(), ProximaError> {
    let pinned_server = Listener::builder()
        .bind(pinned_bind)
        .accept("h2")
        .handle(into_handle(ConstantOk))
        .serve()
        .await?;
    pinned_server.stop();
    Ok(())
}
```

An h2 client dialing this address still works. An h1 client dialing the
*same* address gets nothing — no status line, connection just closes —
because `"h1"` was never in this listener's candidate set to begin with.
`examples/any_listener.rs` proves both directly:

```text
.accept("h2") still serves a real h2 client on its own port 127.0.0.1:59024
.accept("h2") never classifies an h1 client on 127.0.0.1:59024 — 0 bytes came back, no status line
```

## 5. The whole picture, side by side

| call | binds | accepts |
|---|---|---|
| `.any()` | one port | every registered candidate |
| `.accepts(&["h1", "h2"])` | one port | exactly the named subset |
| `.accept("h2")` | one port | exactly one candidate |

That's the entire vocabulary for part 2. Nothing here needed a config file,
a second port, or a deny list — those come in part 3, one at a time, as a
single toy listener grows into a production one.

## Where to go next

- [Part 3: growing it into production](./06-listener-production.md) —
  telemetry, a scanner deny-list with a DoS blacklist, request-level
  admission that actually sheds load, client-side resilience, and the
  same-port-vs-separate-port decision, all on the same shape you just
  learned.
- [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) — the
  deep dive on everything `Listener::builder()` does BEYOND `.any()`
  (`.tcp()`/`.udp()`/`.quic()`/`.tls()`/`.grpc()`/`.pgwire(query)`, why TLS
  composes as a decorator, the two places the builder honestly can't mirror
  `Client`). Not required for this on-ramp — read it if you want the full
  builder story.
- [Part 4: composing the sugar](./07-sugar-composition.md) — the three
  type-specific axis families (transport/security/protocol) and how they
  compose, once you're past `.any()`/`.accept()` and want to pick a wire
  on purpose.
- [Part 8: any protocol, any transport](./11-any-transport-agnostic.md) —
  `.any()` classifying a UDP-sourced connection through the SAME
  classifier this page taught, once a registered candidate asks for one.
  Best read after [part 6](./09-extend-your-own-protocol.md), which teaches
  the `AnyProtocol` trait this extends.
