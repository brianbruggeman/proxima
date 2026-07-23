# Listener on-ramp, part 2: the universal listener

**Prerequisites:** [part 1: hello](./04-listener-hello.md) — you should be
comfortable with "handler = `async fn`, mount it, serve it."

**You will:** bind ONE port that answers both HTTP/1.1 and HTTP/2 clients,
without ever telling it which one to expect — then narrow it back down to
exactly one wire, in one line, when you want that instead.

**New concepts (in order):** `Listener::builder()` · `.any()` · `.accept(name)`
/ `.accepts([...])`.

Every code block below is real and compiles — the full, runnable file is
`examples/any_listener.rs` (`cargo run --example any_listener --features
http1-native`; verified tokio-free with `cargo tree --features http1-native
-e normal -i tokio`, empty result).

## 1. The problem this solves

Part 1's `RunConfig::http(bind)` already speaks both HTTP/1.1 and HTTP/2 —
but only because both ride the *same* negotiation your browser already
knows: TLS's ALPN extension picks the version *after* the connection opens.
Plaintext h2 ("h2c", prior-knowledge, no TLS) has no such negotiation — a
plaintext listener has to look at the bytes themselves to tell an h1
request line (`GET / HTTP/1.1\r\n...`) apart from h2's fixed 24-byte
connection preface. That's what `.any()` does: it looks.

## 2. Meet `Listener::builder()`

Part 1 used `App` directly. `Listener::builder()` is a second front door
onto the exact same serve machinery — a fluent chain that ends in
`.serve()`, which itself calls `App::new()` + `app.mount` + `app.serve`
under the hood (`src/listener/handle.rs:433-439`'s own doc says so
directly — no second serve loop was invented for this). You reach for it
when you want the `.any()`/`.accept()` family below; plain `App` is still
correct for everything part 1 taught.

```rust
use proxima::{Listener, ListenerBuilderEntry, into_handle};

let server = Listener::builder()
    .bind(bind)
    .any()
    .handle(into_handle(my_handler))
    .serve()
    .await?;
```

`.handle(pipe)` is the one thing every `Listener::builder()` chain needs
that `App::mount` also needed — where to dispatch. `into_handle` wraps
your handler into the uniform shape `.handle()` expects
(`src/listener/handle.rs:177-180`).

## 3. `.any()`: stop picking

`.any()` (`src/listener/handle.rs:271-276`) accepts every protocol
candidate currently registered on this `App` — by default, that's h1 and
h2 prior-knowledge (`src/app.rs:275-282`, `App::new`'s own registration).
Each accepted connection is classified from its own leading bytes and
routed to whichever candidate matches:

```rust
let server = Listener::builder()
    .bind(bind)
    .any()
    .handle(into_handle(my_handler))
    .serve()
    .await?;
```

One bind. One handler. Both a plain h1 client and a native h2 client
connecting to the *same* address get routed correctly — proven directly in
`examples/any_listener.rs`:

```rust
let any_server = Listener::builder()
    .bind(any_bind)
    .any()
    .handle(into_handle(ConstantOk))
    .serve()
    .await?;

// a plain h1 client (std::net::TcpStream) gets a real HTTP/1.1 response
// a native h2 client (H2ClientUpstream) gets a real HTTP/2 response
// — same address, same handler, both work.
```

Running that example produces exactly this:

```
.any() classified a plain h1 client correctly on 127.0.0.1:53524
.any() classified a native h2 client correctly on the SAME port 127.0.0.1:53524
```

### How the sniff actually works (one paragraph, not a deep dive)

Each candidate protocol answers one question against the bytes seen so
far: "is this mine, not yet enough to tell, or definitely not mine?"
(`ProbeVerdict::{Match, NeedMore, No}`,
`proxima-listen/src/any/probe.rs`). Candidates are checked in priority
order (default `100` for both h1 and h2; ties are fine), and a low-priority
match is held back as long as a higher-priority candidate could still win —
so nothing is misrouted while classification is still ambiguous
(`proxima-listen/src/any/classifier.rs`'s "priority-ordered-wait" rule).
You don't need to reimplement any of this — it's exactly what `.any()`
already does; the paragraph above is here so the word "classifier" doesn't
feel like magic later.

## 4. `.accept(name)` / `.accepts([...])`: narrow it back down

Sometimes you want the opposite of `.any()` — pin a port to exactly one
wire. Two one-liners:

```rust
// exactly one candidate
.accept("h2")

// a named subset of candidates
.accepts(&["h1", "h2"])
```

`.accept(name)` is sugar over `.accepts(&[name])` with one entry
(`src/listener/handle.rs:289-295`); `.accepts` restricts the SAME
classifier to a named subset instead of every registered candidate
(`:278-287`). This is a SEPARATE bind from `.any()`'s — a full,
side-by-side comparison of when to use which is part 3's closing section.
For now, the one-liner:

```rust
let pinned_server = Listener::builder()
    .bind(pinned_bind)
    .accept("h2")
    .handle(into_handle(ConstantOk))
    .serve()
    .await?;
```

An h2 client dialing this address still works. An h1 client dialing the
*same* address gets nothing — no status line, connection just closes —
because `"h1"` was never in this listener's candidate set to begin with.
`examples/any_listener.rs` proves both directly:

```
.accept("h2") still serves a real h2 client on its own port 127.0.0.1:55567
.accept("h2") never classifies an h1 client on 127.0.0.1:55567 — 0 bytes came back, no status line
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
