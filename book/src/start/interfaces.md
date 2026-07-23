# dial, serve, run — the three faces

`hello` (the chapter before this one) already used one of proxima's three
user-facing interfaces without naming it: `App`. Before you go any further —
before the pipe algebra, before a single primitive — meet the other two, and
see why all three are the *same shape* wearing a different hat.

Every proxima program is built from three doors onto the one substrate:

| interface  | direction   | question it answers                          |
|------------|-------------|-----------------------------------------------|
| `Client`   | dial out    | "I need to talk to something out there."      |
| `Listener` | serve in    | "Something out there needs to talk to me."    |
| `App`      | compose+run | "Wire the pipes together and keep it running."|

`Client` and `Listener` are mirror images: both are fluent builders over the
*exact same* spec-accumulation trait, so a `Client` chain and a `Listener`
chain read like the same sentence with the verb reversed. `App` is what turns
a `Listener`'s spec into a bound socket and keeps it running until told to
stop — the thing `hello` used to answer real HTTP requests.

## One seam, two builders

Both `ClientBuilder` and `ListenerBuilder` accumulate into the same kind of
value underneath: a plain JSON map of key → setting. A method call like
`.http(url)` or `.tls()` does nothing but write one key into that map; the
config chapter later in this book shows the identical map coming from a
config file instead of a builder chain — fluent code and config are two
views of the same data, never two competing ways to say the same thing. The
base seam is two methods, `set`/`push` (`proxima-config/src/sugar/builder.rs:51–57`):

```rust
pub trait SpecBuilder: Sized {
    fn set(self, key: &str, value: impl Into<Value>) -> Self;
    fn push(self, key: &str, value: impl Into<Value>) -> Self;
}
```

That's all `SpecBuilder` is. The fluent methods on top (`.http()`, `.tcp()`,
`.tls()`, `.kafka()`, …) are **not** one blanket trait reaching across every
builder — each axis is its own TYPE-SPECIFIC trait, implemented once per
concrete builder: `ClientTransportExt`/`ClientProtocolExt`/`ClientSecurityExt`
for `ClientBuilder` (`src/client/transport.rs`, `src/client/protocol.rs`,
`src/client/security.rs`), `ListenerTransportExt`/`ListenerProtocolExt` for
`ListenerBuilder` (`src/listener/transport.rs`, `src/listener/protocol.rs`).
A prior version of this crate had a blanket pair here (`ProtocolSugar`/
`TransportSugar`, `impl<B: SpecBuilder> ProtocolSugar for B {}`) — retired,
precisely because "reaches every `SpecBuilder`, including foreign-crate ones
a caller never meant to touch" turned out to be too much reach
(`proxima-config/src/sugar/builder.rs`'s own module doc says so directly).
The client's own transport axis, for the shape (`src/client/transport.rs:14–35`):

```rust
pub trait ClientTransportExt: Sized {
    fn tcp(self) -> Self;
    fn udp(self) -> Self;
    fn quic(self) -> Self;
    fn proxy(self, url: impl Into<String>) -> Self;
}
```

`ClientBuilder` implements `ClientTransportExt`; `ListenerBuilder` implements
the SEPARATE `ListenerTransportExt` (same method names, different trait, own
impl). Every `.http(url)`/`.tls()`/`.quic()` method is on the page because
you imported the SPECIFIC trait that provides it for the SPECIFIC builder
you're calling it on — `use proxima::prelude::*;` brings every first-party
axis trait into scope at once, or import them individually. Nothing is
implicit magic, and — unlike the retired blanket design — a type can no
longer accidentally inherit an axis method that makes no sense for it (a
`ListenerBuilder` cannot reach the client-only `.proxy(url)` through
`ClientTransportExt`, because it never implements that trait).

HTTP/3 rides this same seam as a MODIFIER, not a fourth protocol key:
`.http(url).quic()` (client) / `.http(bind).quic()` (listener) is h3 — there
is no separate `.h3()` method. `.quic()` writes `transport: "quic"`; the
loader checks that key alongside `http` and dispatches to the native h3
upstream/listener instead of the ordinary h1/h2 path. See
`docs/tutorials/07-sugar-composition.md` for the full composition story
(transport × security × protocol, and the honest failure mode when an
invalid combination is requested).

## `Client` — dial out

The one-liner (`src/client/handle.rs:133–137`):

```rust
pub fn http(url: impl Into<String>) -> Result<Self, ProximaError> {
    let mut spec = serde_json::Map::new();
    spec.insert("http".to_string(), Value::String(url.into()));
    Self::from_value(Value::Object(spec))
}
```

and the fluent form, for when you also need a transport pick, a proxy, or
auth (`src/client/handle.rs:157–160`):

```rust
#[must_use]
pub fn builder() -> ClientBuilder {
    ClientBuilder::default()
}
```

`Client::builder().https(url).tls().build()` resolves lazily on the first
`.call(method, path).send()` — no socket opens until you actually dial. A
real call site, from the `proxy` example (`examples/proxy/main.rs:94`):

```rust
let client = Client::http(format!("http://{origin_bind}"))?;
```

That's the whole config for a reverse proxy's upstream: `client` above is
callable directly, so forwarding a request is just handing it to `client` and
returning what comes back — no field copying, no rewriting. (The next
chapter names exactly what makes `Client` callable like that — a `Pipe` — so
this chapter doesn't need to.) See [proxy](../applied/proxy.md) for the
complete, compiled program.

## `Listener` — serve in

`Listener` itself is defined in `proxima-listen`, a lower crate this one
depends on. Rust only lets you add an inherent `impl SomeType { .. }` block
from the crate that *defines* `SomeType` — the "orphan rule" — so this crate
cannot write a second `impl Listener { fn builder() }` next to the one
`proxima-listen` already has. The fluent `.builder()`/`.http(bind)` entry
points are added instead as a trait, `ListenerBuilderEntry`, blanket-
implemented for that foreign type — the mirror is a trait import, exactly
like `ProtocolSugar`/`TransportSugar` (`src/listener/handle.rs:21–61`):

```rust
pub trait ListenerBuilderEntry {
    #[must_use]
    fn builder() -> ListenerBuilder;

    #[must_use]
    fn http(bind: SocketAddr) -> ListenerBuilder;
}

impl ListenerBuilderEntry for Listener {
    fn builder() -> ListenerBuilder {
        ListenerBuilder::default()
    }

    fn http(bind: SocketAddr) -> ListenerBuilder {
        ListenerBuilder::default().bind(bind).http(bind.to_string())
    }
}
```

Line up the two one-liners and the mirror is exact — dial vs. bind, url vs.
address:

```text
Client::http(url)   -> ClientBuilder   -> .build()
Listener::http(bind) -> ListenerBuilder -> .handle(pipe).serve()
```

The one honest difference: a client's url already names *what* to dial. A
listener has nothing to dispatch to until you give it one — `.handle(pipe)`
is required before `.serve()`, the listener-side sibling of the client's
upstream url. A full, compiled example, `.h2()` through
`Listener::builder()` (`examples/h2_native_server.rs:73–78`):

```rust
let server = Listener::builder()
    .bind(bind)
    .h2()
    .handle(into_handle(ConstantOk))
    .serve()
    .await?;
```

The complete file, including the h2 client round trip that proves it:

```rust
{{#include ../../../examples/h2_native_server.rs}}
```

`.serve()` composes `App::new()` + `App::mount` + `App::serve` underneath —
the exact idiom the next section teaches, automated behind the builder.
`.h2()` and `.pgwire(query)` are listener-only axes with no client-side twin
at all (a listener speaks h2 with nothing to dial, or terminates a typed SQL
wire a client never carries). `.tls(TlsConfig)` DOES have a client-side twin
(`ClientSecurityExt::tls()`) — but a different shape, not the same method
reused: the client's is bare (zero-arg, ALPN negotiation does the actual
work), the listener's takes real cert material (`proxima_tls::TlsConfig`) —
a listener terminates TLS, a client only *picks* it for a url it already
has. `docs/tutorials/02-listener-builder.md` is the deep dive on all three
axes, on why TLS composes as a decorator rather than a spec field, and on
the two places the mirror is honestly asymmetric.

A listener can also skip picking a protocol entirely: `.any()` binds one
socket and classifies each connection's own leading bytes against every
registered candidate (h1, h2 prior-knowledge by default), routing each
connection to whichever one matches — `.accept(name)`/`.accepts([...])`
narrow that same classifier back down to one wire or a named subset when
you want that instead. `docs/tutorials/05-listener-universal.md` teaches
`.any()` from zero; `docs/tutorials/06-listener-production.md` grows it
into a production shape — a scanner deny-list backed by a real DoS
blacklist (`.deny(name, literal)` + `.blacklist(config)`), request-level
admission that renders a real 503 on the wire, and the same-port-vs-
separate-port decision this chapter's `Listener` section doesn't otherwise
answer.

## `App` — compose + run

`hello`'s entire wiring, again (`examples/hello/main.rs:48,49,53,58`):

```rust
let app = App::new()?;
app.mount("/", hello)?;
let server = app.serve(RunConfig::http(bind)).await?;
server.run_until_signal().await;
```

`App::new()` adopts whichever runtime `#[proxima::main]` already booted (or
builds a default prime runtime if none was booted). `App::mount` accepts four
different shapes at the call site — a bare `async fn`, a handler-shaped
pipe, an already-built `PipeHandle`, or a registered pipe's name — dispatched
through one trait, `IntoMountTarget`, so `.mount(path, target)` never forces
you to wrap anything by hand (`src/app.rs:1231`):

```rust
pub trait IntoMountTarget<Via> {
    fn into_mount_target(self) -> MountTarget;
}
```

`app.serve(RunConfig::http(bind))` is the terminal that plays the same role
`Listener::builder()...serve()` plays one level down — in fact
`ListenerBuilder::serve` calls `App::new()` + `.mount` + `.serve` itself, so
`App` is the composition every listener chain bottoms out in. Two knobs let
you swap what actually executes the pipes and accepts sockets without
touching a single handler: `App::with_runtime` (an `Arc<dyn Runtime>`) and
`App::with_acceptor_factory` (an `Arc<dyn AcceptorFactory>`) — see
`docs/tutorials/03-native-runtime.md` for the full ambient-runtime story,
including the one rule that trips up multi-`App` programs: booting one
runtime for `main` silently becomes the runtime *every* `App` you build
inside it adopts, unless each app opts out explicitly.

## What's next

- [the pipe algebra](../algebra/index.md) — the `Pipe` trait every handler
  behind `Client`, `Listener`, and `App` ultimately is.
- [proxy](../applied/proxy.md) / [gateway](../applied/gateway.md) — compiled
  programs combining all three faces.
- `docs/tutorials/02-listener-builder.md` — the deep dive on `Listener`'s
  builder: `resolve_listen_protocol`, the TLS decorator, and the two honest
  asymmetries against `Client`.
- `docs/tutorials/03-native-runtime.md` — the deep dive on `App`'s runtime
  seam: the `Runtime` trait, `http1` vs. `http1-native`, and the
  ambient-runtime adoption rule.
- `docs/tutorials/04-listener-hello.md` onward — a standalone, faster
  on-ramp straight to a running `Listener`, for a reader who wants to skip
  straight to `.any()`/`.accept()`/`.deny()`/`.blacklist()` without reading
  Foundations first. It continues into `docs/tutorials/07-sugar-composition.md`
  (the transport/security/protocol axes, composed), `08-protocol-fleet.md`
  (memcached/DNS/Kafka/MQTT/AMQP, client and listener), and
  `09-extend-your-own-protocol.md` (plugging in a protocol proxima doesn't
  ship, with zero edits to this crate).
- [add your own protocol](../extend/protocol.md) / [the protocol fleet](../protocols/fleet.md)
  — this book's own chapters on the same two topics.
