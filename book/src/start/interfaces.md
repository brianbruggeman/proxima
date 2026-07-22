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
methods that write into that map are not hand-rolled per builder; they come
from two axis traits, blanket-implemented over anything that can `set`/`push`
a key (`proxima-config/src/sugar/builder.rs:47–57,62–83,89–122`):

```rust
pub trait SpecBuilder: Sized {
    fn set(self, key: &str, value: impl Into<Value>) -> Self;
    fn push(self, key: &str, value: impl Into<Value>) -> Self;
}

pub trait ProtocolSugar: SpecBuilder {
    fn http(self, url: impl Into<String>) -> Self { self.set("http", url.into()) }
    fn https(self, url: impl Into<String>) -> Self { self.set("http", url.into()) }
    fn grpc(self, url: impl Into<String>) -> Self { self.set("grpc", url.into()) }
}

pub trait TransportSugar: SpecBuilder {
    fn auto(self) -> Self { self.set("transport", "auto") }
    fn tcp(self) -> Self { self.set("transport", "tcp") }
    fn tls(self) -> Self { self.set("transport", "tls") }
    fn h3(self) -> Self { self.set("transport", "h3") }
    fn proxy(self, url: impl Into<String>) -> Self { self.set("proxy", url.into()) }
}
```

`ClientBuilder` and `ListenerBuilder` each implement only `set`/`push`; every
`.http(url)`/`.tls()`/`.h3()` method on both types is the identical blanket
method, imported into scope with one `use
proxima::{ProtocolSugar, TransportSugar};`. Nothing is implicit — the method
is on the page because you imported the trait.

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
`.tls(config)`, `.h2()`, and `.pgwire(query)` are listener-only axes with no
client-side twin (a listener terminates real cert material or speaks a typed
SQL wire; a client only ever *picks* a transport for a url it already has) —
`docs/tutorials/02-listener-builder.md` is the deep dive on those three axes,
on why TLS composes as a decorator rather than a spec field, and on the two
places the mirror is honestly asymmetric.

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
  Foundations first.
