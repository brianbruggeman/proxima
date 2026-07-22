# Foundations, part 3: the Listener builder, mirrored from the Client

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md) §13 (a pipe that answers web requests, `into_handle`, `App::mount`, `app.serve(RunConfig)`) and [Foundations, part 2](./01-ergonomics.md) §8 (`App::mount`'s four accepted shapes). You should also already be comfortable pointing a `proxima::Client` at an upstream — `Client::http(url)` or `Client::builder().http(url).tls().build()` — the way [Build an API gateway](./build-an-api-gateway.md) uses it (`Client::http(format!("http://{origin_api_bind}"))?`, `examples/gateway/main.rs:77`). This document does not re-teach `Client` from scratch, but §1 below gives a short, cited recap of exactly the parts the Listener side mirrors — there is no dedicated "Client builder" tutorial elsewhere in this tree yet, so this section is that recap.

**You will learn:** that `Listener` — a serve-side value that already existed in `proxima-listen` — grows a fluent builder, `Listener::builder()` / `Listener::http(bind)`, built from the *exact same* spec-accumulation trait `Client::builder()` uses; why that had to be a trait blanket-impl'd onto a foreign type rather than a second `Listener` struct; how a `.tcp()`/`.tls()`/`.h3()`/`.grpc()` choice resolves down to one concrete `ListenProtocol`, mirroring how the client's `load()` resolves a spec to a `PipeHandle`; why TLS is not a field on any spec but a decorator composed at `.serve()` time; and the two places — and only two — where a listener's builder honestly cannot mirror the client's, because a listener's inputs are different in kind.

**New concepts (in order):** `SpecBuilder` (the seam `ProtocolSugar`/`TransportSugar` are blanket-built on) · `ListenerBuilderEntry` (`Listener::builder()`/`Listener::http(bind)`) · `resolve_listen_protocol` · `TlsListenProtocol` (the TLS decorator) · `ListenerSpec::protocol` (the escape hatch) · the two asymmetric axes.

Every code block below is either copied verbatim from a real file in this repository (cited by `file:line`, checked against `git rev-parse HEAD` = `0ac7a565` on `main`) or a unit test that `cargo nextest run` in this repo actually passes — re-run yourself with `CARGO_TARGET_DIR=/tmp/cargo_target cargo nextest run -p proxima listener::handle::tests --features http1,http2,http3,tls,pgwire`. Nothing here is invented. Where I found the *source's own* doc comments citing a stale line number, I say so rather than silently copying the error forward — see the "heads-up" callouts. (This document was originally checked against `9d3b3c4f`, one commit before `.h2()`/`.pgwire(query)` and the `.tls()`+`.pgwire()` guard landed in `0ac7a565`; every citation and code block below has been re-verified against `0ac7a565`, and §4/§5/§9 now cover the two new axes.)

## Contents

1. What you already know: `Client::builder()`, recapped
2. Meet the mirror: `Listener::builder()` / `Listener::http(bind)`
3. Why a trait, not a second `Listener` type
4. One coin, two faces: the side-by-side axis table
5. From spec to a concrete protocol: `resolve_listen_protocol`
6. Why the listener names wire versions and the client doesn't
7. TLS as a composed layer, not a spec field
8. The general escape hatch: `ListenerSpec::protocol`
9. `.serve()`: what it actually composes
10. The two honest asymmetries
11. A full walkthrough, compiled and run
12. Where to go next

## 1. What you already know: `Client::builder()`, recapped

A `proxima::Client` dials an upstream. The one-liner is `Client::http(url)`, and the fluent form is `Client::builder()...build()` (`src/client/handle.rs:133–137,158–160`):

```rust
pub fn http(url: impl Into<String>) -> Result<Self, ProximaError> {
    let mut spec = serde_json::Map::new();
    spec.insert("http".to_string(), Value::String(url.into()));
    Self::from_value(Value::Object(spec))
}

#[must_use]
pub fn builder() -> ClientBuilder {
    ClientBuilder::default()
}
```

`ClientBuilder` is not a hand-rolled struct with a pile of ad-hoc setters. It wraps one `serde_json::Map<String, Value>` (`src/client/handle.rs:444–459`, the `#[derive(Default)] pub struct ClientBuilder` fields) — the same JSON `Value` a `[[pipe]]` TOML table deserializes to — and every fluent method just writes a key into that map. `.http(url)` writes the `http` key, `.tls()` writes `transport: "tls"`, and so on. Two axis *traits*, blanket-implemented over anything that can accumulate such a map, supply those methods (`proxima-config/src/sugar/builder.rs:47–122`):

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
impl<B: SpecBuilder> ProtocolSugar for B {}

pub trait TransportSugar: SpecBuilder {
    fn auto(self) -> Self { self.set("transport", "auto") }
    fn tcp(self) -> Self { self.set("transport", "tcp") }
    fn tls(self) -> Self { self.set("transport", "tls") }
    fn h3(self) -> Self { self.set("transport", "h3") }
    fn proxy(self, url: impl Into<String>) -> Self { self.set("proxy", url.into()) }
}
impl<B: SpecBuilder> TransportSugar for B {}
```

`ClientBuilder` implements only `set`/`push` (`src/client/handle.rs:648–663`); `ProtocolSugar`/`TransportSugar` fall out for free from the blanket impls. The methods are on the page only because you `use proxima_config::sugar::{ProtocolSugar, TransportSugar}` (or the umbrella re-export, `use proxima::{ProtocolSugar, TransportSugar}`) — nothing is implicit magic. This is why a fluent chain and a literal config `Value` are provably the same spec, checked directly in the client's own test suite (`verbs_map_to_methods_and_builder_lowers_axes`, `src/client/handle.rs:1012`, parity block at `1045–1057`, abbreviated):

```rust
let fluent = Client::builder().http("http://api.example.com").tls().proxy("http://127.0.0.1:8080").build()?;
let config = Client::from_value(json!({
    "http": "http://api.example.com", "transport": "tls", "proxy": "http://127.0.0.1:8080",
}))?;
assert_eq!(fluent.inner.spec, config.inner.spec);
```

That's the whole shape you already know: one accumulated JSON spec, two axis traits giving it fluent methods, and a terminal (`.build()`) that freezes it. Everything below is the identical shape, reused — not reinvented — for the serve side.

## 2. Meet the mirror: `Listener::builder()` / `Listener::http(bind)`

`proxima-listen` already had a real serve-side value called `Listener` (`proxima-listen/src/handle.rs:136–143`) — bind address, protocol name, shutdown policy, spec, and a dispatch `PipeHandle`, produced by `ListenerSpec::attach(dispatch)` and driven by `Listener::run_with_runtime`. What landed on `main` (commit `49940cf5`, `feat(listen): listener builder mirroring the client + compositional tls`) is a fluent front door onto that same value, in `src/listener/handle.rs`:

```rust
pub trait ListenerBuilderEntry {
    fn builder() -> ListenerBuilder;
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
(`src/listener/handle.rs:36–61`)

Read that side by side with §1's `Client::http`/`Client::builder`. Same two entry points, same names, same shape: a one-liner that pre-fills the common case, and a bare `builder()` for everything else. `Listener::http(bind)` even reuses `ProtocolSugar::http` internally — it calls `.http(bind.to_string())`, the identical method `ClientBuilder` gets from the same blanket impl, just handed a stringified bind address instead of a dial URL (more on why in §6).

Bring both entry points into scope together — `use proxima::{Listener, ListenerBuilderEntry};` — and you can write the mirror of the manual `App::new()?; app.mount(...)?; app.serve(RunConfig::http(bind)).await?;` shape from `examples/hello/main.rs:42–47` in one chain. This is copied verbatim from the module's own doc comment (`src/listener/mod.rs:46–62`) — note it lives inside a rustdoc ```` ```ignore ```` fence, so it is illustrative, not compiled by `cargo test --doc`:

```rust
use proxima::{Listener, ListenerBuilderEntry, TransportSugar, into_handle};

let server = Listener::builder()
    .bind("127.0.0.1:8080".parse()?)
    .tcp()
    .handle(into_handle(my_pipe))
    .serve()
    .await?;
server.run_until_signal().await;

// mirrors `Client::http(url)`:
let server = Listener::http("127.0.0.1:8080".parse()?)
    .handle(into_handle(my_pipe))
    .serve()
    .await?;
```

`my_pipe` is a placeholder for whatever `Handler` you already mount today. I did not stop at "illustrative" — I wrote a concrete, compiling, actually-run version of both forms (plus the manual `App`/`RunConfig` baseline) against this exact commit; that full listing, and how I verified it, is §11.

## 3. Why a trait, not a second `Listener` type

The obvious question: why is `Listener::builder()` a trait method reached through `ListenerBuilderEntry`, rather than a plain inherent `impl Listener { fn builder() -> ... }`? Rust's orphan rule is the answer, and the source states it directly (`src/listener/handle.rs:21–35`):

> `Listener` is defined in `proxima-listen`, a crate this one depends on but that cannot depend back on `App`/`Server`; Rust's orphan rule forbids an inherent `impl Listener { fn builder() }` from this crate, so the entry points are a local trait blanket-impl'd for the foreign type instead of a second `Listener` type living here.

`ListenerBuilder::serve()` (§9) has to construct an `App`, mount a router, and return a `Server` — all types that live in the umbrella `proxima` crate, several layers above `proxima-listen`. `proxima-listen` cannot know about `App`; only the umbrella crate can wire the two together. But the umbrella crate doesn't own the `Listener` type it wants to extend. The escape hatch Rust actually gives you for "add a method to a type I don't own" is a trait, defined in the crate doing the extending, implemented once for the foreign type. That's exactly the same idiom `ProtocolSugar`/`TransportSugar` already used in §1 — `impl<B: SpecBuilder> ProtocolSugar for B {}` extends *every* `SpecBuilder`, including ones defined in other crates, the same way `impl ListenerBuilderEntry for Listener` extends one specific foreign type. `src/listener/mod.rs`'s own module doc makes the same point in one line: "Rust's orphan rule forbids this crate from adding an inherent method to a type it doesn't own." No new `Listener` struct was minted to work around this — that would have been the RISC violation (principle 1: don't add a type next to an existing one when the existing one still works).

## 4. One coin, two faces: the side-by-side axis table

`ListenerBuilder` (`src/listener/handle.rs:84–103`) accumulates its own `serde_json::Map<String, Value>`, exactly like `ClientBuilder` does — plus two fields with no client-side twin, `tls` (`#[cfg(feature = "tls")]`) and `pgwire_query` (`#[cfg(feature = "pgwire")]`), both accumulated separately from `spec` because neither a `TlsConfig` nor a `PgPipeHandle` fits a `serde_json::Value` spec key (§7 covers `tls`; §5/§9 cover `pgwire_query`) — and implements the identical `SpecBuilder` seam (`src/listener/handle.rs:506–521`):

```rust
impl proxima_config::sugar::SpecBuilder for ListenerBuilder {
    fn set(self, key: &str, value: impl Into<Value>) -> Self {
        self.spec(key, value.into())
    }

    fn push(mut self, key: &str, value: impl Into<Value>) -> Self {
        let entry = self.spec.entry(key.to_string()).or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(array) = entry {
            array.push(value.into());
        }
        self
    }
}
```

That means `use proxima::{ProtocolSugar, TransportSugar}` lights up `.http()`/`.https()`/`.tcp()`/`.tls()`/`.h3()`/`.auto()` on `ListenerBuilder` for free — the *same* blanket impls from §1, not a forked listener-specific DSL. A unit test proves it directly, by making the identical `ProtocolSugar`/`TransportSugar` calls a `ClientBuilder` chain would and checking the spec keys land identically (`listener_builder_mirrors_client_builder_axis_keys`, `src/listener/handle.rs:616–634`):

```rust
let fluent = ListenerBuilder::default()
    .https("127.0.0.1:8080")
    .spec("transport", json!("tls"));
assert_eq!(fluent.spec.get("http").and_then(Value::as_str), Some("127.0.0.1:8080"));
assert_eq!(fluent.spec.get("transport").and_then(Value::as_str), Some("tls"));
```

`src/listener/mod.rs`'s module doc lays out the full picture as a table (`src/listener/mod.rs:30–39`) — since this document's first pass, `.h2()` and `.pgwire(query)` landed too, and the source's own table now already carries both rows, plus the `.tls()` correction this document's first pass had to call out separately. The table below is that same table, with the two new axes:

| axis | `ClientBuilder` | `ListenerBuilder` |
|---|---|---|
| `.tcp()` / `.auto()` | real (`TransportSugar`) | real — resolves to the h1+h2 ALPN combiner (`"http"`) |
| `.tls()` | real, zero-arg (`TransportSugar`) — the dial url comes from a separately-chained `.http(url)` | **shadowed**: inherent `.tls(TlsConfig)` — real cert material required; resolves to the same `"http"` combiner |
| `.h3()` | real (`TransportSugar`) | real — resolves to `"h3-native"`, self-registered |
| `.proxy(url)` | real | no listener meaning — `.serve()` hard-errors if present |
| `.http(url)` / `.https(url)` | real (dials the url) | real — carries the **bind** address as a string, read back by `bind_from_spec` when `.bind(addr)` wasn't called directly |
| `.grpc(url)` / `.grpc()` | real, url-carrying | **shadowed**: inherent url-less `.grpc()` — resolves to `"h2"` (gRPC rides h2), self-registered like `.h3()` |
| *(no client twin)* | — | inherent `.h2()` — the other name for the same shared `"h2"` protocol `.grpc()` resolves to; h2 has no url-carrying client-side twin the way `.tls()`/`.h3()` do |
| *(no client twin)* | — | inherent `.pgwire(query)` (feature `pgwire`) — carries a typed SQL query engine (`proxima_pgwire::PgPipeHandle`) directly, not a marker key; self-registers a *fresh* `PgWireListenProtocol` on every `.serve()` rather than sharing one instance (§5, §9) |

> **Note (previously a heads-up):** the first pass of this document, checked against `9d3b3c4f`, found the source table's `.tls()` row for the client column read "real, url-carrying (`TransportSugar`)" — wrong, since `TransportSugar::tls` (§1's quote, `proxima-config/src/sugar/builder.rs:104–106`) is `fn tls(self) -> Self { self.set("transport", "tls") }`, zero arguments. That source doc comment (`src/listener/mod.rs:33`) has been corrected directly as part of landing this document, alongside the `.h2()`/`.pgwire(query)` rows above; the table here now matches the source exactly.

Two of the original rows say "shadowed" — Rust lets an inherent method on a type hide a trait method of the same name reached via blanket impl, so `.tls(cfg)`/`.grpc()` on `ListenerBuilder` are real, `#[cfg]`-gated inherent methods (`src/listener/handle.rs:144–147` for `.grpc()`, `:207–212` for `.tls()`) that take priority over the same-named `TransportSugar`/`ProtocolSugar` trait methods. `.h2()`/`.pgwire(query)` don't shadow anything — there is no blanket `TransportSugar::h2()` or `ProtocolSugar::pgwire()` to shadow in the first place, so they are plain inherent methods with no trait-method collision to resolve. §10 explains why `.tls()`/`.grpc()`, specifically, had to diverge from the client.

> **Drift note (added after `.any()`/`.deny()`/`.blacklist()` landed, `86c9302f`):** this document's axis table above predates that landing and does not list `.any()`/`.accept(name)`/`.accepts([...])`/`.any_handler(name, handler)`/`.any_on_reject(hook)`/`.deny(name, literal)`/`.denies([...])`/`.blacklist(config)` (all `#[cfg(any(feature = "http1", feature = "http1-native"))]`, `src/listener/handle.rs:259–398`) — a real, silent gap in this table, not a redesign of anything it already says. The listener on-ramp series (`docs/tutorials/04-listener-hello.md` onward) is the maintained teaching surface for that whole axis; [part 2](./05-listener-universal.md) covers `.any()`/`.accept()`/`.accepts()`, [part 3](./06-listener-production.md) covers `.deny()`/`.blacklist()` and everything built on top of them.

## 5. From spec to a concrete protocol: `resolve_listen_protocol`

One term this section needs before its code makes sense: a `ListenProtocol` (`proxima-listen`) is the listen-side trait each wire implementation — `HttpListenProtocol` (h1+h2), `H2ListenProtocol`, `H3NativeListenProtocol` — implements; a `ListenRegistry` looks these up by name. (If you've read [Build a plugin](./build-a-plugin.md), this is structurally the listen-side twin of that tutorial's `PipeFactory`/registry pattern — not a prerequisite for this document, just a pointer if the shape looks familiar.)

On the client side, `load()` reads the accumulated spec `Value` and dispatches on which key is present — `value.get("http")` picks the `"http"` (or `"http-tokio"`) factory, `value.get("grpc")` picks `"grpc"` (`src/load.rs:488,499`):

```rust
let (handle, kv_backend): (PipeHandle, Option<Arc<dyn KvHandle>>) =
    if let Some(http) = value.get("http") {
        let canonical = canonical_http(http, value)?;
        let key = match value.get("wire").and_then(Value::as_str) {
            Some("tokio") if context.registry.get("http-tokio").is_ok() => "http-tokio",
            _ => "http",
        };
        let factory = context.registry.get(key)?;
        (factory.build(&canonical, None).await?, None)
    } else if let Some(grpc) = value.get("grpc") {
        let canonical = canonical_http(grpc, value)?;
        let factory = context.registry.get("grpc")?;
        (factory.build(&canonical, None).await?, None)
    } else if let Some(synth) = value.get("synth") {
        // ...
```

> **Note (previously a heads-up):** the first pass of this document found `src/listener/handle.rs`'s own doc comment above `resolve_listen_protocol` citing `src/load.rs:455`, when the real lines are `:488` (the `http` arm) and `:499` (the `grpc` arm). That doc comment (`src/listener/handle.rs:344`) has been corrected directly as part of landing this document — the function and the mirroring claim were always correct, only the line number the source's own comment named had drifted.

`resolve_listen_protocol` is the listen-side twin of that same dispatch (`src/listener/handle.rs:360–378`) — since this document's first pass, it grew two more arms, `.pgwire(query)` and `.h2()`:

```rust
fn resolve_listen_protocol(
    spec: &serde_json::Map<String, Value>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    if spec.contains_key("pgwire_axis") {
        return Ok(("pgwire".to_string(), None));
    }
    if spec.contains_key("grpc") || spec.contains_key("h2") {
        return h2_listen_protocol();
    }
    if spec.get("transport").and_then(Value::as_str) == Some("h3") {
        return h3_native_listen_protocol();
    }
    // default / `.tcp()` / `.auto()`: the ALPN h1+h2 combiner. `.tls()`
    // composes as a decorator OVER whatever this resolves — see
    // `compose_tls` — so it never changes what's resolved here.
    Ok(("http".to_string(), None))
}
```

`.pgwire(query)` is checked **first**: it carries a typed query engine no other axis combination can produce, so it always wins, and it never returns an extra protocol here — `.serve()` (§9) registers its own fresh `PgWireListenProtocol` directly, since resolving needs the `query` handle this spec-only function never sees (`ListenerBuilder` keeps `pgwire_query` off the `spec` map entirely, §4). `.grpc()`/`.h2()` share one branch and resolve to the identical `"h2"` protocol — they're two names for the same wire, kept separate because `.grpc()` mirrors the client's `ProtocolSugar::grpc` naming while `.h2()` names the transport directly, and gRPC rides h2 either way.

Six real unit tests exercise this directly and pass on `main` today (`src/listener/handle.rs:548–614`, verified by `cargo nextest run` output reproduced in the report):

- `resolve_listen_protocol_defaults_to_http_and_opts_into_grpc_via_the_same_key_load_reads` — `.tcp()` and bare `transport: "tls"` both resolve to `"http"`, with no self-registered protocol (`extra.is_none()`), because `"http"` is already in `App::new()`'s default registry.
- `grpc_axis_resolves_to_h2_and_self_registers` (needs `http2`) — `.grpc()` resolves to `"h2"` and *does* carry a protocol to register, because h2-as-a-listen-protocol is not in `App::new()`'s default set.
- `h3_axis_resolves_to_h3_native_and_self_registers` (needs `http3`) — same shape for `.h3()` → `"h3-native"`.
- `h2_axis_resolves_to_the_same_shared_h2_protocol_as_grpc` (needs `http2`) — `.h2()` resolves to the identical `"h2"` name and carried `Arc` as `.grpc()`.
- `pgwire_axis_resolves_to_pgwire_and_carries_nothing_here` (needs `pgwire`) — the `pgwire_axis` marker resolves to `"pgwire"` with no carried protocol (the fresh `PgWireListenProtocol` is `.serve()`'s job, not this function's).
- `pgwire_axis_takes_priority_over_grpc_and_h3` (needs `pgwire`) — `.pgwire(query)` combined with `.grpc()` and `.h3()` still resolves to `"pgwire"`, proving the priority order above.

## 6. Why the listener names wire versions and the client doesn't

Look again at §5's two dispatch functions. The client's picks between exactly two registry keys (`"http"`/`"http-tokio"`, `"grpc"`) — no branch on `h1` vs `h2` vs `h3`. The listener's picks between three (`"http"`, `"h2"`, `"h3-native"`) *plus* threads a `transport` value the client-side dispatch never even reads. This is not an oversight; it's a structural difference between dialing and binding.

A client dials one TCP (or QUIC) connection to one upstream. Whether that connection ends up speaking HTTP/1.1 or HTTP/2 over it is negotiated **after** the connection is open, via TLS's ALPN extension — the `Transport` enum's own doc comment says exactly this (`src/client/handle.rs:410–428`): `Tls` is "TLS over TCP (h1/h2 by ALPN)". One factory (`"http"`), one socket, one `PipeHandle` — the wire version is a runtime negotiation result inside that one factory, not a different registry entry.

A listener has no negotiation to defer to, because *it* is the side that decides what the socket looks like before any client shows up. `"http"` (h1+h2 combiner), `"h2"`, and `"h3-native"` are three genuinely different `ListenProtocol` implementations — `HttpListenProtocol` (`src/listeners/mod.rs:54`), `H2ListenProtocol` (`:58`), `H3NativeListenProtocol` (`:65`) — each owning its **own accept loop and its own socket type**: `"http"`/`"h2"` bind a TCP listener and `accept()` connections; `"h3-native"` binds a **UDP** socket and drives QUIC datagrams, not a TCP accept loop at all. There is no ALPN negotiation that could turn a bound UDP-QUIC listener into a TCP one after the fact — the choice has to be made before `.serve()` binds anything, which is exactly why it's a compile-time-selected registry key rather than a runtime ALPN outcome. `resolve_listen_protocol` picking a **name** (used both as the registry lookup key and, for h2/h3, as the concrete `Arc<dyn ListenProtocol>` self-registered onto the fresh `App`) is that decision, made once, before `bind()` is ever called.

## 7. TLS as a composed layer, not a spec field

This is the part of the mirror that is *not* symmetric with the client at all, and it used to be, which is instructive. Before commit `49940cf5`, `ListenerSpec` (and `Listener`) carried TLS as a plain optional field, set by a `with_tls` setter (`proxima-listen/src/handle.rs`, as of `49940cf5^`):

```rust
#[derive(Clone)]
pub struct ListenerSpec {
    pub bind: SocketAddr,
    pub protocol_name: String,
    pub shutdown: ShutdownPolicy,
    pub spec: Value,
    #[cfg(feature = "tls")]
    pub tls: Option<proxima_tls::TlsConfig>,
}

impl ListenerSpec {
    /// Terminate TLS at this listener. Available under the `tls`
    /// feature; the listener serializes the config into the spec
    /// JSON so the HTTP protocol picks it up and wraps accepted
    /// sockets with a `TlsAcceptor` before they reach hyper.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn with_tls(mut self, tls: proxima_tls::TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }
    // ...
}
```

That field is gone as of `49940cf5`, replaced by a decorator. The reason, in the source's own words — split today across the `ListenerSpec::protocol` field's own doc comment (`proxima-listen/src/handle.rs:57–76`) and a shorter restatement inside `run_with_runtime` (`:175–182`):

> A protocol resolved at construction time instead of by name through the `ListenRegistry` at serve time. ... This is also how TLS composes: there is deliberately no `tls` field on this struct (a typed `Option<TlsConfig>` slot would make TLS a property of every protocol variant — a protocol × tls matrix). TLS termination is instead `TlsListenProtocol`, a `ListenProtocol` DECORATOR that wraps whatever concrete protocol is carried here — on/off is the presence of that wrapper, composed the same way any other concrete protocol reaches this field: through `Self::protocol`.
>
> — and, at the call site inside `run_with_runtime`: TLS is NOT a field read here — a `TlsListenProtocol` (if `self.protocol` carries one) injects its own marker into the spec it hands its wrapped inner protocol, inside its own `serve`.

Walk that "matrix" claim through concretely: a field on `ListenerSpec` is one slot, shared no matter which protocol you picked — but the *code that reads it* still has to live somewhere, and that somewhere was `HttpListenProtocol`'s own `serve` (it reads the `__proxima_tls` spec marker directly, `proxima-http/src/listener/mod.rs`'s `serve_default`, line 451: `proxima_tls::config_from_spec_value(spec.get(proxima_tls::SPEC_KEY))`). Add `h2`/`h3-native` as protocols and TLS-for-h2, TLS-for-h3 either both need their own copy of that same TLS-reading code, or the field silently does nothing for them — one axis (protocol) times another (tls on/off) has to be handled somewhere, and a struct field forces that somewhere to be *inside* each protocol implementation. A decorator sidesteps the multiplication entirely: `TlsListenProtocol` is generic over `Arc<dyn ListenProtocol>`, wraps *any* of them uniformly, and only one implementation of "stamp the TLS marker into the spec" needs to exist, ever (`proxima-listen/src/handle.rs:499–536`):

```rust
#[cfg(feature = "tls")]
pub struct TlsListenProtocol {
    inner: Arc<dyn ListenProtocol>,
    tls: proxima_tls::TlsConfig,
}

#[cfg(feature = "tls")]
impl TlsListenProtocol {
    #[must_use]
    pub fn new(inner: Arc<dyn ListenProtocol>, tls: proxima_tls::TlsConfig) -> Self {
        Self { inner, tls }
    }
}

#[cfg(feature = "tls")]
impl ListenProtocol for TlsListenProtocol {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let mut spec_with_tls = spec.clone();
        attach_tls_to_spec(&mut spec_with_tls, &self.tls);
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .serve(bind, dispatch, &spec_with_tls, context, shutdown)
                .await
        })
    }
}
```
(`proxima-listen/src/handle.rs:499–536`, unabbreviated — the constructor block between the struct and the `ListenProtocol` impl was left out of this document's first pass)

`.serve()` clones whatever spec it was handed, stamps the same `__proxima_tls` marker key `attach_tls_to_spec` always used (this part is unchanged — only *where* the stamping happens moved), and hands that spec to the wrapped protocol's own `serve`. `name()` delegates straight to `inner.name()`, so anything downstream that keys off the protocol's registered name (like `Listener::run_with_runtime`'s `is_http` check for its SO_REUSEPORT-spread decision) sees straight through the wrapper — you get TLS termination without the wrapper pretending to be a *different* protocol.

`ListenerBuilder::tls(config)` (the `.tls(TlsConfig)` from §4's table) is the fluent front door onto this decorator — it does **not** write a spec key at all (`src/listener/handle.rs:207–212`, and proven directly by a unit test, `tls_composes_a_decorator_instead_of_writing_a_spec_key`, `src/listener/handle.rs:689–701`):

```rust
#[cfg(feature = "tls")]
#[must_use]
pub fn tls(mut self, tls: proxima_tls::TlsConfig) -> Self {
    self.tls = Some(tls);
    self
}
```

```rust
// test: tls_composes_a_decorator_instead_of_writing_a_spec_key
let fluent = ListenerBuilder::default().tls(proxima_tls::TlsConfig::self_signed());
assert!(fluent.tls.is_some(), ".tls(config) must accumulate on the builder, not the spec");
assert!(
    !fluent.spec.contains_key(proxima_tls::SPEC_KEY),
    "TLS must not be a spec key any more — it composes as a decorator at .serve() time"
);
```

`.serve()` reads that separately-accumulated `self.tls` field and calls `compose_tls`, which wraps whatever `resolve_listen_protocol` already picked (`src/listener/handle.rs:417–446`, abbreviated — the current source interleaves a comment explaining the registry-collision rename between the code lines below):

```rust
#[cfg(feature = "tls")]
fn compose_tls(
    tls: Option<proxima_tls::TlsConfig>,
    name: String,
    protocol: Option<Arc<dyn ListenProtocol>>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let Some(tls) = tls else {
        return Ok((name, protocol));
    };
    let inner = match protocol {
        Some(protocol) => protocol,
        None => http_listen_protocol_for_tls()?,
    };
    let registry_name = format!("{name}+tls");
    let wrapped: Arc<dyn ListenProtocol> = Arc::new(proxima_listen::TlsListenProtocol::new(inner, tls));
    let named: Arc<dyn ListenProtocol> = Arc::new(NamedListenProtocol { name: registry_name.clone(), inner: wrapped });
    Ok((registry_name, Some(named)))
}
```

One detail worth teaching in its own right: the registry key gets renamed to `"{name}+tls"` (e.g. `"http+tls"`). Why not just re-register under `"http"`? Because `App::new()` *already* registered a plain, non-TLS `HttpListenProtocol` under `"http"` (`src/app.rs:252–266`, `App::new`'s own registration list) — and `TlsListenProtocol::name()` delegates to the wrapped protocol's name, which is also `"http"`. Registering a second, TLS-wrapping protocol under the identical key `"http"` in the same registry would either collide or silently shadow the plain one. `NamedListenProtocol` (`src/listener/handle.rs:452–473`) is a small wrapper whose only job is to override the registry *key* without touching `serve` — the TLS-terminating decorator runs unchanged underneath; only the name used to look it up differs. `compose_tls_wraps_the_resolved_protocol_and_renames_the_registry_key` (`src/listener/handle.rs:703–719`) is the test proving `name == wrapped.name()` always holds, which is the invariant that keeps this registration collision-free.

## 8. The general escape hatch: `ListenerSpec::protocol`

`compose_tls` above (and `resolve_listen_protocol`'s h2/h3 arms) both produce the same shape: a registry-name string, plus an *optional*, already-built `Arc<dyn ListenProtocol>` to self-register instead of relying on a by-name lookup. That shape isn't invented by the umbrella crate — it's exactly what `ListenerSpec::protocol` already exists for, one layer down in `proxima-listen` (`proxima-listen/src/handle.rs:91–110`):

```rust
/// Escape hatch for any concrete [`ListenProtocol`] — h1/h2/h3-native
/// (whose types live in `proxima-http`, a crate that already depends on
/// this one, so this crate cannot host named per-protocol constructors
/// for them without a cyclic dependency), or a caller's own out-of-crate
/// impl. The mirror of the client's `.protocol(impl ClientProtocol)`
/// escape hatch.
#[must_use]
pub fn protocol(bind: SocketAddr, protocol: Arc<dyn ListenProtocol>) -> Self {
    Self {
        bind,
        protocol_name: protocol.name().to_string(),
        shutdown: ShutdownPolicy::drain_30s(),
        spec: Value::Null,
        protocol: Some(protocol),
    }
}
```

This is the listen-side twin of the client's `ClientBuilder::protocol(impl ClientProtocol)` (`src/client/handle.rs:543–551`) — the seam that lets a wire protocol defined *outside* this crate (a private/vendored `ListenProtocol`, or one of `h1`/`h2`/`h3-native` themselves, which `proxima-listen` cannot name directly without a cyclic dependency on `proxima-http`) plug into a `Listener` without a registry lookup by name. `Listener::run_with_runtime` checks for a carried `Arc` first and only falls back to `registry.get(&self.protocol_name)` if none was carried (`proxima-listen/src/handle.rs:183–186`) — proven directly by `run_with_runtime_resolves_carried_protocol_without_registry_entry`, which runs a fake protocol against a deliberately **empty** registry and shows it still serves. `ListenerBuilder::serve` (§9) is a caller of this exact seam: every non-default `resolve_listen_protocol`/`compose_tls` result (and, since this document's first pass, every `.pgwire(query)` registration too) flows through the umbrella's own `App::register_listen_protocol` (`src/app.rs:895`), not through `ListenerSpec::protocol` directly, but it's the identical "carry the concrete `Arc`, don't make the caller round-trip through a typo-able string" idea.

## 9. `.serve()`: what it actually composes

`ListenerBuilder::serve()` is a terminal — it consumes the builder and returns a running `Server`. Reading its full body (`src/listener/handle.rs:244–291`) end to end is the fastest way to see that it invents no new serve loop; it is the exact `App::new()? -> app.mount(...)? -> app.serve(config).await?` idiom from `examples/hello/main.rs:42–47`, automated — with two new steps since this document's first pass, both flagged inline below:

```rust
pub async fn serve(self) -> Result<Server, ProximaError> {
    reject_dead_axes(&self.spec)?;
    #[cfg(all(feature = "tls", feature = "pgwire"))]
    if self.tls.is_some() && self.pgwire_query.is_some() {
        return Err(ProximaError::Config(
            "Listener::builder(): .pgwire(query) manages its own TLS upgrade \
             (proxima-pgwire's `listen` feature); .tls(config) would double-wrap it \
             with the wrong (http) protocol underneath — drop one"
                .into(),
        ));
    }
    let bind = self.bind.or_else(|| bind_from_spec(&self.spec)).ok_or_else(|| {
        ProximaError::Config(
            "Listener::builder(): .bind(addr) (or .http(bind.to_string())) is required before .serve()".into(),
        )
    })?;
    let dispatch = self.dispatch.ok_or_else(|| {
        ProximaError::Config("Listener::builder(): .handle(pipe) is required before .serve()".into())
    })?;
    #[cfg(feature = "pgwire")]
    let pgwire_query = self.pgwire_query;
    let (protocol, extra_protocol) = resolve_listen_protocol(&self.spec)?;
    #[cfg(feature = "tls")]
    let (protocol, extra_protocol) = compose_tls(self.tls, protocol, extra_protocol)?;
    let app = App::new()?;
    if let Some(protocol) = extra_protocol {
        app.register_listen_protocol(protocol)?;
    }
    #[cfg(feature = "pgwire")]
    if let Some(query) = pgwire_query {
        app.register_listen_protocol(Arc::new(proxima_pgwire::PgWireListenProtocol::new(
            "pgwire", query,
        )))?;
    }
    app.mount("/{*path}", MountTarget::Handle(dispatch))?;
    let config = RunConfig { bind, protocol, spec: Value::Object(self.spec) };
    app.serve(config).await
}
```

Step by step, matching what §1–§8 already taught:

1. `reject_dead_axes` (§10) — fail fast on spec keys the listener side has no wiring for, before any socket work.
2. **New since the first pass:** the `.tls(config)` + `.pgwire(query)` guard — `.pgwire(query)` manages its own TLS upgrade under `proxima-pgwire`'s own `listen` feature, so composing the generic `TlsListenProtocol` decorator on top would wrap the WRONG (http) protocol underneath; `.serve()` hard-errors rather than silently double-wrapping (`pgwire_and_tls_together_hard_error_instead_of_wrapping_the_wrong_protocol`, `src/listener/handle.rs:729–754`).
3. Resolve `bind` — either `.bind(addr)` directly, or recovered from the `http` spec key `bind_from_spec` reads back (§10 explains why this recovery step exists at all).
4. Require `.handle(pipe)` — the one input every listener has that a client never does (§10). `.pgwire(query)` still requires it too, for the same uniform validation path, even though `PgWireListenProtocol::serve` never calls it once a constructor-supplied `query` is present.
5. `resolve_listen_protocol` (§5) picks the registry name, plus an optional self-registering `Arc`.
6. `compose_tls` (§7) wraps that in `TlsListenProtocol` if `.tls(config)` was called; otherwise a no-op pass-through (`compose_tls_is_a_noop_when_no_tls_was_requested`, `src/listener/handle.rs:721–727`).
7. A fresh `App::new()` (`src/app.rs:253`) — self-registers the extra protocol if one was carried (`App::register_listen_protocol`, `src/app.rs:895`).
8. **New since the first pass:** if `.pgwire(query)` carried a query engine, register a *fresh* `PgWireListenProtocol` carrying it — `App::new()`'s static registration list (§7) cannot pre-register a protocol it doesn't yet have a query engine for, unlike the one shared `h2`/`h3-native` instance `.grpc()`/`.h2()`/`.h3()` register.
9. `app.mount("/{*path}", MountTarget::Handle(dispatch))` — every path routes to `.handle(pipe)`, using the exact catch-all glob convention (`/{*path}`) the rest of the router uses for shorthand single-pipe listeners (`src/app.rs`'s `router.add(Mount::new("/{*path}", target))` sites, `src/app.rs:1055,1111`).
10. `app.serve(RunConfig { bind, protocol, spec })` — the identical `App::serve` (`src/app.rs:829–837`) `examples/hello` calls directly.

> **Note (previously a heads-up):** the first pass of this document found the source's own doc comment above `.serve()` citing the catch-all convention as also used at `src/app.rs:925,981`, when the private helper that reads `[[listen]]` TOML entries (`bind_listener`) is actually at `src/app.rs:992`, and the `/{*path}` shorthand pattern itself is written at `:1055` and `:1111`. That doc comment (`src/listener/handle.rs:231`) has been corrected directly as part of landing this document — the claim itself (this is the same catch-all convention used elsewhere, not a new one) was always correct, only the line numbers had drifted.

One genuine behavioral gap the source is explicit about, worth carrying forward rather than discovering the hard way: `App::serve` returns as soon as the listener lane is *spawned* — not once it's actually accepting connections — whereas `proxima_listen::handle::Listener::run_with_runtime` (the lower-level driver `ListenerBuilder` does **not** go through) blocks for a per-lane ready acknowledgment first. A caller that dials immediately after `Listener::builder()....serve()` resolves can race a not-yet-listening socket. The fix (threading the same ready-ack into `App::run_until_signal`) is out of scope for this landing; today's workaround is a bounded poll-connect retry loop, the same one `tests/e2e/listener_client_interop.rs`'s `wait_until_listening` uses.

## 10. The two honest asymmetries

`src/listener/mod.rs`'s module doc calls these out directly, and both trace back to one fact: **a listener binds; a client dials.**

**Bind is mandatory, and has no url to fall back on.** A `Client` always has a URL — `.http(url)` *is* the dial target, full stop. A `Listener` needs a socket address to bind, and nothing in `ProtocolSugar`/`TransportSugar` was designed to carry one — `.http(url)` in the client's own vocabulary means "the address to *connect to*." `Listener::http(bind)` (§2) reuses that same method for a different purpose (stringify the bind address into the same `http` spec key, §4's table), which is why `.bind(addr)` exists as its own inherent method (`src/listener/handle.rs:119–123`) and why `bind_from_spec` (`src/listener/handle.rs:300–304`) exists purely to recover a bind address from that string when `.bind()` itself was never called:

```rust
fn bind_from_spec(spec: &serde_json::Map<String, Value>) -> Option<SocketAddr> {
    spec.get("http").and_then(Value::as_str).and_then(|value| value.parse().ok())
}
```

`.serve()` requires one or the other be present, and says so precisely rather than failing on a `None.unwrap()` deep in a socket call (`serve_without_bind_errors_before_touching_a_socket`, `src/listener/handle.rs:656–663`, and the client has no equivalent test because it has no equivalent failure mode — a `Client` with no url just never resolves a factory, it doesn't need a bind check).

**`.tls(TlsConfig)` and `.grpc()` are shadowed, not reused, and for two different reasons.** `.tls()` on the client is bare — zero arguments, because ALPN negotiation (§6) does the actual work; all `.tls()` needs to do is flip the transport marker. A listener terminating TLS needs real key material — a certificate and a private key, or a self-signed generator — which the bare `TransportSugar::tls()` has no argument slot for. So `ListenerBuilder` defines its own `.tls(proxima_tls::TlsConfig)` (§7), an inherent method that *shadows* the zero-arg blanket one. `reject_dead_axes` (`src/listener/handle.rs:315–330`) is the safety net for the case where the `tls` feature itself is off and the shadowing inherent method doesn't exist to intercept the call — a bare `.tls()` would otherwise silently set `transport: "tls"` and terminate nothing:

```rust
if spec.get("transport").and_then(Value::as_str) == Some("tls") && !tls_marker_present(spec) {
    return Err(ProximaError::Config(
        "Listener::builder(): bare .tls() only sets a marker key and terminates nothing; \
         call .tls(TlsConfig::self_signed() | TlsConfig::pem(..) | TlsConfig::files(..)?), \
         which requires the `tls` feature".into(),
    ));
}
```

`.grpc()` shadows for the opposite reason: the client's `.grpc(url)` carries a *dial target* — there is nowhere else for that url to live. A listener has nothing to dial; it already has `.handle(pipe)` on hand and only needs to flip which wire protocol it resolves to (§5's `spec.contains_key("grpc")` check). So `ListenerBuilder::grpc()` is a zero-argument inherent method (`src/listener/handle.rs:135–147`) that just sets a boolean marker, shadowing the client's url-carrying one. `.h2()` (§4) is the same shape, minus the shadowing — there is no blanket `TransportSugar::h2()` at all, so it needs no divergence, just a plain inherent method. `.pgwire(query)` (§4, §5, §9) is the odd one out among the new axes: not a marker key at all, but a typed `PgPipeHandle` accumulated on the builder the same way `.tls(config)` accumulates a `TlsConfig` — a listener-only shape with no client-side concept to shadow or extend.

`.proxy(url)` — an *actual* client-only concept, an egress CONNECT tunnel before the dial — has no listener meaning at all, and nothing shadows it: `TransportSugar::proxy` is still reachable on `ListenerBuilder` through the ordinary blanket impl, because there is no negative-impl mechanism in Rust to remove a trait method selectively. `reject_dead_axes` is what actually stops it from doing something wrong silently — `.serve()` hard-errors if the `proxy` key is present at all, rather than binding a listener that quietly ignores a caller's `.proxy(url)` call (`proxy_axis_hard_errors_at_serve_instead_of_silently_ignoring_it`, `src/listener/handle.rs:677–687`).

## 11. A full walkthrough, compiled and run

**Update since the first pass:** the same landing that added `.h2()`/`.pgwire(query)` also shipped two real, citable examples that exercise `Listener::builder()` end to end against a real client — grep `examples/` for `ListenerBuilderEntry`/`Listener::builder` yourself and you'll find them:

- `examples/h2_native_server.rs` — `Listener::builder().bind(bind).h2().handle(into_handle(ConstantOk)).serve().await?` (line 73), proven with a real `H2ClientUpstream` round trip, no tokio anywhere in the request path (`cargo run --example h2_native_server`; `cargo tree --example h2_native_server -e normal -i tokio` is empty).
- `examples/pgwire_server.rs` — `Listener::builder().bind(bind).pgwire(into_pg_handle(EchoPipe)).handle(into_handle(NeverDispatch)).serve().await?` (line 136), proven with a real PostgreSQL-wire `PgClient` over a plain `TcpStream` (`cargo run --example pgwire_server --features pgwire`).

Neither example demonstrates the specific comparison this section teaches, though — `Listener::builder()` vs. the one-liner `Listener::http(bind)`, side by side with the manual `App`/`RunConfig` baseline they both mirror — so that three-way comparison still isn't cited from a shipped example. I wrote it, registered it as a temporary `[[example]]` in this worktree's `Cargo.toml`, built it with `cargo build --example`, ran it with `cargo run --example` (exit code 0, no panics), and then removed both the file and the `Cargo.toml` entry before finishing — none of that scratch work is committed. Treat this block as *verified to compile and run against `0ac7a565`*, not as an existing, citable file:

```rust
use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use proxima::{App, Listener, ListenerBuilderEntry, ProximaError, Request, Response, RunConfig, TransportSugar, into_handle};

#[proxima::piped(send)]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));

    // manual shape (examples/hello/main.rs) — what you already know
    let app = App::new()?;
    app.mount("/", hello)?;
    let server = app.serve(RunConfig::http(bind)).await?;
    drop(server);

    // builder shape, same result
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(hello))
        .serve()
        .await?;
    drop(server);

    // one-liner, mirroring Client::http(url)
    let server = Listener::http(bind).handle(into_handle(hello)).serve().await?;
    drop(server);

    Ok(())
}
```

Note `hello` had to be `#[proxima::piped(send)]` here, not the bare `async fn` `App::mount` itself accepts directly (Foundations part 2 §8's `ViaFn`). `App::mount`'s `IntoMountTarget` wraps a bare fn in an internal `FnHandler` adapter for you; `into_handle` (needed for `.handle(pipe)`, since `ListenerBuilder::handle` takes `impl Into<PipeHandle>`, `src/listener/handle.rs:125–133`) has a stricter bound — `Implementor: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + 'static` (`proxima-primitives/src/pipe/handler.rs:86–88`) — which a bare fn item doesn't satisfy on its own. This is a real, mechanical distinction worth carrying forward, not a quirk of my scratch file: `App::mount(path, target)` takes anything `IntoMountTarget` covers (four shapes, §1's recap points at Foundations part 2 §8); `ListenerBuilder::handle(handle)` takes something already reduced to a `PipeHandle`, one shape only.

I'd suggest promoting a trimmed version of this — probably folded into `examples/hello`'s own README as a second "the builder way" section, or a new `examples/listener_builder/main.rs` alongside `examples/h2_native_server.rs`/`examples/pgwire_server.rs` — as a follow-up; see the report for why I didn't invent one wholesale here (owner ratification call, not mine to make unilaterally).

## 12. Where to go next

- [Listener on-ramp, part 2: the universal listener](./05-listener-universal.md) and [part 3](./06-listener-production.md) — the `.any()`/`.accept()`/`.accepts()`/`.deny()`/`.blacklist()` axis this document's own table (§4) predates; see the drift note after §4's table.
- [Foundations, part 2](./01-ergonomics.md) §8, if `IntoMountTarget`'s four shapes (referenced in §11 above) weren't already solid.
- [Build an API gateway](./build-an-api-gateway.md) for `Client::http`/`Client::builder` used against a real upstream, end to end.
- `examples/h2_native_server.rs` and `examples/pgwire_server.rs` (§11) — the two real, runnable examples exercising `.h2()` and `.pgwire(query)` through `Listener::builder()` against a real client, if the scratch three-way comparison in §11 left the shape wanting a shipped, `cargo run`-able reference.
- `docs/configuration.md`'s "typed listener configs" section for a *third*, unrelated way to reach a running listener — `HttpListener::http(addr)` / `HttpsListener::https(addr, cert, key)` (`src/settings/listener.rs:33–47,74`). Don't confuse it with this document's `Listener::builder()`: those are `Into<RunConfig>` typed config shapes for `App::serve(impl Into<RunConfig>)`, one layer above the spec/registry resolution this document covers, and they carry no `.tcp()`/`.tls()`/`.h3()`/`.grpc()` axis sugar at all.
- `proxima-listen`'s own crate docs for `ListenProtocol`, `ListenRegistry`, and `Listener::run_with_runtime`'s per-core SO_REUSEPORT fan-out — this document treated `run_with_runtime` as a fact (§9's readiness-race caveat), not something to re-derive; a future tutorial on multi-core listener fan-out would build on it directly.
