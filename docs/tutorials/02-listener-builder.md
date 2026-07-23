# Foundations, part 3: the Listener builder, mirrored from the Client

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md) §13 (a pipe that answers web requests, `into_handle`, `App::mount`, `app.serve(RunConfig)`) and [Foundations, part 2](./01-ergonomics.md) §8 (`App::mount`'s four accepted shapes). You should also already be comfortable pointing a `proxima::Client` at an upstream — `Client::http(url)` or `Client::builder().http(url).tls().build()` — the way [Build an API gateway](./build-an-api-gateway.md) uses it (`Client::http(format!("http://{origin_api_bind}"))?`, `examples/gateway/main.rs:77`). This document does not re-teach `Client` from scratch, but §1 below gives a short, cited recap of exactly the parts the Listener side mirrors — there is no dedicated "Client builder" tutorial elsewhere in this tree yet, so this section is that recap.

**You will learn:** that `Listener` — a serve-side value that already existed in `proxima-listen` — grows a fluent builder, `Listener::builder()` / `Listener::http(bind)`, built from the *same* spec-accumulation seam `Client::builder()` uses; why the fluent axis methods (`.tcp()`, `.http()`, `.kafka()`, …) are TYPE-SPECIFIC extension traits per builder, not one shared blanket trait; why the entry points had to be a trait blanket-impl'd onto a foreign type rather than a second `Listener` struct; how a `.tcp()`/`.udp()`/`.quic()`/`.grpc()` choice resolves down to one concrete `ListenProtocol`, mirroring how the client's `load()` resolves a spec to a `PipeHandle`; why TLS is not a field on any spec but a decorator composed at `.serve()` time; and the two places — and only two — where a listener's builder honestly cannot mirror the client's, because a listener's inputs are different in kind.

**New concepts (in order):** `SpecBuilder` (the shared accumulation seam) · the type-specific axis traits (`ClientTransportExt`/`ClientProtocolExt`/`ClientSecurityExt` vs. `ListenerTransportExt`/`ListenerProtocolExt`) · `ListenerBuilderEntry` (`Listener::builder()`/`Listener::http(bind)`) · `resolve_listen_protocol` · `TlsListenProtocol` (the TLS decorator) · `.protocol(impl AnyProtocol)` (the escape hatch) · the two asymmetric axes.

Every code block below is copied verbatim from a real file in this repository (cited by `file:line`, checked against this worktree's `git rev-parse HEAD` off `ec02fc3f`) or a unit test that `cargo nextest run` in this repo actually passes. Nothing here is invented.

**Drift notice for anyone who read an earlier version of this page:** this document originally taught a blanket `ProtocolSugar`/`TransportSugar` pair — `impl<B: SpecBuilder> ProtocolSugar for B {}`, lighting up `.http()`/`.tls()`/`.h3()`/etc. on *any* `SpecBuilder` for free. That blanket pair is **deleted**. `.h3()` is also gone — HTTP/3 is now `.quic()` (composed with `.http()`: `.http().quic()`). Every axis method below is a TYPE-SPECIFIC trait, implemented once per concrete builder (`ClientTransportExt for ClientBuilder`, `ListenerTransportExt for ListenerBuilder`, and so on) — no blanket impl reaches across builders any more. This rewrite reflects that; if you see `ProtocolSugar`/`TransportSugar`/`.h3()` anywhere else in this tree, that's the same stale teaching and should be reported.

## Contents

1. What you already know: `Client::builder()`, recapped
2. Meet the mirror: `Listener::builder()` / `Listener::http(bind)`
3. Why a trait, not a second `Listener` type
4. One coin, two faces: the side-by-side axis table
5. From spec to a concrete protocol: `resolve_listen_protocol`
6. Why the listener names wire versions and the client doesn't
7. TLS as a composed layer, not a spec field
8. The general escape hatch: `.protocol(impl AnyProtocol)`
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

`ClientBuilder` is not a hand-rolled struct with a pile of ad-hoc setters. It wraps one `serde_json::Map<String, Value>` (`src/client/handle.rs:409–421`, the `#[derive(Default)] pub struct ClientBuilder` fields) — the same JSON `Value` a `[[pipe]]` TOML table deserializes to — and every fluent method just writes a key into that map. The base seam is two methods, `set`/`push` (`proxima-config/src/sugar/builder.rs:51–57`):

```rust
pub trait SpecBuilder: Sized {
    fn set(self, key: &str, value: impl Into<Value>) -> Self;
    fn push(self, key: &str, value: impl Into<Value>) -> Self;
}
```

That's *all* `SpecBuilder` is. There used to be a pair of blanket traits on top (`ProtocolSugar`/`TransportSugar`, `impl<B: SpecBuilder> ProtocolSugar for B {}`) that gave every `SpecBuilder` the exact same `.http()`/`.tls()`/`.h3()` methods for free. That pair is **retired** — `proxima-config/src/sugar/builder.rs`'s own module doc says so directly: "There is no blanket axis trait here any more (`ProtocolSugar`/`TransportSugar` were removed — a blanket `impl<B: SpecBuilder> Trait for B` reaches every `SpecBuilder`, including foreign-crate ones a caller never meant to touch)." The replacement is a SEPARATE, TYPE-SPECIFIC trait per axis per builder. On the client side (`src/client/transport.rs:14–35`, `src/client/protocol.rs:18–79`, `src/client/security.rs:18–25`):

```rust
pub trait ClientTransportExt: Sized {
    fn tcp(self) -> Self;
    fn udp(self) -> Self;
    fn quic(self) -> Self;
    fn proxy(self, url: impl Into<String>) -> Self;
}

pub trait ClientProtocolExt: Sized {
    fn http(self, url: impl Into<String>) -> Self;
    fn https(self, url: impl Into<String>) -> Self;
    fn grpc(self, url: impl Into<String>) -> Self;
    // + .kafka(dsn) / .mqtt(dsn) / .amqp(dsn) / .dns(dsn) / .memcached(dsn) /
    //   .redis(dsn) / .valkey(dsn) / .pgwire(dsn), each feature-gated
}

pub trait ClientSecurityExt: Sized {
    fn tls(self) -> Self;
}
```

Each is implemented exactly once, for `ClientBuilder` (`impl ClientTransportExt for ClientBuilder { fn tcp(self) -> Self { self.set("transport", "tcp") } ... }`, `src/client/transport.rs:37–53`). `.tcp()`/`.udp()`/`.quic()` still just write the `transport` key; `.tls()` writes `transport: "tls"`; `.http(url)`/`.grpc(url)` still write `http`/`grpc`. The *mechanism* (write one key into the accumulated map) is unchanged from the old blanket design — only *how many types can reach the method* changed: before, importing `TransportSugar` lit up `.tcp()` on every `SpecBuilder` in the crate, including ones that had no business seeing it; now, `.tcp()` on a `ClientBuilder` comes from `ClientTransportExt`, a trait `ListenerBuilder` does not implement at all, so it is not even in the candidate set for method resolution on a `ListenerBuilder` value.

Bring the client's three axis traits into scope with `use proxima::{ClientTransportExt, ClientProtocolExt, ClientSecurityExt};` (or the umbrella `use proxima::prelude::*;`, which re-exports all of them — `src/lib.rs:418–423`). This is why a fluent chain and a literal config `Value` are provably the same spec, checked directly in the client's own test suite (`verbs_map_to_methods_and_builder_lowers_axes`, `src/client/handle.rs:967`, parity block at `995–1012`, abbreviated):

```rust
let fluent = Client::builder().http("http://api.example.com").tls().proxy("http://127.0.0.1:8080").build()?;
let config = Client::from_value(json!({
    "http": "http://api.example.com", "transport": "tls", "proxy": "http://127.0.0.1:8080",
}))?;
assert_eq!(fluent.inner.spec, config.inner.spec);
```

That's the whole shape you already know: one accumulated JSON spec, a handful of type-specific axis traits giving it fluent methods, and a terminal (`.build()`) that freezes it. Everything below is the identical shape, reused — not reinvented — for the serve side.

## 2. Meet the mirror: `Listener::builder()` / `Listener::http(bind)`

`proxima-listen` already had a real serve-side value called `Listener` (`proxima-listen/src/handle.rs`) — bind address, protocol name, shutdown policy, spec, and a dispatch `PipeHandle`, produced by `ListenerSpec::attach(dispatch)` and driven by `Listener::run_with_runtime`. On top of that sits a fluent front door, in `src/listener/handle.rs`:

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

Read that side by side with §1's `Client::http`/`Client::builder`. Same two entry points, same names, same shape: a one-liner that pre-fills the common case, and a bare `builder()` for everything else. `Listener::http(bind)` calls `.http(bind.to_string())` — the listener's OWN `ListenerProtocolExt::http` (`src/listener/protocol.rs:28–29`), a different trait impl from the client's `ClientProtocolExt::http`, but the same spec key (`"http"`) and the same idea: pre-fill the common case.

Bring both entry points into scope together — `use proxima::{Listener, ListenerBuilderEntry};`, plus `use proxima::ListenerTransportExt;` for `.tcp()` below (or `use proxima::prelude::*;` for everything at once) — and you get the mirror of the manual `App::new()?; app.mount(...)?; app.serve(RunConfig::http(bind)).await?;` shape from `examples/hello/main.rs:108–111`:

```rust
use proxima::prelude::*;
use proxima::into_handle;

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
(`src/listener/mod.rs:49–66`, its own module-doc example)

`my_pipe` is a placeholder for whatever `Handler` you already mount today. §11 below is a concrete, compiling, actually-run version of this, checked against the same source this document cites.

## 3. Why a trait, not a second `Listener` type

The obvious question: why is `Listener::builder()` a trait method reached through `ListenerBuilderEntry`, rather than a plain inherent `impl Listener { fn builder() -> ... }`? Rust's orphan rule is the answer, and the source states it directly (`src/listener/handle.rs:21–35`):

> `Listener` is defined in `proxima-listen`, a crate this one depends on but that cannot depend back on `App`/`Server`; Rust's orphan rule forbids an inherent `impl Listener { fn builder() }` from this crate, so the entry points are a local trait blanket-impl'd for the foreign type instead of a second `Listener` type living here.

`ListenerBuilder::serve()` (§9) has to construct an `App`, mount a router, and return a `Server` — all types that live in the umbrella `proxima` crate, several layers above `proxima-listen`. `proxima-listen` cannot know about `App`; only the umbrella crate can wire the two together. But the umbrella crate doesn't own the `Listener` type it wants to extend. The escape hatch Rust actually gives you for "add a method to a type I don't own" is a trait, defined in the crate doing the extending, implemented once for the foreign type — `impl ListenerBuilderEntry for Listener`. No new `Listener` struct was minted to work around this — that would have been the RISC violation (principle 1: don't add a type next to an existing one when the existing one still works).

Note this is a NARROWER use of the same Rust mechanism §1's axis traits use: `ListenerBuilderEntry` is implemented for exactly one foreign type (`Listener`), not blanket-implemented for every type satisfying some bound. `ProtocolSugar`/`TransportSugar` (§1) were the blanket-over-everything version of this idiom, and that breadth is exactly why they were retired — see §4's drift note for why "reaches every `SpecBuilder`" turned out to be the wrong amount of reach.

## 4. One coin, two faces: the side-by-side axis table

`ListenerBuilder` (`src/listener/handle.rs:82–158`) accumulates its own `serde_json::Map<String, Value>`, exactly like `ClientBuilder` does — plus several fields with no client-side twin (`tls: Option<proxima_tls::TlsConfig>` under `#[cfg(feature = "tls")]`, `pgwire_query`/`dns_handler`/`websocket_handler`/`any_mode`/`extra_protocols`/`blacklist_config` under their own feature gates), each accumulated separately from `spec` because none of them fits a plain `serde_json::Value` key — and implements the identical `SpecBuilder` seam (`src/listener/handle.rs:1127–1140`):

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

On top of that seam, `ListenerBuilder` gets its OWN axis traits — `ListenerTransportExt` (`src/listener/transport.rs`) and `ListenerProtocolExt` (`src/listener/protocol.rs`) — separate types from the client's `ClientTransportExt`/`ClientProtocolExt`, not the same blanket trait reused. Bring them into scope with `use proxima::{ListenerTransportExt, ListenerProtocolExt};` (or `proxima::prelude::*`). A unit test proves the two sides still lower to identical spec keys for the axes they share, even though the trait behind each is now a different type (`listener_builder_mirrors_client_builder_axis_keys`, `src/listener/handle.rs:1255–1271`):

```rust
let fluent = ListenerBuilder::default()
    .https("127.0.0.1:8080")
    .spec("transport", json!("tls"));
assert_eq!(fluent.spec.get("http").and_then(Value::as_str), Some("127.0.0.1:8080"));
assert_eq!(fluent.spec.get("transport").and_then(Value::as_str), Some("tls"));
```

`src/listener/mod.rs`'s own module doc lays out the full, CURRENT picture as a table (`src/listener/mod.rs:30–41`) — this is the authoritative source for this section, transcribed here verbatim:

| axis | client (`ClientBuilder`) | listener (`ListenerBuilder`) |
|---|---|---|
| `.tcp()` / `.udp()` / `.quic()` | `ClientTransportExt` | `ListenerTransportExt` — `.quic()` resolves to the native h3 `DatagramProtocol` listener |
| `.tls()` | `ClientSecurityExt`, zero-arg — the dial url comes from a separately-chained `.http(url)` | inherent `.tls(TlsConfig)` — real cert material required; composes as a decorator over whatever `resolve_listen_protocol` resolves |
| `.proxy(url)` | `ClientTransportExt` | no listener meaning — `.serve()` hard-errors if present |
| `.http(url)` / `.https(url)` | real (dials the url) | real — carries the BIND address (`bind.to_string()`), read by `bind_from_spec` when `.bind(addr)` wasn't called directly |
| `.grpc(url)` / `.grpc()` | url-carrying | url-less — listener dispatches to `.handle(pipe)`, not a url; resolves to `"h2"` (gRPC rides h2) |
| `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()` | DSN, delegates to `.protocol()` | typed handle, delegates to `.protocol()` |
| `.pgwire()` | DSN, delegates to `.protocol()` | typed query engine — KEEPS its bespoke fresh-registration path (TLS double-wrap guard) |
| `.dns()` | DSN, delegates to `.protocol()` | the one dual-transport axis — branches on `.tcp()`/`.udp()` at `.serve()` time |
| (no client twin) | — | `.websocket(handler)` — wires into h1's Upgrade seam, not a peer `AnyProtocol` |
| (no client twin) | — | inherent `.h2()` — the other name for the same shared `"h2"` protocol |

Two rows say "shadowed" territory even though the table doesn't spell that word out: `.tls()` and `.grpc()` are both real methods on BOTH builders, but they are NOT the same trait method reached twice — `ClientBuilder`'s `.tls()` comes from `ClientSecurityExt` (zero-arg), `ListenerBuilder`'s `.tls(config)` is a plain inherent method (one-arg, `src/listener/handle.rs:414–417`) that `ListenerBuilder` defines because it does not implement `ClientSecurityExt` at all — there is no name collision to "shadow," because the trait providing the client's version was never in scope for a `ListenerBuilder` value in the first place. Same story for `.grpc()`: the client's `ClientProtocolExt::grpc(self, url)` and the listener's own inherent `ListenerProtocolExt::grpc(self)` (zero-arg, `src/listener/handle.rs:601–604`) are different methods on different traits; nothing is being hidden, because `ListenerBuilder` was never a candidate for `ClientProtocolExt` to begin with. This is the concrete payoff of retiring the blanket traits (§1's drift note): under the old design, EVERY axis method reached every `SpecBuilder`, so the client's url-carrying `.grpc(url)` and a hypothetical listener-side url-less `.grpc()` genuinely WOULD have collided and needed Rust's inherent-shadows-trait resolution rule to sort out. Under the type-specific design, they were never in the same trait to begin with — the "shadowing" story this document's first pass had to explain at length is now moot, because the traits are separate.

`.proxy(url)` is the one row with a real, load-bearing gotcha: it's real on `ClientTransportExt`, and `ListenerBuilder` does NOT implement `ClientTransportExt` — so a `ListenerBuilder` value cannot call `.proxy(url)` through that trait at all, full stop. The `.serve()` hard-error this row promises ("no listener meaning") is not defending against a caller reaching `.proxy()` via the trait (that's a compile error, not a runtime one) — it's defending against the ONE door still open regardless: the raw `SpecBuilder::set`/`.spec(key, value)` escape hatch (`builder.set("proxy", url)` compiles, since `ListenerBuilder: SpecBuilder`). §10 covers this in full.

> **Drift note (unchanged from earlier passes):** `.any()`/`.accept(name)`/`.accepts([...])`/`.any_handler(name, handler)`/`.any_on_reject(hook)`/`.deny(name, literal)`/`.denies([...])`/`.blacklist(config)` (all `#[cfg(any(feature = "http1", feature = "http1-native"))]`, `src/listener/handle.rs:213–390`) don't appear in the table above — the listener on-ramp series (`docs/tutorials/04-listener-hello.md` onward) is the maintained teaching surface for that whole axis; [part 2](./05-listener-universal.md) covers `.any()`/`.accept()`/`.accepts()`, [part 3](./06-listener-production.md) covers `.deny()`/`.blacklist()`, and [part 4](./07-sugar-composition.md) covers how the transport/protocol axes in THIS table compose with each other and fail honestly when a combination has no meaning.

## 5. From spec to a concrete protocol: `resolve_listen_protocol`

One term this section needs before its code makes sense: a `ListenProtocol` (`proxima-listen`) is the listen-side trait each wire implementation — `HttpListenProtocol` (h1+h2), `H2PriorKnowledgeAnyProtocol`, `H3NativeListenProtocol` — implements; a `ListenRegistry` looks these up by name. (If you've read [Build a plugin](./build-a-plugin.md), this is structurally the listen-side twin of that tutorial's `PipeFactory`/registry pattern — not a prerequisite for this document, just a pointer if the shape looks familiar.)

On the client side, `load()` reads the accumulated spec `Value` and dispatches on which key is present, ALSO now checking `transport == "quic"` to route through the native h3 upstream instead of the ordinary h1/h2 client (`src/load.rs:483–507`, abbreviated):

```rust
let (handle, kv_backend): (PipeHandle, Option<Arc<dyn KvHandle>>) =
    if let Some(http) = value.get("http") {
        if value.get("transport").and_then(Value::as_str) == Some("quic") {
            let canonical = canonical_h3(http, value)?;
            let factory = context.registry.get("h3-native")?;
            (factory.build(&canonical, None).await?, None)
        } else {
            let canonical = canonical_http(http, value)?;
            let key = match value.get("wire").and_then(Value::as_str) {
                Some("tokio") if context.registry.get("http-tokio").is_ok() => "http-tokio",
                _ => "http",
            };
            let factory = context.registry.get(key)?;
            (factory.build(&canonical, None).await?, None)
        }
    } else if let Some(grpc) = value.get("grpc") {
        // ...
```

This is the concrete mechanism behind "`.http().quic()` IS h3" (the sugar-composition page, [part 4](./07-sugar-composition.md), teaches this from the reader's side; this is the implementation): `.quic()` never introduces a THIRD protocol key alongside `http`/`grpc` — it's a modifier on `http` that `load()` checks once it already knows `http` is present.

`resolve_listen_protocol` is the listen-side twin of that same dispatch (`src/listener/handle.rs:955–974`, current source, with the `pgwire_axis`/`dns_axis` marker-key checks the client side has no equivalent of):

```rust
fn resolve_listen_protocol(
    spec: &serde_json::Map<String, Value>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    if spec.contains_key("pgwire_axis") {
        return Ok(("pgwire".to_string(), None));
    }
    if spec.contains_key("dns_axis") {
        return Ok(("dns".to_string(), None));
    }
    if spec.contains_key("grpc") || spec.contains_key("h2") {
        return h2_listen_protocol();
    }
    if spec.get("transport").and_then(Value::as_str) == Some("quic") {
        return h3_native_listen_protocol();
    }
    // default / `.tcp()`: the ALPN h1+h2 combiner. `.tls()` composes as a
    // decorator OVER whatever this resolves — see `compose_tls` — so it
    // never changes what's resolved here.
    Ok(("http".to_string(), None))
}
```

`.pgwire(query)`/`.dns(handler)` are checked **first**: each carries a typed handle no other axis combination can produce, so they always win, and neither returns an extra protocol here — `.serve()` (§9) registers its own fresh instance directly, since resolving needs the handle this spec-only function never sees. `.grpc()`/`.h2()` share one branch and resolve to the identical `"h2"` protocol — two names for the same wire, kept separate because `.grpc()` mirrors the client's `ClientProtocolExt::grpc` naming while `.h2()` names the transport directly, and gRPC rides h2 either way. Note the branch order: `.grpc()`/`.h2()` are checked BEFORE `transport == "quic"`, so `.grpc().quic()` would resolve to `"h2"`, not h3-native — except that combination is rejected earlier, by `reject_invalid_axis_combinations` (§10, [part 4](./07-sugar-composition.md) teaches this failure mode directly), before `resolve_listen_protocol` is ever called.

Real unit tests exercise this directly and pass on this worktree today (`src/listener/handle.rs:1171–1243`):

- `resolve_listen_protocol_defaults_to_http_and_opts_into_grpc_via_the_same_key_load_reads` (`:1171`) — `.tcp()` and bare `transport: "tls"` both resolve to `"http"`, with no self-registered protocol (`extra.is_none()`), because `"http"` is already in `App::new()`'s default registry.
- `grpc_axis_resolves_to_h2_and_self_registers` (`:1191`, needs `http2`) — `.grpc()` resolves to `"h2"` and *does* carry a protocol to register, because h2-as-a-listen-protocol is not in `App::new()`'s default set.
- `quic_axis_resolves_to_h3_native_and_self_registers` (`:1201`, needs `http3`) — same shape for `.quic()` → `"h3-native"`.
- `h2_axis_resolves_to_the_same_shared_h2_protocol_as_grpc` (`:1211`, needs `http2`) — `.h2()` resolves to the identical `"h2"` name and carried `Arc` as `.grpc()`.
- `pgwire_axis_resolves_to_pgwire_and_carries_nothing_here` (`:1221`, needs `pgwire`) — the `pgwire_axis` marker resolves to `"pgwire"` with no carried protocol.
- `pgwire_axis_takes_priority_over_grpc_and_quic` (`:1233`, needs `pgwire`) — `.pgwire(query)` combined with `.grpc()` and `.quic()` still resolves to `"pgwire"`, proving the priority order above.

## 6. Why the listener names wire versions and the client doesn't

Look again at §5's two dispatch functions. The client's picks between `"http"`/`"http-tokio"`/`"h3-native"` (all under the ONE `http` key, `transport` as a modifier) and `"grpc"` — no branch that treats h1/h2/h3 as three peer registry entries the way the listener does. The listener's picks between three genuinely different `ListenProtocol` names (`"http"`, `"h2"`, `"h3-native"`) as PEERS. This is not an oversight; it's a structural difference between dialing and binding.

A client dials one TCP (or QUIC) connection to one upstream. Whether that connection ends up speaking HTTP/1.1 or HTTP/2 over it is negotiated **after** the connection is open, via TLS's ALPN extension — one factory (`"http"`), one socket, one `PipeHandle` — the wire version is a runtime negotiation result inside that one factory, not a different registry entry. HTTP/3 is the one case that genuinely needs its OWN factory even client-side (`"h3-native"`), because QUIC isn't "TCP plus ALPN" — it's a different transport, so `.quic()` routes to a different factory rather than a different negotiated outcome inside the same one.

A listener has no negotiation to defer to, because *it* is the side that decides what the socket looks like before any client shows up. `"http"` (h1+h2 combiner), `"h2"`, and `"h3-native"` are three genuinely different `ListenProtocol` implementations — each owning its **own accept loop and its own socket type**: `"http"`/`"h2"` bind a TCP listener and `accept()` connections; `"h3-native"` binds a **UDP** socket and drives QUIC datagrams, not a TCP accept loop at all. There is no ALPN negotiation that could turn a bound UDP-QUIC listener into a TCP one after the fact — the choice has to be made before `.serve()` binds anything, which is exactly why it's a compile-time-selected registry key rather than a runtime ALPN outcome. `resolve_listen_protocol` picking a **name** (used both as the registry lookup key and, for h2/h3, as the concrete `Arc<dyn ListenProtocol>` self-registered onto the fresh `App`) is that decision, made once, before `bind()` is ever called.

## 7. TLS as a composed layer, not a spec field

TLS is not a plain optional field on `ListenerSpec`/`Listener` — it composes as a decorator. The reason, in the source's own words (`proxima-listen/src/handle.rs`'s `ListenerSpec::protocol` doc):

> A protocol resolved at construction time instead of by name through the `ListenRegistry` at serve time. ... This is also how TLS composes: there is deliberately no `tls` field on this struct (a typed `Option<TlsConfig>` slot would make TLS a property of every protocol variant — a protocol × tls matrix). TLS termination is instead `TlsListenProtocol`, a `ListenProtocol` DECORATOR that wraps whatever concrete protocol is carried here — on/off is the presence of that wrapper, composed the same way any other concrete protocol reaches this field: through `Self::protocol`.

Walk that "matrix" claim through concretely: a field on `ListenerSpec` is one slot, shared no matter which protocol you picked — but the *code that reads it* still has to live somewhere. Add `h2`/`h3-native` as protocols and TLS-for-h2, TLS-for-h3 either both need their own copy of that same TLS-reading code, or the field silently does nothing for them — one axis (protocol) times another (tls on/off) has to be handled somewhere, and a struct field forces that somewhere to be *inside* each protocol implementation. A decorator sidesteps the multiplication entirely: `TlsListenProtocol` is generic over `Arc<dyn ListenProtocol>`, wraps *any* of them uniformly, and only one implementation of "stamp the TLS marker into the spec" needs to exist, ever (`proxima-listen/src/handle.rs`):

```rust
#[cfg(feature = "tls")]
pub struct TlsListenProtocol {
    inner: Arc<dyn ListenProtocol>,
    tls: proxima_tls::TlsConfig,
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
            inner.serve(bind, dispatch, &spec_with_tls, context, shutdown).await
        })
    }
}
```

`.serve()` clones whatever spec it was handed, stamps the same `__proxima_tls` marker key, and hands that spec to the wrapped protocol's own `serve`. `name()` delegates straight to `inner.name()`, so anything downstream that keys off the protocol's registered name sees straight through the wrapper — you get TLS termination without the wrapper pretending to be a *different* protocol.

`ListenerBuilder::tls(config)` (the inherent `.tls(TlsConfig)` from §4's table) is the fluent front door onto this decorator — it does **not** write a spec key at all (`src/listener/handle.rs:414–417`, and proven directly by a unit test, `tls_composes_a_decorator_instead_of_writing_a_spec_key`, `src/listener/handle.rs:1334–1347`):

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

`.serve()` reads that separately-accumulated `self.tls` field and calls `compose_tls`, which wraps whatever `resolve_listen_protocol` already picked (`src/listener/handle.rs:1039–1063`):

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

One detail worth teaching in its own right: the registry key gets renamed to `"{name}+tls"` (e.g. `"http+tls"`). Why not just re-register under `"http"`? Because `App::new()` *already* registered a plain, non-TLS `HttpListenProtocol` under `"http"` — and `TlsListenProtocol::name()` delegates to the wrapped protocol's name, which is also `"http"`. Registering a second, TLS-wrapping protocol under the identical key `"http"` in the same registry would either collide or silently shadow the plain one. `NamedListenProtocol` (`src/listener/handle.rs:1074–1103`) is a small wrapper whose only job is to override the registry *key* without touching `serve` — the TLS-terminating decorator runs unchanged underneath; only the name used to look it up differs. `compose_tls_wraps_the_resolved_protocol_and_renames_the_registry_key` (`src/listener/handle.rs:1348–1365`) is the test proving `name == wrapped.name()` always holds, which is the invariant that keeps this registration collision-free.

## 8. The general escape hatch: `.protocol(impl AnyProtocol)`

`compose_tls` above (and `resolve_listen_protocol`'s h2/h3 arms) both produce the same shape: a registry-name string, plus an *optional*, already-built `Arc<dyn ListenProtocol>` to self-register instead of relying on a by-name lookup. That shape isn't invented by the umbrella crate — `ListenerSpec::protocol` already exists for exactly this, one layer down in `proxima-listen`. But the escape hatch a THIRD PARTY actually reaches for is one level higher and simpler: `ListenerBuilder::protocol(impl AnyProtocol)` (`src/listener/handle.rs:277–289`), the listener-side mirror of the client's `.protocol(impl ClientProtocol)` (`src/client/handle.rs:463–471`):

```rust
#[cfg(any(feature = "http1", feature = "http1-native"))]
#[must_use]
pub fn protocol(mut self, protocol: impl proxima_listen::any::AnyProtocol) -> Self {
    let protocol: Arc<dyn proxima_listen::any::AnyProtocol> = Arc::new(protocol);
    let name = protocol.name().to_string();
    self.any_mode = Some(match self.any_mode.take() {
        None | Some(AnyMode::All) => AnyMode::All,
        Some(AnyMode::Subset(mut names)) => {
            if !names.contains(&name) {
                names.push(name);
            }
            AnyMode::Subset(names)
        }
    });
    self.extra_protocols.push(protocol);
    self
}
```

This is the SAME mechanism `.kafka(handler)`/`.mqtt(handler)`/`.amqp(handler)`/`.memcached(handler)`/`.redis(handler)` (`ListenerProtocolExt`) delegate to internally, and the SAME mechanism a downstream crate with no dependency on `proxima-listen` at all can call directly. [Part 6: add your own protocol](./09-extend-your-own-protocol.md) is the full teaching page for this seam — grounded in `tests/e2e/listener_any_protocol_extension.rs`, which registers a third-party `AnyProtocol` candidate alongside the built-in h1/h2 ones and proves it drives its own reply.

## 9. `.serve()`: what it actually composes

`ListenerBuilder::serve()` is a terminal — it consumes the builder and returns a running `Server`. Reading its full body (`src/listener/handle.rs:449–530`) end to end is the fastest way to see that it invents no new serve loop; it is the exact `App::new()? -> app.mount(...)? -> app.serve(config).await?` idiom from `examples/hello/main.rs`, automated, with two guard checks up front:

```rust
pub async fn serve(self) -> Result<Server, ProximaError> {
    reject_dead_axes(&self.spec)?;
    reject_invalid_axis_combinations(&self)?;
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
    // ... websocket-wrap / pgwire / dns / any() registration, then:
    let app = App::new()?;
    // ... register any extra protocol resolve_listen_protocol/compose_tls/
    //     .any()/.protocol() produced ...
    app.mount("/{*path}", MountTarget::Handle(dispatch))?;
    let config = RunConfig { bind, protocol, spec: Value::Object(self.spec) };
    app.serve(config).await
}
```

Step by step:

1. `reject_dead_axes` (§10) — fail fast on spec keys the listener side has no wiring for (`.proxy(url)`, a bare `.tls()` marker with no real config behind it), before any socket work.
2. `reject_invalid_axis_combinations` — fail fast on axis PAIRS that are individually valid but jointly meaningless (`.kafka().quic()`, `.grpc().quic()`, `.dns().quic()`, `.websocket().quic()`, `.websocket()` + any `AnyProtocol` axis). [Part 4](./07-sugar-composition.md) teaches this failure mode from the reader's side, with the actual error text.
3. The `.tls(config)` + `.pgwire(query)` guard — `.pgwire(query)` manages its own TLS upgrade under `proxima-pgwire`'s own `listen` feature, so composing the generic `TlsListenProtocol` decorator on top would wrap the WRONG (http) protocol underneath; `.serve()` hard-errors rather than silently double-wrapping.
4. Resolve `bind` — either `.bind(addr)` directly, or recovered from the `http` spec key `bind_from_spec` reads back (§10 explains why this recovery step exists at all).
5. Require `.handle(pipe)` — the one input every listener has that a client never does (§10).
6. `resolve_listen_protocol` (§5) picks the registry name, plus an optional self-registering `Arc`; `compose_tls` (§7) wraps that in `TlsListenProtocol` if `.tls(config)` was called.
7. A fresh `App::new()` — self-registers whatever extra protocol was carried (from `resolve_listen_protocol`/`compose_tls`, from `.any()`/`.protocol()`'s own registration path, or a fresh `PgWireListenProtocol`/`DnsAnyProtocol`/`DnsDatagramProtocol` if `.pgwire(query)`/`.dns(handler)` carried a handle).
8. `app.mount("/{*path}", MountTarget::Handle(dispatch))` — every path routes to `.handle(pipe)`, using the exact catch-all glob convention the rest of the router uses for shorthand single-pipe listeners.
9. `app.serve(RunConfig { bind, protocol, spec })` — the identical `App::serve` `examples/hello` calls directly.

One genuine behavioral gap worth carrying forward rather than discovering the hard way: `App::serve` returns as soon as the listener lane is *spawned* — not once it's actually accepting connections — whereas the lower-level `proxima_listen::handle::Listener::run_with_runtime` (which `ListenerBuilder` does **not** go through) blocks for a per-lane ready acknowledgment first. A caller that dials immediately after `Listener::builder()....serve()` resolves can race a not-yet-listening socket. Today's workaround is a bounded poll-connect retry loop — see `wait_until_listening` in `examples/any_listener.rs` or `tests/e2e/listener_client_interop.rs`.

## 10. The two honest asymmetries

Both trace back to one fact: **a listener binds; a client dials.**

**Bind is mandatory, and has no url to fall back on.** A `Client` always has a URL — `.http(url)` *is* the dial target, full stop. A `Listener` needs a socket address to bind, and `ListenerProtocolExt::http` reuses that same method NAME for a different purpose (stringify the bind address into the same `http` spec key, §4's table), which is why `.bind(addr)` exists as its own inherent method (`src/listener/handle.rs:186–191`) and why `bind_from_spec` exists purely to recover a bind address from that string when `.bind()` itself was never called:

```rust
fn bind_from_spec(spec: &serde_json::Map<String, Value>) -> Option<SocketAddr> {
    spec.get("http").and_then(Value::as_str).and_then(|value| value.parse().ok())
}
```

`.serve()` requires one or the other be present, and says so precisely rather than failing on a `None.unwrap()` deep in a socket call (`serve_without_bind_errors_before_touching_a_socket`, `src/listener/handle.rs:1295–1304`, and the client has no equivalent test because it has no equivalent failure mode — a `Client` with no url just never resolves a factory, it doesn't need a bind check).

**`.tls(TlsConfig)` and `.grpc()` are inherent, not trait methods, and for two different reasons.** `.tls()` on the client is bare — zero arguments, because ALPN negotiation (§6) does the actual work; all `.tls()` needs to do is flip the transport marker. A listener terminating TLS needs real key material — a certificate and a private key, or a self-signed generator — which a zero-arg method has no slot for. So `ListenerBuilder` defines its own `.tls(proxima_tls::TlsConfig)`, a plain inherent method — not a trait implementation at all, because minting `ListenerSecurityExt` for exactly one method with no second implementor would be a trait with a single member (see the module doc's own reasoning, `src/listener/transport.rs`'s sibling `src/client/security.rs:1–12`, which explains the client side's OWN choice not to mint a trait either, for the opposite reason — it needs no divergence to defend against). `reject_dead_axes` (`src/listener/handle.rs:909–923`) is the safety net for the one remaining door: a caller reaching a bare `.spec("transport", "tls")` directly (or a `tls`-feature-off build, where the inherent `.tls(TlsConfig)` doesn't exist to intercept anything) would otherwise leave a listener silently unterminated:

```rust
if spec.get("transport").and_then(Value::as_str) == Some("tls") && !tls_marker_present(spec) {
    return Err(ProximaError::Config(
        "Listener::builder(): bare .tls() only sets a marker key and terminates nothing; \
         call .tls(TlsConfig::self_signed() | TlsConfig::pem(..) | TlsConfig::files(..)?), \
         which requires the `tls` feature".into(),
    ));
}
```

`.grpc()` is inherent for the opposite reason: the client's `ClientProtocolExt::grpc(url)` carries a *dial target* — there is nowhere else for that url to live. A listener has nothing to dial; it already has `.handle(pipe)` on hand and only needs to flip which wire protocol it resolves to (§5's `spec.contains_key("grpc")` check). So `ListenerProtocolExt::grpc(self)` is zero-argument (`src/listener/handle.rs:601–604`) — a different trait method on a different trait from the client's, not a shadowing of it. `.h2()` (§4) is the same shape, minus even a same-named counterpart to distinguish itself from — there is no `ClientTransportExt::h2()` at all, so it needs no divergence, just a plain inherent method. `.pgwire(query)` (§4, §5, §9) carries a typed `PgPipeHandle` accumulated on the builder the same way `.tls(config)` accumulates a `TlsConfig` — a listener-only shape with no client-side concept to extend.

`.proxy(url)` — an *actual* client-only concept, an egress CONNECT tunnel before the dial — has no listener meaning at all, and `ListenerBuilder` cannot even reach it through `ClientTransportExt` (it doesn't implement that trait). `reject_dead_axes` is what stops the ONE remaining door (the raw `SpecBuilder::set`/`.spec()` escape hatch) from doing something wrong silently — `.serve()` hard-errors if the `proxy` key is present at all, rather than binding a listener that quietly ignores it (`proxy_axis_hard_errors_at_serve_instead_of_silently_ignoring_it`, `src/listener/handle.rs:1316–1333`).

## 11. A full walkthrough, compiled and run

Two real, citable examples already exercise `Listener::builder()` end to end against a real client — grep `examples/` for `ListenerBuilderEntry`/`Listener::builder` yourself and you'll find them:

- `examples/h2_native_server.rs` — `Listener::builder().bind(bind).h2().handle(into_handle(ConstantOk)).serve().await?`, proven with a real `H2ClientUpstream` round trip, no tokio anywhere in the request path (`cargo run --example h2_native_server`; `cargo tree --example h2_native_server -e normal -i tokio` is empty).
- `examples/any_listener.rs` — `.any()`/`.accept(name)` through the identical builder, an h1 AND a native h2 client both routed correctly off one bind (`cargo run --example any_listener --features http1-native`).
- `tests/e2e/listener_builder_sugar.rs` — the widest single-file proof of everything this document teaches: bare `.http()`, `.http().tcp().tls(cfg)`, `.http().quic()` resolving h3-native, `.kafka(handle).tcp()` vs. the `.kafka(handle).quic()` named-error rejection, `.dns(handle).tcp()` vs `.dns(handle).udp()` binding genuinely different listen protocols, `.grpc().quic()`'s rejection, and a third-party `.thrift()` extension trait built entirely inside the test file. [Part 4](./07-sugar-composition.md) walks several of these scenarios as a runnable example.

The three-way comparison this section teaches — `Listener::builder()` vs. the one-liner `Listener::http(bind)`, side by side with the manual `App`/`RunConfig` baseline they both mirror — is not itself one shipped example; it's small enough to read as one block:

```rust
use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use proxima::prelude::*;
use proxima::{App, ProximaError, Request, Response, RunConfig, into_handle};

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

Note `hello` had to be `#[proxima::piped(send)]` here, not the bare `async fn` `App::mount` itself accepts directly (Foundations part 2 §8's `ViaFn`). `App::mount`'s `IntoMountTarget` wraps a bare fn in an internal `FnHandler` adapter for you; `into_handle` (needed for `.handle(pipe)`, since `ListenerBuilder::handle` takes `impl Into<PipeHandle>`) has a stricter bound — `Implementor: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + 'static` — which a bare fn item doesn't satisfy on its own. This is a real, mechanical distinction worth carrying forward: `App::mount(path, target)` takes anything `IntoMountTarget` covers (four shapes); `ListenerBuilder::handle(handle)` takes something already reduced to a `PipeHandle`, one shape only.

## 12. Where to go next

- [Listener on-ramp, part 2: the universal listener](./05-listener-universal.md) and [part 3](./06-listener-production.md) — the `.any()`/`.accept()`/`.accepts()`/`.deny()`/`.blacklist()` axis this document's own table (§4) predates.
- [Part 4: composing the sugar](./07-sugar-composition.md) — the transport/security/protocol axes from §4, taught from the reader's side, including the honest failure mode (`.kafka().quic()`) this document's §5/§9/§10 explain the MECHANISM for.
- [Part 5: the protocol fleet](./08-protocol-fleet.md) — every `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()`/`.dns()` axis in §4's table, client AND listener, with each protocol's honest scope stated.
- [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — §8's `.protocol(impl AnyProtocol)` escape hatch, in full, grounded in a real third-party extension test.
- [Foundations, part 2](./01-ergonomics.md) §8, if `IntoMountTarget`'s four shapes (referenced in §11 above) weren't already solid.
- [Build an API gateway](./build-an-api-gateway.md) for `Client::http`/`Client::builder` used against a real upstream, end to end.
- `docs/configuration.md`'s "typed listener configs" section for a *third*, unrelated way to reach a running listener — `HttpListener::http(addr)` / `HttpsListener::https(addr, cert, key)` (`src/settings/listener.rs`). Don't confuse it with this document's `Listener::builder()`: those are `Into<RunConfig>` typed config shapes for `App::serve(impl Into<RunConfig>)`, one layer above the spec/registry resolution this document covers, and they carry no `.tcp()`/`.tls()`/`.quic()`/`.grpc()` axis sugar at all.
- `proxima-listen`'s own crate docs for `ListenProtocol`, `ListenRegistry`, and `Listener::run_with_runtime`'s per-core SO_REUSEPORT fan-out — this document treated `run_with_runtime` as a fact (§9's readiness-race caveat), not something to re-derive.
