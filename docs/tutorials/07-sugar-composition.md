# Listener on-ramp, part 4: composing the sugar

**Prerequisites:** [part 1: hello](./04-listener-hello.md), [part 2: the universal listener](./05-listener-universal.md), [part 3: production](./06-listener-production.md). You should be comfortable with `Listener::builder()...serve()`, `.any()`/`.accept(name)`, and reading a `ProximaError`.

**You will:** stop letting `.any()` sniff the wire for you, and pick one on purpose — composing three independent axes (transport, security, protocol) on the SAME `Listener::builder()`/`Client::builder()` chain — and see the exact error text proxima gives you when a composition genuinely has no meaning.

**New concepts (in order):** the three axis families (`ClientTransportExt`/`ListenerTransportExt`, `ClientSecurityExt`, `ClientProtocolExt`/`ListenerProtocolExt`) · `.http().quic()` as h3 · `ProximaError::Config` as the honest failure mode for an invalid composition.

Every code block below is copied verbatim from `examples/sugar_composition.rs`, and every printed line is the ACTUAL output of running it on this machine, in this repository — nothing here is invented.

```sh
cargo run --example sugar_composition --features "http1-native,tls,http3,kafka-listener,dns-listener"
```

## 1. `use proxima::prelude::*;` — one import, every axis

Every method this page teaches comes from one of a handful of small traits: `ClientTransportExt`/`ListenerTransportExt` (`.tcp()`/`.udp()`/`.quic()`), `ClientSecurityExt` (`.tls()`, client-side), and `ClientProtocolExt`/`ListenerProtocolExt` (`.http()`/`.grpc()`/`.kafka()`/…). `use proxima::prelude::*;` brings all of them into scope at once (`src/lib.rs:418–423`) — that's the only import this whole file needs beyond the specific types each section constructs:

```rust
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::tls::TlsConfig;
use proxima::{ProximaError, SendPipe};
```

There is **no** single blanket trait that gives you every method on every builder — `ListenerBuilder` and `ClientBuilder` each implement their OWN axis traits (a `ListenerBuilder` cannot call the client-only `.proxy(url)`, because it never implements `ClientTransportExt` at all). [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) is the deep dive on why; this page is about what you can DO with the axes, once you have them.

## 2. Transport: `.tcp()` / `.udp()` / `.quic()`

The default combiner (h1+h2 over TCP) needs no axis call at all — but calling `.tcp()` explicitly says so on the page rather than leaving it implicit:

```rust
let server_1 = Listener::builder()
    .bind(bind_1)
    .tcp()
    .handle(into_handle(FixedOk))
    .serve()
    .await?;

let client = Client::builder()
    .http(format!("http://{bind_1}"))
    .tcp()
    .build()?;
let response = client.call("GET", "/").send().await?;
```

Running this produces exactly:

```
§1: .http().tcp() listener + .http(url).tcp() client -> 200
```

## 3. Security: `.http().tcp().tls(cfg)`

TLS is a SEPARATE axis from transport and protocol — it composes on top of whatever `.tcp()`/`.http()` already picked, as a decorator, not a spec key ([part 3 of Foundations](./02-listener-builder.md) §7 is the mechanism; this is the reader-facing shape):

```rust
let server_2 = Listener::builder()
    .bind(bind_2)
    .tcp()
    .tls(TlsConfig::self_signed())
    .handle(into_handle(FixedOk))
    .serve()
    .await?;
```

```
§2: .http().tcp().tls(cfg) serves on 127.0.0.1:55001 — TLS terminates as a decorator over
    the SAME h1+h2 combiner, not a spec key (see tests/e2e/listener_client_interop.rs for
    the full handshake proof)
```

`TlsConfig::self_signed()` (`proxima-tls/src/imp.rs:57`) generates a throwaway self-signed cert in-process — fine for this teaching example and for tests, never for a real deployment (`TlsConfig::pem(..)`/`TlsConfig::files(..)` load real cert material). The listener's `.tls(TlsConfig)` takes real key material because a listener terminates the connection; the client's own `.tls()` (§5 below) is bare — zero arguments — because ALPN negotiation, not the client, does the actual work.

## 4. `.http().quic()` IS h3 — there is no separate `.h3()` method

This is the single most important fact this page teaches: HTTP/3 is not a fourth protocol key alongside `http`/`grpc` — it's `.quic()`, composed onto `.http()`, on EITHER builder:

```rust
let server_3 = Listener::builder()
    .bind(bind_3)
    .quic()
    .spec("dev_self_signed", json!(true))
    .spec("dev_sans", json!(["localhost"]))
    .handle(into_handle(FixedOk))
    .serve()
    .await?;
```

```
§3: .http(bind).quic() resolves to the native h3-native DatagramProtocol listener on
    127.0.0.1:55003 — a real UDP bind, not the ALPN h1+h2 combiner. There is no separate
    `.h3()` method any more; `.quic()` on `.http()` IS h3.
```

If you've read an older piece of proxima teaching material that mentions `.h3()` as its own method — that's stale; report it. `.quic()` is the only spelling, and it works identically on `Client::builder().http(url).quic()` (dials h3-native) and `Listener::builder().quic()` (binds h3-native).

## 5. `.dns(handler).udp()` vs `.dns(handler).tcp()` — the one dual-transport axis

Every other protocol axis (`.kafka()`, `.mqtt()`, `.http()`, …) is single-transport: it either rides TCP always, or picks its OWN factory regardless of `.tcp()`/`.udp()`/`.quic()`. `.dns(handler)` is the one exception — it genuinely branches on the transport axis at `.serve()` time, binding a DIFFERENT kind of socket depending which you picked:

```rust
let server_tcp = Listener::builder()
    .bind(bind_tcp)
    .tcp()
    .handle(into_handle(FixedOk))
    .dns(stub_handle())
    .serve()
    .await?;
// a raw TCP connect succeeds here

let server_udp = Listener::builder()
    .bind(bind_udp)
    .udp()
    .handle(into_handle(FixedOk))
    .dns(stub_handle())
    .serve()
    .await?;
// a raw TCP connect to THIS bind must fail — it's a UDP socket
```

```
§4: .dns(handler).tcp() accepts a raw TCP connect on 127.0.0.1:55004
§4: .dns(handler).udp() refuses a raw TCP connect on 127.0.0.1:55006 — a genuinely
    different listen protocol from the .tcp() variant above, not the same socket
```

`.tcp()` (the default) resolves a TCP-shaped `AnyListenProtocol` candidate speaking DNS-over-TCP (RFC 1035 §4.2.2 framing); `.udp()` resolves a completely different `DatagramProtocolListenProtocol` wrapping `DnsDatagramProtocol` — classic DNS-over-UDP. Neither is "more correct" — real DNS resolvers speak both, on the same port number, over two different transports. `.quic()` for DNS is a config error (§6): DNS-over-QUIC (DoQ) is unimplemented.

## 6. The failure mode: an invalid composition is a named error, never a silent degrade

Not every axis combination has a meaning. `.kafka(handler)` delegates to `.protocol(impl AnyProtocol)`, and `AnyProtocol::drive` takes a `Box<dyn StreamConnection>` — a byte stream. `.quic()` binds a UDP-datagram socket, not a byte stream. Combining them is not "an inefficiency" or "an edge case that degrades gracefully" — it is a request for something that cannot exist, and `.serve()` says so BEFORE touching a single socket:

```rust
let outcome = Listener::builder()
    .bind(bind_bad)
    .quic()
    .handle(into_handle(FixedOk))
    .kafka(stub_handle())
    .serve()
    .await;
```

This is the ACTUAL error text, printed by `examples/sugar_composition.rs` on this machine:

```
§5: .kafka(handler).quic() -> named ProximaError::Config:
    config: Listener::builder(): .kafka()/.mqtt()/.amqp()/.memcached()/.redis()/.any()/.accept()/.protocol() are TCP-only (AnyProtocol::drive takes Box<dyn StreamConnection>); combining with .quic()/.udp() has no meaning — use .tcp() (the default)
```

Same story for gRPC — it rides h2, never h3, so `.grpc().quic()` is rejected identically:

```
§6: .grpc().quic() -> named ProximaError::Config:
    config: Listener::builder(): .grpc()/.h2() + .quic(): gRPC rides h2, not QUIC; drop .quic() (the default h1+h2 ALPN combiner already carries h2)
```

Every invalid composition this crate knows about is rejected the SAME way — a `ProximaError::Config` naming the two axes in conflict and the fix, returned from `.serve()` before `bind()` or `App::new()` ever run (`reject_invalid_axis_combinations`, `src/listener/handle.rs:690`). There is no combination that silently downgrades to a "close enough" wire.

## What you built

Every section above ran against the SAME `Listener::builder()`/`Client::builder()` shape [part 1](./04-listener-hello.md) taught — no new serve loop, no new client type. You now have the full composition vocabulary: three independent axes, composing freely except where the source says explicitly why not.

## Where to go next

- [Part 5: the protocol fleet](./08-protocol-fleet.md) — every `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()`/`.dns()` axis this page only sketched, taught fully (client AND listener, honest scope per protocol).
- [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — the SAME `.protocol()` seam `.kafka()`/`.mqtt()`/… delegate to, reachable from a crate that never imports `proxima-listen`.
- [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) — the deep dive on WHY these are type-specific traits instead of one blanket one, and the exact source (`resolve_listen_protocol`, `reject_invalid_axis_combinations`) behind every behavior this page demonstrated.
