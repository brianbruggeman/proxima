# Listener on-ramp, part 4: composing the sugar

**Prerequisites:** [part 1: hello](./04-listener-hello.md), [part 2: the universal listener](./05-listener-universal.md), [part 3: production](./06-listener-production.md). You should be comfortable with `Listener::builder()...serve()`, `.any()`/`.accept(name)`, and reading a `ProximaError`.

**You will:** stop letting `.any()` sniff the wire for you, and pick one on purpose — composing three independent axes (transport, security, protocol) on the SAME `Listener::builder()`/`Client::builder()` chain — and see the exact error text proxima gives you when a composition genuinely has no meaning.

**New concepts (in order):** the three axis families (`ClientTransportExt`/`ListenerTransportExt`, `ClientSecurityExt`, `ClientProtocolExt`/`ListenerProtocolExt`) · `.http().quic()` as h3 · `ProximaError::Config` as the honest failure mode for an invalid composition.

Every code block below is copied verbatim from `examples/sugar_composition.rs`, or is that same real code wrapped in just enough signature scaffolding (a `SocketAddr` parameter instead of the real file's own `free_loopback_addr()?` helper, a real `async fn` in place of a `main`-body fragment) to type-check standalone — flagged inline, every time, right where it happens. Every printed line is the ACTUAL output of running the real file on this machine, in this repository — nothing here is invented.

```sh
cargo run --example sugar_composition --features "http1-native,tls,http3,kafka-listener,dns-listener"
```

## 1. `use proxima::prelude::*;` — one import, every axis

Every method this page teaches comes from one of a handful of small traits: `ClientTransportExt`/`ListenerTransportExt` (`.tcp()`/`.udp()`/`.quic()`), `ClientSecurityExt` (`.tls()`, client-side), and `ClientProtocolExt`/`ListenerProtocolExt` (`.http()`/`.grpc()`/`.kafka()`/…). `use proxima::prelude::*;` brings all of them into scope at once (`src/lib.rs:523–529`) — that's the only import this whole file needs beyond the specific types each section constructs:

```rust
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::tls::TlsConfig;
use proxima::{ProximaError, SendPipe};
```

There is **no** single blanket trait that gives you every method on every builder — `ListenerBuilder` and `ClientBuilder` each implement their OWN axis traits (a `ListenerBuilder` cannot call the client-only `.proxy(url)`, because it never implements `ClientTransportExt` at all). [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) is the deep dive on why; this page is about what you can DO with the axes, once you have them.

## 2. Transport: `.tcp()` / `.udp()` / `.quic()`

The default combiner (h1+h2 over TCP) needs no axis call at all — but calling `.tcp()` explicitly says so on the page rather than leaving it implicit. `FixedOk` (the handler every section on this page reuses) is defined here, once — every later section's excerpt gets it for free:

```rust
struct FixedOk;

impl SendPipe for FixedOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Ok(Response::ok("sugar-composition-ok"))
    }
}

// `bind_1: SocketAddr` here instead of the real file's own
// `free_loopback_addr()?` helper — everything else is unedited.
async fn tcp_axis(bind_1: SocketAddr) -> Result<(), ProximaError> {
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
    assert_eq!(response.status(), 200);
    server_1.stop();
    Ok(())
}
```

Running the real file (which calls this exact shape from `main`, then prints the result) produces exactly:

```
§1: .http().tcp() listener + .http(url).tcp() client -> 200
```

## 3. Security: `.http().tcp().tls(cfg)`

TLS is a SEPARATE axis from transport and protocol — it composes on top of whatever `.tcp()`/`.http()` already picked, as a decorator, not a spec key ([part 3 of Foundations](./02-listener-builder.md) §7 is the mechanism; this is the reader-facing shape). Both sides of `.tls()` show up here: the listener's real `TlsConfig`, and the client's own zero-arg assertion right after it:

```rust
async fn tls_axis(bind_2: SocketAddr) -> Result<(), ProximaError> {
    let server_2 = Listener::builder()
        .bind(bind_2)
        .tcp()
        .tls(TlsConfig::self_signed())
        .handle(into_handle(FixedOk))
        .serve()
        .await?;

    // The client's own `.tls()` (`ClientSecurityExt`) is the zero-arg twin —
    // no cert material, because ALPN negotiation, not the client, does the
    // actual work. `.build()` only accumulates the `transport: "tls"` spec
    // key and never touches a socket, so this composes without the client
    // needing to trust `server_2`'s self-signed cert (a real `.send()`
    // against it is out of scope here — see
    // `tests/e2e/listener_client_interop.rs` for a real handshake, done
    // with a raw `rustls` connector that accepts the test cert on purpose).
    Client::builder()
        .https(format!("https://{bind_2}"))
        .tls()
        .build()?;

    server_2.stop();
    Ok(())
}
```

```
§2: .http().tcp().tls(cfg) serves on 127.0.0.1:65025 — TLS terminates as a decorator over
    the SAME h1+h2 combiner, not a spec key (see tests/e2e/listener_client_interop.rs for
    the full handshake proof)
§2: .https(url).tls() client builds against 127.0.0.1:65025 — ClientSecurityExt::tls() only
    writes the transport: "tls" spec key; the assertion is checked on the first .send(), not
    here
```

`TlsConfig::self_signed()` (`proxima-tls/src/imp.rs:61`) generates a throwaway self-signed cert in-process — fine for this teaching example and for tests, never for a real deployment (`TlsConfig::pem(..)`/`TlsConfig::files(..)` load real cert material). The listener's `.tls(TlsConfig)` takes real key material because a listener terminates the connection; the client's own `.tls()` is bare — zero arguments, as the block above just proved — because ALPN negotiation, not the client, does the actual work. [Part 3 of Foundations](./02-listener-builder.md) §4's axis table has both sides side by side, with the citations for each.

## 4. `.http().quic()` IS h3 — there is no separate `.h3()` method

This is the single most important fact this page teaches: HTTP/3 is not a fourth protocol key alongside `http`/`grpc` — it's `.quic()`, composed onto `.http()`, on EITHER builder:

```rust
use serde_json::json;

async fn quic_axis(bind_3: SocketAddr) -> Result<(), ProximaError> {
    let server_3 = Listener::builder()
        .bind(bind_3)
        .quic()
        .spec("dev_self_signed", json!(true))
        .spec("dev_sans", json!(["localhost"]))
        .handle(into_handle(FixedOk))
        .serve()
        .await?;
    server_3.stop();
    Ok(())
}
```

```
§3: .http(bind).quic() resolves to the native h3-native DatagramProtocol listener on
    127.0.0.1:65027 — a real UDP bind, not the ALPN h1+h2 combiner. There is no separate
    `.h3()` method any more; `.quic()` on `.http()` IS h3.
```

If you've read an older piece of proxima teaching material that mentions `.h3()` as its own method — that's stale; report it. `.quic()` is the only spelling, and it works identically on `Client::builder().http(url).quic()` (dials h3-native) and `Listener::builder().quic()` (binds h3-native).

## 5. `.dns(handler)` — one call, both transports, one port

Real DNS resolvers speak both DNS-over-TCP and DNS-over-UDP on the SAME port
number. `.dns(handler)` used to make you pick — `.tcp()` (default) resolved a
TCP `AnyListenProtocol` candidate, `.udp()` a completely different
`DatagramProtocolListenProtocol` — genuinely two non-composable listen
protocols, chosen by which transport method you happened to chain. That
branch is gone: `.dns(handler)` now registers TWO `AnyProtocol` candidates
(DNS-over-TCP, RFC 1035 §4.2.2's 2-byte length prefix; DNS-over-UDP, RFC 1035
§4.2.1's raw message) under ONE `.any()`-fanned listener, and neither
`.tcp()` nor `.udp()` changes what gets bound:

```rust
use proxima_dns::{DnsAnswer, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, into_dns_handle};

async fn dns_axis(bind: SocketAddr) -> Result<(), ProximaError> {
    struct NameErrorDns;

    impl SendPipe for NameErrorDns {
        type In = DnsPipeRequest;
        type Out = DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
            Ok(DnsPipeReply::typed(200, DnsAnswer::name_error()))
        }
    }

    fn stub_handle() -> DnsPipeHandle {
        into_dns_handle(NameErrorDns)
    }

    let server = Listener::builder()
        .bind(bind)
        .handle(into_handle(FixedOk))
        .dns(stub_handle())
        .serve()
        .await?;
    // a real DNS-over-TCP query AND a real DNS-over-UDP query, sent to the
    // SAME address, both resolve — no .tcp()/.udp() call needed at all.
    server.stop();
    Ok(())
}
```

```
§4: .dns(handler) on 127.0.0.1:65028 answers BOTH a DNS-over-TCP query (id 7) and a
    DNS-over-UDP query (id 7) on the SAME port — no .tcp()/.udp() call needed
```

`.quic()` for DNS is still a config error (§6): DNS-over-QUIC (DoQ) is
unimplemented — that mechanism is genuinely absent, unlike the old
`.tcp()`/`.udp()` split, which was a design choice this crate no longer
makes you deal with. [Part 8: any protocol, any transport](./11-any-transport-agnostic.md)
is the deep dive on the mechanism `.dns(handler)` is built on
(`AnyProtocol::wants_datagram`) — worth reading before you write your OWN
datagram-shaped protocol candidate.

## 6. The failure mode: an invalid composition is a named error, never a silent degrade

Not every axis combination has a meaning. `.quic()` binds a UDP endpoint whose connections are demultiplexed by QUIC's own Destination Connection ID (DCID) — a completely different mechanism from the byte-prefix classifier `.kafka(handler)`/`.any()`/`.protocol()` are built on. Combining them is not "an inefficiency" or "an edge case that degrades gracefully" — it is a request for something that cannot exist, and `.serve()` says so BEFORE touching a single socket:

```rust
async fn kafka_quic_is_rejected(bind_bad: SocketAddr) -> Result<(), ProximaError> {
    use proxima_kafka::wire::{ApiVersionsResponse, RequestBody, ResponseBody};
    use proxima_kafka::{KafkaPipeHandle, into_kafka_handle};

    struct StubKafka;

    impl SendPipe for StubKafka {
        type In = RequestBody;
        type Out = ResponseBody;
        type Err = ProximaError;

        async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
            Ok(ResponseBody::ApiVersions(ApiVersionsResponse::supported()))
        }
    }

    fn stub_handle() -> KafkaPipeHandle {
        into_kafka_handle(StubKafka)
    }

    let outcome = Listener::builder()
        .bind(bind_bad)
        .quic()
        .handle(into_handle(FixedOk))
        .kafka(stub_handle())
        .serve()
        .await;
    assert!(outcome.is_err(), ".kafka(handler).quic() must not silently serve");
    Ok(())
}
```

This is the ACTUAL error text, printed by `examples/sugar_composition.rs` on this machine:

```
§5: .kafka(handler).quic() -> named ProximaError::Config:
    config: Listener::builder(): .kafka()/.mqtt()/.amqp()/.memcached()/.redis()/.any()/.accept()/.protocol() have no QUIC connection-demux support (QUIC multiplexes connections by DCID, a different mechanism from this byte-prefix classifier); use .tcp() (the default) — a registered candidate whose AnyProtocol::wants_datagram() is true is already reachable over UDP with no .udp() call needed
```

Notice what's absent from that text: `.udp()`. Unlike `.quic()`, pairing `.udp()` with `.kafka()`/`.any()`/`.protocol()` is no longer rejected — it is simply redundant, since a registered candidate's own [`AnyProtocol::wants_datagram`](./11-any-transport-agnostic.md) already decides whether `.any()` binds a UDP socket, with no `.udp()` call needed either way. `.quic()` stays rejected because it names a genuinely different, absent mechanism (QUIC's DCID demux), not a missing config flag.

Same story for gRPC — it rides h2, never h3, so `.grpc().quic()` is rejected identically:

```rust
async fn grpc_quic_is_rejected(bind_6: SocketAddr) -> Result<(), ProximaError> {
    let outcome = Listener::builder()
        .bind(bind_6)
        .quic()
        .grpc()
        .handle(into_handle(FixedOk))
        .serve()
        .await;
    assert!(outcome.is_err(), ".grpc().quic() must not silently serve");
    Ok(())
}
```

```
§6: .grpc().quic() -> named ProximaError::Config:
    config: Listener::builder(): .grpc()/.h2() + .quic(): gRPC rides h2, not QUIC; drop .quic() (the default h1+h2 ALPN combiner already carries h2)
```

Every invalid composition this crate knows about is rejected the SAME way — a `ProximaError::Config` naming the two axes in conflict and the fix, returned from `.serve()` before `bind()` or `App::new()` ever run (`reject_invalid_axis_combinations`, `src/listener/handle.rs:677`). There is no combination that silently downgrades to a "close enough" wire.

## What you built

Every section above ran against the SAME `Listener::builder()`/`Client::builder()` shape [part 1](./04-listener-hello.md) taught — no new serve loop, no new client type. You now have the full composition vocabulary: three independent axes, composing freely except where the source says explicitly why not.

## Where to go next

- [Part 5: the protocol fleet](./08-protocol-fleet.md) — every `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()`/`.dns()` axis this page only sketched, taught fully (client AND listener, honest scope per protocol).
- [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — the SAME `.protocol()` seam `.kafka()`/`.mqtt()`/… delegate to, reachable from a crate that never imports `proxima-listen`.
- [Part 7: conflaguration as first-class](./10-conflaguration.md) — once you've picked axes with the builder shape this page taught, this is the same knobs (a listener's admission/blacklist config, a protocol's own server config) expressed as a typed `Settings` struct or a TOML file instead of a fluent chain.
- [Part 8: any protocol, any transport](./11-any-transport-agnostic.md) — the mechanism §5's `.dns(handler)` is built on: how `.any()`'s classifier reaches a UDP-sourced connection through the identical `probe`/`drive` contract a TCP one uses, and how to opt your OWN `AnyProtocol` candidate in.
- [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) — the deep dive on WHY these are type-specific traits instead of one blanket one, and the exact source (`resolve_listen_protocol`, `reject_invalid_axis_combinations`) behind every behavior this page demonstrated.
