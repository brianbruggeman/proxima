# Listener on-ramp, part 6: add your own protocol

**Prerequisites:** [part 4: composing the sugar](./07-sugar-composition.md), [part 5: the protocol fleet](./08-protocol-fleet.md). You should be comfortable with `.kafka(handler)`/`.dns(handler)` as protocol axes, and with the idea that they "delegate to `.protocol()`" without yet knowing what that means.

**You will:** plug a protocol proxima has never heard of into the SAME open universal listener that classifies h1 vs h2, alongside the built-in candidates, with ZERO edits to this crate — and understand precisely WHY Rust allows this.

**New concepts (in order):** `AnyProtocol` (`probe` + `drive`) · `ProbeVerdict` · `Listener::builder().any().protocol(impl AnyProtocol)` · the one-line ext trait pattern · why first-party protocols (`.kafka()`, `.mqtt()`) are not special.

If you read one page in this whole series, make it this one. Every protocol in [part 5](./08-protocol-fleet.md) — kafka, mqtt, amqp, memcached, redis — is built on EXACTLY the mechanism this page teaches. The fleet is a demonstration of the seam, not a fixed menu bolted onto the crate from the inside.

Every code block below is copied verbatim from `examples/extend_protocol.rs` and `tests/e2e/listener_any_protocol_extension.rs` (the same shape, proven as a `#[proxima::test]`), and every printed line is the ACTUAL output of running the example:

```sh
cargo run --example extend_protocol --features http1-native
```
```sh
cargo tree --example extend_protocol --features http1-native -e normal -i tokio
# empty — tokio-free end to end, same as any_listener.rs
```

## 1. The two questions `AnyProtocol` answers

Recall from [part 2](./05-listener-universal.md) that `.any()` binds ONE socket and classifies each connection's own leading bytes against every registered candidate. Each candidate is asked exactly two questions:

1. **"Is this prefix you?"** — `probe(&self, prefix: &[u8]) -> ProbeVerdict`. Pure, sans-IO: no I/O, no allocation beyond the verdict itself. Answers `Match { consumed }`, `NeedMore { at_least }`, or `No`.
2. **"Drive this already-accepted stream."** — `drive(&self, stream, handler, spec, peer, admission) -> impl Future<Output = Result<(), ProximaError>>`, called exactly once, after `probe` won.

That's the whole trait (`proxima-listen/src/any/probe.rs:134–195`, re-exported as `proxima::AnyProtocol`/`proxima::prelude::AnyProtocol`):

```rust
pub trait AnyProtocol: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn priority(&self) -> u16 { 100 }
    fn max_prefix_bytes(&self) -> usize;
    fn probe(&self, prefix: &[u8]) -> ProbeVerdict;
    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>>;
}
```

`ListenProtocol` (the trait h1/h2/h3-native implement) owns a BIND and an accept loop — "run one socket." `AnyProtocol` owns neither — it's asked to classify a prefix, then drive ONE already-accepted stream. `Listener::builder().any()`'s single accept loop is the ONE thing that owns the socket; every registered candidate, first-party or not, is a peer under it.

The three parameters this page's example doesn't use, briefly, so nothing in the signature is a mystery: `handler` is your business pipe, type-erased to `AnyHandler` (an `Arc<dyn Any + Send + Sync>`) because different candidates want different handler SHAPES (h1/h2 want a `Request -> Response` pipe; a future redis-shaped candidate would want a `Frame -> Frame` one) — your own `drive` downcasts it back to whatever concrete type you expect. `spec` is the listener's accumulated JSON spec (the same `Value` `.spec(key, value)` writes into). `admission` is the listener-wide request/connection cap — call `admission.request_admit()` at your OWN protocol's natural request boundary (per-command, per-message, whatever your wire's unit of work is) if you want to participate in `max_in_flight_requests` shedding ([part 3](./06-listener-production.md) §4 teaches this from the caller's side).

`drive`'s signature returns `Pin<Box<dyn Future<...>>>` rather than an RPITIT `impl Future` — a deliberate, documented exception to proxima's box-free-by-default rule: `AnyProtocol` is an OPEN, unbounded dyn set (any downstream crate can add a candidate at runtime), so it has to be object-safe, and RPITIT methods aren't. This is the "open/unbounded dyn set" exception the house rules name explicitly, not a shortcut.

## 2. Author `PingPongProtocol` — exactly as a downstream crate would

`PingPongProtocol` below never imports `proxima_listen` directly — only `proxima::prelude` (for `AnyProtocol` + `ProbeVerdict`) and `proxima::{listen, stream}` (for the trait's own signature types: `ConnAdmission`, `AnyHandler`, `PeerInfo`, `StreamConnection`) — all reachable through the umbrella crate. A real third party never needs `proxima-listen` as a direct Cargo dependency either:

```rust
const PING: &[u8] = b"PINGPONG/1 PING\r\n";
const PONG: &[u8] = b"PINGPONG/1 PONG\r\n";

struct PingPongProtocol;

impl AnyProtocol for PingPongProtocol {
    fn name(&self) -> &str {
        "pingpong"
    }

    fn max_prefix_bytes(&self) -> usize {
        PING.len()
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(PING.len());
        if prefix[..compare_len] != PING[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < PING.len() {
            return ProbeVerdict::NeedMore { at_least: PING.len() };
        }
        ProbeVerdict::Match { consumed: PING.len() }
    }

    fn drive<'a>(
        &'a self,
        mut stream: Box<dyn StreamConnection>,
        _handler: proxima::listen::any::AnyHandler,
        _spec: &'a Value,
        _peer: Option<PeerInfo>,
        _admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            use futures::AsyncWriteExt as _;
            stream.write_all(PONG).await?;
            stream.close().await?;
            Ok(())
        })
    }
}
```

`probe` recognizes a fixed 17-byte literal — real protocols probe on something more structural (an h1 request line's method+space, h2's fixed 24-byte connection preface, kafka's length-prefixed header), but the SHAPE (compare against known bytes, ask for more, or reject) is identical no matter how simple or complex the real wire is.

## 3. Register it: `.any().protocol(candidate)`

```rust
let server = Listener::builder()
    .bind(bind)
    .tcp()
    .handle(into_handle(LegitOk))
    .any()
    .protocol(PingPongProtocol)
    .serve()
    .await?;
```

Two things happen here that matter: `.any()` accepts every FIRST-PARTY candidate this `App` registers by default (h1, h2 prior-knowledge); `.protocol(PingPongProtocol)` (`ListenerBuilder::protocol`, `src/listener/handle.rs:277–289`) adds `PingPongProtocol` to that SAME accepted set — it does not replace it, narrow it, or require a separate bind. Running this proves BOTH halves at once:

```
.any().ping_pong(PingPongProtocol) still routes legit h1 traffic on 127.0.0.1:54857
a PINGPONG/1 connection on the SAME 127.0.0.1:54857 is classified and driven by PingPongProtocol's own drive() -> "PINGPONG/1 PONG\r\n"
```

A legit h1 client dialing the SAME address still gets a real HTTP/1.1 response — registering an external candidate never shadows or narrows away the first-party set. A connection opening with the `PINGPONG/1 PING\r\n` literal is classified against `PingPongProtocol` and driven by ITS OWN `drive`, which this example authored, not proxima.

(`examples/extend_protocol.rs`'s own `main` actually calls `.any().ping_pong(PingPongProtocol)`, not `.any().protocol(PingPongProtocol)` directly — §4 is the one extra line that gets you there, and why it's legal to add.)

## 4. The one-line ext trait: making it read like a first-party axis

`.protocol(impl AnyProtocol)` already works standing alone (as shown above) — but a real crate usually wraps it in its own ext trait so `.kafka(handler)`-style call sites read the same way. This is the ACTUAL trait `examples/extend_protocol.rs` defines and uses:

```rust
trait PingPongExt: Sized {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self;
}

impl PingPongExt for ListenerBuilder {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self {
        self.protocol(protocol)
    }
}

// call site, in `main`:
let server = Listener::builder()
    .bind(bind)
    .tcp()
    .handle(into_handle(LegitOk))
    .any()
    .ping_pong(PingPongProtocol)
    .serve()
    .await?;
```

**Why this is legal Rust — the part worth understanding, not just copying:** Rust's orphan rule forbids `impl ForeignTrait for ForeignType` (implementing a trait you don't own, for a type you don't own) — but it's satisfied the moment EITHER the trait or the type is local. `PingPongExt` is defined right here, in your crate; `ListenerBuilder` is defined in proxima. The trait is local, so the impl is legal, even though neither side has ever heard of the other. This is the EXACT same idiom `ListenerProtocolExt::kafka`/`.mqtt`/`.amqp`/`.memcached`/`.redis` use internally (`src/listener/protocol.rs`) — those are not special-cased into the crate; they are the SAME pattern, written once, by the crate that happens to also be the umbrella crate. `tests/e2e/listener_builder_sugar.rs`'s own `third_party_sugar` module proves this identically, with a `TestThriftExt` trait built entirely inside a test file, working exactly like the shipped ones.

## 5. Why the fleet is a demonstration, not a fixed menu

Nothing in [part 5](./08-protocol-fleet.md) required proxima's own source to change to add a NEW protocol. `KafkaAnyProtocol`, `MqttAnyProtocol`, `AmqpAnyProtocol`, `MemcachedAnyProtocol`, `RedisAnyProtocol` are all ordinary `impl AnyProtocol for ...` blocks, living in their OWN crates (`proxima-kafka`, `proxima-mqtt`, …), each paired with exactly the same one-line ext trait pattern §4 just taught. If your protocol isn't in the fleet, you are not blocked — you are in the SAME position `proxima-kafka` was in before it existed: author an `AnyProtocol`, mint a one-line ext trait, publish your crate, `use your_crate::YourProtocolExt;`.

## What you built

A third-party wire protocol, registered onto the SAME open universal listener as h1/h2, with zero edits to any file in this crate — and the understanding of exactly why Rust allows it. This is the seam every protocol in [part 5](./08-protocol-fleet.md) is built on.

## Where to go next

- [Part 5: the protocol fleet](./08-protocol-fleet.md) if you arrived here first — see the same mechanism used five more times, for real wires.
- [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) §8 — the escape hatch one layer down (`ListenerSpec::protocol`), for the rarer case of a caller who needs to bypass `.any()`'s classifier entirely.
- `tests/e2e/listener_any_protocol_extension.rs` — the equivalent proof as a `#[proxima::test]`, if you want to see this pattern inside a test harness instead of a `main`.
