# protocol — add your own wire, with zero core edits

*(builds on: the pipe algebra, [dial/serve/run](../start/interfaces.md))*

Every protocol proxima ships — kafka, mqtt, amqp, memcached, redis — is built
on ONE open seam: `AnyProtocol`, a trait with two methods (`probe` + `drive`)
that lets a downstream crate register a new wire onto the SAME open
universal listener that already classifies h1 vs h2. The fleet is a
demonstration of this seam, not a fixed menu bolted onto the crate from the
inside — if your protocol isn't shipped, you are not blocked.

## The two questions

`Listener::builder().any()` binds one socket and classifies each
connection's own leading bytes against every registered candidate. Each
candidate answers exactly two questions — "is this prefix you?" (`probe`,
pure and sans-IO) and, once it wins, "drive this already-accepted stream"
(`drive`, exactly once):

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

`drive` returns a boxed future rather than an RPITIT `impl Future` — the one
deliberate exception to proxima's box-free-by-default rule that genuinely
applies: `AnyProtocol` is an OPEN, unbounded `dyn` set (any crate can add a
candidate at runtime), and RPITIT methods aren't object-safe.

## Author one, register it

The complete, compiled, runnable file below never imports `proxima-listen`
directly — only `proxima::prelude` and `proxima::{listen, stream}`, both
reachable through the umbrella crate. `PingPongProtocol::probe` recognizes a
fixed literal; a real protocol probes on something more structural (h2's
24-byte connection preface, kafka's length-prefixed header), but the SHAPE
is identical:

```rust
{{#include ../../../examples/extend_protocol.rs}}
```

Run it yourself:

```sh
cargo run --example extend_protocol --features http1-native
```

```
.any().ping_pong(PingPongProtocol) still routes legit h1 traffic on 127.0.0.1:54857
a PINGPONG/1 connection on the SAME 127.0.0.1:54857 is classified and driven by PingPongProtocol's own drive() -> "PINGPONG/1 PONG\r\n"

extend_protocol: a third-party AnyProtocol candidate works with zero core edits
```

Registering `PingPongProtocol` never shadows or narrows the built-in h1/h2
set — a legit h1 client dialing the SAME bind still gets a real HTTP/1.1
response. That's the proof this seam is additive, not a fork.

## Why the one-line ext trait is legal

`.any().protocol(PingPongProtocol)` already works standing alone. The file
above wraps it in one further line, `PingPongExt::ping_pong`, so the call
site reads exactly like `.kafka(handler)`:

```rust
trait PingPongExt: Sized {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self;
}

impl PingPongExt for ListenerBuilder {
    fn ping_pong(self, protocol: impl AnyProtocol) -> Self {
        self.protocol(protocol)
    }
}
```

Rust's orphan rule forbids `impl ForeignTrait for ForeignType` — but it's
satisfied the moment EITHER side is local. `PingPongExt` is local; the impl
for the foreign `ListenerBuilder` is legal. This is the SAME idiom
`ListenerProtocolExt::kafka`/`.mqtt`/`.amqp`/`.memcached`/`.redis` use
internally — not a simplified stand-in for the real mechanism, the real
mechanism itself, reused.

## What's next

- [the protocols chapter](../protocols/fleet.md) — five real wires built on
  exactly this seam, each with its own honest scope boundary.
- `docs/tutorials/09-extend-your-own-protocol.md` — this chapter's prose
  companion, with the full `AnyProtocol`/`ProbeVerdict` walkthrough.
- `tests/e2e/listener_any_protocol_extension.rs` — the equivalent proof as a
  `#[proxima::test]`.
