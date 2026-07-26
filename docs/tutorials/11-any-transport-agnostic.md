# Listener on-ramp, part 8: any protocol, any transport

**Prerequisites:** [Part 2: the universal listener](./05-listener-universal.md) (`.any()`, `ProbeVerdict`, "the listener sniffs it") and [Part 6: add your own protocol](./09-extend-your-own-protocol.md) (`AnyProtocol`'s two questions — `probe`, `drive`). You should be comfortable authoring a candidate that recognizes a fixed literal and drives its own reply.

**You will:** register a UDP-datagram candidate alongside the built-in, TCP-only h1 candidate on the SAME `.any()` listener — one bind, one port number, and the caller never writes `.tcp()` or `.udp()` anywhere.

**New concepts (in order):** `AnyProtocol::wants_datagram()` (the one new question — everything else you already know from part 6) · why `probe`/`drive` did not have to change shape to gain this · the accept-step fan-in, and why it costs nothing when no candidate opts in · the honest OS constraint this hides but cannot eliminate.

Every code block below is real and compiles — the full, runnable file is `examples/any_transport_agnostic.rs` (`cargo run --example any_transport_agnostic --features http1-native`; verified tokio-free with `cargo tree --features http1-native -e normal -i tokio`, empty result). Both scenarios are also proven end to end as `#[proxima::test]`s in `tests/e2e/listener_any_transport_agnostic.rs`.

## 1. The problem this solves

Part 6 taught `AnyProtocol` as two pure questions: "is this prefix you?" (`probe`) and "drive this already-accepted stream" (`drive`). Neither question ever mentioned a transport — `probe` takes `&[u8]`, `drive` takes a `Box<dyn StreamConnection>`. But until recently, only ONE thing could ever produce that `Box<dyn StreamConnection>` for `.any()` to hand a candidate: a TCP accept. `AnyProtocol::drive`'s signature took `Box<dyn StreamConnection>`, and the only path that filled it in was `.any()`'s own TCP accept loop — so a candidate whose real wire is UDP (DNS being the obvious one) could never register with `.any()` at all. It needed its own, separate, UDP-only listener (`proxima_listen::stream::DatagramProtocol`), and a caller who wanted that protocol reachable over BOTH transports had to stand up two listeners and hope they agreed.

That restriction is gone. `.any()` now classifies over TCP AND UDP, on the SAME port number, through the IDENTICAL `probe`/`drive` contract — because classification was never really about transport to begin with. `probe` only ever looked at bytes. What changed is what feeds it those bytes.

## 2. The one new question: `AnyProtocol::wants_datagram()`

A candidate opts in with one method (`proxima-listen/src/any/probe.rs:265–287`):

```rust
fn wants_datagram(&self) -> bool {
    false
}
```

Default `false`. Every candidate you've already met — h1, h2, kafka, redis, `PingPongProtocol` from part 6 — never overrides it, and nothing about them changes: an all-`false` candidate set makes `.any()`'s bind behavior BYTE-IDENTICAL to before this method existed. No UDP socket is ever opened unless at least one registered candidate asks for one.

A candidate whose own wire is naturally connectionless — DNS, a custom datagram RPC — overrides it to `true`. That is the entire opt-in. `probe` and `drive` do not grow a new parameter, a new return type, or a datagram-flavored twin method. You write the SAME trait you wrote in part 6.

## 3. What does NOT change, and why that's the point

Re-read `ProbeVerdict` (`proxima-listen/src/any/probe.rs:91–118`) — nothing there is TCP-specific. `Match { consumed }` / `NeedMore { at_least }` / `No` describe a decision about accumulated bytes, never about how those bytes arrived. A datagram simply arrives WHOLE rather than incrementally: where a TCP-fed candidate might see 4 bytes, then 12, then the full frame across several `probe` calls, a UDP-fed candidate typically sees its entire message on the first call, because a `recv_from` either returns the whole datagram or nothing. `probe` doesn't need to know which happened — it just gets called again with a longer prefix, same as always, if it asks for more.

`drive` is the more interesting case, because it still receives a `Box<dyn StreamConnection>` — the SAME stream-shaped handle. For a UDP-sourced connection, this is a one-shot adapter over the single already-received datagram: one `read` returns the whole message, the next `read` is `Ok(0)` (EOF) — a UDP message has no more bytes coming, and a stream reports "no more bytes" the same way. Every `write` your candidate makes before it closes the connection is buffered, not sent immediately — a request/reply protocol may write a response in more than one call (say, a header then a body), and UDP has no notion of a partial send, so those writes coalesce into exactly ONE outbound datagram, shipped back to the original sender the moment your `drive` closes the stream. `LiteralUdpProtocol` below never calls anything datagram-specific — it calls `read_to_end` then `write_all` then `close`, the exact three calls a TCP-fed candidate would make.

This is the teaching point worth sitting with: **the transport determines how bytes arrive; it never changes what classifying or driving means.** You already knew the whole mechanism from part 6. `wants_datagram` is the only new fact.

## 4. The worked example: one port, two transports, no transport call

`LiteralUdpProtocol` is the datagram-side twin of part 6's `PingPongProtocol` — same shape, one extra method:

```rust
struct LiteralUdpProtocol {
    name: &'static str,
    priority: u16,
    literal: &'static [u8],
    reply: &'static [u8],
}

impl AnyProtocol for LiteralUdpProtocol {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> u16 {
        self.priority
    }

    fn max_prefix_bytes(&self) -> usize {
        self.literal.len()
    }

    fn wants_datagram(&self) -> bool {
        true
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(self.literal.len());
        if prefix[..compare_len] != self.literal[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < self.literal.len() {
            return ProbeVerdict::NeedMore { at_least: self.literal.len() };
        }
        ProbeVerdict::Match { consumed: self.literal.len() }
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
            use futures::{AsyncReadExt as _, AsyncWriteExt as _};
            let mut received = Vec::new();
            stream.read_to_end(&mut received).await?;
            stream.write_all(self.reply).await?;
            stream.close().await?;
            Ok(())
        })
    }
}
```

Registered on the SAME `.any()` chain the built-in h1 candidate already answers on — `.protocol()` is exactly the escape hatch part 6 taught, no new method to learn:

```rust
let server = Listener::builder()
    .bind(bind)
    .handle(into_handle(LegitOk))
    .any()
    .protocol(LiteralUdpProtocol {
        name: "udpx",
        priority: 100,
        literal: b"UDPX/1\r\n",
        reply: b"UDPX/1 200 OK\r\nhello-from-datagram-candidate",
    })
    .serve()
    .await?;
```

Read that chain again: no `.tcp()`, no `.udp()`. Running it produces exactly this:

```
.any() on 127.0.0.1:52053: a TCP client still gets routed to h1 -> HTTP/1.1 200
.any() on the SAME 127.0.0.1:52053: a UDP datagram is classified and driven by LiteralUdpProtocol -> "UDPX/1 200 OK\r\nhello-from-datagram-candidate"
```

A plain `std::net::TcpStream` connect gets h1's real HTTP/1.1 response, exactly as in part 2. A plain `std::net::UdpSocket::send_to` at the IDENTICAL address gets `LiteralUdpProtocol`'s reply. Nothing downstream of `.any()`'s accept step — admission, the blacklist, the classifier itself — can tell which transport produced either connection.

## 5. Priority and ambiguity: the same rule you already trust

Part 2 taught the classifier's priority-ordered-wait rule for stream candidates: a lower-priority match is held back as long as a higher-priority candidate could still win, and two candidates tied at the SAME winning priority resolve to `ClassifyOutcome::AmbiguousMatch` (`proxima-listen/src/any/classifier.rs:39–68`) rather than an arbitrary pick. That rule is arbitration over `AnyProtocol` candidates, full stop — it was never specific to TCP, and registering four datagram candidates on one UDP socket exercises the identical code path:

```rust
let server = Listener::builder()
    .bind(bind)
    .handle(into_handle(LegitOk))
    .any()
    .protocol(LiteralUdpProtocol { name: "hipri", priority: 200, literal: b"HIPRI/1\r\n", reply: b"HIPRI-WINS" })
    .protocol(LiteralUdpProtocol { name: "lopri", priority: 100, literal: b"LOPRI/1\r\n", reply: b"LOPRI-WINS" })
    .protocol(LiteralUdpProtocol { name: "tied-a", priority: 150, literal: b"AMBIG/1\r\n", reply: b"TIED-A-WINS" })
    .protocol(LiteralUdpProtocol { name: "tied-b", priority: 150, literal: b"AMBIG/1\r\n", reply: b"TIED-B-WINS" })
    .serve()
    .await?;
```

```
4 datagram candidates, one socket, on 127.0.0.1:52056: hipri/lopri each answer their own disjoint literal regardless of priority order
tied-a/tied-b share both a literal AND a priority: the datagram is DROPPED (no reply within 500ms), never routed to either one
```

`hipri`/`lopri` answer their own disjoint literals independently of registration or priority order. `tied-a`/`tied-b` share both a literal and a priority — the listener's own dispatch (`proxima_http::any_listener::classify_and_drive_plaintext`'s `ClassifyOutcome::AmbiguousMatch` arm, `proxima-http/src/any_listener.rs:1484–1494`) logs the collision and drops the datagram rather than guessing. This is reachable from EITHER transport `.any()` binds — a stream candidate set with a genuine priority collision behaves identically.

## 6. The honest constraint this hides but cannot eliminate

**One port number, two sockets, invisible to the caller.** `.any()`'s own doc says this plainly (`proxima-listen/src/any/probe.rs:265–271`): a candidate that opts into `wants_datagram` needs `.any()` to ALSO bind a UDP socket on the SAME port number for it to be reachable. TCP:N and UDP:N are two distinct sockets at the OS level — different protocol, different kernel object, different bind call — and this API hides that seam behind one `.bind(addr)`, but it cannot make the two sockets into one. When no candidate wants a datagram, `.any()` binds exactly the one TCP socket it always did (`AcceptDriver::Plain`, internal — see §8); the moment one does, it binds both (`AcceptDriver::Fanned`, also internal), racing a TCP accept against a UDP receive with `FanIn`/`Select` (`proxima-http/src/any_listener.rs:867–889`).

Two consequences worth stating plainly:

- **TLS + a datagram candidate is a config error, not a silent gap.** TLS assumes a multi-round-trip byte stream; a UDP-sourced connection is one already-complete datagram with no handshake to terminate. `.any()` refuses this combination at `.serve()` time rather than pretending to secure it (`proxima-http/src/any_listener.rs:940–948`).
- **`.quic()` is still rejected under `.any()`; `.udp()` is not.** QUIC multiplexes many logical connections over one UDP socket by Destination Connection ID (DCID) — a completely different demultiplexing mechanism from this byte-prefix classifier, out of scope here (`src/listener/handle.rs:683–698`). `.udp()`, by contrast, no longer means anything to `.any()` at all: a registered candidate's own `wants_datagram()` already decides whether a UDP socket gets bound, so pairing `.any()` with `.udp()` is redundant, never an error.

## 7. The concrete payoff: `.dns(handler)` needed no branch at all

[Part 4](./07-sugar-composition.md) §5 and [Part 5](./08-protocol-fleet.md) §2 used to describe `.dns(handler)` as "the one dual-transport axis" — `.serve()` read `spec["transport"]` and picked ONE of two non-composable listen protocols, a TCP `AnyListenProtocol` or a standalone UDP `DatagramProtocolListenProtocol`. That branch is retired (`src/listener/handle.rs:536–559`): `.dns(handler)` now registers `proxima_dns::DnsAnyProtocol` (DNS-over-TCP) and `proxima_dns::DnsUdpAnyProtocol` (DNS-over-UDP, `wants_datagram() == true`) as two ordinary `AnyProtocol` candidates under one `.any()`-fanned listener. `DnsUdpAnyProtocol` is not internal plumbing — it is the exact pattern this page just taught you, written by the same crate that ships `DnsAnyProtocol`, registered the same way `LiteralUdpProtocol` was registered above. If you write your OWN datagram-shaped protocol, `DnsUdpAnyProtocol`'s source (`proxima-dns/src/udp_any_protocol.rs`) is a second, real-world worked example beyond this page's.

## 8. What's internal here, named only so you can trace it, never to `use`

Two things make this work under the hood, and neither is public API — you cannot `use` them, and you never need to:

- `proxima_http::any_listener::AcceptSource` (private, `proxima-http/src/any_listener.rs:697`) — the fan-in's two variants, `Tcp` and `Datagram`.
- `proxima_http::any_listener::DatagramAsStream` (private, `:777`) — the one-shot `StreamConnection` adapter §3 described.
- `proxima_http::any_listener::AcceptDriver` (private, `:867`) — `Plain` (the byte-identical TCP-only path) or `Fanned` (`FanIn<AcceptSource, Select, 2>`), chosen once at `.serve()` time from whether any candidate's `wants_datagram()` is `true`.

Everything you can actually call is public and already taught: `AnyProtocol` (with its new `wants_datagram` method), `Listener::builder().any().protocol(..)`, and the sugar (`.dns(handler)`) that uses this underneath.

## What's next

- [Part 2: the universal listener](./05-listener-universal.md) and [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — if any of `probe`/`drive`/priority/`.protocol()` above felt unfamiliar, these are the pages that taught them first.
- [Part 4: composing the sugar](./07-sugar-composition.md) §5 and [Part 5: the protocol fleet](./08-protocol-fleet.md) §2 — `.dns(handler)` as a caller sees it, now corrected to match this page's mechanism.
- `tests/e2e/listener_any_transport_agnostic.rs` — the same two scenarios this page walked, as `#[proxima::test]` assertions.
- `proxima-dns/src/udp_any_protocol.rs` — a second, real-world `wants_datagram` implementation beyond this page's `LiteralUdpProtocol`.
