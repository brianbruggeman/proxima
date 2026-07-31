---
name: guiding-principles
description: proxima-quic and proxima-h3 overlay axioms. The workspace-wide principles moved to slot-0/AGENTS.md and are now loaded into every session by the session-agents hook — this skill carries only the domain overlays on top of them. Use when working inside proxima-quic or proxima-h3, when the user says "what does axiom X say", "guiding principles", "north star", "what's our rule on X", or when a proposal in those domains needs testing against the binding rules. Also pulled in by `/disciplined-component`.
---

# guiding-principles

The 21 workspace principles live in `slot-0/AGENTS.md` under
`# guiding principles (binding)`. They are loaded into every session by
`~/.claude/hooks/session-agents`, so they bind whether or not this skill is
invoked — which is the point: a north star that has to be summoned bound
nothing outside `proxima/`.

This file is the domain layer. The proxima-quic and proxima-h3 axioms below
add to the workspace principles when working inside those domains; if a domain
axiom conflicts with a workspace principle, the workspace principle wins.

New directives go where they bind: workspace-wide ones are appended to
`slot-0/AGENTS.md` with a date, QUIC/H3-specific ones here. Do not copy
workspace principles back into this file — one source, no drift.

`/disciplined-component` composes with both layers: its 13-point gate is the
mechanical enforcement layer for principles 1, 3, 4, 11, 12, 16.

---

## proxima-quic axioms (overlay)

These axioms add on top of the workspace principles when working on
proxima-quic / proxima-h3 / TLS-bridge code. Workspace principles
that bind especially hard for QUIC work: 1, 3, 4, 5, 8, 9, 11, 15.

### A. RFC compliance is the spec; quinn/h3 are only the bench incumbents

We are NOT a quinn re-implementation. We conform to RFC 9000 / 9001 /
9002 / 9114 / 9204 / 9221 / 9297 + multipath draft. Where quinn or h3
diverges from spec, we follow spec and document the divergence in
`edges.md`. Where the spec leaves room for implementation choice
(scheduler design, ACK frequency, etc.), we pick a design and document
the rationale per component row.

`rfc-reference.md` is the source of truth for "which RFC section does
this implement". Every component row cites RFC §x.y.

### B. Sans-IO `Instant` and `Rng` are caller-owned

The proto crates cannot depend on `std::time::Instant` (tier-3 path
forbids std) and cannot bring their own RNG (would couple every
embedded user to a specific crate).

- `proxima_quic_proto::Instant` is a sealed `u64`-micros newtype with
  local `Duration` arithmetic. Caller passes one in per `poll`.
- `proxima_quic_proto::Rng` is a trait the caller implements (default
  blanket impl for `rand_core::CryptoRng` available behind a feature;
  no_std-friendly).

Locked down in C5 (`Rng`) and C11 (`Instant`) design passes — each via
`/research-rigor` self-play tournament.

### C. Connection state machine = discriminated enum (pattern A)

```rust
pub enum ConnectionState {
    Initial(InitialState),         // owns: client_dcid, initial_secrets, retry_token
    Handshake(HandshakeState),     // owns: handshake_secrets, peer_tp
    Established(EstablishedState), // owns: app_secrets, streams, ack, cc
    Closing(ClosingState),         // owns: close_frame, drain_deadline
    Draining(DrainingState),       // owns: drain_deadline only
    Closed,                        // unit
}
```

Transitions consume the old state and produce the new one. Misreaching
dead state is a compile error because the data isn't there.

Key update and handshake-only sub-flows use **typestate type
parameters** (pattern B) where exactly one path through the states
exists.

### D. No `Box<dyn Trait>` in the proto crates

Period. Trait objects acceptable in the I/O facade for runtime
polymorphism, not in proto. This means the congestion-control trait
is consumed via const-generic type parameter or via discriminated enum
`Congestion::NewReno(_) | Cubic(_) | Bbrv2(_)`, not via `Box<dyn
CongestionControl>`.

### E. Multipath + RFC 9221 + ECN are first-class, not feature-gated

The spec includes them; the implementation includes them. They are not
optional extensions in v1. The on-the-wire shape is shaped to support
them from C2 (packet header) and C3 (frame codec) onward.

### F. TLS 1.3 lives in proto, not the facade

`proxima-quic-proto::tls` houses the rustls + aws-lc-rs bridge (or the
inline TLS state machine, if the spike fallback fires). The std-tier
facade `proxima-quic` does NOT have a separate TLS path. Reason: the
sans-IO contract requires TLS to be drivable from any I/O loop, not
just our facade.

### G. Runtime-agnostic at the facade boundary

`proxima-quic` and `proxima-h3` expose `poll_*` methods returning
`Poll<...>`. Drivable by `prime` (production), `tokio` (compat-only,
feature-gated), `embassy`, or any custom executor. CI gate confirms
production builds have zero transitive tokio symbols.

### H. The cutover commit (Phase D2) is one atomic change

C41 deletes `quinn`, `quinn-proto`, `h3`, `h3-quinn` from the workspace
`Cargo.toml`, rewires `proxima/src/lib.rs` + `proxima/src/listeners/mod.rs`
+ `proxima-h3/src/{listener,server,upstream}.rs` + `proxima-h3/Cargo.toml`
+ `proxima-quic/Cargo.toml` + every profile TOML, and flips
`quic_impl=native` + `h3_impl=native` as workspace defaults. Either it
all lands or none of it does. Backtracking is via single git revert.

---

## proxima-h3 axioms (overlay)

These add on top of the workspace principles + the proxima-quic
overlay when working on H3 specifically.

### A. QPACK dynamic table is the only unbounded H3 state

Everything else in H3 is fixed-shape per request. The QPACK dynamic
table can grow up to `qpack_max_table_capacity` (sized via
`prime-runtime.toml [h3] qpack_max_table_capacity`). Use
`heapless::IndexMap<N>` with the cap from sized.rs at tier-1; alloc
allowed only if compile-time cap doesn't fit. Document the choice in
C33 / C34 design pass.

### B. H3 request state machine = typestate (pattern B)

Each request follows exactly one path:

```
Idle → HeadersSent → BodyStreaming → TrailersSent → Done
```

or, on the response side:

```
Idle → HeadersReceived → BodyReceiving → TrailersReceived → Done
```

Typestate type parameters enforce the transition at compile time. The
server-side connection (C35) owns a typed request table keyed by
`StreamId`; the client-side connection (C36) owns the symmetric one
for outbound requests. Forbidden: a runtime `state: RequestState` enum
field accessed via `match` on every operation.

### C. H3 frame codec mirrors the QUIC frame codec shape

Sans-IO `parse(&[u8]) -> Option<(Frame<'_>, usize)>` with borrowed
views; encode into `&mut [u8]`. Tier-3 target. No `Vec` in the parser
hot path.

### D. H3 SETTINGS exchange is a one-shot at connection open

After the SETTINGS frames exchange in both directions (RFC 9114 §7.2.4),
the negotiated settings are immutable for the connection lifetime.
Encode this as a typestate transition on the connection itself —
`H3Connection<Negotiating>` → `H3Connection<Established>`. The
`Established` form is the only one with stream-open / request-send
methods.

### E. H3-Datagrams (RFC 9297) compose with RFC 9221, not a separate path

C37 implements H3-Datagrams by sitting on top of the
`proxima-quic-proto::datagram` module (C25). The H3-Datagram quarter-
stream-id mux lives in the H3 layer; the wire transport is the QUIC
DATAGRAM frame. No new transport-layer code in proxima-h3-proto for
this.

### F. Server push is implemented but defaults disabled

RFC 9114 §4.6 server push: the wire format is implemented (C35
emits PUSH_PROMISE; C36 receives + accepts/rejects). But the default
`H3 ServerConfig::server_push` is `Disabled`. Push has dubious
real-world value; we ship the wire support so consumers can opt in.

### G. Extended CONNECT (RFC 9220) is for future MASQUE + WebSocket

C38 implements extended CONNECT mostly for the future MASQUE +
WebSocket-over-HTTP/3 plumbing. v1 ships the wire support and the
state-machine hooks; downstream consumers (an MASQUE proxy or
WebSocket adapter crate) build on it.

---


---

## How to use this file

- When opening a new discipline log, `edges.md`, or design doc, state which
  workspace principles (`slot-0/AGENTS.md`) and which axioms below the new
  component engages, and how.
- When a subagent (or future-you) proposes a shortcut, check it against
  workspace principles 1, 3, 6, 11, and 15 at minimum. For QUIC work, also
  axioms A, C, D, F, G. For sans-IO, principle 11 binds in full.
- This file plus `slot-0/AGENTS.md` are the source of truth for the
  architectural rules of thumb and sans-IO discipline that
  `/disciplined-component` references. Do not duplicate the rules into the
  gate template.
