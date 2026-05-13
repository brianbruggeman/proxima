# guiding principles — proxima-h3 (overlay)

Workspace principles previously lived in
`pty-tester/docs/proxima-pty/guiding-principles.md`, which no longer
exists in this checkout. Pending consolidation into a single workspace
doc, the binding overlay is the QUIC overlay
([`docs/proxima-quic/guiding-principles.md`](../proxima-quic/guiding-principles.md))
plus the `kind:5` entries in
[`ai_docs/invariants.jsonl`](../../ai_docs/invariants.jsonl). This
file adds H3-specific axioms only.

**Workspace principles that bind H3 work specifically** (cross-reference):

- **Principle 11 — Sans-IO state machine** (enum-shaped, low/no alloc,
  extreme benching). Every H3 frame codec + request state machine in
  C32-C38 satisfies this.
- **Principle 12 — Per-crate build-time-configurable constants.**
  `proxima-protocols/http3_codec.toml` mirrors the
  `proxima-protocols/quic.toml` pattern (both crate-root tomls now
  that h3/quic are modules of `proxima-protocols`, not standalone
  crates). All H3 caps (max_concurrent_requests, qpack table caps,
  blocked_streams) reference `crate::sized::*`.
- **Principle 13 — Skill budget per component.** QPACK
  encoder/decoder (C33/C34) needs `/algorithm-development` (HPACK-
  derived but with QUIC-specific blocked-stream handling). H3
  Extended CONNECT (C38) needs `/security-review` if it ships
  WebSocket support.
- **Principle 14 — Incumbent wins on correctness.** Every H3 wire
  test MUST source expected bytes from a named primary source
  (RFC 9114 / RFC 9204 appendix, captured packet from h2/h3
  upstream impl, etc.) — not from memory. Parity failures against
  the `h3` reference crate or RFC vectors are MY bug until proven
  otherwise via the 6-step debugging discipline.

## H3-specific axioms

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
`proxima-protocols::quic::datagram` module (C25). The H3-Datagram
quarter-stream-id mux lives in the H3 layer; the wire transport is the
QUIC DATAGRAM frame. No new transport-layer code in
`proxima-protocols::http3_codec` for this.

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
