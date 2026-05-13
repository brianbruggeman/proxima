# guiding principles — proxima-quic + proxima-h3 rewrite

## Crate consolidation note (2026-07)

Crates named below were folded into consolidated crates:

- `proxima-quic-proto` → `proxima-protocols::quic`
- `proxima-h2-codec` → `proxima-protocols::http2_codec`
- `proxima-h3` (std stack) → `proxima-http::http3`

References below use the pre-consolidation names as written at the
time. Full rename map:
[`docs/decomposition/consolidation.md`](../decomposition/consolidation.md).

This file is the QUIC-and-H3-specific overlay. The workspace-wide
principles previously lived in `pty-tester/docs/proxima-pty/guiding-principles.md`
but that file no longer exists in this checkout; pending consolidation
into a single workspace `guiding-principles.md`, the binding overlay
for this initiative is THIS file plus the rules-of-thumb encoded in
[`ai_docs/invariants.jsonl`](../../ai_docs/invariants.jsonl) (search
for `kind:5` entries with `applies_to` covering `quic-component` /
`h3-component` / `sans-io` / `hot-path`).

Plan file: [`i-need-a-full-adaptive-seal.md`](../../../../../.claude/plans/i-need-a-full-adaptive-seal.md)

## Workspace principles that bind especially hard here

- **Principle 1 — RISC reuse first.** AWS-LC-RS for crypto (aead, hkdf,
  digest, x25519). rustls (no_std + alloc, `quic` feature) for TLS 1.3.
  `arrayvec` + `heapless` for fixed-cap state. `http` crate (no_std +
  alloc) for HTTP semantics. an external crypto crate composes aws-lc-rs already.
  Mirror `proxima-h2-codec` patterns for the sans-IO codec shape; mirror
  `proxima-mqtt::decode_remaining_length` for varint scaffolding.
- **Principle 3 — no_std + alloc + alloc-free tiers.** Tier-3 (bare
  no_std + no alloc) is the **aspiration** for every leaf module. Goal:
  ≥60% of `proxima-quic-proto` module count compiles at tier-3.
- **Principle 4 — 1st-class config AND fluent builder.** Every config
  type (Endpoint, Server, Client, Transport, Congestion, Loss, Ecn,
  Datagram, Multipath, ZeroRtt; H3 Server, Client, Qpack) derives
  `Builder + Deserialize + Serialize + Settings + Validate + ConfigDisplay`.
  Parity test fixture verifies env-loader / builder / TOML / default.
- **Principle 5 — \*DK polymorphism in the reactor.** The UDP datagram
  source (C28) lands as a **new source kind** in `prime::os::reactor`,
  not a fork. Future kinds (pidfd, io_uring SQE, DPDK ring, SPDK queue,
  AF_XDP) become "implement `ReadinessSource` for X".
- **Principle 8 — multi-axis constraint generator.** `quic_impl` /
  `h3_impl` / `tls` structural axes added to `proxima-build::Profile`.
  `[quic]` + `[h3]` sizing axes added to `prime-runtime.toml`.
- **Principle 9 — real-world data in tests.** rcgen for live cert
  material; captured QUIC handshakes from Cloudflare / Google / quiche
  / s2n-quic / qlog corpora for wire-protocol parser tests.
- **Principle 11 — sans-IO: enum FSM, low/no alloc, extreme benching,
  extreme perf.** Binding from C1 onward. Six-clause gate, no exceptions.

## QUIC- and H3-specific axioms

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
`Poll<...>`. Drivable in principle by `prime`, `tokio`, `embassy`, or
any custom executor.

**Caveat (current state):** the `native` feature on both facades
depends directly on `tokio` today — the UDP accept loop in
`proxima-http/src/http3/native/listen.rs` and the per-connection loop in
`proxima-quic/src/native/listener.rs` use `tokio::net::UdpSocket` +
`tokio::spawn`. Replacing this with a `prime::os::reactor` Datagram
source is tracked in `docs/proxima-quic/edges.md` ("Tokio transitive
leak via prime's std feature (C31)") and is NOT a prerequisite for
the dual-surface decision below. The `TOKIO_FREE_FACADE_ENFORCE` gate
cell stays opt-in; running it today fails because tokio is in the
tree, by design, until prime grows the Datagram reactor source.

### H. Dual-surface: native AND quinn-compat ship together

`proxima-quic` and `proxima-h3` both expose two top-level features —
`native` (over the proto crates) and `quinn-compat` (over upstream
quinn / h3 / h3-quinn). Both default-on. Consumers select per build:

- `--features native --no-default-features` — tokio-coupled but
  quinn-free; runs through the sans-IO proto crates.
- `--features quinn-compat --no-default-features` — uses the upstream
  quinn ecosystem unchanged.
- Default — both available; consumer picks at runtime via
  `proxima-listen`'s spec.

There is no atomic-cutover commit that deletes `quinn` from the
workspace. The legacy bridge stays available indefinitely so external
consumers of the quinn ecosystem retain a supported path and so the
native facade can grow at its own pace (multipath, 0-RTT under load,
etc.) without forcing every downstream off quinn first.

The always-on `proxima-h3 quinn-free (native feature only)` CI cell
verifies that a `--features native` build of `proxima-h3` carries
zero quinn in its dep tree — so the two surfaces stay structurally
separable even though they ship together.

---

## How to use this file

When updating `discipline.md` or `edges.md` here, re-read both this
file and the workspace `guiding-principles.md` first. When proposing
a new component row, write down which workspace principles and which
QUIC-specific axioms it engages. When a subagent or future-you
proposes a shortcut, check it against workspace principles 1, 3, 6,
11 plus QUIC axioms A, C, D, F, G.

When the user adds new directives that affect QUIC/H3 specifically,
they get appended here AND the date is noted. Workspace-wide
directives go in the workspace file.
