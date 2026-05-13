# C20 — Anti-amplification (paper proof)

Per [RFC 9000 §8.1]. The DDoS-amplification mitigation: a server MUST
NOT send more than `3 × bytes_received_from_unvalidated_client` until
it has validated the client's source address.

This component was implemented incrementally across C11 (counter
primitive + InitialState wire-up) + C11.5 (Handshake-arrival
validation hook). C20 formalises the integration + paper-proves the
two binding invariants.

[RFC 9000 §8.1]: https://www.rfc-editor.org/rfc/rfc9000#section-8.1

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## State

`AntiAmplificationCounter` (in `crate::anti_amplification`) per
RFC 9000 §8.1:

- `received_from_peer: u64` — bytes inbound from the peer.
- `sent_to_peer: u64` — bytes we've sent.
- `address_validated: bool` — has the peer's address been validated?
  - **Client side**: `true` at construction. The client's own
    address is implicitly validated by the act of receiving the
    server's responses.
  - **Server side**: `false` at construction. Flips to `true` when:
    1. A Handshake-encrypted packet arrives from the client (RFC
       9000 §8.1 — receipt of a Handshake-AEAD-protected packet
       cryptographically validates the client's address); OR
    2. A successful address-validation token arrives (RFC 9000
       §8.1.3) — Retry-token verify path; C19.

## Invariants (binding)

**I1.** While `!address_validated`, `send_budget() = 3 × received - sent`,
saturating at zero.

**I2.** `address_validated` is a one-way latch. Once `true`, never
flips back to `false`. The counter's `send_budget()` returns
`u64::MAX` (effectively unlimited).

## Wire-up across the FSM

| State | record_received | mark_address_validated | can_send guard |
|---|---|---|---|
| InitialState (client) | `handle_initial_datagram` updates on inbound | always validated at construction | `poll_transmit_initial` checks `can_send(MIN_INITIAL_DATAGRAM_BYTES)` (no-op for client) |
| HandshakeState | `handle_handshake_datagram` updates on inbound + calls `mark_address_validated` per RFC §8.1 | flipped by `handle_handshake_datagram` | n/a (validated) |
| EstablishedState onward | n/a (validated) | n/a | n/a |

For the client side, the counter is essentially redundant (client
never has a budget constraint). The real value emerges when the
**server-side connection** lands — at that point the same counter
gates outbound transmits from the server until the first
Handshake-encrypted client packet arrives.

## Worked example (server side — for the eventual server connection)

State at t=0 (server-side Initial): `received=0, sent=0, validated=false`.

| Event | Action | Counter after |
|---|---|---|
| Client Initial 1200 B inbound | `record_received(1200)` | rcvd=1200, sent=0, send_budget=3600 |
| Server tries to send 1200 B | `can_send(1200)` ✓; `record_sent(1200)` | rcvd=1200, sent=1200, send_budget=2400 |
| Server tries to send 3000 B | `can_send(3000)` ✗ (only 2400 available) | unchanged |
| Server sends 2400 B | `can_send(2400)` ✓; `record_sent(2400)` | rcvd=1200, sent=3600, send_budget=0 |
| Server tries to send 1 B | `can_send(1)` ✗ | unchanged (blocked) |
| Client Initial 1200 B inbound | `record_received(1200)` | rcvd=2400, sent=3600, send_budget=3600 |
| Handshake-encrypted client packet inbound | `record_received(...)` + `mark_address_validated()` | latched: send_budget=u64::MAX |
| All subsequent sends | `can_send(_)` ✓ always | no constraint |

The 3× ratio prevents the server from being weaponised as a
reflective DDoS amplifier — a spoofed-source attack maxes out at
3× amplification, well below the typical ratios attackers exploit
(DNS reflection ~50×, NTP monlist ~500×).

## Code site

- `proxima-quic-proto/src/anti_amplification.rs` — `AntiAmplificationCounter`.
- `proxima-quic-proto/src/connection/state.rs` — fields on
  InitialState + HandshakeState.
- `proxima-quic-proto/src/connection/mod.rs`:
  - `new_client` constructs with `Side::Client` (validated).
  - `handle_initial_datagram` / `handle_handshake_datagram` call
    `record_received`.
  - `handle_handshake_datagram` calls `mark_address_validated`
    per RFC §8.1.
  - `poll_transmit_initial` consults `can_send` before building.
  - `poll_transmit_initial` calls `record_sent` on successful
    build.

## What's NOT in C20 v1

- **Server-side connection construction** — defers to whenever
  server-side FSM lands. At that point a new `new_server()`
  constructor uses `Side::Server` (un-validated) and the existing
  counter behaviour kicks in.
- **Retry-token-based pre-validation** (RFC 9000 §8.1.3) — defers
  to C19 (retry tokens). On successful token verify, server-side
  `new_server()` constructs with `mark_address_validated()`
  pre-flipped.
- **Anti-amplification PADDING tax** — RFC 9000 §14.1 — server's
  initial response should be padded to maximise the 3× budget's
  effective use. Defers to the server-side egress path.

## Tier

`anti_amplification` module is tier-3 — single struct of u64 + bool
+ Side; pure functions; no allocations.

## Self-critique

- **Pass 1 — paper before code**: code preceded paper in this
  case (C11 wire-up was straightforward) but the paper formalises
  the invariants for future review + server-side wire-up.
- **Pass 2 — algorithm walk produces exact expected output**:
  yes; worked example covers the 6-step server budget exhaustion +
  latch path.
- **Pass 3 — code maps step-by-step to algorithm**: yes; each
  method on `AntiAmplificationCounter` corresponds to a named step.
- **Pass 4 — test uses exact inputs from worked example**: existing
  `anti_amplification::tests::server_send_budget_*` cover the
  arithmetic; the FSM-integration walk lives in C11.5's
  Handshake→Established test which asserts
  `mark_address_validated()` fires.
- **Pass 5 — would the test fail on bugs**: yes; flipping the
  `Side::Client`/`Server` boolean at construction would break the
  client-side "send_budget=∞" test; missing the `mark_address_validated`
  call would leave the budget bounded forever.
- **Pass 6 — paper linked to test**: yes; test docstrings
  reference RFC 9000 §8.1 + this paper.
