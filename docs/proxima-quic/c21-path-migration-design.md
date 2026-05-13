# C21 — Path migration + PATH_CHALLENGE/PATH_RESPONSE (paper proof)

Per [RFC 9000 §8.2] + [§9]. Path validation lets a connection
verify that its peer can receive packets at a NEW address before
migrating traffic to that path.

[RFC 9000 §8.2]: https://www.rfc-editor.org/rfc/rfc9000#section-8.2
[§9]: https://www.rfc-editor.org/rfc/rfc9000#section-9

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Scope

**C21 v1**:
- `PathChallenger` per-path state: outstanding 8-byte tokens we sent
  in `PATH_CHALLENGE` + the per-path validation status.
- On inbound `Frame::PathChallenge` — automatically respond with
  `PATH_RESPONSE` carrying the SAME 8-byte data per RFC 9000 §8.2.2.
  (The response is OPPORTUNISTIC and rides on the same path that
  delivered the challenge.)
- On inbound `Frame::PathResponse` — match the 8-byte data against
  outstanding challenges; on match, mark the path validated; on no
  match, log + drop.
- `Connection::initiate_path_challenge(rng)` — caller-initiated;
  generates a fresh 8-byte token (via the principle-13 `/research-rigor`
  Rng — `rand_core::CryptoRng + RngCore`), enqueues a PATH_CHALLENGE
  frame for the next outbound 1-RTT packet, records the token.

**C21 deferred**:
- Multi-path / connection migration with anti-amplification gating
  on the new path (RFC 9000 §9.4) — defers alongside C26 multipath.
- NAT rebind detection — depends on the I/O facade observing source-
  address changes; defers to facade work.
- 1-RTT egress of PATH_CHALLENGE/PATH_RESPONSE frames — defers to the
  1-RTT ingress/egress slice.

## Per principle 14 (incumbent wins on correctness)

PATH_CHALLENGE / PATH_RESPONSE frame format per RFC 9000 §19.17 +
§19.18 — already done in C3 (`Frame::PathChallenge { data: [u8; 8] }`
+ `Frame::PathResponse { data: [u8; 8] }`). The challenge-response
matching logic is well-specified (data must be byte-identical).

## Worked example

State: per-path challenger with two outstanding tokens.

| Event | Action | State after |
|---|---|---|
| `initiate_path_challenge(rng)` → token T1 | enqueue PATH_CHALLENGE(T1); record (T1, now) | outstanding=[T1] |
| `initiate_path_challenge(rng)` → token T2 | enqueue PATH_CHALLENGE(T2); record (T2, now) | outstanding=[T1, T2] |
| inbound PATH_CHALLENGE(C1) | auto-enqueue PATH_RESPONSE(C1) | response pending |
| inbound PATH_RESPONSE(T2) | match → remove T2; mark path validated | outstanding=[T1]; validated=true |
| inbound PATH_RESPONSE(T_unknown) | no match → log + drop | outstanding=[T1]; validated unchanged |
| timer fires past challenge expiry | declare T1 abandoned | outstanding=[] |

## Security review

- **Token unpredictability**: 8 bytes from a CryptoRng. RFC 9000 §8.2.1
  requires "the data MUST be unpredictable" — 8 bytes from a CSPRNG
  satisfies (~2^64 collision resistance).
- **Reflection attack**: an attacker could echo back a PATH_CHALLENGE
  the client sent to validate the attacker's spoofed source. Mitigation
  per RFC §8.2.2: PATH_RESPONSE must arrive at the path being
  validated; the FSM tracks per-path responses (the I/O facade tags
  inbound datagrams with the path identifier).
- **Token reuse**: per RFC §8.2.1, "An endpoint MUST NOT reuse a
  Path Challenge token across multiple PATH_CHALLENGE frames" —
  enforced by the CSPRNG (collisions are astronomically unlikely).
  Recorded tokens are matched-and-removed; no reuse.

## Rng integration (principle 13 decision)

Per the C19.1 deferred-edge decision: use `rand_core::CryptoRng +
rand_core::RngCore`. Caller passes `&mut R: CryptoRng + RngCore` to
`initiate_path_challenge`. C21 lands the `rand_core` dep (if not
already present from C19.1 — but we never landed C19.1 yet so this
is the first integration).

For test scaffolding, `rand_core::OsRng` works on host; tests inject
a deterministic `SeedableRng` (e.g. `rand_chacha::ChaCha8Rng`).

## Code site

- `proxima-quic-proto/src/path/mod.rs` — re-exports.
- `proxima-quic-proto/src/path/challenger.rs` — `PathChallenger`
  with outstanding-token tracking + validation state.
- `proxima-quic-proto/src/connection/mod.rs` — `initiate_path_challenge`
  method + `parse_and_apply_established` integration for inbound
  PATH_CHALLENGE / PATH_RESPONSE.

## Sizing (principle 12)

```toml
[path]
# Maximum outstanding PATH_CHALLENGE tokens awaiting response per
# path. RFC 9000 doesn't mandate; quinn uses ~8.
max_outstanding_challenges = 8
# Max paths tracked per connection (single-path v1; multipath C26
# raises this to match max_paths_per_connection from prime-runtime).
max_paths_per_connection = 1
```

## Tier

`path::*` is tier-3 (small bounded arrayvec + Copy POD).

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes;
  6-row worked example covers issue / inbound-challenge-response /
  match / no-match / expiry.
- **Pass 3 — code maps step-by-step**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; failing to enforce
  byte-equality on the 8-byte token, allowing token reuse, missing
  the "validated" flag flip would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
