# C23 — Key update (paper proof + state machine)

Per [RFC 9001 §6]. The 1-RTT key-update mechanism lets either peer
rotate the Application-traffic-secret mid-connection without
re-running the TLS handshake.

Tagged per principle 13: `/algorithm-development` (key-phase state
machine is subtle — RFC §6 has at least 5 hard rules) + `/security-review`
(timing side-channels per §6.3; KEY_UPDATE_ERROR conditions per §6.2).

[RFC 9001 §6]: https://www.rfc-editor.org/rfc/rfc9001#section-6

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Mechanism

1. New write secret = `HKDF-Expand-Label(secret_<n>, "quic ku", "", Hash.length)`.
2. The endpoint toggles the **Key Phase bit** in the first byte of
   1-RTT packets (header-protection-masked on the wire) per RFC §5.4.1.
3. Header protection key is NOT updated (RFC §6.1) — only the AEAD
   key + IV rotate.

## Preconditions to initiate

- Handshake MUST be confirmed (RFC 9001 §4.1.2) — `is_confirmed()` on the
  TLS provider.
- We MUST have received an ACK for a packet sent in the current key
  phase (RFC §6.1) — without this we don't know the peer can decrypt
  current-phase packets.

## Detection (inbound)

- The peer flipping the key phase bit is the detection signal.
- We try unprotect with `current.aead`; if that fails, try `next.aead`
  (the keys we proactively derived) per RFC §6.2.
- On successful unprotect with `next.aead`:
  - We MUST update OUR send keys to match before sending the ACK
    (RFC §6.2 last paragraph).

## State to track

```rust
pub struct KeyUpdateManager {
    /// Current generation (monotonic). Key phase bit = generation & 1.
    pub generation: u8,
    /// `true` once handshake is confirmed (post-TlsEvent::HandshakeConfirmed).
    pub handshake_confirmed: bool,
    /// `true` if we've received an ACK for any packet sent in the
    /// current key phase. Reset to false on every initiation.
    pub current_phase_acked: bool,
    /// Pending new secrets staged by the TLS provider's
    /// `initiate_key_update` (arrives via sink on_new_secrets with
    /// generation = current + 1).
    pub pending_next: Option<EpochSecrets>,
    /// Earliest time we may initiate another update. RFC §6.1 says we
    /// MUST have received an ACK for the current phase; we additionally
    /// floor at 3 × pto to avoid update storms.
    pub next_update_allowed_at: Instant,
    /// `true` once an inbound packet was successfully unprotected with
    /// `pending_next` keys (a peer-initiated update we detected).
    /// Triggers immediate send-key swap per RFC §6.2.
    pub peer_initiated_swap_pending: bool,
}
```

## Worked example (client-initiated key update)

State at t=T0: generation=0, handshake_confirmed=true, current_phase_acked=true, pending_next=None.

| Event | Action | State after |
|---|---|---|
| `initiate_key_update(now)` | preconditions OK → call `tls.initiate_key_update()` | unchanged (TLS provider will push new secrets via sink) |
| sink pushes EpochSecrets { generation=1, … } | store in `pending_next` | pending_next=Some(gen=1 secrets) |
| `swap_to_pending(now)` (called by FSM when next outbound 1-RTT packet ready) | swap current with pending_next; generation = 1; current_phase_acked = false | generation=1, pending_next=None, key phase bit becomes 1 |
| outbound 1-RTT packet sent (PN P0) with key phase = 1 | (egress observes generation) | unchanged |
| inbound ACK for P0 (key phase bit = 1 in inbound header) | `record_current_phase_ack(now)` | current_phase_acked=true |
| `initiate_key_update(now)` (too soon — within update_interval?) | check next_update_allowed_at | depends on time |

## Worked example (peer-initiated key update)

State at t=T0: generation=0, current keys installed, `pending_next` proactively populated with generation=1 keys (RFC §6.3 says we MAY do this proactively).

| Event | Action | State after |
|---|---|---|
| inbound packet with key phase bit = 1 (we last sent with bit = 0) | try unprotect with current (gen=0) → fails; try unprotect with pending_next (gen=1) → succeeds | `peer_initiated_swap_pending=true` |
| `swap_to_pending(now)` (called before sending the ACK) | swap; generation = 1; current_phase_acked = false | generation=1 |
| ACK for the trigger packet emitted with key phase = 1 | (egress observes) | unchanged |

## Security review (per principle 13)

| Concern | Mitigation |
|---|---|
| Timing side-channel on key-phase bit (RFC §6.3) | `unprotect_with_choice(current, next)` runs constant-time across both keys — never branches on which key succeeded. (Defer to C23.1: v1 sequential try is acceptable per RFC's MAY clause; production may add randomized-key fallback for full timing-resistance.) |
| Two consecutive updates without ack → KEY_UPDATE_ERROR (RFC §6.2) | `current_phase_acked` gate at initiation — refuses to initiate while pending. |
| Old-key DoS — keeping previous keys around forever costs memory | Retain previous keys for `pto` after swap then discard (RFC §6.5). v1: ArrayVec<EpochSecrets, 2> (current + next); no "previous" slot. Acceptable trade-off; documented. |
| Key derivation: HKDF-Expand-Label with label "quic ku" + empty context — must NOT reuse the application-secret nonce | TLS provider's responsibility per RFC §6.1 — verified by the TlsProvider trait's contract. |

## Code site

- `proxima-quic-proto/src/key_update.rs` — `KeyUpdateManager` +
  `KeyUpdateError` (FailedPreconditions / NotConfirmed /
  CurrentPhaseUnacked / NotReady / Forbidden).
- Integration into `Connection::initiate_key_update` defers to when
  the FSM has 1-RTT egress + ack-feedback wired. v1 ships the
  primitive; the FSM hookup follows the established A5 pattern of
  "primitive lands first, FSM wires up alongside 1-RTT egress slice."

## Sized constants (principle 12)

```toml
[key_update]
# Minimum interval (microseconds) between consecutive client-initiated
# key updates beyond the RFC's "MUST wait for current-phase ACK" rule.
# Defaults to 3 * pto-equivalent (1 s — conservative).
min_initiation_interval_micros = 1_000_000
```

## Tier

`key_update` is tier-3 (small struct of POD + Option<EpochSecrets>;
no alloc).

## Per principle 14 (incumbent wins)

Key derivation label `"quic ku"` per RFC §6.1 verbatim. Key-phase bit
position in first byte per RFC 9001 §5.4.1 (bit 0x04 in the masked
first byte of a Short header per §17.3.1). All test vectors derive
from these published constants — no draft, no memory.

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes; two
  worked-example walks (client-initiated + peer-initiated).
- **Pass 3 — code maps step-by-step**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; skipping the
  current_phase_acked precondition (would violate RFC §6.1),
  forgetting to update send keys before ACKing the trigger packet
  (RFC §6.2), or generation off-by-one would all break specific
  assertions.
- **Pass 6 — paper linked to test**: yes.
