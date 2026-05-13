# C24 — 0-RTT state machine + replay-protection policy

Per [RFC 9001 §4.6] (Enabling/Accepting/Rejecting 0-RTT) + §9.2
(Replay Attacks with 0-RTT) + TLS 1.3 [RFC 8446 §8].

Tagged per principle 13: `/algorithm-development` (the early-data
state machine has 4 transitions + a per-side asymmetry +
acceptance/rejection coupling that RFC §4.6.2 explicitly calls
"connection error of type PROTOCOL_VIOLATION") + `/security-review`
(0-RTT IS the replay-attack vector — RFC §9.2 dedicates an entire
section to it).

[RFC 9001 §4.6]: https://www.rfc-editor.org/rfc/rfc9001#section-4.6
[RFC 8446 §8]: https://www.rfc-editor.org/rfc/rfc8446#section-8

## Scope split

| Slice | Scope | Lands here |
|---|---|---|
| **C24.0** (this row) | Per-connection 0-RTT state machine + replay-protection policy + ticket carrier + 0-RTT-ACK-after-reject detection. **Wire-format ZeroRtt-epoch keying + Epoch enum extension defer.** | tier-3 v1 |
| C24.1 | `Epoch::ZeroRtt` enum variant + per-epoch routing through ack / packet_number / congestion / FSM dispatcher | defers — touches every switch-match on Epoch (substantial cross-cutting change; needs paired update) |
| C24.2 | `derive_zero_rtt_keys` in the TLS provider trait (per RFC §5.2.1 — client_early_traffic_secret) | defers; couples to C24.1's Epoch::ZeroRtt landing |
| C24.3 | Client retransmission of unACKed 0-RTT data as 1-RTT data after handshake confirmation (RFC §4.6.2 last paragraph) | defers to 1-RTT egress slice |

## C24.0 — 0-RTT state machine + policy

### Why this matters

Per RFC §9.2: "Disabling 0-RTT entirely is the most effective defense
against replay attack." But 0-RTT is also the QUIC feature applications
explicitly want for sub-RTT latency. The right shape is:
**make the choice explicit** at the connection layer via a typed
policy, so the application protocol's RFC-§9.2-mandated replay analysis
maps to a concrete value the runtime can enforce.

### Replay-protection policy

```rust
pub enum ZeroRttPolicy {
    /// 0-RTT MUST be used if the resumption ticket allows it.
    /// Caller (the application protocol) attests that it has
    /// implemented adequate replay-mitigation (e.g. HTTP/3 with
    /// idempotent-methods-only enforcement; CONNECT-UDP with
    /// per-tunnel session keying).
    Required,
    /// 0-RTT is permitted if the server accepts it. Caller has
    /// implemented per-RFC-§9.2 replay-mitigation analysis.
    Allowed,
    /// 0-RTT is disabled even if the resumption ticket would
    /// allow it. The most effective defense per RFC §9.2.
    Disabled,
}
```

Default is `Disabled` — RFC §9.2 explicitly recommends this as the
strongest defense. Caller opts in.

### State machine (per side)

```
                    ┌──────────────────┐
                    │   NotAttempted   │  (no resumption ticket)
                    └────────┬─────────┘
                             │ prepare_resumption(ticket, secrets, tps)
                             │ AND policy ∈ {Required, Allowed}
                             ▼
                    ┌──────────────────┐
                    │    Attempting    │  (client only — server sees this
                    └────────┬─────────┘   state on no event)
                             │
              ┌──────────────┴──────────────┐
              │                             │
   server     │                             │ server rejects
   accepts    ▼                             ▼ (EncryptedExtensions
   (early_   ┌──────────────────┐         ┌──────────────────┐  without early_data
   data ext  │     Accepted     │         │     Rejected     │  OR HelloRetryRequest)
   sent)     └──────────────────┘         └──────────────────┘
                                                   │
                                                   │ inbound ACK for any
                                                   │ 0-RTT packet
                                                   ▼
                                          PROTOCOL_VIOLATION
                                          (RFC §4.6.2 — "client SHOULD
                                          treat receipt of an
                                          acknowledgment for a 0-RTT
                                          packet as a connection error
                                          of type PROTOCOL_VIOLATION")
                                          → emit CONNECTION_CLOSE

                    ┌──────────────────┐
                    │     Disabled     │  (policy = Disabled, or no ticket)
                    └──────────────────┘
```

### Per-connection state

```rust
pub struct ZeroRttManager {
    policy: ZeroRttPolicy,
    status: ZeroRttStatus,
    /// Server-issued NewSessionTicket bytes from the previous
    /// connection. Opaque server-private; client passes verbatim
    /// in the next ClientHello's pre_shared_key extension.
    resumption_ticket: Option<ResumptionTicket>,
}

pub enum ZeroRttStatus {
    NotAttempted,                   // no ticket OR policy = Disabled
    Attempting,                     // client only; pending server EE
    Accepted,                       // server EE included early_data
    Rejected,                       // server EE omitted early_data OR sent HRR
    Disabled,                       // policy = Disabled, even with ticket
}
```

`ResumptionTicket` is a length-prefixed byte string capped at
[`MAX_RESUMPTION_TICKET_LEN`] (build-time const; default 1024 — covers
typical TLS 1.3 tickets).

## Worked example (client-initiated 0-RTT — accept path)

State at t=T0: fresh ZeroRttManager, no ticket, no prior connection.

| t  | Event | Action | State after |
|----|-------|--------|-------------|
| T0 | construct ZeroRttManager(policy=Allowed) | — | status=NotAttempted, policy=Allowed |
| T1 | prior session ends; server sent NewSessionTicket | caller calls `prepare_resumption(ticket_bytes)` | status=Attempting, ticket=Some(...) |
| T2 | gate check before emitting 0-RTT packet | `may_send_zero_rtt()` returns true | unchanged |
| T3 | client sends 0-RTT packet | (egress, no state change) | unchanged |
| T4 | server's EncryptedExtensions arrives with early_data extension | `note_server_accepted()` | status=Accepted |
| T5 | gate check for subsequent 0-RTT | `may_send_zero_rtt()` returns true (accepted) | unchanged |

## Worked example (client-initiated 0-RTT — reject path)

State at t=T0: ZeroRttManager(policy=Allowed), ticket prepared, status=Attempting.

| t  | Event | Action | State after |
|----|-------|--------|-------------|
| T0 | start | (Attempting) | status=Attempting |
| T1 | client sends 0-RTT packet PN=42 | (egress) | unchanged |
| T2 | server's EncryptedExtensions arrives WITHOUT early_data | `note_server_rejected()` | status=Rejected |
| T3 | gate check | `may_send_zero_rtt()` returns false | unchanged |
| T4 | inbound ACK arrives for PN=42 (0-RTT epoch) | `record_zero_rtt_ack_received()` returns `Err(ProtocolViolation)` | status=Rejected (terminal — caller closes connection) |

## Worked example (policy=Disabled — no 0-RTT even with ticket)

| t  | Event | Action | State after |
|----|-------|--------|-------------|
| T0 | construct ZeroRttManager(policy=Disabled) | — | status=Disabled, policy=Disabled |
| T1 | caller calls `prepare_resumption(ticket_bytes)` | returns `Err(PolicyDisabled)` | unchanged |
| T2 | gate check | `may_send_zero_rtt()` returns false | unchanged |

## Security review (per principle 13)

| Concern | RFC reference | Mitigation in C24.0 |
|---|---|---|
| Application-level replay (idempotent vs non-idempotent operations) | §9.2 | NOT the connection layer's problem — RFC explicitly delegates to the application protocol. Connection layer enforces the binary policy the application chose. |
| QUIC-frame-level replay (PADDING, PING, ACK, STREAM, etc.) | §9.2 | RFC says "Processing of QUIC frames is idempotent and cannot result in invalid connection states if frames are replayed, reordered, or lost." No mitigation needed at the frame layer. |
| Server reject + client sends more 0-RTT | §4.6.2 | `may_send_zero_rtt()` gate returns false once status=Rejected. Caller MUST consult before egress. |
| 0-RTT ACK after server reject | §4.6.2 | `record_zero_rtt_ack_received` returns `Err(ProtocolViolation)` when status=Rejected; caller emits CONNECTION_CLOSE. |
| Resumption-ticket leak (server compromise after the fact) | §9.2 | NOT solved by 0-RTT mechanism — solved by TLS 1.3 forward secrecy (Application keys are derived AFTER 1-RTT confirmation; 0-RTT is the deliberate exception). Documented. |
| Disable-by-default | §9.2 | `ZeroRttPolicy::Disabled` is the `Default` impl. Caller MUST opt in. |
| Cross-connection state leakage (e.g. transport parameters changed since ticket issued) | §4.6.3 | NOT solved here — the TLS provider validates the cached transport-params match the new connection's. C24.0 carries the ticket bytes; validation defers to C24.1's TLS-provider integration. |
| Replay across versions / connections that share NewSessionTicket | §9.2 | NOT solved here — server-side single-use-ticket enforcement is the server's job. C24.0 is purely client-side ticket carriage. |

**Sign-off**: composition-flaw scan complete. The remaining residual
risks (application-level replay, cross-connection TP validation,
ticket reuse) are explicitly out-of-scope per the RFC-mandated layering.

## Tier

C24.0 is tier-3 (POD policy + status + bounded ResumptionTicket
byte string). No alloc.

## Per principle 14 (incumbent wins)

State-machine transitions taken verbatim from RFC §4.6.2:
- "A server accepts 0-RTT by sending an early_data extension" → Attempting → Accepted
- "A server rejects 0-RTT by sending the EncryptedExtensions without an early_data extension" → Attempting → Rejected
- "When 0-RTT is rejected ... client SHOULD treat receipt of an acknowledgment for a 0-RTT packet as a connection error of type PROTOCOL_VIOLATION" → Rejected + 0-RTT-ack → ProtocolViolation

No invention. Worked examples are direct translations of the RFC's
named cases.

## Sized constants (principle 12)

```toml
[zero_rtt]
# Maximum length of a server-issued NewSessionTicket carried as the
# client's resumption credential. TLS 1.3 tickets are typically
# 100-300 bytes; 1024 leaves headroom for extension data.
max_resumption_ticket_len = 1024
```

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes (3 worked examples — accept, reject, disabled).
- **Pass 3 — code maps step-by-step**: planned for the impl below.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; allowing 0-RTT egress after rejection, accepting an ack for a rejected 0-RTT packet without flagging, or letting prepare_resumption succeed when policy=Disabled would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
