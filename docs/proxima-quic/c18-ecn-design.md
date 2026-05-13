# C18 — ECN (paper proof)

Per [RFC 9000 §13.4] + [RFC 8311]. ECN ("Explicit Congestion
Notification") lets routers mark packets `CE` (Congestion Experienced)
instead of dropping them. The receiver reports per-codepoint counts in
ACK frames; the sender treats every increment in the `CE` count as a
congestion event (equivalent to declaring those packets lost from a
cwnd-reduction perspective).

[RFC 9000 §13.4]: https://www.rfc-editor.org/rfc/rfc9000#section-13.4
[RFC 8311]: https://www.rfc-editor.org/rfc/rfc8311

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## ECN codepoints (RFC 3168 §5)

| codepoint | bits | name |
|---|---|---|
| 0b00 | Not-ECT | not ECN-capable |
| 0b01 | ECT(1) | ECN-capable (legacy / Cubic-default) |
| 0b10 | ECT(0) | ECN-capable (QUIC's recommended default) |
| 0b11 | CE     | Congestion Experienced |

QUIC senders typically mark outbound with **ECT(0)** per RFC 9000
§13.4.1; routers along the path may rewrite to `CE` (0b11). Receivers
count each codepoint and reflect via the ACK_ECN frame (frame type
0x03, the ECN variant of ACK).

## State per epoch

```rust
pub struct EcnState {
    /// Mode advertised on outbound packets. Starts as `Attempting`
    /// (we send ECT(0)). After a successful validation handshake
    /// (RFC 9000 §13.4.2) it advances to `Capable`. If validation
    /// fails (e.g. the peer's ACK_ECN counts don't match what we
    /// sent), it falls back to `Disabled`.
    pub mode: EcnMode,
    /// Counts of CE marks the sender has acknowledged (from the
    /// peer's ACK_ECN frames). Compared each ACK to detect new
    /// CE-marked packets.
    pub ce_acked: u64,
    /// Total ECT(0) outbound packets we've sent on this epoch.
    pub ect0_sent: u64,
    /// Total ECT(1) outbound (we never set this, but track it for
    /// validation).
    pub ect1_sent: u64,
}

pub enum EcnMode { Attempting, Capable, Disabled }
```

## ECN validation (RFC 9000 §13.4.2)

The sender must verify the receiver is faithfully reporting ECN counts.
Per RFC 9000 §13.4.2:

1. If `ack.ect0 + ack.ecn_ce < ect0_sent`: peer is under-counting →
   `mode = Disabled`.
2. If `ack.ect1 > ect1_sent`: peer claims more ECT(1) than we sent →
   `mode = Disabled`.
3. If `ack.ecn_ce` decreased relative to our cached `ce_acked`: nonsensical →
   `mode = Disabled`.
4. Otherwise: every increment in `ack.ecn_ce` over our cached `ce_acked`
   is a CE event → signal congestion to the controller.

After three consecutive valid ACKs (heuristic), `mode = Capable`.

## On packet sent (per-epoch)

```
on_packet_sent(codepoint):
  match codepoint:
    Ect0: ect0_sent += 1
    Ect1: ect1_sent += 1
    NotEct | Ce: no-op (we don't set these)
```

## On ACK_ECN received

```
on_ack_with_ecn(counts: EcnCounts, controller: &mut impl CongestionController, now, pto):
  if mode == Disabled: return
  // Validate
  if counts.ect0 + counts.ecn_ce < ect0_sent
     OR counts.ect1 > ect1_sent
     OR counts.ecn_ce < ce_acked:
       mode = Disabled
       return
  // CE delta → congestion event
  let new_ce = counts.ecn_ce - ce_acked
  if new_ce > 0:
    controller.on_ecn_ce_seen(now, pto)
  ce_acked = counts.ecn_ce
  // Advance validation
  if mode == Attempting && validation_consecutive_oks >= 3:
    mode = Capable
```

The `on_ecn_ce_seen` trait method is the new addition to
`CongestionController`; default impl treats it as a single "loss
event" with a synthetic SentPacket (size 0) to trigger the cwnd
reduction without affecting bytes_in_flight.

## Worked example

State at t=0: `mode=Attempting, ect0_sent=0, ce_acked=0`.

| Event | Action | After |
|---|---|---|
| Send PN 0 with ECT(0) | `ect0_sent += 1` | ect0_sent=1 |
| Send PN 1 with ECT(0) | `ect0_sent += 1` | ect0_sent=2 |
| ACK_ECN: {ect0:2, ect1:0, ecn_ce:0} | validate ✓; new_ce=0; no controller call | ce_acked=0 |
| Send PN 2 with ECT(0) | `ect0_sent += 1` | ect0_sent=3 |
| ACK_ECN: {ect0:2, ect1:0, ecn_ce:1} | validate (2+1=3≥3 ✓); new_ce=1; `controller.on_ecn_ce_seen` | ce_acked=1 |

The controller's cwnd halves on the second ACK (the CE event).

## Code site

- `proxima-quic-proto/src/ecn/mod.rs` — `EcnCodepoint` enum + `EcnState` + `EcnMode`.
- Extend `CongestionController` trait with `fn on_ecn_ce_seen(&mut self, now, pto) { ... default }`.
- NewReno + Cubic impl `on_ecn_ce_seen` to apply a single cwnd reduction (same code path as a loss event, no bytes subtracted).
- FSM `Connection<P>` carries `ecn: [EcnState; 3]` per epoch.
- `poll_transmit_*` calls `ecn[epoch].on_packet_sent(Ect0)`.
- `handle_*_datagram` ACK parsing: when ACK frame carries `Some(EcnCounts)`, call `ecn[epoch].on_ack_with_ecn(counts, &mut self.congestion, now, pto)`.
- `DatagramWrite` gains an `ecn_codepoint: EcnCodepoint` field for the I/O facade to consume (the kernel sets the IP TOS byte based on this).

## Tier

`ecn` module is tier-3 — single struct of u64 counters + an enum.

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes; worked-example table covers the two-ACK sequence ending in CE.
- **Pass 3 — code maps step-by-step to algorithm**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: yes;
  `ecn_validation_passes_then_ce_signals_congestion` will encode the table.
- **Pass 5 — would the test fail on bugs**: yes; missing the
  `ect0 + ecn_ce ≥ ect0_sent` check would let a CE go undetected;
  swapping `ce_acked` increment / controller call order would break
  the test's controller-cwnd assertion.
- **Pass 6 — paper linked to test**: yes.
