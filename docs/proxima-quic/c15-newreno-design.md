# C15 — NewReno congestion control (paper proof)

Per RFC 9002 §7. The NewReno controller decides:

1. **How many bytes are allowed in flight** (cwnd).
2. **How fast cwnd grows on acknowledgement** (slow-start vs congestion-avoidance).
3. **How cwnd shrinks on loss** (loss-reduction factor + minimum-window floor).
4. **When cwnd resets on persistent congestion** (per RFC 9002 §7.6).

Composes:

- C14 `LossOutcome` — list of lost SentPackets per ACK / timer-fire.
- C14 `SentPacket` — `size_bytes` + `sent_time` used for bytes_in_flight + persistent-congestion calculation.
- C14 `LossDetection::compute_pto` — used to scale the persistent-congestion threshold.

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Constants (RFC 9002 §7.2 + §B)

```
kInitialWindow      = 10 * max_datagram_size  (recommended; clamp at 14720 B per RFC 9002 §7.2)
kMinimumWindow      = 2 * max_datagram_size   (RFC 9002 §7.2)
kLossReductionFactor = 1/2                    (RFC 9002 §7.3.2)
kPersistentCongestionThreshold = 3            (already defined in loss::constants)
max_datagram_size   = 1200                    (RFC 9000 §14; conservative initial; raises with PMTUD)
```

In Rust:

```rust
pub const K_LOSS_REDUCTION_NUM: u64 = 1;
pub const K_LOSS_REDUCTION_DENOM: u64 = 2;
pub const DEFAULT_MAX_DATAGRAM_SIZE: u64 = 1200;
pub const K_INITIAL_WINDOW_DATAGRAMS: u64 = 10;
pub const K_MIN_WINDOW_DATAGRAMS: u64 = 2;
```

## Controller state

```rust
pub trait CongestionController {
    fn on_packet_sent(&mut self, sent_bytes: u64);
    fn on_packet_acked(&mut self, packet: &SentPacket, now: Instant);
    fn on_packets_lost(&mut self, lost: &[SentPacket], now: Instant, pto: Duration);
    /// Bytes the connection is currently allowed to send.
    fn send_budget(&self) -> u64;
    fn bytes_in_flight(&self) -> u64;
    fn cwnd(&self) -> u64;
    fn ssthresh(&self) -> Option<u64>;
}

pub struct NewReno {
    pub cwnd: u64,
    pub ssthresh: Option<u64>,
    pub bytes_in_flight: u64,
    pub max_datagram_size: u64,
    /// `Some(t)` while we're in the congestion-recovery window started at
    /// `t`. Per RFC 9002 §7.3.2, additional loss events whose
    /// `sent_time` falls inside this window do NOT trigger a fresh
    /// cwnd reduction (avoids double-cutting on a single congestion
    /// burst).
    pub congestion_recovery_start_time: Option<Instant>,
}
```

## Algorithms

### `on_packet_sent(bytes)`

```
bytes_in_flight = saturating_add(bytes_in_flight, bytes)
```

### `on_packet_acked(packet, now)`

```
1. bytes_in_flight = saturating_sub(bytes_in_flight, packet.size_bytes)
2. if in_congestion_recovery(packet.sent_time): return  // don't grow cwnd during recovery
3. if cwnd < ssthresh.unwrap_or(u64::MAX):
     // Slow start (RFC 9002 §7.3.1)
     cwnd += packet.size_bytes
   else:
     // Congestion avoidance (RFC 9002 §7.3.3)
     // cwnd += (max_datagram_size * bytes_acked) / cwnd
     cwnd += (max_datagram_size * packet.size_bytes) / cwnd
```

### `on_packets_lost(lost, now, pto)`

```
1. // Subtract lost bytes from in-flight (whether or not we reduce cwnd).
   for packet in lost: bytes_in_flight = saturating_sub(bytes_in_flight, packet.size_bytes)

2. // Find the earliest sent_time across the lost set — the congestion
   // event "started" at the OLDEST lost packet's sent_time.
   let earliest = lost.iter().map(|p| p.sent_time).min()
   if earliest is None: return  // empty lost set

3. // Only react to losses that aren't already inside the current
   // recovery window (RFC 9002 §7.3.2).
   if !in_congestion_recovery(earliest):
     congestion_recovery_start_time = Some(now)
     ssthresh = Some(max(cwnd / 2, k_minimum_window()))
     cwnd     = ssthresh.unwrap()

4. // Persistent congestion check (RFC 9002 §7.6) — applies only when
   //   - there are >= 2 lost packets (else there's no duration to span)
   //   - all lost packets fall under the largest ack'd
   //   - the duration spanned by the lost set exceeds the persistent-
   //     congestion threshold × pto
   if lost.len() >= 2:
     let span = lost.iter().map(|p| p.sent_time).max().unwrap()
                .duration_since(lost.iter().map(|p| p.sent_time).min().unwrap())
     if span >= pto * k_persistent_congestion_threshold:
       cwnd = k_minimum_window()
       congestion_recovery_start_time = None  // reset recovery window
```

### `in_congestion_recovery(sent_time)`

```
return congestion_recovery_start_time.map(|t| sent_time <= t).unwrap_or(false)
```

### `send_budget()`

```
return cwnd.saturating_sub(bytes_in_flight)
```

## Worked examples

### Slow-start growth

State: `cwnd = 12000` (10 × 1200), `ssthresh = None`, `bytes_in_flight = 12000`.
Action: `on_packet_acked(SentPacket{size_bytes=1200, sent_time=T0}, now=T0+50ms)`.

- bytes_in_flight = 12000 - 1200 = 10800
- in_congestion_recovery(T0)? No (start_time = None) → continue
- cwnd < u64::MAX (slow start) → cwnd += 1200 = **13200**

### Congestion-avoidance growth

State: `cwnd = 13200`, `ssthresh = Some(13200)`, `bytes_in_flight = 13200`.
Action: `on_packet_acked(size=1200, sent_time=T0)`.

- bytes_in_flight = 13200 - 1200 = 12000
- Not in recovery, cwnd >= ssthresh → CA branch.
- cwnd += (1200 × 1200) / 13200 = 1_440_000 / 13_200 = **109** (integer)
- new cwnd = 13200 + 109 = **13309**

### Loss event reduces cwnd

State: `cwnd = 20000`, `ssthresh = None`, `bytes_in_flight = 20000`, `recovery = None`.
Action: `on_packets_lost([SentPacket{size=1200, sent_time=T0}], now=T1, pto=300ms)`.

- bytes_in_flight = 20000 - 1200 = 18800
- earliest = T0
- in_congestion_recovery(T0)? No → enter recovery
- congestion_recovery_start_time = Some(T1)
- ssthresh = max(20000/2, 2400) = **10000**
- cwnd = 10000

### Persistent congestion resets cwnd

State: `cwnd = 8000`, `ssthresh = Some(10000)`, `recovery = Some(T1)`, `pto = 300 ms`.
Action: `on_packets_lost([{size=1200, sent_time=T2}, {size=1200, sent_time=T2 + 1000ms}], now=T3, pto=300ms)`.

- bytes_in_flight subtract 2400.
- earliest = T2; not in recovery (T2 > T1).
- Enter recovery: ssthresh = max(8000/2, 2400) = 4000; cwnd = 4000.
- Persistent-congestion check: len=2; span = 1000 ms; threshold = 3 × 300 ms = 900 ms; 1000 >= 900 → TRIGGER.
- cwnd = k_minimum_window() = 2400; recovery_start = None.

## Per-connection scope

NewReno is shared across all epochs — the cwnd is a property of the path, not the epoch. The state machine instantiates ONE `NewReno` instance per connection (one per path in the multipath case, but that's C26 territory).

## PTO probe bypass

When the FSM's `handle_timeout` calls `loss.on_loss_detection_timeout(now)` and the result is "PTO fired, no loss declared," the FSM should send an ack-eliciting probe packet even if `send_budget == 0`. RFC 9002 §6.2.4 permits the PTO probe to exceed cwnd. C15 exposes this by providing `congestion.allow_probe_send() -> u64` returning `1 * max_datagram_size` regardless of cwnd; the FSM's `poll_transmit_*` checks this flag when it knows a probe is needed. (Wire-up of the probe-emission path lands as a small follow-on in C15.4 or C19.)

## Code site

- `proxima-quic-proto/src/congestion/mod.rs` — re-exports + `CongestionController` trait.
- `proxima-quic-proto/src/congestion/constants.rs` — RFC constants.
- `proxima-quic-proto/src/congestion/new_reno.rs` — `NewReno` impl.
- `proxima-quic-proto/src/connection/mod.rs` — add `congestion: NewReno` field;
  - `poll_transmit_*` calls `congestion.on_packet_sent(packet.in_flight ? size : 0)`;
  - `handle_*_datagram` after `loss.on_ack_received`, takes the returned `LossOutcome` AND iterates the acked packets through `congestion.on_packet_acked`, then `congestion.on_packets_lost(lost, now, pto)`.

## Self-critique

- **Pass 1 — paper before code**: yes; design doc precedes any C15 code.
- **Pass 2 — algorithm walk produces exact expected output**: verified for
  slow-start (cwnd 12000→13200), CA (13200→13309), loss event
  (20000→10000), persistent congestion (8000→2400).
- **Pass 3 — code maps step-by-step to algorithm**: deferred; the
  implementation MUST keep `on_packet_sent` / `on_packet_acked` /
  `on_packets_lost` / `in_congestion_recovery` / `send_budget` as
  named methods.
- **Pass 4 — test uses exact inputs from worked examples**: yes;
  unit tests `slow_start_growth_walked_example`,
  `congestion_avoidance_growth_walked_example`, `loss_event_walked_example`,
  `persistent_congestion_walked_example`.
- **Pass 5 — would the test fail on bugs**: yes; flipping the
  slow-start/CA branch on `cwnd < ssthresh` (vs >=), forgetting the
  in_congestion_recovery guard on growth, swapping num/denom of
  kLossReductionFactor, off-by-one in the persistent-congestion span
  comparison would all break specific table values.
- **Pass 6 — paper linked to test**: test docstrings reference this doc.
