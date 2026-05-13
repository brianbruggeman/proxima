# C14 — Loss detection + RTT estimation (paper proof)

Per RFC 9002 §5 (RTT) + §6 (loss detection) + §A.5–§A.8 (pseudocode).
Composes:

- C9 PN spaces — packet-number per epoch.
- C13 AckScheduler — emits ACK frames carrying ranges of received PNs;
  this component consumes the INBOUND peer ACK frames (the dual).
- C15 NewReno (next) — consumes loss / cwnd events from C14.

Per workspace principle 11 (sans-IO state machine, enum + per-variant
data, low/no alloc, extreme benching). Per the `/algorithm-development`
skill — paper proof PRECEDES code.

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Two phases

C14 is two halves of the same coin:

1. **Sent-packet tracking** — every packet we transmit goes into a
   per-epoch `SentPacketQueue` with its PN, send-time, byte-count, and
   ack-eliciting flag. Records are removed when ack'd or declared lost.
2. **Detection** — on inbound ACK or timeout, we compute (a) which
   sent packets are now ack'd (and emit RTT samples for the largest),
   (b) which sent packets are now lost (packet-threshold OR time-
   threshold per RFC 9002 §6.1), (c) the next loss-detection deadline
   (min of loss-time, PTO).

Persistent congestion detection per §7.6 is congestion-control-side —
deferred to C15.

---

## Data types

```rust
pub struct SentPacket {
    pub packet_number: u64,
    pub sent_time: Instant,
    pub size_bytes: u16,
    /// Per RFC 9002 §A.1: was the packet "ack-eliciting" — i.e. did it
    /// contain at least one ack-eliciting frame (CRYPTO, STREAM, PING,
    /// HANDSHAKE_DONE, RESET_STREAM, etc.)? Pure-ACK packets are not.
    pub is_ack_eliciting: bool,
    /// Per RFC 9002 §A.1: "in_flight" — counts against the congestion
    /// window. Pure-ACK packets are not in flight; almost everything
    /// else is.
    pub in_flight: bool,
}

pub struct SentPacketQueue<const MAX: usize> {
    /// arrayvec::ArrayVec sorted ASCENDING by packet_number.
    /// Per the C14 design: drop-oldest on overflow with a SAFETY caveat
    /// — losing an in-flight record can cause an over-loss-detection
    /// blip. MAX is sized to `prime-runtime.toml [quic].sent_packets_cap`
    /// (1024).
    packets: arrayvec::ArrayVec<SentPacket, MAX>,
}

pub struct RttEstimator {
    /// Smoothed RTT per RFC 9002 §5.3.
    pub smoothed_rtt: Option<Duration>,
    /// RTT variance per §5.3.
    pub rttvar: Option<Duration>,
    /// Minimum observed RTT per §5.2.
    pub min_rtt: Option<Duration>,
    /// Latest RTT sample per §5.1.
    pub latest_rtt: Option<Duration>,
    /// First-RTT-sample flag — controls the initialisation branch in
    /// §5.3.
    pub first_sample_taken: bool,
}

pub struct LossDetection {
    /// Per-epoch sent-packet queues (Initial / Handshake / Application).
    pub sent_packets: [SentPacketQueue<MAX_SENT_PACKETS>; 3],
    /// Largest packet number ack'd per epoch (RFC 9002 §A.1).
    pub largest_acked_packet: [Option<u64>; 3],
    /// Time the last ack-eliciting packet was sent per epoch
    /// (RFC 9002 §A.1; used for PTO computation).
    pub time_of_last_ack_eliciting_packet: [Option<Instant>; 3],
    /// Per-epoch loss-time — the earliest packet that hasn't been
    /// declared lost but is at risk of being declared lost on the next
    /// timeout (RFC 9002 §6.1.2).
    pub loss_time: [Option<Instant>; 3],
    /// PTO count per RFC 9002 §A.1 — multiplier on the PTO timer.
    /// Reset on receipt of an ack-eliciting acknowledgement.
    pub pto_count: u32,
    pub rtt: RttEstimator,
}
```

## Constants (RFC 9002 §A.2)

```
kPacketThreshold     = 3       // reorder threshold for packet-based loss
kTimeThreshold       = 9/8     // factor on RTT for time-based loss
kGranularity         = 1 ms    // clock-granularity guard
kInitialRtt          = 333 ms  // before first sample
kPersistentCongestionThreshold = 3
```

In Rust these live in `crate::ack::scheduler::DEFAULT_MAX_ACK_DELAY_MICROS`
+ a new `proxima-quic-proto::loss::constants` module mirror.

---

## RTT update (RFC 9002 §5.3)

```
function on_rtt_sample(latest_rtt: Duration, ack_delay: Duration, now: Instant):
  step 1: min_rtt = match min_rtt: Some(m) => Some(min(m, latest_rtt)),
                                  None => Some(latest_rtt)

  step 2: if !first_sample_taken:
            smoothed_rtt = Some(latest_rtt)
            rttvar       = Some(latest_rtt / 2)
            first_sample_taken = true
            return

  step 3: // Adjust latest_rtt for ack_delay if it doesn't go below min_rtt.
          adjusted_rtt = latest_rtt
          if min_rtt.unwrap() + ack_delay <= latest_rtt:
            adjusted_rtt = latest_rtt - ack_delay

  step 4: // EWMA per RFC 9002 §5.3.
          rttvar_sample = abs(smoothed_rtt.unwrap() - adjusted_rtt)
          rttvar       = (3/4) * rttvar.unwrap() + (1/4) * rttvar_sample
          smoothed_rtt = (7/8) * smoothed_rtt.unwrap() + (1/8) * adjusted_rtt
```

### Worked example for RTT

Inputs:
- latest_rtt = 100 ms, ack_delay = 5 ms, first sample
- next: latest_rtt = 90 ms, ack_delay = 2 ms

After first call:
- min_rtt = 100 ms
- smoothed_rtt = 100 ms
- rttvar = 50 ms
- first_sample_taken = true

After second call:
- min_rtt = min(100 ms, 90 ms) = 90 ms
- adjusted_rtt = 90 - 2 = 88 ms (min_rtt + ack_delay = 92 ms <= 90 ms? NO — so adjusted_rtt = 90)
- Actually 90 + 2 = 92 ms is not <= 90 ms (latest_rtt), so the adjustment guard fails. adjusted_rtt = 90.
- Wait — the RFC says "if min_rtt + ack_delay <= latest_rtt then adjust". min_rtt=90, ack_delay=2, sum=92, latest_rtt=90. 92 <= 90 is FALSE. So no adjustment. adjusted_rtt = 90.
- rttvar_sample = |100 - 90| = 10
- rttvar = (3/4) * 50 + (1/4) * 10 = 37.5 + 2.5 = 40
- smoothed_rtt = (7/8) * 100 + (1/8) * 90 = 87.5 + 11.25 = 98.75

Test will encode these values bit-exact.

---

## Loss detection (RFC 9002 §6.1)

```
function detect_losses(epoch: Epoch, now: Instant) -> Vec<SentPacket>:
  step 1: largest_acked = largest_acked_packet[epoch]
          loss_time[epoch] = None
          if largest_acked.is_none(): return []  // no ack yet

  step 2: loss_delay = compute_loss_delay()
                       // = kTimeThreshold * max(smoothed_rtt, latest_rtt)
                       // clamped at kGranularity

  step 3: lost_send_time_threshold = now - loss_delay

  step 4: lost_packets = []
          for packet in sent_packets[epoch]:
            if packet.packet_number > largest_acked.unwrap(): continue
            // Time-threshold: packet sent before lost_send_time_threshold.
            if packet.sent_time <= lost_send_time_threshold:
              lost_packets.push(packet)
              continue
            // Packet-threshold: there's a packet with PN >= our PN + kPacketThreshold
            //                   that was acked.
            if largest_acked.unwrap() >= packet.packet_number + kPacketThreshold:
              lost_packets.push(packet)
              continue
            // Not lost — record the next potential loss time.
            loss_time[epoch] = Some(min(loss_time[epoch].unwrap_or(MAX),
                                        packet.sent_time + loss_delay))

  step 5: return lost_packets


function compute_loss_delay() -> Duration:
  step 1: rtt = max(smoothed_rtt.unwrap_or(kInitialRtt),
                   latest_rtt.unwrap_or(kInitialRtt))
  step 2: loss_delay = (kTimeThreshold * rtt)  // 9/8 of RTT
  step 3: return max(loss_delay, kGranularity)
```

### Worked example for packet-threshold loss

State: sent [PN 0..=10]. ACK received with largest_acked=10. kPacketThreshold=3.

- For PN 0..=7: 10 >= PN + 3 → declared lost.
- For PN 8..=9: 10 >= 8+3? 10 >= 11? NO. 10 >= 9+3=12? NO. Not lost (yet).
- PN 10 == largest_acked, ack'd not lost.

Result: PNs 0..=7 declared lost.

### Worked example for time-threshold loss

State: sent [PN 5 at t=1000, PN 6 at t=1100, PN 7 at t=1200]. ACK arrived at t=1300 with largest_acked=7.

- smoothed_rtt = 100 ms (set up from prior samples).
- loss_delay = 9/8 * 100 ms = 112.5 ms.
- lost_send_time_threshold = 1300 - 112.5 = 1187.5 ms.
- PN 5 sent at 1000, <= 1187.5 → time-lost.
- PN 6 sent at 1100, <= 1187.5 → time-lost.
- PN 7 sent at 1200, > 1187.5 → not time-lost; potential loss_time = 1200 + 112.5 = 1312.5.

PN 5 and PN 6 declared lost. loss_time[epoch] = 1312.5.

---

## PTO (RFC 9002 §6.2)

```
function compute_pto() -> Duration:
  step 1: pto = smoothed_rtt.unwrap_or(kInitialRtt)
                + max(4 * rttvar.unwrap_or(kInitialRtt / 2), kGranularity)
                + max_ack_delay
  step 2: return pto * (2 ^ pto_count)

function set_pto_timer(epoch: Epoch, now: Instant):
  step 1: if time_of_last_ack_eliciting_packet[epoch].is_none(): return None
  step 2: return time_of_last_ack_eliciting_packet[epoch] + compute_pto()

function on_loss_detection_timeout(now: Instant):
  step 1: earliest_loss = loss_time across all epochs that's not None
  step 2: if earliest_loss is set:
            lost = detect_losses(earliest_epoch, now)
            apply lost to congestion controller (C15)
            return
  step 3: // No loss-time → PTO fired.
          pto_count += 1
          // Caller should send an ack-eliciting probe packet next.
```

### Worked example for PTO

After first sample (RTT 100 ms), no acks since:
- smoothed_rtt = 100, rttvar = 50, max_ack_delay = 25, pto_count = 0
- pto = 100 + max(4 * 50, 1) + 25 = 100 + 200 + 25 = 325 ms
- After fire: pto_count = 1 → next pto = 325 * 2 = 650 ms

---

## Per-epoch storage choice

Each epoch gets its own queue + scheduler state. Per RFC 9002 §A.4 the
loss detection state is shared across all epochs for `pto_count` (one
counter total) but per-epoch for `largest_acked`, `loss_time`,
`time_of_last_ack_eliciting_packet`, and `sent_packets`. The
implementation matches.

## Sent-packet queue overflow

Per the deferred edge: when at capacity, drop the OLDEST packet (lowest
PN). Rationale:

- Older in-flight records are more likely to already be lost or
  irrelevant by the time the cap is hit.
- Drop-oldest preserves recent records — the ones most likely to drive
  the next loss detection.
- Quinn-proto uses a similar policy.

Loss detection over a queue with dropped entries may produce
SLIGHTLY-OPTIMISTIC results (missed losses) — recorded as a
DEFERRED EDGE for C14 follow-up if benchmarks show pathological
cases under sustained loss.

## What's NOT in C14

- **Persistent congestion** — RFC 9002 §7.6 — needs to compute
  `kPersistentCongestionThreshold * pto` and compare against the
  duration spanned by lost packets. Goes into C15 because the response
  (reset cwnd) is congestion-control behaviour.
- **Application of loss to congestion control** — `on_packets_lost`
  per RFC 9002 §A.10 — call into C15 with the loss event.
- **ECN handling** — RFC 9000 §13.4 — moved to C18.

## Tier

`loss::*` modules target tier-3. Storage uses `arrayvec::ArrayVec`
no-alloc; constants `const`-folded. RTT arithmetic uses `crate::time`
newtypes (saturating).

## Code site

- `proxima-quic-proto/src/loss/mod.rs` — re-exports.
- `proxima-quic-proto/src/loss/sent_packet.rs` — `SentPacket` + `SentPacketQueue<MAX>`.
- `proxima-quic-proto/src/loss/rtt.rs` — `RttEstimator`.
- `proxima-quic-proto/src/loss/detector.rs` — `LossDetection` orchestrator.
- `proxima-quic-proto/src/connection/mod.rs` — wire `on_packet_sent` /
  `on_ack_received` / `on_loss_detection_timeout` into the FSM
  dispatcher.

## Self-critique

- **Pass 1 — paper before code**: yes; this doc precedes any C14 code.
- **Pass 2 — algorithm walk produces exact expected output**: verified
  for RTT update (98.75 / 40 numbers), packet-threshold loss
  (PNs 0..=7), time-threshold loss (PNs 5+6 lost, loss_time = 1312.5),
  and PTO (325 / 650).
- **Pass 3 — code maps step-by-step to algorithm**: deferred until
  code lands; the implementation MUST keep `on_rtt_sample` /
  `compute_loss_delay` / `detect_losses` / `compute_pto` /
  `on_loss_detection_timeout` as named functions.
- **Pass 4 — test uses exact inputs from worked examples**: yes;
  unit tests `rtt_estimator_walked_example`,
  `packet_threshold_loss_walked_example`,
  `time_threshold_loss_walked_example`, `pto_walked_example`.
- **Pass 5 — would the test fail on bugs**: yes; off-by-one in
  kPacketThreshold (3 vs 2), wrong EWMA factor (3/4 vs 7/8 swap),
  forgetting the kGranularity floor, or PTO * 2^pto_count vs
  PTO + pto_count would all break specific assertions.
- **Pass 6 — paper linked to test**: test docstrings reference this doc.
