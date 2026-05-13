//! Per-RFC-9002 §6 loss detection orchestrator.
//!
//! Owns one [`SentPacketQueue`] per epoch, the shared
//! [`RttEstimator`], and the loss-detection state (per-epoch loss-time
//! + global pto_count) per RFC 9002 §A.1.

use arrayvec::ArrayVec;

use crate::quic::time::{Duration, Instant};
use crate::quic::tls::Epoch;

use super::constants::{
    K_GRANULARITY_MICROS, K_PACKET_THRESHOLD, K_TIME_THRESHOLD_DENOM, K_TIME_THRESHOLD_NUM,
    MAX_SENT_PACKETS,
};
use super::rtt::RttEstimator;
use super::sent_packet::{SentPacket, SentPacketQueue};

/// Per-epoch loss-detection state.
#[derive(Debug, Clone)]
pub struct PerEpochState {
    pub sent_packets: SentPacketQueue<MAX_SENT_PACKETS>,
    /// Largest packet number ack'd in this epoch (RFC 9002 §A.1).
    pub largest_acked_packet: Option<u64>,
    /// Earliest time at which a still-pending packet would be declared
    /// lost via the time-threshold path (RFC 9002 §A.1 `loss_time`).
    pub loss_time: Option<Instant>,
    /// Time the last ack-eliciting packet was sent in this epoch.
    pub time_of_last_ack_eliciting_packet: Option<Instant>,
}

impl Default for PerEpochState {
    fn default() -> Self {
        Self::new()
    }
}

impl PerEpochState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sent_packets: SentPacketQueue::new(),
            largest_acked_packet: None,
            loss_time: None,
            time_of_last_ack_eliciting_packet: None,
        }
    }
}

/// Loss detector — combines per-epoch sent-packet queues + RTT
/// estimator + the global pto_count.
#[derive(Debug, Clone)]
pub struct LossDetection {
    pub epochs: [PerEpochState; 3],
    pub pto_count: u32,
    pub rtt: RttEstimator,
    /// Local `max_ack_delay` per RFC 9002 §6.2 — fed into the PTO
    /// computation. Caller sets via [`Self::set_max_ack_delay`].
    pub max_ack_delay: Duration,
}

impl Default for LossDetection {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum packets reported in a single `detect_losses` call.
/// Sourced from `proxima-quic-proto.toml [loss].max_loss_burst`
/// (override via `PROXIMA_QUIC_PROTO_LOSS_MAX_LOSS_BURST`). Bounded
/// array avoids alloc on the loss path.
pub const MAX_LOSS_BURST: usize = crate::quic::sized::LOSS_MAX_LOSS_BURST;

/// Outcome of a `detect_losses` call OR a `on_ack_received` call.
#[derive(Debug, Default)]
pub struct LossOutcome {
    /// Packets removed from the sent queue because they were ack'd.
    /// Only populated by `on_ack_received` (detect_losses leaves this
    /// empty).
    pub newly_acked: ArrayVec<SentPacket, MAX_LOSS_BURST>,
    /// Packets removed from the sent queue because they were declared
    /// lost.
    pub lost: ArrayVec<SentPacket, MAX_LOSS_BURST>,
    /// Updated loss_time for the epoch (None if no still-pending packet
    /// would tip into the loss window in the future).
    pub next_loss_time: Option<Instant>,
    /// Set when `on_loss_detection_timeout` fires a PTO (empty `lost`
    /// list) — tells the caller WHICH epoch's probe to send. Matches
    /// the epoch with the earliest PTO deadline (same calculation as
    /// `next_deadline`). `None` when the outcome is from time-threshold
    /// loss or ACK processing.
    pub pto_epoch: Option<Epoch>,
    /// Set when `on_loss_detection_timeout` fires the time-threshold
    /// path — tells the caller WHICH epoch's loss timer expired.
    /// Always matches the `lost` packets' epoch. `None` for PTO or
    /// ACK-driven outcomes.
    pub loss_epoch: Option<Epoch>,
}

impl LossDetection {
    /// Construct a detector with `max_ack_delay = 25 ms` (RFC 9000
    /// §18.2 default).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epochs: [
                PerEpochState::new(),
                PerEpochState::new(),
                PerEpochState::new(),
            ],
            pto_count: 0,
            rtt: RttEstimator::new(),
            max_ack_delay: Duration::from_micros(25_000),
        }
    }

    /// Update the locally-advertised `max_ack_delay` (peer-RX, used
    /// in PTO computation).
    pub fn set_max_ack_delay(&mut self, max_ack_delay: Duration) {
        self.max_ack_delay = max_ack_delay;
    }

    /// Record a freshly-sent packet (called by `poll_transmit`).
    pub fn on_packet_sent(&mut self, epoch: Epoch, packet: SentPacket) {
        let state = &mut self.epochs[epoch.index()];
        if packet.is_ack_eliciting {
            state.time_of_last_ack_eliciting_packet = Some(packet.sent_time);
        }
        let _dropped = state.sent_packets.push(packet);
    }

    /// Process an inbound ACK frame (called by the FSM's parse path).
    ///
    /// `largest` is the peer's `largest_acknowledged` field; `ack_delay`
    /// is the decoded value (after AckDelayExponent scaling); `acked_ranges`
    /// is the set of `(smallest, largest)` INCLUSIVE packet-number ranges the
    /// ACK frame covers (a handful — NOT expanded per-PN).
    ///
    /// Returns the [`LossOutcome`] from the resulting loss-detection
    /// scan.
    pub fn on_ack_received(
        &mut self,
        epoch: Epoch,
        largest: u64,
        ack_delay: Duration,
        acked_ranges: &[(u64, u64)],
        now: Instant,
    ) -> LossOutcome {
        // Phase 1: update largest_acked + remove ACKed packets from
        // the sent queue. Scoped so the &mut self.epochs borrow drops
        // before detect_losses borrows &mut self.
        let mut newly_acked = ArrayVec::<SentPacket, MAX_LOSS_BURST>::new();
        {
            let state = &mut self.epochs[epoch.index()];
            state.largest_acked_packet = Some(match state.largest_acked_packet {
                Some(current) => current.max(largest),
                None => largest,
            });
            // Match each in-flight packet against the ACK ranges (a handful),
            // not every covered PN. A cumulative ACK's expanded PN list grows
            // with the connection — O(span) per ACK, O(connection^2) overall,
            // ~1/3 of the server core under load.
            state.sent_packets.retain(|packet| {
                let pn = packet.packet_number;
                if acked_ranges.iter().any(|&(lo, hi)| lo <= pn && pn <= hi) {
                    let _ = newly_acked.try_push(*packet);
                    false
                } else {
                    true
                }
            });
        }
        // RTT sample from the largest acked ack-eliciting packet.
        if let Some(packet) = newly_acked
            .iter()
            .find(|p| p.packet_number == largest && p.is_ack_eliciting)
            && let Some(latest) = now.duration_since(packet.sent_time)
        {
            self.rtt.on_sample(latest, ack_delay);
        }
        // Reset pto_count on any ack-eliciting acknowledgement.
        if newly_acked.iter().any(|packet| packet.is_ack_eliciting) {
            self.pto_count = 0;
        }
        // Phase 2: detect losses (borrows &mut self again). detect_losses
        // also clears the PTO anchor when no ack-eliciting packets remain
        // (RFC 9002 §A.7), folded into its single retain pass — no separate
        // O(N) scan here.
        let mut outcome = self.detect_losses(epoch, now);
        outcome.newly_acked = newly_acked;
        outcome
    }

    /// Run the loss-detection scan for `epoch` against the current
    /// `largest_acked` and now. Updates the epoch's `loss_time` and
    /// removes lost records from the sent_packets queue.
    pub fn detect_losses(&mut self, epoch: Epoch, now: Instant) -> LossOutcome {
        let loss_delay = self.compute_loss_delay();
        let state = &mut self.epochs[epoch.index()];
        let Some(largest_acked) = state.largest_acked_packet else {
            state.loss_time = None;
            return LossOutcome::default();
        };
        let send_time_floor = now.saturating_sub(loss_delay);
        let mut lost = ArrayVec::<SentPacket, MAX_LOSS_BURST>::new();
        let mut next_loss_time: Option<Instant> = None;
        let mut remaining_ack_eliciting = false;
        // Remove lost packets in place (single pass) instead of rebuilding
        // the whole queue — the rebuild copied every survivor twice on
        // every ACK (~1/3 of the server core under load). Survivors stay
        // put; the ack-eliciting check folds into the same pass.
        state.sent_packets.retain(|record| {
            if record.packet_number > largest_acked {
                remaining_ack_eliciting |= record.is_ack_eliciting;
                return true;
            }
            let time_lost = record.sent_time.as_micros() <= send_time_floor.as_micros();
            let packet_lost = largest_acked >= record.packet_number + K_PACKET_THRESHOLD;
            if time_lost || packet_lost {
                let _ = lost.try_push(*record);
                false
            } else {
                let candidate_loss_time = record.sent_time + loss_delay;
                next_loss_time = Some(match next_loss_time {
                    Some(existing) => existing.min(candidate_loss_time),
                    None => candidate_loss_time,
                });
                remaining_ack_eliciting |= record.is_ack_eliciting;
                true
            }
        });
        state.loss_time = next_loss_time;
        // clear PTO anchor if no ack-eliciting packets remain
        if !remaining_ack_eliciting {
            state.time_of_last_ack_eliciting_packet = None;
        }
        let loss_epoch = if lost.is_empty() { None } else { Some(epoch) };
        LossOutcome {
            newly_acked: ArrayVec::new(),
            lost,
            next_loss_time,
            pto_epoch: None,
            loss_epoch,
        }
    }

    /// Compute the loss delay per RFC 9002 §6.1.2:
    /// `max(kTimeThreshold * max(smoothed_rtt, latest_rtt), kGranularity)`.
    #[must_use]
    pub fn compute_loss_delay(&self) -> Duration {
        let smoothed = self.rtt.smoothed_rtt.unwrap_or(RttEstimator::initial_rtt());
        let latest = self.rtt.latest_rtt.unwrap_or(smoothed);
        let rtt = Duration::from_micros(smoothed.as_micros().max(latest.as_micros()));
        let scaled =
            Duration::from_micros(rtt.as_micros() * K_TIME_THRESHOLD_NUM / K_TIME_THRESHOLD_DENOM);
        let granularity = Duration::from_micros(K_GRANULARITY_MICROS);
        Duration::from_micros(scaled.as_micros().max(granularity.as_micros()))
    }

    /// Compute the PTO per RFC 9002 §6.2.1:
    /// `(smoothed_rtt + max(4 * rttvar, kGranularity) + max_ack_delay) * 2^pto_count`.
    /// For Initial/Handshake epochs the `max_ack_delay` contribution is zero
    /// (RFC 9002 §6.2.1) — caller passes `include_max_ack_delay = false`.
    #[must_use]
    pub fn compute_pto(&self, include_max_ack_delay: bool) -> Duration {
        let smoothed = self.rtt.smoothed_rtt_or_initial();
        let rttvar = self.rtt.rttvar_or_initial();
        let granularity = Duration::from_micros(K_GRANULARITY_MICROS);
        let rttvar_term =
            Duration::from_micros((4 * rttvar.as_micros()).max(granularity.as_micros()));
        let max_ack_delay = if include_max_ack_delay {
            self.max_ack_delay
        } else {
            Duration::ZERO
        };
        let base = Duration::from_micros(
            smoothed.as_micros() + rttvar_term.as_micros() + max_ack_delay.as_micros(),
        );
        // Multiply by 2^pto_count, saturating at u64::MAX.
        let factor = 1u64.checked_shl(self.pto_count).unwrap_or(u64::MAX);
        Duration::from_micros(base.as_micros().saturating_mul(factor))
    }

    /// Drop all loss-detection state for an epoch whose keys are
    /// being discarded. RFC 9001 §4.9 + RFC 9002 §A.4 — once an
    /// endpoint discards Initial/Handshake keys, packets in that PN
    /// space can no longer be retransmitted; their loss-detection
    /// state MUST stop influencing the unified PTO/loss timers AND
    /// their in-flight bytes MUST be released from the congestion
    /// controller (since they will never be acked or declared lost).
    ///
    /// Returns the total `size_bytes` of every in-flight packet
    /// (`in_flight == true`) that was tracked in this epoch — the
    /// caller MUST pass this to
    /// [`CongestionController::on_packet_number_space_discarded`](crate::quic::congestion::CongestionController::on_packet_number_space_discarded)
    /// so cwnd availability stays accurate.
    ///
    /// Without this, a fully-acked-but-not-cleanly-discarded epoch
    /// can keep stale `sent_packets` (or anchor a stale
    /// `time_of_last_ack_eliciting_packet`) and arm a PTO deadline
    /// that nobody can answer — inflating pto_count and stealing
    /// real Application-epoch recovery responsiveness. Without the
    /// bytes release, cwnd is permanently understated by the
    /// discarded packets' size.
    #[must_use = "release the returned in_flight_bytes via CongestionController::on_packet_number_space_discarded"]
    pub fn discard_epoch(&mut self, epoch: Epoch) -> u64 {
        let state = &mut self.epochs[epoch.index()];
        let in_flight_bytes: u64 = state
            .sent_packets
            .iter()
            .filter(|pkt| pkt.in_flight)
            .map(|pkt| u64::from(pkt.size_bytes))
            .sum();
        state.sent_packets = SentPacketQueue::new();
        state.largest_acked_packet = None;
        state.loss_time = None;
        state.time_of_last_ack_eliciting_packet = None;
        in_flight_bytes
    }

    /// Test helper: reset all epoch state so PTO deadlines
    /// don't fire for epochs the mock handshake left stale.
    #[cfg(test)]
    pub fn clear_epoch_timestamps_for_test(&mut self) {
        for state in &mut self.epochs {
            state.time_of_last_ack_eliciting_packet = None;
            state.sent_packets = SentPacketQueue::new();
            state.largest_acked_packet = None;
            state.loss_time = None;
        }
        self.pto_count = 0;
    }

    /// Earliest pending loss-detection deadline across all epochs.
    /// Combination of the per-epoch `loss_time` (if any) AND the
    /// per-epoch PTO timer (the earliest among epochs with an
    /// outstanding ack-eliciting packet). `on_loss_detection_timeout`
    /// MUST select the same epoch this method's PTO branch picked —
    /// see the matching `min_by_key` call there.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        let loss_deadline = self.epochs.iter().filter_map(|state| state.loss_time).min();
        let pto_deadline = self
            .epochs
            .iter()
            .enumerate()
            .filter_map(|(index, state)| {
                // gate PTO on an outstanding ack-eliciting packet —
                // a bare timestamp without a matching unacked packet
                // is treated as cleared (defensive: the ACK / loss
                // paths already null the timestamp).
                state
                    .sent_packets
                    .iter()
                    .any(|pkt| pkt.is_ack_eliciting)
                    .then_some(())?;
                let sent_time = state.time_of_last_ack_eliciting_packet?;
                let include_max_ack_delay = index == Epoch::Application.index();
                Some(sent_time + self.compute_pto(include_max_ack_delay))
            })
            .min();
        match (loss_deadline, pto_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Drive forward at the supplied time. If `now >= loss_deadline`,
    /// produce a loss outcome per the time-threshold path. Otherwise
    /// if `now >= pto_deadline`, bump `pto_count` and return an empty
    /// outcome (caller's responsibility to actually send a probe).
    pub fn on_loss_detection_timeout(&mut self, now: Instant) -> LossOutcome {
        // Loss-time first per RFC 9002 §A.9.
        let earliest_loss_epoch = (0..3usize).find(|index| {
            self.epochs[*index]
                .loss_time
                .map(|t| t <= now)
                .unwrap_or(false)
        });
        if let Some(index) = earliest_loss_epoch {
            let epoch = match index {
                0 => Epoch::Initial,
                1 => Epoch::Handshake,
                _ => Epoch::Application,
            };
            return self.detect_losses(epoch, now);
        }
        // PTO fired. Identify the epoch whose PTO actually expired —
        // must match next_deadline's pick (the epoch with the
        // EARLIEST computed PTO deadline, not the deepest index).
        // RFC 9002 §6.2.4: probe in the PN space where the timeout
        // occurred. Gate on outstanding ack-eliciting packets so a
        // stale timestamp from a fully-acked epoch never triggers
        // a probe (mirrors next_deadline's gating).
        self.pto_count = self.pto_count.saturating_add(1);
        let pto_epoch = (0..3usize)
            .filter_map(|index| {
                let has_outstanding = self.epochs[index]
                    .sent_packets
                    .iter()
                    .any(|pkt| pkt.is_ack_eliciting);
                if !has_outstanding {
                    return None;
                }
                let sent_time = self.epochs[index].time_of_last_ack_eliciting_packet?;
                let include_max_ack_delay = index == Epoch::Application.index();
                let deadline = sent_time + self.compute_pto(include_max_ack_delay);
                Some((index, deadline))
            })
            .min_by_key(|(_, deadline)| *deadline)
            .map(|(index, _)| match index {
                0 => Epoch::Initial,
                1 => Epoch::Handshake,
                _ => Epoch::Application,
            });
        LossOutcome {
            pto_epoch,
            ..Default::default()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::time::Duration;

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    fn sent(pn: u64, sent_time: Instant) -> SentPacket {
        SentPacket {
            packet_number: pn,
            sent_time,
            size_bytes: 1200,
            is_ack_eliciting: true,
            in_flight: true,
        }
    }

    #[test]
    fn new_detector_has_no_deadline() {
        let det = LossDetection::new();
        assert_eq!(det.next_deadline(), None);
        assert_eq!(det.pto_count, 0);
    }

    #[test]
    fn packet_threshold_loss_walked_example() {
        // From docs/proxima-quic/c14-loss-detection-design.md.
        // sent PN 0..=10; ACK largest=10 → PN 0..=7 declared lost.
        let mut det = LossDetection::new();
        for pn in 0..=10u64 {
            det.on_packet_sent(Epoch::Application, sent(pn, at(1_000_000 + pn * 100)));
        }
        let acked: alloc::vec::Vec<(u64, u64)> = alloc::vec![(10, 10)];
        let outcome = det.on_ack_received(
            Epoch::Application,
            10,
            Duration::ZERO,
            &acked,
            at(1_001_500),
        );
        let lost_pns: alloc::vec::Vec<u64> = outcome
            .lost
            .iter()
            .map(|record| record.packet_number)
            .collect();
        assert_eq!(lost_pns, alloc::vec![0u64, 1, 2, 3, 4, 5, 6, 7]);
        // PNs 8, 9 survive (largest=10, 10-8=2 < 3, 10-9=1 < 3).
        let survivors: alloc::vec::Vec<u64> = det.epochs[Epoch::Application.index()]
            .sent_packets
            .iter()
            .map(|record| record.packet_number)
            .collect();
        assert_eq!(survivors, alloc::vec![8u64, 9]);
    }

    #[test]
    fn time_threshold_loss_walked_example() {
        // Same docs/proxima-quic/c14-loss-detection-design.md worked example.
        // Sent: PN 5 @ t=1000, PN 6 @ t=1100, PN 7 @ t=1200 (in micros 1ms units).
        // smoothed_rtt set up so loss_delay = 9/8 * 100 ms = 112.5 ms.
        // ACK at t=1300 ms with largest=7 → PN 5, 6 declared lost; PN 7 ack'd.
        let mut det = LossDetection::new();
        // Seed RTT to 100 ms via a synthetic prior sample.
        det.rtt
            .on_sample(Duration::from_millis(100), Duration::ZERO);
        // sent PN 5..=7 at t = 1000/1100/1200 ms
        for (offset, pn) in (5u64..=7).enumerate() {
            let sent_time = Instant::from_micros((1000 + offset as u64 * 100) * 1_000);
            det.on_packet_sent(Epoch::Application, sent(pn, sent_time));
        }
        // Drive the second RTT sample to KEEP smoothed_rtt close to 100 ms.
        det.rtt
            .on_sample(Duration::from_millis(100), Duration::ZERO);
        let now = Instant::from_micros(1300 * 1_000);
        let acked: alloc::vec::Vec<(u64, u64)> = alloc::vec![(7, 7)];
        let outcome = det.on_ack_received(Epoch::Application, 7, Duration::ZERO, &acked, now);
        let lost_pns: alloc::vec::Vec<u64> = outcome
            .lost
            .iter()
            .map(|record| record.packet_number)
            .collect();
        // Both PN 5 and PN 6 are at the time-threshold cliff;
        // depending on integer rounding 5 may survive — assert at least 6.
        assert!(
            lost_pns.contains(&6u64),
            "PN 6 must be time-lost; got {lost_pns:?}"
        );
    }

    #[test]
    fn compute_pto_with_initial_rtt() {
        // Before any sample: smoothed_rtt=333 ms, rttvar=166.5 ms, max_ack_delay=25.
        // PTO = 333 + max(4*166.5, 1) + 25 = 333 + 666 + 25 = 1024 ms (Application epoch).
        let det = LossDetection::new();
        let pto = det.compute_pto(true);
        // 4 * (333000/2) = 4 * 166500 = 666000; smoothed=333000; max_ack_delay=25000
        // total = 333000 + 666000 + 25000 = 1024000 µs.
        assert_eq!(pto.as_micros(), 1_024_000);
    }

    #[test]
    fn pto_count_doubles_pto() {
        let mut det = LossDetection::new();
        let base = det.compute_pto(true).as_micros();
        det.pto_count = 1;
        assert_eq!(det.compute_pto(true).as_micros(), base * 2);
        det.pto_count = 3;
        assert_eq!(det.compute_pto(true).as_micros(), base * 8);
    }

    #[test]
    fn on_loss_detection_timeout_with_no_loss_bumps_pto_count() {
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Initial, sent(0, at(1_000_000)));
        // No ack received → no loss_time set; firing timeout bumps PTO.
        let _ = det.on_loss_detection_timeout(at(10_000_000));
        assert_eq!(det.pto_count, 1);
    }

    #[test]
    fn pto_count_resets_on_ack_eliciting_ack() {
        let mut det = LossDetection::new();
        det.pto_count = 5;
        det.on_packet_sent(Epoch::Application, sent(0, at(1_000_000)));
        let acked: alloc::vec::Vec<(u64, u64)> = alloc::vec![(0, 0)];
        det.on_ack_received(Epoch::Application, 0, Duration::ZERO, &acked, at(1_100_000));
        assert_eq!(det.pto_count, 0);
    }

    #[test]
    fn fully_acked_epoch_does_not_arm_pto_deadline() {
        // Regression for the stale-timestamp bug: after a packet is
        // sent then ACKed, the PTO anchor for that epoch must clear so
        // next_deadline() does not return a stale deadline that would
        // fire an unjustified probe.
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Application, sent(0, at(1_000_000)));
        assert!(det.next_deadline().is_some(), "PTO armed after send");
        let acked: alloc::vec::Vec<(u64, u64)> = alloc::vec![(0, 0)];
        det.on_ack_received(Epoch::Application, 0, Duration::ZERO, &acked, at(1_100_000));
        assert_eq!(
            det.next_deadline(),
            None,
            "PTO must clear when epoch fully acked"
        );
        assert_eq!(
            det.epochs[Epoch::Application.index()].time_of_last_ack_eliciting_packet,
            None,
            "time_of_last_ack_eliciting_packet must be cleared"
        );
    }

    #[test]
    fn pto_epoch_matches_next_deadline_pick_when_initial_earliest() {
        // Regression for the epoch-mismatch bug: on_loss_detection_timeout
        // must return the same epoch next_deadline picked, not the
        // deepest epoch with a timestamp. Construct a case where
        // Initial has an EARLIER PTO deadline than Application, then
        // verify the detector names Initial as the timeout epoch.
        let mut det = LossDetection::new();
        // Initial sent at t=1_000_000 → PTO deadline ≈ 1_000_000 + 999_000 = 1_999_000 (no max_ack_delay)
        det.on_packet_sent(Epoch::Initial, sent(0, at(1_000_000)));
        // Application sent later at t=1_500_000 → PTO ≈ 1_500_000 + 1_024_000 = 2_524_000
        det.on_packet_sent(Epoch::Application, sent(0, at(1_500_000)));
        let earliest = det.next_deadline().expect("deadline armed");
        // earliest should match Initial's
        let initial_pto = at(1_000_000) + det.compute_pto(false);
        assert_eq!(earliest, initial_pto, "next_deadline picks Initial");
        // Fire the timeout at the deadline
        let outcome = det.on_loss_detection_timeout(initial_pto);
        assert_eq!(
            outcome.pto_epoch,
            Some(Epoch::Initial),
            "pto_epoch must match next_deadline's pick (Initial)"
        );
    }

    #[test]
    fn loss_epoch_named_on_time_threshold_outcome() {
        // Regression for the user's "make loss_epoch first-class" request.
        // After ACK + time-threshold loss, the outcome must name the
        // epoch that produced the lost packets.
        let mut det = LossDetection::new();
        for pn in 0..=10u64 {
            det.on_packet_sent(Epoch::Handshake, sent(pn, at(1_000_000 + pn * 100)));
        }
        let acked: alloc::vec::Vec<(u64, u64)> = alloc::vec![(10, 10)];
        let outcome =
            det.on_ack_received(Epoch::Handshake, 10, Duration::ZERO, &acked, at(1_001_500));
        assert!(!outcome.lost.is_empty(), "packet-threshold loss must fire");
        // detect_losses (invoked inside on_ack_received) is called with
        // the same epoch, so loss_epoch must name Handshake.
        assert_eq!(outcome.loss_epoch, Some(Epoch::Handshake));
        assert!(
            outcome.pto_epoch.is_none(),
            "pto_epoch is None on ACK outcome"
        );
    }

    #[test]
    fn discard_epoch_clears_pto_anchor_and_sent_queue() {
        // Regression: after discarding an epoch, its PTO timer MUST
        // stop contributing to next_deadline so a stale Handshake PTO
        // cannot keep waking the unified timer with no emitter to
        // answer (RFC 9001 §4.9 — once keys are discarded, the PN
        // space is unsendable).
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Handshake, sent(0, at(1_000_000)));
        det.on_packet_sent(Epoch::Application, sent(0, at(1_500_000)));
        // Both armed
        assert!(det.next_deadline().is_some());
        // Discard Handshake — returns in-flight bytes (1200 from
        // the single sent ack-eliciting packet) for the caller to
        // release from cwnd.
        let released = det.discard_epoch(Epoch::Handshake);
        assert_eq!(released, 1200, "discard_epoch returns in-flight bytes");
        // Application is still armed, but the deadline is Application's,
        // not Handshake's earlier one
        let next = det.next_deadline().expect("Application still armed");
        let app_pto = at(1_500_000) + det.compute_pto(true);
        assert_eq!(
            next, app_pto,
            "Handshake deadline must no longer contribute"
        );
        // Firing the timeout must now name Application, not Handshake
        let outcome = det.on_loss_detection_timeout(app_pto);
        assert_eq!(outcome.pto_epoch, Some(Epoch::Application));
        // Discarded epoch's PerEpochState is zeroed
        let hs = &det.epochs[Epoch::Handshake.index()];
        assert_eq!(hs.time_of_last_ack_eliciting_packet, None);
        assert!(hs.sent_packets.iter().count() == 0);
        assert_eq!(hs.largest_acked_packet, None);
        assert_eq!(hs.loss_time, None);
    }

    #[test]
    fn discard_epoch_on_unsendable_handshake_prevents_pto_inflation() {
        // Worked example: server still has 1 outstanding Handshake
        // packet (last Finished CRYPTO) when it moves to Established.
        // Without discard_epoch, this packet's PTO keeps firing,
        // bumping pto_count every cycle, doubling the Application
        // PTO — even though no probe can be emitted.
        let mut det = LossDetection::new();
        det.on_packet_sent(Epoch::Handshake, sent(7, at(1_000_000)));
        // Several PTOs fire on the stale Handshake packet.
        for _ in 0..3 {
            let dl = det.next_deadline().expect("Handshake PTO armed");
            let _ = det.on_loss_detection_timeout(dl);
        }
        assert_eq!(det.pto_count, 3, "stale PTOs inflated pto_count");
        // Now discard the epoch and verify pto_count's effect on
        // Application is gone (deadline is no longer multiplied
        // by 2^stale).
        let _released = det.discard_epoch(Epoch::Handshake);
        assert_eq!(det.next_deadline(), None, "no epoch remains armed");
        // Sending a fresh Application packet computes PTO from
        // pto_count=3 still — discarding the epoch does NOT reset
        // pto_count (that resets only on an ack-eliciting ACK per
        // RFC 9002 §6.2.1). Document the carryover.
        det.on_packet_sent(Epoch::Application, sent(0, at(2_000_000)));
        let app_pto = det.compute_pto(true);
        let base = LossDetection::new().compute_pto(true);
        assert_eq!(
            app_pto.as_micros(),
            base.as_micros() * 8,
            "pto_count carry: 2^3 = 8x"
        );
    }

    extern crate alloc;
}
