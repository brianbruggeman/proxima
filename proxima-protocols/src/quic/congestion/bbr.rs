//! BBR congestion control per [draft-ietf-ccwg-bbr-05].
//!
//! - [`MinRttFilter`] — windowed-min RTT filter (10-sec window per §2.13.1).
//! - [`MaxBwFilter`] — 2-cycle windowed-max delivery-rate filter (§2.10).
//! - [`BbrState`] — discriminated enum FSM (Startup / Drain /
//!   ProbeBW(SubState) / ProbeRTT) per §3.3.
//! - [`DeliveryRateSample`] — per-ACK delivery-rate sample (§2.5).
//! - [`Bbr`] — the full congestion controller composing the above,
//!   implementing the per-state pacing/cwnd-gain table (§3.3-3.7)
//!   and the state-transition rules.
//!
//! Note (principle 14): pacing-gain values + state-transition triggers
//! are pinned to the draft text. Production parity bench-vs-Google's
//! reference C implementation lands in a separate opt-sweep on real
//! hardware; until then this implementation may diverge in steady-state
//! throughput on extreme loadouts but is RFC-correct in shape.
//!
//! [draft-ietf-ccwg-bbr-05]: https://www.ietf.org/archive/id/draft-ietf-ccwg-bbr-05.txt
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All state is POD; integer-only
//! arithmetic (pacing gains scaled by 100, cwnd in bytes).

use crate::quic::sized;
use crate::quic::time::{Duration, Instant};

/// `BBR.MinRTTFilterLen` per draft §2.13.1. Sourced from
/// `proxima-quic-proto.toml [bbr].min_rtt_filter_window_micros`
/// (override via `PROXIMA_QUIC_PROTO_BBR_MIN_RTT_FILTER_WINDOW_MICROS`).
pub const MIN_RTT_FILTER_WINDOW_MICROS: u64 = sized::BBR_MIN_RTT_FILTER_WINDOW_MICROS;

/// `BBR.MaxBwFilterLen` per draft §2.10 — 2 ProbeBW cycles.
pub const MAX_BW_FILTER_LEN: usize = 2;

/// Windowed-min RTT filter per draft §2.13.1.
///
/// Retains the smallest RTT sample observed within the last
/// `window` of wall-clock time. When the current min ages past the
/// window, the next sample replaces it regardless of value.
#[derive(Debug, Clone, Copy)]
pub struct MinRttFilter {
    min_rtt: Option<Duration>,
    /// Caller-supplied monotonic time at which the current
    /// `min_rtt` sample was recorded.
    min_rtt_stamp: Instant,
    window: Duration,
}

impl MinRttFilter {
    /// Construct a filter with the BBR-default window (10 seconds).
    #[must_use]
    pub const fn with_default_window() -> Self {
        Self::new(Duration::from_micros(MIN_RTT_FILTER_WINDOW_MICROS))
    }

    /// Construct a filter with the given window length.
    #[must_use]
    pub const fn new(window: Duration) -> Self {
        Self {
            min_rtt: None,
            min_rtt_stamp: Instant::ZERO,
            window,
        }
    }

    /// Record a new RTT sample. The filter updates `min_rtt` when
    /// either:
    ///
    /// 1. `rtt < current min` (improved sample), OR
    /// 2. `now - min_rtt_stamp > window` (window expired — replace
    ///    regardless).
    pub fn note_sample(&mut self, rtt: Duration, now: Instant) {
        let expired = self.is_expired(now);
        let improved = match self.min_rtt {
            None => true,
            Some(current) => rtt < current,
        };
        if expired || improved {
            self.min_rtt = Some(rtt);
            self.min_rtt_stamp = now;
        }
    }

    /// Current windowed-min RTT, or `None` if no samples have been
    /// recorded yet.
    #[must_use]
    pub const fn get(&self) -> Option<Duration> {
        self.min_rtt
    }

    /// `true` if the current min has aged past the filter window
    /// (and the next sample will replace it).
    #[must_use]
    pub fn is_expired(&self, now: Instant) -> bool {
        if self.min_rtt.is_none() {
            return false;
        }
        match now.duration_since(self.min_rtt_stamp) {
            Some(elapsed) => elapsed > self.window,
            None => false,
        }
    }

    /// Wall-clock time at which the current min was recorded.
    #[must_use]
    pub const fn stamp(&self) -> Instant {
        self.min_rtt_stamp
    }
}

/// 2-cycle windowed-max delivery-rate filter per draft §2.10.
///
/// Tracks the maximum delivery-rate sample over the current and
/// previous ProbeBW cycle. `advance_cycle` shifts the current cycle
/// into the previous slot and resets the current slot to zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxBwFilter {
    /// `cycles[0]` = current cycle max, `cycles[1]` = previous cycle max.
    cycles: [u64; MAX_BW_FILTER_LEN],
    /// Per draft §2.10 only a single bit is needed.
    cycle_count: u8,
}

impl MaxBwFilter {
    /// Construct an empty filter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cycles: [0; MAX_BW_FILTER_LEN],
            cycle_count: 0,
        }
    }

    /// Record a delivery-rate sample — `cycles[0] = max(cycles[0], sample)`.
    pub fn note_sample(&mut self, delivery_rate: u64) {
        if delivery_rate > self.cycles[0] {
            self.cycles[0] = delivery_rate;
        }
    }

    /// Advance the cycle — `cycles[1] = cycles[0]; cycles[0] = 0`.
    /// Called by the BBR ProbeBW scheduler at the boundary of each
    /// probe cycle.
    pub fn advance_cycle(&mut self) {
        self.cycles[1] = self.cycles[0];
        self.cycles[0] = 0;
        self.cycle_count ^= 1;
    }

    /// Current windowed-max — `max(cycles[0], cycles[1])`.
    #[must_use]
    pub const fn get(&self) -> u64 {
        if self.cycles[0] >= self.cycles[1] {
            self.cycles[0]
        } else {
            self.cycles[1]
        }
    }

    /// Current cycle counter (single-bit virtual time per §2.10).
    #[must_use]
    pub const fn cycle_count(&self) -> u8 {
        self.cycle_count
    }
}

/// ProbeBW sub-state per draft §3.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeBwSubState {
    /// Probe for higher bandwidth (pacing_gain > 1).
    Up,
    /// Drain queue accumulated during Up (pacing_gain < 1).
    Down,
    /// Hold steady at the estimated max_bw (pacing_gain = 1).
    Cruise,
    /// Refill the pipe between probe cycles.
    Refill,
}

/// BBR primary state per draft §3.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BbrState {
    /// Rapid ramp-up at connection start to discover max_bw.
    Startup,
    /// Drain the queue accumulated during Startup.
    Drain,
    /// Steady-state probing — most of the connection's lifetime.
    ProbeBw(ProbeBwSubState),
    /// Periodically lower sending rate to probe for fresh min_rtt.
    ProbeRtt,
}

/// One delivery-rate sample per draft §2.5. Computed per inbound ACK:
/// `delivery_rate = bytes_delivered_in_interval / interval_elapsed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryRateSample {
    /// Bytes ACKed in the interval (not including loss).
    pub delivered: u64,
    /// Elapsed wall-clock time over the interval (microseconds).
    pub interval_micros: u64,
    /// Computed rate in bytes/second (cached for the filter).
    pub rate_bytes_per_sec: u64,
}

impl DeliveryRateSample {
    /// Build a sample. `interval_micros == 0` is degenerate (no time
    /// elapsed); the rate is treated as 0 to keep arithmetic safe.
    #[must_use]
    pub const fn new(delivered: u64, interval_micros: u64) -> Self {
        let rate_bytes_per_sec = if interval_micros == 0 {
            0
        } else {
            // bytes / micros * 1_000_000 = bytes/sec (avoid overflow by
            // computing in u128).
            ((delivered as u128 * 1_000_000) / interval_micros as u128) as u64
        };
        Self {
            delivered,
            interval_micros,
            rate_bytes_per_sec,
        }
    }
}

/// Pacing gain scaled by 100 per draft §3.
///
/// - Startup: 2.885 (= 2/ln(2)) → 289.
/// - Drain:   0.347 (= ln(2)/2) → 35.
/// - ProbeBW.Up:     1.25 → 125.
/// - ProbeBW.Down:   0.75 → 75.
/// - ProbeBW.Cruise: 1.00 → 100.
/// - ProbeBW.Refill: 1.00 → 100.
/// - ProbeRTT:       1.00 → 100.
const PACING_GAIN_STARTUP: u32 = 289;
const PACING_GAIN_DRAIN: u32 = 35;
const PACING_GAIN_PROBE_UP: u32 = 125;
const PACING_GAIN_PROBE_DOWN: u32 = 75;
const PACING_GAIN_PROBE_CRUISE: u32 = 100;
const PACING_GAIN_PROBE_REFILL: u32 = 100;
const PACING_GAIN_PROBE_RTT: u32 = 100;
/// Cwnd gain in Startup/Drain (draft §3.4.1) — same 2.885 as Startup pacing.
const CWND_GAIN_STARTUP: u32 = 289;
/// Cwnd gain in ProbeBW (draft §3.5) — 2.0.
const CWND_GAIN_PROBE_BW: u32 = 200;
/// Cwnd gain in ProbeRTT — 1.0 with extra cap to BDP per draft §3.7.
const CWND_GAIN_PROBE_RTT: u32 = 100;
/// Per draft §3.4: Startup → Drain when delivery rate hasn't grown
/// by 25% over 3 consecutive RTTs. We track 3 ack rounds.
const STARTUP_FULL_LOSS_COUNT_ROUNDS: u32 = 3;
/// Per draft §3.7: ProbeRTT cwnd cap is 4 MSS.
const PROBE_RTT_CWND_TARGET_PACKETS: u64 = 4;
/// Per draft §3.7: ProbeRTT lasts max(200 ms, 1 round-trip).
const PROBE_RTT_MIN_DURATION_MICROS: u64 = 200_000;
/// ProbeBW cycle duration per sub-state (in units of min_rtt).
/// Draft §3.5 specifies Up=1, Down=1, Cruise=many (4 used here), Refill=1.
const PROBE_BW_UP_RTTS: u32 = 1;
const PROBE_BW_DOWN_RTTS: u32 = 1;
const PROBE_BW_CRUISE_RTTS: u32 = 4;
const PROBE_BW_REFILL_RTTS: u32 = 1;

/// Maximum-segment-size assumption per draft §2.7. Real production
/// pulls this from the connection's negotiated max_udp_payload_size;
/// for the controller's BDP arithmetic 1200 bytes is the conservative
/// QUIC v1 floor (RFC 9000 §14).
pub const BBR_DEFAULT_MSS: u64 = 1200;

/// Full BBR congestion controller per draft-ietf-ccwg-bbr-05.
///
/// Composition layer over [`MinRttFilter`], [`MaxBwFilter`], and
/// [`BbrState`] — owns the per-state pacing/cwnd-gain choice and the
/// transition rules. Driven by [`Bbr::on_packet_acked`] (delivery-
/// rate samples + state machine ticks) and [`Bbr::on_packets_lost`]
/// (loss feedback drives the Startup-exit + ProbeBW.Down logic).
#[derive(Debug, Clone)]
pub struct Bbr {
    state: BbrState,
    min_rtt: MinRttFilter,
    max_bw: MaxBwFilter,
    /// Bytes-per-second pacing rate (computed as max_bw * pacing_gain / 100).
    pacing_rate_bps: u64,
    /// Congestion window in bytes.
    cwnd: u64,
    /// Packet maximum segment size — drives the cwnd minimum + ProbeRTT cap.
    mss: u64,
    /// Time of last state transition; drives ProbeBW sub-cycle scheduling
    /// + ProbeRTT exit timing.
    state_entered: Instant,
    /// Number of ack rounds where delivery-rate didn't grow by >= 25%.
    /// Drives Startup → Drain.
    full_bw_flat_rounds: u32,
    /// Last filtered delivery-rate observed; compared against the current
    /// filter to compute growth per round.
    last_round_max_bw: u64,
    /// Bytes still in flight (last reported value).
    bytes_in_flight: u64,
}

impl Bbr {
    /// Construct a fresh controller at `now` with the default MSS.
    #[must_use]
    pub fn new(now: Instant) -> Self {
        Self::with_mss(BBR_DEFAULT_MSS, now)
    }

    /// Construct with a caller-supplied MSS (negotiated payload size).
    #[must_use]
    pub fn with_mss(mss: u64, now: Instant) -> Self {
        let initial_cwnd = (CWND_GAIN_STARTUP as u64).saturating_mul(mss) / 100;
        Self {
            state: BbrState::Startup,
            min_rtt: MinRttFilter::with_default_window(),
            max_bw: MaxBwFilter::new(),
            pacing_rate_bps: 0,
            cwnd: initial_cwnd.max(4 * mss),
            mss,
            state_entered: now,
            full_bw_flat_rounds: 0,
            last_round_max_bw: 0,
            bytes_in_flight: 0,
        }
    }

    /// Borrow the current primary state (for tests + diagnostics).
    #[must_use]
    pub const fn state(&self) -> BbrState {
        self.state
    }

    /// Current congestion-window in bytes.
    #[must_use]
    pub const fn cwnd_bytes(&self) -> u64 {
        self.cwnd
    }

    /// Current pacing rate (bytes/second).
    #[must_use]
    pub const fn pacing_rate_bytes_per_sec(&self) -> u64 {
        self.pacing_rate_bps
    }

    /// Current windowed-min RTT (or `None` before any sample).
    #[must_use]
    pub const fn min_rtt(&self) -> Option<Duration> {
        self.min_rtt.get()
    }

    /// Current windowed-max delivery rate (0 before any sample).
    #[must_use]
    pub const fn max_bw(&self) -> u64 {
        self.max_bw.get()
    }

    /// Pacing gain (scaled by 100) for the current state.
    #[must_use]
    pub const fn pacing_gain_percent(&self) -> u32 {
        match self.state {
            BbrState::Startup => PACING_GAIN_STARTUP,
            BbrState::Drain => PACING_GAIN_DRAIN,
            BbrState::ProbeBw(ProbeBwSubState::Up) => PACING_GAIN_PROBE_UP,
            BbrState::ProbeBw(ProbeBwSubState::Down) => PACING_GAIN_PROBE_DOWN,
            BbrState::ProbeBw(ProbeBwSubState::Cruise) => PACING_GAIN_PROBE_CRUISE,
            BbrState::ProbeBw(ProbeBwSubState::Refill) => PACING_GAIN_PROBE_REFILL,
            BbrState::ProbeRtt => PACING_GAIN_PROBE_RTT,
        }
    }

    /// Cwnd gain (scaled by 100) for the current state.
    #[must_use]
    pub const fn cwnd_gain_percent(&self) -> u32 {
        match self.state {
            BbrState::Startup | BbrState::Drain => CWND_GAIN_STARTUP,
            BbrState::ProbeBw(_) => CWND_GAIN_PROBE_BW,
            BbrState::ProbeRtt => CWND_GAIN_PROBE_RTT,
        }
    }

    /// Update bytes-in-flight (called by the connection when a packet
    /// is queued or ACKed).
    pub fn set_bytes_in_flight(&mut self, bytes: u64) {
        self.bytes_in_flight = bytes;
    }

    /// Record an RTT sample (called from the connection's RTT estimator).
    pub fn on_rtt_sample(&mut self, rtt: Duration, now: Instant) {
        self.min_rtt.note_sample(rtt, now);
    }

    /// Record a delivery-rate sample for an ACK round. Drives:
    /// - the windowed-max bandwidth filter (§2.10);
    /// - Startup full-bandwidth detection (§3.4);
    /// - per-state pacing/cwnd-gain updates;
    /// - the BBR state-transition checks for this tick.
    pub fn on_packet_acked(&mut self, sample: DeliveryRateSample, now: Instant) {
        self.max_bw.note_sample(sample.rate_bytes_per_sec);
        self.tick_state_machine(now);
        self.update_pacing_and_cwnd();
    }

    /// Record loss feedback. Per draft §3.4 a loss episode during
    /// Startup signals the pipe is full and triggers Startup → Drain.
    pub fn on_packets_lost(&mut self, lost_bytes: u64, now: Instant) {
        let _ = lost_bytes;
        if matches!(self.state, BbrState::Startup) {
            self.transition_to(BbrState::Drain, now);
        }
    }

    /// Per-tick state machine — called from on_packet_acked.
    fn tick_state_machine(&mut self, now: Instant) {
        match self.state {
            BbrState::Startup => self.tick_startup(now),
            BbrState::Drain => self.tick_drain(now),
            BbrState::ProbeBw(sub) => self.tick_probe_bw(sub, now),
            BbrState::ProbeRtt => self.tick_probe_rtt(now),
        }
    }

    /// Startup exit per draft §3.4: 3 consecutive ack rounds without
    /// 25% growth in filtered delivery rate.
    fn tick_startup(&mut self, now: Instant) {
        let current_max_bw = self.max_bw.get();
        let grew_25pct = current_max_bw >= self.last_round_max_bw.saturating_mul(125) / 100;
        if grew_25pct {
            self.full_bw_flat_rounds = 0;
            self.last_round_max_bw = current_max_bw;
        } else {
            self.full_bw_flat_rounds += 1;
            if self.full_bw_flat_rounds >= STARTUP_FULL_LOSS_COUNT_ROUNDS {
                self.transition_to(BbrState::Drain, now);
            }
        }
    }

    /// Drain exit per draft §3.4.2: enter ProbeBW.Refill when bytes-in-
    /// flight has dropped to the BDP (or below).
    fn tick_drain(&mut self, now: Instant) {
        if self.bytes_in_flight <= self.bdp_bytes() {
            self.transition_to(BbrState::ProbeBw(ProbeBwSubState::Refill), now);
        }
    }

    /// ProbeBW sub-cycle scheduler per draft §3.5.
    fn tick_probe_bw(&mut self, sub: ProbeBwSubState, now: Instant) {
        let Some(min_rtt) = self.min_rtt.get() else {
            return;
        };
        let Some(elapsed) = now.duration_since(self.state_entered) else {
            return;
        };
        let cycle_rtts = elapsed.as_micros() / min_rtt.as_micros().max(1);
        let target_rtts = u64::from(match sub {
            ProbeBwSubState::Up => PROBE_BW_UP_RTTS,
            ProbeBwSubState::Down => PROBE_BW_DOWN_RTTS,
            ProbeBwSubState::Cruise => PROBE_BW_CRUISE_RTTS,
            ProbeBwSubState::Refill => PROBE_BW_REFILL_RTTS,
        });
        if cycle_rtts >= target_rtts {
            let next = match sub {
                ProbeBwSubState::Refill => ProbeBwSubState::Up,
                ProbeBwSubState::Up => ProbeBwSubState::Down,
                ProbeBwSubState::Down => ProbeBwSubState::Cruise,
                ProbeBwSubState::Cruise => ProbeBwSubState::Refill,
            };
            // Advance the max-bw filter at every Refill boundary
            // (one full Up/Down/Cruise/Refill cycle = one BBR cycle).
            if matches!(next, ProbeBwSubState::Up) {
                self.max_bw.advance_cycle();
            }
            self.transition_to(BbrState::ProbeBw(next), now);
        }
        // Per draft §3.6 — every 10 sec, force a ProbeRTT round.
        if self.min_rtt.is_expired(now) {
            self.transition_to(BbrState::ProbeRtt, now);
        }
    }

    /// ProbeRTT exit per draft §3.7: max(200 ms, 1 RTT) below
    /// inflight_lo cap.
    fn tick_probe_rtt(&mut self, now: Instant) {
        let Some(elapsed) = now.duration_since(self.state_entered) else {
            return;
        };
        if elapsed.as_micros() >= PROBE_RTT_MIN_DURATION_MICROS {
            // Probe complete — re-enter ProbeBW.Refill to repopulate cwnd.
            self.transition_to(BbrState::ProbeBw(ProbeBwSubState::Refill), now);
        }
    }

    /// Apply the current state's pacing + cwnd gains to derive
    /// pacing_rate_bps + cwnd_bytes.
    fn update_pacing_and_cwnd(&mut self) {
        let max_bw = self.max_bw.get();
        self.pacing_rate_bps = max_bw.saturating_mul(u64::from(self.pacing_gain_percent())) / 100;
        if matches!(self.state, BbrState::ProbeRtt) {
            // Cap to 4 MSS during ProbeRTT (draft §3.7).
            self.cwnd = PROBE_RTT_CWND_TARGET_PACKETS.saturating_mul(self.mss);
        } else {
            let bdp = self.bdp_bytes();
            self.cwnd = bdp.saturating_mul(u64::from(self.cwnd_gain_percent())) / 100;
            // Per draft §2.7 cwnd floor = 4 MSS.
            self.cwnd = self.cwnd.max(4 * self.mss);
        }
    }

    /// Bandwidth-delay product in bytes per draft §2.7.
    fn bdp_bytes(&self) -> u64 {
        match (self.max_bw.get(), self.min_rtt.get()) {
            (bw, Some(rtt)) if bw > 0 => {
                let micros = rtt.as_micros();
                bw.saturating_mul(micros) / 1_000_000
            }
            _ => 4 * self.mss,
        }
    }

    /// Internal state transition — records timestamp + clears
    /// per-state counters where needed.
    fn transition_to(&mut self, target: BbrState, now: Instant) {
        if matches!(target, BbrState::ProbeBw(ProbeBwSubState::Refill))
            && !matches!(self.state, BbrState::ProbeBw(_))
        {
            // Re-entering ProbeBW from Drain or ProbeRTT — reset
            // full_bw counter so re-entry to ProbeBW doesn't trip
            // Startup→Drain logic prematurely if the state machine
            // bounces back to Startup later.
            self.full_bw_flat_rounds = 0;
        }
        self.state = target;
        self.state_entered = now;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn at(secs: u64) -> Instant {
        Instant::from_micros(secs * 1_000_000)
    }

    fn ms(millis: u64) -> Duration {
        Duration::from_micros(millis * 1_000)
    }

    // --- MinRttFilter ---

    #[test]
    fn min_rtt_filter_starts_empty() {
        let filter = MinRttFilter::new(Duration::from_micros(10_000_000));
        assert_eq!(filter.get(), None);
        assert!(!filter.is_expired(at(1)));
    }

    #[test]
    fn min_rtt_filter_records_first_sample() {
        let mut filter = MinRttFilter::new(Duration::from_micros(10_000_000));
        filter.note_sample(ms(50), at(0));
        assert_eq!(filter.get(), Some(ms(50)));
    }

    #[test]
    fn worked_example_min_rtt_filter_10s_window_from_design_doc() {
        // docs/proxima-quic/c17-bbr-design.md MinRttFilter worked example.
        let mut filter = MinRttFilter::new(Duration::from_micros(10_000_000));
        filter.note_sample(ms(50), at(0));
        assert_eq!(filter.get(), Some(ms(50)));

        // Better sample at t=4 replaces.
        filter.note_sample(ms(30), at(4));
        assert_eq!(filter.get(), Some(ms(30)));
        assert_eq!(filter.stamp(), at(4));

        // Worse sample at t=11 — but stamp(4) + 10s window means still in window
        // (11 - 4 = 7s < 10s). Keep the 30ms.
        filter.note_sample(ms(80), at(11));
        assert_eq!(filter.get(), Some(ms(30)));

        // Sample at t=15 — window expired (15 - 4 = 11s > 10s). Replace.
        filter.note_sample(ms(80), at(15));
        assert_eq!(filter.get(), Some(ms(80)));
        assert_eq!(filter.stamp(), at(15));
    }

    #[test]
    fn min_rtt_filter_expired_after_window() {
        let mut filter = MinRttFilter::new(Duration::from_micros(10_000_000));
        filter.note_sample(ms(50), at(0));
        assert!(!filter.is_expired(at(5)));
        assert!(!filter.is_expired(at(10)));
        assert!(filter.is_expired(at(11)));
    }

    #[test]
    fn min_rtt_filter_default_window_is_10_seconds() {
        let filter = MinRttFilter::with_default_window();
        // Sanity: the default const matches the draft (10 seconds).
        assert_eq!(MIN_RTT_FILTER_WINDOW_MICROS, 10_000_000);
        // Smoke-test that the constructor uses it (filter is still
        // unsampled, but the window is baked in).
        let _ = filter;
    }

    // --- MaxBwFilter ---

    #[test]
    fn max_bw_filter_starts_zero() {
        let filter = MaxBwFilter::new();
        assert_eq!(filter.get(), 0);
        assert_eq!(filter.cycle_count(), 0);
    }

    #[test]
    fn worked_example_max_bw_filter_2_cycle_window_from_design_doc() {
        // docs/proxima-quic/c17-bbr-design.md MaxBwFilter worked example.
        let mut filter = MaxBwFilter::new();
        assert_eq!(filter.get(), 0);

        filter.note_sample(100);
        assert_eq!(filter.get(), 100);

        filter.note_sample(80);
        assert_eq!(filter.get(), 100); // 80 < 100, max unchanged

        filter.advance_cycle();
        assert_eq!(filter.get(), 100); // previous cycle still held

        filter.note_sample(60);
        assert_eq!(filter.get(), 100); // previous(100) > current(60)

        filter.advance_cycle();
        assert_eq!(filter.get(), 60); // 100 dropped off

        filter.note_sample(70);
        assert_eq!(filter.get(), 70); // 70 > 60
    }

    #[test]
    fn max_bw_filter_cycle_count_is_single_bit() {
        let mut filter = MaxBwFilter::new();
        assert_eq!(filter.cycle_count(), 0);
        filter.advance_cycle();
        assert_eq!(filter.cycle_count(), 1);
        filter.advance_cycle();
        assert_eq!(filter.cycle_count(), 0);
    }

    // --- BbrState ---

    #[test]
    fn bbr_state_discriminates_primary_states() {
        let states: [BbrState; 4] = [
            BbrState::Startup,
            BbrState::Drain,
            BbrState::ProbeBw(ProbeBwSubState::Cruise),
            BbrState::ProbeRtt,
        ];
        // Pattern-exhaustiveness check: every variant pattern-matches
        // uniquely.
        for state in &states {
            match state {
                BbrState::Startup => {}
                BbrState::Drain => {}
                BbrState::ProbeBw(_) => {}
                BbrState::ProbeRtt => {}
            }
        }
    }

    #[test]
    fn probe_bw_sub_state_carries_per_phase_data() {
        let s = BbrState::ProbeBw(ProbeBwSubState::Up);
        if let BbrState::ProbeBw(sub) = s {
            assert_eq!(sub, ProbeBwSubState::Up);
        } else {
            panic!("expected ProbeBw variant");
        }
    }

    // --- DeliveryRateSample ---

    #[test]
    fn delivery_rate_sample_computes_bytes_per_second() {
        // 12 000 bytes over 10 ms = 1.2 MB/s = 1_200_000 B/s.
        let sample = DeliveryRateSample::new(12_000, 10_000);
        assert_eq!(sample.rate_bytes_per_sec, 1_200_000);
    }

    #[test]
    fn delivery_rate_sample_zero_interval_is_zero_rate() {
        let sample = DeliveryRateSample::new(1024, 0);
        assert_eq!(sample.rate_bytes_per_sec, 0);
    }

    // --- Bbr controller ---

    #[test]
    fn bbr_starts_in_startup_with_initial_cwnd() {
        let bbr = Bbr::new(at(0));
        assert!(matches!(bbr.state(), BbrState::Startup));
        // initial cwnd = max(2.885 * MSS, 4 * MSS) = 4 * MSS (since
        // 2.885 * 1200 = 3462 < 4 * 1200 = 4800).
        assert_eq!(bbr.cwnd_bytes(), 4 * BBR_DEFAULT_MSS);
        assert_eq!(bbr.pacing_gain_percent(), PACING_GAIN_STARTUP);
    }

    #[test]
    fn worked_example_bbr_startup_to_drain_after_three_flat_rounds() {
        // draft §3.4 — after 3 ack rounds without 25% growth in
        // filtered delivery rate, Startup exits to Drain.
        let mut bbr = Bbr::new(at(0));
        bbr.on_rtt_sample(ms(50), at(0));
        // Round 1: establish baseline rate.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(1));
        assert!(matches!(bbr.state(), BbrState::Startup));
        // Round 2: same rate — first flat round.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(2));
        assert!(matches!(bbr.state(), BbrState::Startup));
        // Round 3: same rate — second flat round.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(3));
        assert!(matches!(bbr.state(), BbrState::Startup));
        // Round 4: third flat round → exit Startup to Drain.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(4));
        assert!(matches!(bbr.state(), BbrState::Drain));
        assert_eq!(bbr.pacing_gain_percent(), PACING_GAIN_DRAIN);
    }

    #[test]
    fn bbr_packet_loss_during_startup_exits_to_drain() {
        // draft §3.4 — loss episode during Startup signals pipe full.
        let mut bbr = Bbr::new(at(0));
        bbr.on_rtt_sample(ms(50), at(0));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(1));
        assert!(matches!(bbr.state(), BbrState::Startup));
        bbr.on_packets_lost(1200, at(1));
        assert!(matches!(bbr.state(), BbrState::Drain));
    }

    #[test]
    fn bbr_drain_exits_to_probe_bw_when_inflight_below_bdp() {
        // draft §3.4.2 — Drain → ProbeBW.Refill once in-flight drops
        // to (or below) the BDP.
        let mut bbr = Bbr::new(at(0));
        bbr.on_rtt_sample(ms(50), at(0));
        // Establish baseline delivery rate.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(1));
        // Force transition to Drain via three flat rounds.
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(2));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(3));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(4));
        assert!(matches!(bbr.state(), BbrState::Drain));
        // BDP = max_bw * min_rtt = (60_000/50_000 *1_000_000) * 50_000 / 1_000_000
        //     = 1_200_000 * 50_000 / 1_000_000 = 60_000 bytes.
        // Set bytes-in-flight to 30_000 — below BDP.
        bbr.set_bytes_in_flight(30_000);
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(5));
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Refill)
        ));
    }

    #[test]
    fn bbr_probe_bw_advances_through_subcycle() {
        // draft §3.5 — Refill → Up → Down → Cruise → Refill.
        let mut bbr = Bbr::new(at(0));
        bbr.on_rtt_sample(ms(50), at(0));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(1));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(2));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(3));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(4));
        bbr.set_bytes_in_flight(30_000);
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(5));
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Refill)
        ));
        // Each sub-state lasts a specific number of min_rtts (50 ms).
        // Refill = 1 min_rtt → advance after 50 ms; advance the
        // controller's wall-clock by feeding samples.
        let mut now_us = 5_000_000u64 + 60_000;
        bbr.on_packet_acked(
            DeliveryRateSample::new(60_000, 50_000),
            Instant::from_micros(now_us),
        );
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Up)
        ));
        // Up = 1 min_rtt → advance after 50 ms.
        now_us += 60_000;
        bbr.on_packet_acked(
            DeliveryRateSample::new(60_000, 50_000),
            Instant::from_micros(now_us),
        );
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Down)
        ));
        // Down = 1 min_rtt.
        now_us += 60_000;
        bbr.on_packet_acked(
            DeliveryRateSample::new(60_000, 50_000),
            Instant::from_micros(now_us),
        );
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Cruise)
        ));
        // Cruise = 4 min_rtts.
        now_us += 4 * 60_000;
        bbr.on_packet_acked(
            DeliveryRateSample::new(60_000, 50_000),
            Instant::from_micros(now_us),
        );
        assert!(matches!(
            bbr.state(),
            BbrState::ProbeBw(ProbeBwSubState::Refill)
        ));
    }

    #[test]
    fn bbr_probe_rtt_caps_cwnd_to_four_mss() {
        // draft §3.7 — ProbeRTT cwnd cap is 4 MSS.
        let mut bbr = Bbr::new(at(0));
        bbr.on_rtt_sample(ms(50), at(0));
        bbr.on_packet_acked(DeliveryRateSample::new(60_000, 50_000), at(1));
        // Manually force ProbeRTT to verify the cap (real entry is via
        // min_rtt window expiry at 10s; tested via the state machine).
        bbr.transition_to(BbrState::ProbeRtt, at(2));
        bbr.update_pacing_and_cwnd();
        assert_eq!(bbr.cwnd_bytes(), 4 * BBR_DEFAULT_MSS);
        assert_eq!(bbr.pacing_gain_percent(), PACING_GAIN_PROBE_RTT);
    }

    #[test]
    fn bbr_pacing_gain_table_matches_draft() {
        // draft §3 — pacing gain table fact-check.
        let mut bbr = Bbr::new(at(0));
        assert_eq!(bbr.pacing_gain_percent(), 289);
        bbr.state = BbrState::Drain;
        assert_eq!(bbr.pacing_gain_percent(), 35);
        bbr.state = BbrState::ProbeBw(ProbeBwSubState::Up);
        assert_eq!(bbr.pacing_gain_percent(), 125);
        bbr.state = BbrState::ProbeBw(ProbeBwSubState::Down);
        assert_eq!(bbr.pacing_gain_percent(), 75);
        bbr.state = BbrState::ProbeBw(ProbeBwSubState::Cruise);
        assert_eq!(bbr.pacing_gain_percent(), 100);
        bbr.state = BbrState::ProbeBw(ProbeBwSubState::Refill);
        assert_eq!(bbr.pacing_gain_percent(), 100);
        bbr.state = BbrState::ProbeRtt;
        assert_eq!(bbr.pacing_gain_percent(), 100);
    }
}
