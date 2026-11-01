use crate::tcp_listener::Endpoint;
use crate::tcp_stack::{ConnId, Outbound, TcpStack};
use core::future::Future;
use crossbeam_queue::SegQueue;
use proxima_protocols::tcp::retx::MaxRetransmit;
use proxima_protocols::tcp::rtt::RtoEstimator;
use proxima_protocols::tcp::time::Instant;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use proxima_core::time::{self, Instant as ClockInstant};
use std::time::Duration;
use thiserror::Error;

/// Initial interval before an unanswered ARP request or SYN is sent again
/// (doubling on each resend), and between dials to a port that refused; the TCP
/// initial RTO.
pub const DEFAULT_RETRANSMIT: Duration = Duration::from_micros(RtoEstimator::RTO_MIN.as_micros());

const BACKOFF_CAP: Duration = Duration::from_micros(RtoEstimator::RTO_MAX.as_micros());
const MAX_DOUBLINGS: u32 = 16;

/// RFC 6298 §5.5: the wait before resend `resends` doubles from `initial`, up to
/// the RFC 6298 §2.5 maximum (an `initial` above that maximum is kept as is).
pub(crate) fn backoff_interval(initial: Duration, resends: u32) -> Duration {
    let doubled = initial.saturating_mul(1 << resends.min(MAX_DOUBLINGS));
    doubled.min(BACKOFF_CAP.max(initial))
}

pub(crate) fn resend_due(initial: Duration, resends: u32, waited: Duration) -> bool {
    waited >= backoff_interval(initial, resends)
}

pub(crate) fn wake_after(remaining: Duration, cx: &mut Context<'_>) {
    let mut deadline = proxima_core::time::sleep(remaining);
    if Pin::new(&mut deadline).poll(cx).is_ready() {
        cx.waker().wake_by_ref();
    }
}

#[derive(Debug, Error)]
pub enum DialError {
    #[error("no syn-ack after {attempts} syn transmissions")]
    SynUnanswered { attempts: u8 },
}

pub(crate) enum SynStep<'syn> {
    Send(&'syn [Outbound]),
    Wait(Duration),
    Exhausted,
}

// Drop cannot wait for a driver lock, so a dropped dial parks its (id, isn)
// here and the next lock holder cancels it against the stack.
pub(crate) struct DialLease {
    cancelled: SegQueue<(ConnId, u32)>,
}

impl DialLease {
    pub(crate) fn new() -> Self {
        Self {
            cancelled: SegQueue::new(),
        }
    }

    pub(crate) fn reclaim(&self, stack: &mut TcpStack) {
        while let Some((id, sequence)) = self.cancelled.pop() {
            stack.cancel_connect(id, sequence);
        }
    }

    pub(crate) fn claim(self: &Arc<Self>) -> DialGuard {
        DialGuard {
            lease: Arc::clone(self),
            started: None,
            syn: Vec::new(),
            syn_sent: None,
            syn_resends: 0,
        }
    }
}

pub(crate) struct DialGuard {
    lease: Arc<DialLease>,
    started: Option<(ConnId, u32)>,
    syn: Vec<Outbound>,
    syn_sent: Option<ClockInstant>,
    syn_resends: u8,
}

impl DialGuard {
    pub(crate) fn connection(&self) -> Option<ConnId> {
        self.started.map(|(id, _)| id)
    }

    pub(crate) fn start(
        &mut self,
        stack: &mut TcpStack,
        peer: Endpoint,
    ) -> io::Result<Vec<Outbound>> {
        let (id, outbound) = stack.connect(peer).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                "all ephemeral TCP ports are in use",
            )
        })?;
        self.started = Some((id, outbound[0].1.seq));
        self.syn.clone_from(&outbound);
        self.syn_sent = Some(time::now());
        Ok(outbound)
    }

    pub(crate) fn syn_interval(&self, initial: Duration) -> Duration {
        backoff_interval(initial, u32::from(self.syn_resends))
    }

    pub(crate) fn next_syn(&mut self, initial: Duration) -> SynStep<'_> {
        let waited = self.syn_sent.map_or(Duration::ZERO, |sent| time::now().saturating_duration_since(sent));
        self.next_syn_after(initial, waited)
    }

    fn next_syn_after(&mut self, initial: Duration, waited: Duration) -> SynStep<'_> {
        let interval = self.syn_interval(initial);
        if waited < interval {
            return SynStep::Wait(interval - waited);
        }
        if self.syn_resends >= MaxRetransmit::ABANDON_THRESHOLD {
            return SynStep::Exhausted;
        }
        self.syn_resends += 1;
        self.syn_sent = Some(time::now());
        SynStep::Send(&self.syn)
    }

    pub(crate) fn syn_transmissions(&self) -> u8 {
        self.syn_resends + 1
    }

    pub(crate) fn owns(&self, stack: &TcpStack) -> bool {
        self.started.is_some_and(|(id, sequence)| {
            stack.active_connect_sequence(id) == Some(sequence)
        })
    }

    pub(crate) fn connected(&self, stack: &TcpStack) -> bool {
        self.started.is_some_and(|(id, sequence)| {
            stack.active_connection_sequence(id) == Some(sequence)
        })
    }

    pub(crate) fn complete(self, stack: &TcpStack) -> io::Result<Self> {
        self.flush(stack)?;
        Ok(self)
    }

    pub(crate) fn read(&self, stack: &mut TcpStack, buffer: &mut [u8]) -> io::Result<usize> {
        let id = self.live_id(stack)?;
        Ok(stack.read(id, buffer))
    }

    pub(crate) fn write(
        &self,
        stack: &mut TcpStack,
        buffer: &[u8],
        now: Instant,
    ) -> io::Result<Vec<Outbound>> {
        let id = self.live_id(stack)?;
        Ok(stack.write(id, buffer, now))
    }

    pub(crate) fn close(&self, stack: &mut TcpStack) -> io::Result<Vec<Outbound>> {
        let id = self.live_id(stack)?;
        Ok(stack.close(id))
    }

    pub(crate) fn flush(&self, stack: &TcpStack) -> io::Result<()> {
        self.live_id(stack).map(|_| ())
    }

    fn live_id(&self, stack: &TcpStack) -> io::Result<ConnId> {
        match self.started {
            Some((id, _)) if self.connected(stack) => Ok(id),
            _ => Err(io::Error::from(io::ErrorKind::ConnectionReset)),
        }
    }
}

impl Drop for DialGuard {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            self.lease.cancelled.push(started);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcp_listener::Inbound;
    use proxima_protocols::inet::tcp::TcpFlags;

    const OUR_IP: [u8; 4] = [127, 0, 0, 1];
    const PEER: Endpoint = Endpoint {
        mac: [2, 0, 0, 0, 0, 2],
        ip: [127, 0, 0, 2],
        port: 4321,
    };

    fn from_peer(
        sequence: u32,
        acknowledgement: u32,
        flags: TcpFlags,
        payload: &[u8],
    ) -> Inbound<'_> {
        Inbound {
            source_mac: PEER.mac,
            source_ip: PEER.ip,
            source_port: PEER.port,
            flags,
            seq: sequence,
            ack: acknowledgement,
            window: 4096,
            payload,
        }
    }

    fn syn_ack() -> TcpFlags {
        TcpFlags {
            syn: true,
            ack: true,
            ..TcpFlags::default()
        }
    }

    fn ack() -> TcpFlags {
        TcpFlags {
            ack: true,
            ..TcpFlags::default()
        }
    }

    fn rst() -> TcpFlags {
        TcpFlags {
            rst: true,
            ..TcpFlags::default()
        }
    }

    fn establish(stack: &mut TcpStack, guard: &DialGuard, sequence: u32) {
        let (local_port, _, _) = guard.connection().expect("dial started");
        let reply = stack.on_inbound_to(
            local_port,
            &from_peer(900, sequence.wrapping_add(1), syn_ack(), &[]),
            Instant::ZERO,
        );
        assert_eq!(reply.len(), 1);
        assert_eq!(reply[0].1.ack, 901);
    }

    #[test]
    fn concurrent_dials_to_one_peer_get_distinct_local_ports_and_independent_data() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut first = lease.claim();
        let mut second = lease.claim();

        let first_syn = first.start(&mut stack, PEER).expect("first dial");
        let second_syn = second.start(&mut stack, PEER).expect("second dial");
        let (first_id, second_id) = (
            first.connection().expect("first id"),
            second.connection().expect("second id"),
        );

        assert_ne!(first_id.0, second_id.0, "each dial owns its local port");
        assert_eq!((first_id.1, first_id.2), (second_id.1, second_id.2));
        establish(&mut stack, &first, first_syn[0].1.seq);
        establish(&mut stack, &second, second_syn[0].1.seq);
        for (guard, local_port, body) in [
            (&first, first_id.0, b"first!!!"),
            (&second, second_id.0, b"second!!"),
        ] {
            let sequence = guard.started.expect("started").1;
            stack.on_inbound_to(
                local_port,
                &from_peer(901, sequence.wrapping_add(1), ack(), body),
                Instant::ZERO,
            );
        }
        let mut buffer = [0; 8];
        assert_eq!(first.read(&mut stack, &mut buffer).expect("first read"), 8);
        assert_eq!(&buffer, b"first!!!");
        assert_eq!(second.read(&mut stack, &mut buffer).expect("second read"), 8);
        assert_eq!(&buffer, b"second!!");
    }

    #[test]
    fn dropped_dial_is_reclaimed_and_its_port_returns_to_the_pool() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut cancelled = lease.claim();
        cancelled.start(&mut stack, PEER).expect("start dial");
        let (port, _, _) = cancelled.connection().expect("dial id");
        assert!(stack.owns_port(port));

        drop(cancelled);
        lease.reclaim(&mut stack);

        assert!(!stack.owns_port(port), "reclaim frees the ephemeral port");
        assert_eq!(stack.peer((port, PEER.ip, PEER.port)), None);
    }

    #[test]
    fn stale_cancellation_leaves_a_replacement_on_the_same_id_untouched() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut stale = lease.claim();
        let old = stale.start(&mut stack, PEER).expect("start stale dial");
        let (port, _, _) = stale.connection().expect("stale id");
        stack.on_inbound_to(port, &from_peer(900, 0, rst(), &[]), Instant::ZERO);
        assert!(!stack.owns_port(port), "reset releases the port");

        let mut replacement = lease.claim();
        let current = replacement.start(&mut stack, PEER).expect("start replacement");
        assert_ne!(old[0].1.seq, current[0].1.seq);
        drop(stale);
        lease.reclaim(&mut stack);

        assert!(replacement.owns(&stack), "stale drop must not cancel the replacement");
    }

    #[test]
    fn a_reset_connection_cannot_be_read_through_a_stale_handle() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut guard = lease.claim();
        let syn = guard.start(&mut stack, PEER).expect("start dial");
        establish(&mut stack, &guard, syn[0].1.seq);
        let (port, _, _) = guard.connection().expect("dial id");
        stack.on_inbound_to(port, &from_peer(901, 0, rst(), &[]), Instant::ZERO);

        let mut untouched = [0x7f; 8];
        let error = guard
            .read(&mut stack, &mut untouched)
            .expect_err("reset handle cannot read");

        assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
        assert_eq!(untouched, [0x7f; 8]);
    }

    #[test]
    fn an_unanswered_syn_waits_out_the_interval_before_it_is_resent() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut guard = lease.claim();
        guard.start(&mut stack, PEER).expect("start dial");

        let step = guard.next_syn(Duration::from_secs(3600));

        assert!(matches!(step, SynStep::Wait(remaining) if remaining > Duration::from_secs(3599)));
        assert_eq!(guard.syn_transmissions(), 1);
    }

    #[test]
    fn an_unanswered_syn_is_resent_unchanged_until_the_abandon_threshold() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let lease = Arc::new(DialLease::new());
        let mut guard = lease.claim();
        let original = guard.start(&mut stack, PEER).expect("start dial");
        let original_sequence = original[0].1.seq;

        for _ in 0..MaxRetransmit::ABANDON_THRESHOLD {
            let SynStep::Send(resent) = guard.next_syn(Duration::ZERO) else {
                panic!("an expired syn below the threshold must be resent");
            };
            assert_eq!(resent.len(), 1);
            assert_eq!(resent[0].0, PEER);
            assert_eq!(resent[0].1.seq, original_sequence, "a resend reuses the isn");
            assert!(resent[0].1.flags.syn);
        }

        assert!(matches!(guard.next_syn(Duration::ZERO), SynStep::Exhausted));
        assert_eq!(guard.syn_transmissions(), MaxRetransmit::ABANDON_THRESHOLD + 1);
    }

    const INITIAL: Duration = Duration::from_millis(10);

    fn started_guard(stack: &mut TcpStack) -> DialGuard {
        let lease = Arc::new(DialLease::new());
        let mut guard = lease.claim();
        guard.start(stack, PEER).expect("start dial");
        guard
    }


    #[test]
    fn the_wait_before_each_resend_doubles_from_the_initial_interval() {
        let intervals: Vec<Duration> = (0..5).map(|resends| backoff_interval(INITIAL, resends)).collect();

        let expected: Vec<Duration> = [10, 20, 40, 80, 160].into_iter().map(Duration::from_millis).collect();
        assert_eq!(intervals, expected);
    }

    #[test]
    fn the_backoff_is_capped_at_the_rfc_6298_maximum() {
        let capped = Duration::from_micros(RtoEstimator::RTO_MAX.as_micros());

        assert_eq!(backoff_interval(Duration::from_secs(20), 1), Duration::from_secs(40));
        assert_eq!(backoff_interval(Duration::from_secs(20), 2), capped);
        assert_eq!(backoff_interval(Duration::from_secs(20), 30), capped);
        assert_eq!(backoff_interval(INITIAL, u32::MAX), capped);
    }

    #[test]
    fn an_initial_interval_above_the_cap_is_never_shortened() {
        assert_eq!(backoff_interval(Duration::from_secs(3600), 5), Duration::from_secs(3600));
    }

    #[test]
    fn a_resend_is_due_exactly_when_the_backed_off_interval_has_elapsed() {
        assert!(!resend_due(Duration::from_secs(10), 1, Duration::from_secs(15)));
        assert!(resend_due(Duration::from_secs(10), 1, Duration::from_secs(20)));
        assert!(resend_due(Duration::from_secs(10), 0, Duration::from_secs(15)));
    }

    #[test]
    fn a_zero_initial_interval_stays_zero_however_many_resends() {
        assert_eq!(backoff_interval(Duration::ZERO, 200), Duration::ZERO);
    }

    #[test]
    fn the_kth_syn_resend_is_withheld_until_initial_times_two_to_the_k_has_elapsed() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let mut guard = started_guard(&mut stack);

        for resend in 0..MaxRetransmit::ABANDON_THRESHOLD {
            let interval = backoff_interval(INITIAL, u32::from(resend));
            let SynStep::Wait(remaining) = guard.next_syn_after(INITIAL, interval / 2) else {
                panic!("half the interval has elapsed before resend {resend}, so it must wait");
            };
            assert_eq!(remaining, interval / 2, "resend {resend} waits out the rest of {interval:?}");

            let just_short = interval - Duration::from_nanos(1);
            assert!(matches!(guard.next_syn_after(INITIAL, just_short), SynStep::Wait(_)), "resend {resend} is not due a nanosecond early");
            assert!(matches!(guard.next_syn_after(INITIAL, interval), SynStep::Send(_)), "resend {resend} fires once {interval:?} has elapsed");
            assert_eq!(guard.syn_interval(INITIAL), (interval * 2).min(BACKOFF_CAP), "the wait after resend {resend} doubles up to the cap");
        }
    }

    #[test]
    fn abandoning_after_the_threshold_still_reports_every_transmission() {
        let mut stack = TcpStack::new(OUR_IP, 1234, 100);
        let mut guard = started_guard(&mut stack);

        for _ in 0..MaxRetransmit::ABANDON_THRESHOLD {
            let due = guard.syn_interval(INITIAL);
            assert!(matches!(guard.next_syn_after(INITIAL, due), SynStep::Send(_)));
        }
        let due = guard.syn_interval(INITIAL);

        assert!(matches!(guard.next_syn_after(INITIAL, due), SynStep::Exhausted));
        assert_eq!(guard.syn_transmissions(), MaxRetransmit::ABANDON_THRESHOLD + 1);
    }
}
