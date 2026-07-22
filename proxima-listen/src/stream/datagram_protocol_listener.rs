//! [`DatagramProtocol`] — a sans-IO state machine seam for connectionless
//! protocols whose timer emits outbound datagrams with no inbound datagram
//! to trigger them (the QUIC-handshake-retransmit shape). This is the
//! stateful sibling of [`super::datagram_listener::DatagramListenProtocol`]:
//! that listener races `recv` against `shutdown` and treats every datagram
//! as a stateless request/reply; a state machine that must retransmit on an
//! idle socket needs a THIRD race arm — a timer — which this module adds.
//!
//! [`DatagramProtocol`] itself touches no socket, no `SendBatch`, no clock:
//! it is fed `now` and inbound bytes, and it fills a caller-owned buffer with
//! outbound bytes. [`DatagramProtocolListenProtocol`] is the one std-tier
//! [`crate::ListenProtocol`] that drives it — binds the socket via the
//! injected [`crate::ServeContext::datagram_factory`] (exactly like
//! `DatagramListenProtocol`), and every tick races the batched recv, the
//! protocol's own [`DatagramProtocol::next_deadline`], and shutdown
//! (mirroring `DatagramListenProtocol`'s two-way race, plus a timer arm).
//! The recv/send batching and backpressure flush (`datagram_listener`'s
//! `stage_reply` / `flush_send`, crate-private) are reused verbatim rather
//! than duplicated.
//!
//! A protocol that dispatches work concurrently (e.g. H3-over-QUIC's
//! per-connection request handlers) owns ITS OWN loop and races its own
//! completion source alongside recv/timer/shutdown there — see
//! `H3NativeListenProtocol::serve` in
//! `proxima-http/src/http3/native/listen.rs`. This generic driver stays
//! minimal: genuinely connectionless protocols have no background
//! dispatch to race.
//!
//! No fixed tick: the reactor arms the EXACT next protocol deadline, or
//! nothing when [`DatagramProtocol::next_deadline`] returns `None` — an idle
//! state machine costs zero wakeups. [`DatagramProtocol::transmit`]
//! drains unconditionally every iteration (after either arm wakes), so a
//! reply staged by `on_datagram` and a retransmit staged by `on_timeout`
//! both ship through the identical send path.
//!
//! The `build: Fn() -> P` closure IS the per-`serve()` state-machine
//! factory; a bespoke factory trait would add a type with nothing a closure
//! doesn't already express.

use core::fmt::Debug;
use core::marker::PhantomData;
use core::time::Duration;
use std::future::{Future, poll_fn};
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;

use futures::channel::oneshot;
use futures::future::{self, Either, select};
use proxima_telemetry::{debug, warn};
use serde_json::Value;

use crate::{ListenProtocol, ServeContext};
use proxima_core::ProximaError;
use proxima_core::datagram_batch::DefaultDatagramBatch;
use proxima_core::time::Instant;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::stream::DatagramSocketBatchExt;

use super::datagram_listener::{flush_send, stage_reply};

const TRANSMIT_SCRATCH_BYTES: usize = 2048;

/// A connectionless, sans-IO state machine driven by
/// [`DatagramProtocolListenProtocol`]. Implementors never touch a socket —
/// they are told `now` and handed borrowed inbound bytes, and they fill a
/// caller-owned buffer with outbound bytes. This is the seam a stateful
/// protocol whose timer must fire with no inbound datagram (a QUIC
/// handshake retransmit, a RADIUS retry) plugs into; a stateless
/// request/reply protocol keeps using
/// [`super::datagram_listener::DatagramListenProtocol`] instead.
pub trait DatagramProtocol {
    /// Protocol-level failure. Logged and the offending call skipped — a
    /// malformed datagram or a failed transmit must never tear down a
    /// connectionless listener. `'static` because the async methods below
    /// return an opaque RPITIT future — without an explicit outlives bound
    /// on the associated type, the compiler cannot prove `Self::Err` is
    /// well-formed for the elided per-call borrow those futures capture.
    /// Every error type used with this trait is already owned/`'static` in
    /// practice (`Infallible`, `ListenerError`, …), so this is not a new
    /// restriction.
    type Err: Debug + 'static;

    /// Feed one received datagram from `peer`. Async (RPITIT, explicit
    /// `Send` bound) so an implementor can await dispatch inline — e.g. an
    /// H3-over-QUIC protocol driving a request handler to completion before
    /// staging its reply — without the driver boxing the protocol's future
    /// or reopening the 4-tier Send/Unpin split. The explicit `+ Send` is
    /// deliberate: it forces every implementor's future Send, which is
    /// correct for the multi-threaded datagram runtime this trait serves.
    /// No explicit `'_` on the return type — edition-2024 RPITIT captures
    /// every in-scope lifetime by default (both `&mut self`'s and
    /// `datagram`'s); writing a single `'_` here would instead invoke the
    /// older self-only elision rule and produce a future whose captured
    /// `datagram` borrow the trait signature never promised, which fails
    /// to typecheck at every impl site.
    fn on_datagram(&mut self, now: Instant, peer: SocketAddr, datagram: &[u8]) -> impl Future<Output = Result<(), Self::Err>> + Send;

    /// Fire the elapsed timer. Async for the same reason as
    /// [`on_datagram`](Self::on_datagram).
    fn on_timeout(&mut self, now: Instant) -> impl Future<Output = Result<(), Self::Err>> + Send;

    /// Earliest deadline at which [`on_timeout`](Self::on_timeout) must be
    /// called, if any. `None` means the state machine is quiescent — the
    /// serve loop arms no timer and parks on recv + shutdown alone. Stays
    /// SYNC — a pure query over already-held state, no I/O or dispatch to
    /// await.
    fn next_deadline(&self) -> Option<Instant>;

    /// Drain one pending outbound datagram into `buf`. `Ok(Some((len,
    /// peer)))` for a datagram to ship, `Ok(None)` once drained for this
    /// tick. Called in a loop every tick regardless of which race arm woke.
    /// Async for the same reason as [`on_datagram`](Self::on_datagram) —
    /// named `transmit` rather than `poll_transmit` now that it awaits
    /// rather than polls.
    fn transmit(&mut self, now: Instant, buf: &mut [u8]) -> impl Future<Output = Result<Option<(usize, SocketAddr)>, Self::Err>> + Send;
}

fn instant_now<Clk: Clock>(clock: &Clk) -> Instant {
    Instant::from_monotonic(Duration::from_nanos(clock.now_nanos()))
}

/// Connectionless [`ListenProtocol`] over a [`DatagramProtocol`] state
/// machine. `build` is called once per `serve()` to construct a fresh `P` —
/// the per-invocation factory a control plane rebind needs, mirroring
/// `StreamListenerProtocol::with_factory`.
/// Generic over the [`Clock`] the serve loop reads `now` from AND arms its
/// timer sleep from — one clock, no split (mirrors
/// `H3NativeListenProtocol`). Production defaults `Clk` to [`TimeClock`];
/// deterministic tests inject a mock clock via [`Self::with_clock`].
pub struct DatagramProtocolListenProtocol<Build, P, Clk = TimeClock>
where
    Build: Fn() -> P + Send + Sync + 'static,
    P: DatagramProtocol,
{
    label: String,
    build: Build,
    clock: Clk,
    _protocol: PhantomData<fn() -> P>,
}

impl<Build, P> DatagramProtocolListenProtocol<Build, P, TimeClock>
where
    Build: Fn() -> P + Send + Sync + 'static,
    P: DatagramProtocol,
{
    #[must_use]
    pub fn new(label: impl Into<String>, build: Build) -> Self {
        Self {
            label: label.into(),
            build,
            clock: TimeClock,
            _protocol: PhantomData,
        }
    }
}

impl<Build, P, Clk> DatagramProtocolListenProtocol<Build, P, Clk>
where
    Build: Fn() -> P + Send + Sync + 'static,
    P: DatagramProtocol,
{
    /// Materialise with an explicit [`Clock`] — the seam a deterministic
    /// test injects a fake clock through. Production goes via [`Self::new`],
    /// which defaults `Clk` to [`TimeClock`].
    #[must_use]
    pub fn with_clock(label: impl Into<String>, build: Build, clock: Clk) -> Self {
        Self {
            label: label.into(),
            build,
            clock,
            _protocol: PhantomData,
        }
    }
}

impl<Build, P, Clk> ListenProtocol for DatagramProtocolListenProtocol<Build, P, Clk>
where
    Build: Fn() -> P + Send + Sync + 'static,
    P: DatagramProtocol + Send + 'static,
    Clk: Clock + Clone + Send + Sync + 'static,
    Clk::Delay: Send,
{
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        _dispatch: PipeHandle,
        _spec: &Value,
        context: ServeContext,
        mut shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let label = self.label.clone();
        let build = &self.build;
        let clock = self.clock.clone();
        let datagram_factory = context.datagram_factory.clone();
        let ready_signal = context.ready_signal.clone();

        Box::pin(async move {
            let datagram_factory = datagram_factory.ok_or_else(|| {
                ProximaError::Config("datagram protocol listener requires a datagram factory".into())
            })?;
            let mut socket = datagram_factory
                .bind(bind)
                .map_err(|err| ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {err}"))))?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(label = %label, %bind, "datagram protocol listener bound");

            let mut proto = build();
            let mut batch = DefaultDatagramBatch::new();
            let mut transmit_scratch = [0u8; TRANSMIT_SCRATCH_BYTES];

            loop {
                let tick_start = instant_now(&clock);
                let next_delay = proto
                    .next_deadline()
                    .map(|deadline| deadline.saturating_duration_since(tick_start));
                // The race yields an `Either<Either<..>,..>` whose
                // unresolved arms hold pinned borrows of `socket` /
                // `batch.recv` — those borrows must end before the match
                // body below touches those same values again. Collapsing
                // to this small owned enum INSIDE the scoped block
                // (rather than matching on the raw `Either` outside it)
                // is what lets the borrows drop at the block's end
                // instead of needing to outlive it.
                enum RaceEvent {
                    Recv(std::io::Result<usize>),
                    Timeout,
                    Shutdown,
                }
                let event = {
                    let timer = async {
                        match next_delay {
                            Some(delay) => clock.delay(delay).await,
                            None => future::pending::<()>().await,
                        }
                    };
                    let recv = poll_fn(|cx| socket.poll_fill_recv_batch(cx, &mut batch.recv));
                    futures::pin_mut!(recv, timer);
                    match select(select(recv, timer), &mut shutdown).await {
                        Either::Left((Either::Left((result, _)), _)) => RaceEvent::Recv(result),
                        Either::Left((Either::Right(((), _)), _)) => RaceEvent::Timeout,
                        Either::Right(_) => RaceEvent::Shutdown,
                    }
                };

                match event {
                    RaceEvent::Recv(result) => {
                        let filled = result.map_err(|err| {
                            ProximaError::Io(io::Error::other(format!("{label} recv: {err}")))
                        })?;
                        if filled > 0 {
                            socket.drain_recv_to_empty(&mut batch.recv);
                            let now = instant_now(&clock);
                            for view in batch.recv.filled_datagrams() {
                                if let Err(error) = proto.on_datagram(now, view.peer, view.bytes).await {
                                    warn!(
                                        ?error,
                                        label = %label,
                                        peer = %view.peer,
                                        "datagram protocol on_datagram failed; dropping"
                                    );
                                }
                            }
                            batch.recv.clear();
                        }
                    }
                    RaceEvent::Timeout => {
                        let now = instant_now(&clock);
                        if let Err(error) = proto.on_timeout(now).await {
                            warn!(?error, label = %label, "datagram protocol on_timeout failed");
                        }
                    }
                    RaceEvent::Shutdown => {
                        debug!(label = %label, "datagram protocol listener shutting down");
                        return Ok(());
                    }
                }

                let now = instant_now(&clock);
                loop {
                    let outcome = proto.transmit(now, &mut transmit_scratch).await.map_err(|error| {
                        warn!(?error, label = %label, "datagram protocol transmit failed");
                    });
                    match outcome {
                        Ok(Some((len, peer))) => {
                            stage_reply(&mut socket, &mut batch, &transmit_scratch[..len], peer, &label).await?;
                        }
                        Ok(None) | Err(()) => break,
                    }
                }
                flush_send(&mut socket, &mut batch, &label).await?;
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Waker};

    use futures::task::noop_waker;

    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::request::{Request, Response};
    use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
    use proxima_primitives::stream::{DatagramFactory, DatagramSocket};

    use super::*;

    #[derive(Default)]
    struct SocketState {
        inbound: VecDeque<(Vec<u8>, SocketAddr)>,
        sent: Vec<(Vec<u8>, SocketAddr)>,
        waker: Option<Waker>,
    }

    #[derive(Clone)]
    struct SharedSocket {
        state: Arc<Mutex<SocketState>>,
        local: SocketAddr,
    }

    impl SharedSocket {
        fn new(local: SocketAddr) -> Self {
            Self { state: Arc::new(Mutex::new(SocketState::default())), local }
        }

        fn inject(&self, bytes: Vec<u8>, from: SocketAddr) {
            let mut state = self.state.lock().unwrap();
            state.inbound.push_back((bytes, from));
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
        }

        fn sent(&self) -> Vec<(Vec<u8>, SocketAddr)> {
            self.state.lock().unwrap().sent.clone()
        }
    }

    impl DatagramSocket for SharedSocket {
        fn poll_recv_from(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> std::task::Poll<io::Result<(usize, SocketAddr)>> {
            let mut state = self.state.lock().unwrap();
            match state.inbound.pop_front() {
                Some((bytes, from)) => {
                    let len = bytes.len().min(buf.len());
                    buf[..len].copy_from_slice(&bytes[..len]);
                    std::task::Poll::Ready(Ok((len, from)))
                }
                None => {
                    state.waker = Some(cx.waker().clone());
                    std::task::Poll::Pending
                }
            }
        }

        fn poll_send_to(&mut self, _cx: &mut Context<'_>, buf: &[u8], peer: SocketAddr) -> std::task::Poll<io::Result<usize>> {
            self.state.lock().unwrap().sent.push((buf.to_vec(), peer));
            std::task::Poll::Ready(Ok(buf.len()))
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(self.local)
        }
    }

    struct SharedFactory {
        socket: SharedSocket,
    }

    impl DatagramFactory for SharedFactory {
        fn bind(&self, _addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
            Ok(Box::new(self.socket.clone()))
        }
    }

    #[derive(Clone, Copy, Default)]
    struct ReadyClock;

    impl Clock for ReadyClock {
        type Delay = core::future::Ready<()>;

        fn now_nanos(&self) -> u64 {
            0
        }

        fn delay(&self, _dur: Duration) -> Self::Delay {
            core::future::ready(())
        }
    }

    // Emits exactly one outbound datagram off ITS OWN elapsed timer, with no
    // inbound datagram ever feeding it — the QUIC-handshake-retransmit shape
    // this module exists for.
    struct TimerDrivenProto {
        timer_fired: bool,
        emitted: bool,
        on_timeout_calls: Arc<AtomicUsize>,
        reply_peer: SocketAddr,
    }

    impl DatagramProtocol for TimerDrivenProto {
        type Err = Infallible;

        async fn on_datagram(&mut self, _now: Instant, _peer: SocketAddr, _datagram: &[u8]) -> Result<(), Infallible> {
            Ok(())
        }

        async fn on_timeout(&mut self, _now: Instant) -> Result<(), Infallible> {
            self.timer_fired = true;
            self.on_timeout_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn next_deadline(&self) -> Option<Instant> {
            if self.emitted { None } else { Some(Instant::from_monotonic(Duration::ZERO)) }
        }

        async fn transmit(&mut self, _now: Instant, buf: &mut [u8]) -> Result<Option<(usize, SocketAddr)>, Infallible> {
            if !self.timer_fired || self.emitted {
                return Ok(None);
            }
            self.emitted = true;
            let payload = b"retransmit";
            buf[..payload.len()].copy_from_slice(payload);
            Ok(Some((payload.len(), self.reply_peer)))
        }
    }

    type ReceivedLog = Arc<Mutex<Vec<(Vec<u8>, SocketAddr)>>>;

    // Records every inbound datagram and echoes one ack reply per datagram —
    // proves the recv arm still drives on_datagram + a post-recv transmit
    // drain, unchanged by the new timer arm.
    struct EchoAckProto {
        received: ReceivedLog,
        pending_reply: Option<SocketAddr>,
    }

    impl DatagramProtocol for EchoAckProto {
        type Err = Infallible;

        async fn on_datagram(&mut self, _now: Instant, peer: SocketAddr, datagram: &[u8]) -> Result<(), Infallible> {
            self.received.lock().unwrap().push((datagram.to_vec(), peer));
            self.pending_reply = Some(peer);
            Ok(())
        }

        async fn on_timeout(&mut self, _now: Instant) -> Result<(), Infallible> {
            Ok(())
        }

        fn next_deadline(&self) -> Option<Instant> {
            None
        }

        async fn transmit(&mut self, _now: Instant, buf: &mut [u8]) -> Result<Option<(usize, SocketAddr)>, Infallible> {
            match self.pending_reply.take() {
                Some(peer) => {
                    let payload = b"ack";
                    buf[..payload.len()].copy_from_slice(payload);
                    Ok(Some((payload.len(), peer)))
                }
                None => Ok(None),
            }
        }
    }

    // Never reports a deadline — proves an always-`None` next_deadline never
    // spuriously fires the timer arm.
    struct NeverTimeoutProto {
        on_timeout_calls: Arc<AtomicUsize>,
    }

    impl DatagramProtocol for NeverTimeoutProto {
        type Err = Infallible;

        async fn on_datagram(&mut self, _now: Instant, _peer: SocketAddr, _datagram: &[u8]) -> Result<(), Infallible> {
            Ok(())
        }

        async fn on_timeout(&mut self, _now: Instant) -> Result<(), Infallible> {
            self.on_timeout_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn next_deadline(&self) -> Option<Instant> {
            None
        }

        async fn transmit(&mut self, _now: Instant, _buf: &mut [u8]) -> Result<Option<(usize, SocketAddr)>, Infallible> {
            Ok(None)
        }
    }

    // `DatagramProtocolListenProtocol` never reads its dispatch handle — the
    // state machine owns request/reply — so this stub only exists to satisfy
    // the fixed `ListenProtocol::serve` signature.
    struct UnusedPipe;

    impl SendPipe for UnusedPipe {
        type In = Request<bytes::Bytes>;
        type Out = Response<bytes::Bytes>;
        type Err = ProximaError;

        fn call(&self, _request: Request<bytes::Bytes>) -> impl Future<Output = Result<Response<bytes::Bytes>, ProximaError>> + Send {
            async move {
                Ok(Response {
                    status: 200,
                    metadata: HeaderList::new(),
                    payload: bytes::Bytes::new(),
                    stream: None,
                    upgrade: None,
                })
            }
        }
    }

    fn unused_dispatch() -> PipeHandle {
        into_handle(UnusedPipe)
    }

    // Polls up to `passes` times, stopping the instant the future resolves —
    // polling a completed future again panics (`resumed after completion`),
    // so the caller reads the return to know whether shutdown already ended
    // `serve()` rather than blindly re-polling past it.
    fn poll_n(
        serve: &mut Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>>,
        cx: &mut Context<'_>,
        passes: usize,
    ) -> bool {
        for _ in 0..passes {
            if serve.as_mut().poll(cx).is_ready() {
                return true;
            }
        }
        false
    }

    #[proxima::test]
    async fn timer_fires_outbound_datagram_with_no_inbound_ever_fed() {
        let bind = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5301);
        let socket = SharedSocket::new(bind);
        let factory = Arc::new(SharedFactory { socket: socket.clone() });
        let reply_peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)), 44000);
        let on_timeout_calls = Arc::new(AtomicUsize::new(0));
        let build_calls = Arc::clone(&on_timeout_calls);
        let build = move || TimerDrivenProto {
            timer_fired: false,
            emitted: false,
            on_timeout_calls: Arc::clone(&build_calls),
            reply_peer,
        };
        let protocol = DatagramProtocolListenProtocol::with_clock("dgram-proto-timer", build, ReadyClock);
        let spec = serde_json::json!({});
        let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let mut serve = protocol.serve(bind, unused_dispatch(), &spec, context, shutdown_rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let resolved_early = poll_n(&mut serve, &mut cx, 8);
        assert!(!resolved_early, "serve must not resolve before shutdown fires");

        assert!(!socket.sent().is_empty(), "timer-driven poll_transmit must ship an outbound datagram");
        let sent = socket.sent();
        assert_eq!(sent.len(), 1, "exactly one retransmit datagram");
        assert_eq!(sent[0].0, b"retransmit");
        assert_eq!(sent[0].1, reply_peer);
        assert_eq!(on_timeout_calls.load(Ordering::SeqCst), 1, "on_timeout ran exactly once before the emit");
        assert!(socket.state.lock().unwrap().inbound.is_empty(), "no inbound datagram was ever fed");

        let _ = shutdown_tx.send(());
        assert!(poll_n(&mut serve, &mut cx, 8), "serve resolves once shutdown fires");
    }

    #[proxima::test]
    async fn recv_arm_feeds_on_datagram_and_drains_the_reply() {
        let bind = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5302);
        let socket = SharedSocket::new(bind);
        let factory = Arc::new(SharedFactory { socket: socket.clone() });
        let received = Arc::new(Mutex::new(Vec::new()));
        let build_received = Arc::clone(&received);
        let build = move || EchoAckProto { received: Arc::clone(&build_received), pending_reply: None };
        let protocol = DatagramProtocolListenProtocol::with_clock("dgram-proto-recv", build, ReadyClock);
        let spec = serde_json::json!({});
        let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
        let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let mut serve = protocol.serve(bind, unused_dispatch(), &spec, context, shutdown_rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let _ = serve.as_mut().poll(&mut cx);

        let peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 8)), 45000);
        socket.inject(b"hello".to_vec(), peer);
        poll_n(&mut serve, &mut cx, 8);

        assert_eq!(received.lock().unwrap().as_slice(), &[(b"hello".to_vec(), peer)]);
        let sent = socket.sent();
        assert_eq!(sent.len(), 1, "exactly one ack reply");
        assert_eq!(sent[0], (b"ack".to_vec(), peer));
    }

    #[proxima::test]
    async fn next_deadline_none_never_fires_on_timeout_only_shutdown_ends_it() {
        let bind = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5303);
        let socket = SharedSocket::new(bind);
        let factory = Arc::new(SharedFactory { socket: socket.clone() });
        let on_timeout_calls = Arc::new(AtomicUsize::new(0));
        let build_calls = Arc::clone(&on_timeout_calls);
        let build = move || NeverTimeoutProto { on_timeout_calls: Arc::clone(&build_calls) };
        let protocol = DatagramProtocolListenProtocol::with_clock("dgram-proto-quiet", build, ReadyClock);
        let spec = serde_json::json!({});
        let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let mut serve = protocol.serve(bind, unused_dispatch(), &spec, context, shutdown_rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let resolved_early = poll_n(&mut serve, &mut cx, 20);
        assert!(!resolved_early, "parked with no deadline and no shutdown yet");

        assert_eq!(on_timeout_calls.load(Ordering::SeqCst), 0, "no spurious timer fire with next_deadline always None");
        assert!(socket.sent().is_empty());

        let _ = shutdown_tx.send(());
        assert!(poll_n(&mut serve, &mut cx, 4), "shutdown ends serve()");
    }

}
