//! isolation bench: I/O readiness wakeup latency on a prime shard vs tokio.
//!
//! gates `docs/runtime-prime/discipline-reactor-wakeup.md`. measures the
//! read-wake RTT over an ESTABLISHED localhost TCP stream — connect is
//! amortized OUTSIDE the timed loop, so only the wakeup path is timed:
//! `write payload → read_exact payload`, where the read parks the executor
//! until the echo peer returns the bytes. that park→event→re-poll is exactly
//! the path the connect compare-bench implicated.
//!
//! payload sweep:
//!   - 1 B isolates fixed wake/readiness latency.
//!   - 64 B approximates tiny request/control-frame traffic.
//!   - 4 KiB and 16 KiB expose syscall/copy/partial-I/O costs that a pure
//!     wakeup micro can hide.
//!
//! arms:
//!   - tokio — current-thread runtime, reference ceiling (I/O harvested at the
//!     `block_on` wait point, no separate spin, no cross-thread hop).
//!   - prime — one core shard; the read parks the shard, which (baseline)
//!     may spin the inbox before parking on the reactor.
//!
//! Run:
//! `cargo bench -p prime --no-default-features --features
//!  runtime-prime-reactor,runtime-prime-executor,runtime-prime-inbox-alloc
//!  --bench bench_reactor_wakeup -- reactor_wakeup`
//!
//! Future A/B: save this as the baseline, then compare a reactor-harvest
//! experiment with Criterion's `--baseline` flag.
//!
//! all four endpoints set TCP_NODELAY — a tiny ping-pong is otherwise
//! destroyed by Nagle + delayed-ACK.

#![cfg(all(
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "macos", target_os = "linux"),
))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::future::Future;
use std::hint::black_box;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::os::fd::AsRawFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::future::poll_fn;
use futures::io::AsyncRead as FuturesAsyncRead;
use prime::os::core_shard;
use proxima_runtime::CoreId;
use tokio::io::{AsyncRead as TokioAsyncRead, ReadBuf};

const PAYLOAD_SIZES: [usize; 4] = [1, 64, 4096, 16_384];
const RTT_TRACE_PAYLOAD_SIZES: [usize; 3] = [1, 64, 4096];
const RTT_TRACE_DIAGNOSTIC_ITERS: u64 = 10_000;
static COUNTED_READ_REPORTS: AtomicU64 = AtomicU64::new(0);
static RTT_TRACE_REPORTS: AtomicU64 = AtomicU64::new(0);

/// spawn a blocking std echo peer: accept connections sequentially, echo every
/// read bytes back until the client closes, then accept the next. one per arm.
fn spawn_echo_peer() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind echo");
    let addr = listener.local_addr().expect("local_addr");

    std::thread::Builder::new()
        .name("rw-echo".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { continue };
                stream.set_nodelay(true).ok();
                let mut buf = [0u8; 16_384];
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if stream.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        })
        .expect("spawn echo");

    addr
}

#[derive(Default)]
struct EchoCounters {
    read_wait_ns: AtomicU64,
    write_ns: AtomicU64,
    total_ns: AtomicU64,
}

impl EchoCounters {
    fn reset(&self) {
        self.read_wait_ns.store(0, Ordering::Release);
        self.write_ns.store(0, Ordering::Release);
        self.total_ns.store(0, Ordering::Release);
    }

    fn add_read_wait(&self, duration: Duration) {
        self.read_wait_ns
            .fetch_add(duration_to_nanos(duration), Ordering::AcqRel);
    }

    fn add_write(&self, duration: Duration) {
        self.write_ns
            .fetch_add(duration_to_nanos(duration), Ordering::AcqRel);
    }

    fn add_total(&self, duration: Duration) {
        self.total_ns
            .fetch_add(duration_to_nanos(duration), Ordering::AcqRel);
    }

    fn read_wait_duration(&self) -> Duration {
        Duration::from_nanos(self.read_wait_ns.load(Ordering::Acquire))
    }

    fn write_duration(&self) -> Duration {
        Duration::from_nanos(self.write_ns.load(Ordering::Acquire))
    }

    fn total_duration(&self) -> Duration {
        Duration::from_nanos(self.total_ns.load(Ordering::Acquire))
    }
}

fn duration_to_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// spawn a blocking std echo peer and accumulate peer-side timing:
/// waiting for the client bytes to become readable, then writing the echo.
fn spawn_measured_echo_peer(counters: Arc<EchoCounters>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind measured echo");
    let addr = listener.local_addr().expect("local_addr");

    std::thread::Builder::new()
        .name("rw-echo-measured".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { continue };
                stream.set_nodelay(true).ok();
                let mut buf = [0u8; 16_384];
                loop {
                    let read_start = Instant::now();
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let after_read = Instant::now();
                            counters.add_read_wait(after_read.duration_since(read_start));
                            if stream.write_all(&buf[..n]).is_err() {
                                break;
                            }
                            let after_write = Instant::now();
                            counters.add_write(after_write.duration_since(after_read));
                            counters.add_total(after_write.duration_since(read_start));
                        }
                    }
                }
            }
        })
        .expect("spawn measured echo");

    addr
}

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
}

fn configure_attribution_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn bind_loopback_listener() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local_addr");
    (listener, addr)
}

fn connected_std_pair() -> (std::net::TcpStream, std::net::TcpStream) {
    let (listener, addr) = bind_loopback_listener();
    let stream = std::net::TcpStream::connect(addr).expect("connect std client");
    let (peer, _) = listener.accept().expect("accept std peer");
    stream.set_nodelay(true).ok();
    peer.set_nodelay(true).ok();
    stream.set_nonblocking(true).expect("client nonblocking");
    (stream, peer)
}

fn wait_for_duration(slot: &Arc<Mutex<Option<Duration>>>, label: &str) -> Duration {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(total) = *slot.lock().unwrap() {
            break total;
        }
        assert!(Instant::now() < deadline, "{label} timed out");
        std::thread::yield_now();
    }
}

fn wait_for_counted_read_result(
    slot: &Arc<Mutex<Option<(Duration, ReadPollCounts)>>>,
    label: &str,
) -> (Duration, ReadPollCounts) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(result) = *slot.lock().unwrap() {
            break result;
        }
        assert!(Instant::now() < deadline, "{label} timed out");
        std::thread::yield_now();
    }
}

fn wait_for_rtt_trace_result(
    slot: &Arc<Mutex<Option<(RttTraceTotals, ReadPollCounts)>>>,
    label: &str,
) -> (RttTraceTotals, ReadPollCounts) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(result) = *slot.lock().unwrap() {
            break result;
        }
        assert!(Instant::now() < deadline, "{label} timed out");
        std::thread::yield_now();
    }
}

fn report_counted_read(label: &str, payload_len: usize, iters: u64, counts: ReadPollCounts) {
    let Some(payload_index) = PAYLOAD_SIZES
        .iter()
        .position(|candidate| *candidate == payload_len)
    else {
        return;
    };
    let arm_offset = match label {
        "tokio_counted_read_exact" => 0,
        "prime_counted_read_exact" => 4,
        "tokio_yield_counted_read_exact" => 8,
        "prime_yield_counted_read_exact" => 12,
        _ => return,
    };
    let bit = 1_u64 << (arm_offset + payload_index);
    if COUNTED_READ_REPORTS.fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        return;
    }

    eprintln!(
        "diagnostic {label}/{payload_len}: iters={iters} {}",
        counts.describe(iters)
    );
}

#[derive(Clone, Copy, Default)]
struct RttTraceTotals {
    client_total: Duration,
    client_write: Duration,
    client_read_after_write: Duration,
}

fn duration_avg_ns(duration: Duration, iters: u64) -> f64 {
    duration.as_nanos() as f64 / iters as f64
}

fn report_rtt_trace(
    label: &str,
    payload_len: usize,
    iters: u64,
    counters: &EchoCounters,
    totals: RttTraceTotals,
    counts: ReadPollCounts,
) {
    eprintln!(
        "diagnostic reactor_rtt_trace {label}/{payload_len}: iters={iters} \
         client_total={:.1}ns client_write={:.1}ns client_read_after_write={:.1}ns \
         peer_read_wait={:.1}ns peer_write={:.1}ns peer_total={:.1}ns {}",
        duration_avg_ns(totals.client_total, iters),
        duration_avg_ns(totals.client_write, iters),
        duration_avg_ns(totals.client_read_after_write, iters),
        duration_avg_ns(counters.read_wait_duration(), iters),
        duration_avg_ns(counters.write_duration(), iters),
        duration_avg_ns(counters.total_duration(), iters),
        counts.describe(iters)
    );
}

#[cfg(feature = "runtime-prime-reactor-trace")]
fn trace_avg_ns(total: u64, cycles: u64) -> f64 {
    if cycles == 0 {
        0.0
    } else {
        total as f64 / cycles as f64
    }
}

#[cfg(feature = "runtime-prime-reactor-trace")]
fn report_prime_internal_trace(payload_len: usize, snapshot: prime::trace::Snapshot) {
    eprintln!(
        "diagnostic prime_internal_trace/{payload_len}: cycles={} missed={} \
         worker_phase_cycles={} \
         missing=[turn_enter:{} event:{} waker_start:{} waker_end:{} ready_push:{} turn_exit:{} task_poll:{} worker_phase:{}] \
         pending_to_turn_enter={:.1}ns turn_enter_to_event={:.1}ns \
         event_to_waker={:.1}ns waker={:.1}ns waker_to_ready_push={:.1}ns \
         ready_push_to_turn_exit={:.1}ns turn_exit_to_task_poll={:.1}ns \
         task_poll_to_read_ready={:.1}ns pending_to_read_ready={:.1}ns \
         worker=[pending_to_after_tick:{:.1}ns after_tick_to_timer:{:.1}ns \
         timer_to_spin_done:{:.1}ns spin_done_to_arm:{:.1}ns \
         arm_to_recheck:{:.1}ns recheck_to_first_turn:{:.1}ns] \
         reactor_turns=[enters:{} empty:{} nonread_fired:{} \
         timeouts:{} wakeups:{} \
         ignored_read:{} ignored_write:{} stale:{} unknown:{} \
         pending_to_first_turn:{:.1}ns first_turn_to_event:{:.1}ns \
         first_to_last_turn:{:.1}ns] \
         recheck_continue=[count:{} inbox:{} polled:{} fired:{}]",
        snapshot.cycles,
        snapshot.missed,
        snapshot.worker_phase_cycles,
        snapshot.missing_turn_enter,
        snapshot.missing_event_ready,
        snapshot.missing_waker_start,
        snapshot.missing_waker_end,
        snapshot.missing_ready_push,
        snapshot.missing_turn_exit,
        snapshot.missing_task_poll_start,
        snapshot.missing_worker_phase,
        trace_avg_ns(snapshot.pending_to_turn_enter_ns, snapshot.cycles),
        trace_avg_ns(snapshot.turn_enter_to_event_ns, snapshot.cycles),
        trace_avg_ns(snapshot.event_to_waker_start_ns, snapshot.cycles),
        trace_avg_ns(snapshot.waker_ns, snapshot.cycles),
        trace_avg_ns(snapshot.waker_to_ready_push_ns, snapshot.cycles),
        trace_avg_ns(snapshot.ready_push_to_turn_exit_ns, snapshot.cycles),
        trace_avg_ns(snapshot.turn_exit_to_task_poll_ns, snapshot.cycles),
        trace_avg_ns(snapshot.task_poll_to_read_ready_ns, snapshot.cycles),
        trace_avg_ns(snapshot.pending_to_read_ready_ns, snapshot.cycles),
        trace_avg_ns(
            snapshot.pending_to_after_tick_ns,
            snapshot.worker_phase_cycles
        ),
        trace_avg_ns(
            snapshot.after_tick_to_timer_done_ns,
            snapshot.worker_phase_cycles
        ),
        trace_avg_ns(
            snapshot.timer_done_to_spin_done_ns,
            snapshot.worker_phase_cycles
        ),
        trace_avg_ns(
            snapshot.spin_done_to_arm_wakeup_ns,
            snapshot.worker_phase_cycles
        ),
        trace_avg_ns(
            snapshot.arm_wakeup_to_recheck_done_ns,
            snapshot.worker_phase_cycles
        ),
        trace_avg_ns(
            snapshot.recheck_done_to_turn_enter_ns,
            snapshot.worker_phase_cycles
        ),
        snapshot.turn_enters,
        snapshot.pre_event_empty_turns,
        snapshot.pre_event_nonread_fired,
        snapshot.pre_event_timeout_turns,
        snapshot.pre_event_wakeup_events,
        snapshot.pre_event_ignored_read,
        snapshot.pre_event_ignored_write,
        snapshot.pre_event_stale,
        snapshot.pre_event_unknown,
        trace_avg_ns(snapshot.pending_to_first_turn_enter_ns, snapshot.cycles),
        trace_avg_ns(snapshot.first_turn_enter_to_event_ns, snapshot.cycles),
        trace_avg_ns(snapshot.first_to_last_turn_enter_ns, snapshot.cycles),
        snapshot.recheck_continues,
        snapshot.recheck_inbox_drained,
        snapshot.recheck_polled,
        snapshot.recheck_fired
    );
}

fn should_report_rtt_trace(label: &str, payload_len: usize) -> bool {
    let Some(payload_index) = RTT_TRACE_PAYLOAD_SIZES
        .iter()
        .position(|candidate| *candidate == payload_len)
    else {
        return false;
    };
    let arm_offset = match label {
        "tokio" => 0,
        "tokio_thread" => RTT_TRACE_PAYLOAD_SIZES.len(),
        "prime" => RTT_TRACE_PAYLOAD_SIZES.len() * 2,
        _ => return false,
    };
    let bit = 1_u64 << (arm_offset + payload_index);
    RTT_TRACE_REPORTS.fetch_or(bit, Ordering::AcqRel) & bit == 0
}

fn std_read_ready_once(stream: &mut std::net::TcpStream, buf: &mut [u8; 1]) -> Duration {
    loop {
        let start = Instant::now();
        match stream.read(buf) {
            Ok(1) => return start.elapsed(),
            Ok(n) => panic!("std read returned {n} bytes"),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::yield_now();
            }
            Err(err) => panic!("std read failed: {err}"),
        }
    }
}

fn socket2_recv_ready_once(socket: &socket2::Socket, buf: &mut [u8; 1]) -> Duration {
    loop {
        let uninit_buf =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len()) };
        let start = Instant::now();
        match socket.recv(uninit_buf) {
            Ok(1) => return start.elapsed(),
            Ok(n) => panic!("socket2 recv returned {n} bytes"),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::yield_now();
            }
            Err(err) => panic!("socket2 recv failed: {err}"),
        }
    }
}

fn socket2_recv_wouldblock_once(socket: &socket2::Socket, buf: &mut [u8; 1]) -> Duration {
    let uninit_buf = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len()) };
    let start = Instant::now();
    match socket.recv(uninit_buf) {
        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => start.elapsed(),
        Ok(n) => panic!("socket2 recv unexpectedly returned {n} bytes"),
        Err(err) => panic!("socket2 recv failed: {err}"),
    }
}

fn libc_recv_ready_once(stream: &std::net::TcpStream, buf: &mut [u8; 1]) -> Duration {
    loop {
        let start = Instant::now();
        let n = unsafe { libc::recv(stream.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len(), 0) };
        if n == 1 {
            return start.elapsed();
        }
        if n == 0 {
            panic!("libc recv returned EOF");
        }
        if n > 0 {
            panic!("libc recv returned {n} bytes");
        }
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            std::thread::yield_now();
        } else {
            panic!("libc recv failed: {err}");
        }
    }
}

fn libc_read_ready_once(stream: &std::net::TcpStream, buf: &mut [u8; 1]) -> Duration {
    loop {
        let start = Instant::now();
        let n = unsafe { libc::read(stream.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
        if n == 1 {
            return start.elapsed();
        }
        if n == 0 {
            panic!("libc read returned EOF");
        }
        if n > 0 {
            panic!("libc read returned {n} bytes");
        }
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            std::thread::yield_now();
        } else {
            panic!("libc read failed: {err}");
        }
    }
}

fn libc_read_wouldblock_once(stream: &std::net::TcpStream, buf: &mut [u8; 1]) -> Duration {
    let start = Instant::now();
    let n = unsafe { libc::read(stream.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
    if n >= 0 {
        panic!("libc read unexpectedly returned {n} bytes");
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        start.elapsed()
    } else {
        panic!("libc read failed: {err}");
    }
}

async fn prime_poll_read_ready_once(
    stream: &mut prime::os::net::TcpStream,
    buf: &mut [u8],
    expected_len: usize,
) -> (Duration, usize) {
    poll_fn(|context| {
        let start = Instant::now();
        match FuturesAsyncRead::poll_read(Pin::new(&mut *stream), context, buf) {
            Poll::Ready(Ok(n)) => {
                assert!(n > 0, "prime ready poll_read read no bytes");
                assert!(n <= expected_len, "prime ready poll_read overread");
                Poll::Ready((start.elapsed(), n))
            }
            Poll::Ready(Err(err)) => panic!("prime ready poll_read failed: {err}"),
            Poll::Pending => Poll::Pending,
        }
    })
    .await
}

fn prime_try_read_ready_once(
    stream: &mut prime::os::net::TcpStream,
    buf: &mut [u8],
    expected_len: usize,
) -> (Duration, usize) {
    loop {
        let start = Instant::now();
        match stream.try_read(buf) {
            Ok(n) => {
                assert!(n > 0, "prime ready try_read read no bytes");
                assert!(n <= expected_len, "prime ready try_read overread");
                return (start.elapsed(), n);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::yield_now();
            }
            Err(err) => panic!("prime ready try_read failed: {err}"),
        }
    }
}

async fn prime_poll_read_pending_once(
    stream: &mut prime::os::net::TcpStream,
    buf: &mut [u8],
) -> Duration {
    poll_fn(|context| {
        let start = Instant::now();
        match FuturesAsyncRead::poll_read(Pin::new(&mut *stream), context, buf) {
            Poll::Pending => Poll::Ready(start.elapsed()),
            Poll::Ready(Ok(n)) => panic!("prime pending poll_read read {n} bytes"),
            Poll::Ready(Err(err)) => panic!("prime pending poll_read failed: {err}"),
        }
    })
    .await
}

async fn tokio_try_read_ready_once(
    stream: &tokio::net::TcpStream,
    buf: &mut [u8],
    expected_len: usize,
) -> (Duration, usize) {
    loop {
        stream.readable().await.expect("tokio stream readable");
        let start = Instant::now();
        match stream.try_read(buf) {
            Ok(n) => {
                assert!(n > 0, "tokio ready try_read read no bytes");
                assert!(n <= expected_len, "tokio ready try_read overread");
                return (start.elapsed(), n);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(err) => panic!("tokio ready try_read failed: {err}"),
        }
    }
}

async fn tokio_poll_read_pending_once(
    stream: &mut tokio::net::TcpStream,
    buf: &mut [u8],
) -> Duration {
    poll_fn(|context| {
        let mut read_buf = ReadBuf::new(buf);
        let start = Instant::now();
        match TokioAsyncRead::poll_read(Pin::new(&mut *stream), context, &mut read_buf) {
            Poll::Pending => Poll::Ready(start.elapsed()),
            Poll::Ready(Ok(())) => {
                panic!(
                    "tokio pending poll_read read {} bytes",
                    read_buf.filled().len()
                )
            }
            Poll::Ready(Err(err)) => panic!("tokio pending poll_read failed: {err}"),
        }
    })
    .await
}

struct WakeState {
    armed_seq: AtomicU64,
    wake_seq: AtomicU64,
    done_seq: AtomicU64,
    waker: Mutex<Option<Waker>>,
}

impl WakeState {
    fn new() -> Self {
        Self {
            armed_seq: AtomicU64::new(0),
            wake_seq: AtomicU64::new(0),
            done_seq: AtomicU64::new(0),
            waker: Mutex::new(None),
        }
    }
}

struct ExternalWake {
    state: Arc<WakeState>,
    seq: u64,
}

impl std::future::Future for ExternalWake {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.state.wake_seq.load(Ordering::Acquire) >= self.seq {
            return Poll::Ready(());
        }

        *self.state.waker.lock().unwrap() = Some(context.waker().clone());
        self.state.armed_seq.store(self.seq, Ordering::Release);

        if self.state.wake_seq.load(Ordering::Acquire) >= self.seq {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

struct AlwaysReadyRead {
    byte: u8,
}

impl AlwaysReadyRead {
    fn new() -> Self {
        Self { byte: 0x51 }
    }
}

impl FuturesAsyncRead for AlwaysReadyRead {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        for byte in buf.iter_mut() {
            *byte = self.byte;
        }
        self.byte = self.byte.wrapping_add(1);
        Poll::Ready(Ok(buf.len()))
    }
}

#[derive(Clone, Copy, Default)]
struct ReadPollCounts {
    polls: u64,
    pending: u64,
    ready: u64,
    bytes: u64,
}

impl ReadPollCounts {
    fn describe(self, iters: u64) -> String {
        format!(
            "polls/iter={:.3} pending/iter={:.3} ready/iter={:.3} bytes/iter={:.1}",
            self.polls as f64 / iters as f64,
            self.pending as f64 / iters as f64,
            self.ready as f64 / iters as f64,
            self.bytes as f64 / iters as f64
        )
    }
}

struct CountingPrimeRead<'stream, 'counts> {
    stream: &'stream mut prime::os::net::TcpStream,
    counts: &'counts mut ReadPollCounts,
}

impl FuturesAsyncRead for CountingPrimeRead<'_, '_> {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        this.counts.polls += 1;
        match FuturesAsyncRead::poll_read(Pin::new(&mut *this.stream), context, buf) {
            Poll::Ready(Ok(n)) => {
                this.counts.ready += 1;
                this.counts.bytes += n as u64;
                Poll::Ready(Ok(n))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => {
                this.counts.pending += 1;
                Poll::Pending
            }
        }
    }
}

struct CountingTokioRead<'stream, 'counts> {
    stream: &'stream mut tokio::net::TcpStream,
    counts: &'counts mut ReadPollCounts,
}

impl TokioAsyncRead for CountingTokioRead<'_, '_> {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        this.counts.polls += 1;
        match TokioAsyncRead::poll_read(Pin::new(&mut *this.stream), context, buf) {
            Poll::Ready(Ok(())) => {
                let n = buf.filled().len() - before;
                this.counts.ready += 1;
                this.counts.bytes += n as u64;
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => {
                this.counts.pending += 1;
                Poll::Pending
            }
        }
    }
}

async fn prime_read_exact_counted(
    stream: &mut prime::os::net::TcpStream,
    buf: &mut [u8],
    counts: &mut ReadPollCounts,
) {
    let mut reader = CountingPrimeRead { stream, counts };
    futures::io::AsyncReadExt::read_exact(&mut reader, buf)
        .await
        .expect("prime counted read_exact");
}

async fn tokio_read_exact_counted(
    stream: &mut tokio::net::TcpStream,
    buf: &mut [u8],
    counts: &mut ReadPollCounts,
) {
    let mut reader = CountingTokioRead { stream, counts };
    tokio::io::AsyncReadExt::read_exact(&mut reader, buf)
        .await
        .expect("tokio counted read_exact");
}

async fn run_prime_rtt_trace(
    addr: SocketAddr,
    counters: Arc<EchoCounters>,
    payload_len: usize,
    iters: u64,
) -> (RttTraceTotals, ReadPollCounts) {
    use futures::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = prime::os::net::TcpStream::connect(addr)
        .await
        .expect("prime trace connect");

    let payload = vec![0x71; payload_len];
    let mut buf = vec![0u8; payload_len];
    stream.write_all(&payload).await.expect("warm write");
    stream.read_exact(&mut buf).await.expect("warm read");
    counters.reset();
    #[cfg(feature = "runtime-prime-reactor-trace")]
    prime::trace::reset();

    let mut totals = RttTraceTotals::default();
    let mut counts = ReadPollCounts::default();
    for _ in 0..iters {
        let total_start = Instant::now();
        let write_start = Instant::now();
        stream.write_all(&payload).await.expect("trace write");
        let after_write = Instant::now();
        prime_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
        let after_read = Instant::now();

        totals.client_write += after_write.duration_since(write_start);
        totals.client_read_after_write += after_read.duration_since(after_write);
        totals.client_total += after_read.duration_since(total_start);
    }
    assert_eq!(buf, payload);
    (totals, counts)
}

async fn run_tokio_rtt_trace(
    addr: SocketAddr,
    counters: Arc<EchoCounters>,
    payload_len: usize,
    iters: u64,
) -> (RttTraceTotals, ReadPollCounts) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("tokio trace connect");
    stream.set_nodelay(true).expect("nodelay");

    let payload = vec![0x71; payload_len];
    let mut buf = vec![0u8; payload_len];
    stream.write_all(&payload).await.expect("warm write");
    stream.read_exact(&mut buf).await.expect("warm read");
    counters.reset();

    let mut totals = RttTraceTotals::default();
    let mut counts = ReadPollCounts::default();
    for _ in 0..iters {
        let total_start = Instant::now();
        let write_start = Instant::now();
        stream.write_all(&payload).await.expect("trace write");
        let after_write = Instant::now();
        tokio_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
        let after_read = Instant::now();

        totals.client_write += after_write.duration_since(write_start);
        totals.client_read_after_write += after_read.duration_since(after_write);
        totals.client_total += after_read.duration_since(total_start);
    }
    assert_eq!(buf, payload);
    (totals, counts)
}

fn run_tokio_rtt_trace_on_thread(
    addr: SocketAddr,
    counters: Arc<EchoCounters>,
    payload_len: usize,
    iters: u64,
) -> (RttTraceTotals, ReadPollCounts) {
    std::thread::Builder::new()
        .name("tokio-rtt-trace".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .expect("tokio runtime");
            runtime.block_on(run_tokio_rtt_trace(addr, counters, payload_len, iters))
        })
        .expect("spawn tokio rtt trace")
        .join()
        .expect("tokio rtt trace thread")
}

struct ReadWakeState {
    armed_seq: AtomicU64,
    done_seq: AtomicU64,
}

impl ReadWakeState {
    fn new() -> Self {
        Self {
            armed_seq: AtomicU64::new(0),
            done_seq: AtomicU64::new(0),
        }
    }
}

enum ReadWakePhase {
    Arm,
    AwaitRead,
}

struct PrimeReadWake<'stream> {
    stream: &'stream mut prime::os::net::TcpStream,
    state: Arc<ReadWakeState>,
    seq: u64,
    phase: ReadWakePhase,
    buf: [u8; 1],
}

impl std::future::Future for PrimeReadWake<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let stream = &mut *this.stream;
        let buf = &mut this.buf;
        match FuturesAsyncRead::poll_read(Pin::new(stream), context, buf) {
            Poll::Ready(Ok(1)) => Poll::Ready(()),
            Poll::Ready(Ok(n)) => panic!("prime parked read returned {n} bytes"),
            Poll::Ready(Err(err)) => panic!("prime parked read failed: {err}"),
            Poll::Pending => {
                if matches!(this.phase, ReadWakePhase::Arm) {
                    this.phase = ReadWakePhase::AwaitRead;
                    this.state.armed_seq.store(this.seq, Ordering::Release);
                }
                Poll::Pending
            }
        }
    }
}

struct TokioReadWake<'stream> {
    stream: &'stream mut tokio::net::TcpStream,
    state: Arc<ReadWakeState>,
    seq: u64,
    phase: ReadWakePhase,
    buf: [u8; 1],
}

impl std::future::Future for TokioReadWake<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut read_buf = ReadBuf::new(&mut this.buf);
        match TokioAsyncRead::poll_read(Pin::new(&mut *this.stream), context, &mut read_buf) {
            Poll::Ready(Ok(())) if read_buf.filled().len() == 1 => Poll::Ready(()),
            Poll::Ready(Ok(())) => {
                panic!(
                    "tokio parked read returned {} bytes",
                    read_buf.filled().len()
                )
            }
            Poll::Ready(Err(err)) => panic!("tokio parked read failed: {err}"),
            Poll::Pending => {
                if matches!(this.phase, ReadWakePhase::Arm) {
                    this.phase = ReadWakePhase::AwaitRead;
                    this.state.armed_seq.store(this.seq, Ordering::Release);
                }
                Poll::Pending
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ReadWakeTiming {
    WriteToDone,
    PostWriteToDone,
}

fn wake_one_iteration(state: &WakeState, seq: u64) -> Duration {
    while state.armed_seq.load(Ordering::Acquire) < seq {
        std::thread::yield_now();
    }

    let start = Instant::now();
    state.wake_seq.store(seq, Ordering::Release);
    if let Some(waker) = state.waker.lock().unwrap().take() {
        waker.wake();
    }
    while state.done_seq.load(Ordering::Acquire) < seq {
        std::thread::yield_now();
    }
    start.elapsed()
}

fn wait_for_seq(counter: &AtomicU64, seq: u64, label: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while counter.load(Ordering::Acquire) < seq {
        assert!(
            Instant::now() < deadline,
            "{label} timed out waiting for seq {seq}"
        );
        std::thread::yield_now();
    }
}

fn read_wake_one_iteration(
    peer: &mut std::net::TcpStream,
    state: &ReadWakeState,
    seq: u64,
    timing: ReadWakeTiming,
) -> Duration {
    wait_for_seq(&state.armed_seq, seq, "parked read arm");

    let write_start = Instant::now();
    peer.write_all(&[0x39]).expect("peer wake write");
    let post_write = Instant::now();
    wait_for_seq(&state.done_seq, seq, "parked read done");

    match timing {
        ReadWakeTiming::WriteToDone => write_start.elapsed(),
        ReadWakeTiming::PostWriteToDone => post_write.elapsed(),
    }
}

#[derive(Clone, Copy)]
enum RttBreakdownMetric {
    ClientTotal,
    ClientWrite,
    ClientReadAfterWrite,
    PeerReadWait,
    PeerWrite,
    PeerTotal,
}

impl RttBreakdownMetric {
    const ALL: [Self; 6] = [
        Self::ClientTotal,
        Self::ClientWrite,
        Self::ClientReadAfterWrite,
        Self::PeerReadWait,
        Self::PeerWrite,
        Self::PeerTotal,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::ClientTotal => "client_total",
            Self::ClientWrite => "client_write",
            Self::ClientReadAfterWrite => "client_read_after_write",
            Self::PeerReadWait => "peer_read_wait",
            Self::PeerWrite => "peer_write",
            Self::PeerTotal => "peer_total",
        }
    }

    fn select(
        self,
        counters: &EchoCounters,
        client_total: Duration,
        client_write: Duration,
        client_read_after_write: Duration,
    ) -> Duration {
        match self {
            Self::ClientTotal => client_total,
            Self::ClientWrite => client_write,
            Self::ClientReadAfterWrite => client_read_after_write,
            Self::PeerReadWait => counters.read_wait_duration(),
            Self::PeerWrite => counters.write_duration(),
            Self::PeerTotal => counters.total_duration(),
        }
    }
}

fn bench_reactor_wakeup(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reactor_wakeup");
    configure_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes((payload_len * 2) as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let tokio_addr = spawn_echo_peer();
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                        let mut stream = tokio::net::TcpStream::connect(tokio_addr)
                            .await
                            .expect("tokio connect");
                        stream.set_nodelay(true).expect("nodelay");

                        let payload = vec![0x5a; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        stream.write_all(&payload).await.expect("warm write");
                        stream.read_exact(&mut buf).await.expect("warm read");

                        let start = Instant::now();
                        for _ in 0..iters {
                            stream.write_all(&payload).await.expect("write");
                            stream.read_exact(&mut buf).await.expect("read");
                        }
                        let total = start.elapsed();
                        assert_eq!(buf, payload);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let prime_addr = spawn_echo_peer();
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();
                    let addr = prime_addr;

                    handle
                        .dispatch_send(Box::pin(async move {
                            use futures::io::{AsyncReadExt, AsyncWriteExt};
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");

                            let payload = vec![0x5a; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            stream.write_all(&payload).await.expect("warm write");
                            stream.read_exact(&mut buf).await.expect("warm read");

                            let start = Instant::now();
                            for _ in 0..iters {
                                stream.write_all(&payload).await.expect("write");
                                stream.read_exact(&mut buf).await.expect("read");
                            }
                            let total = start.elapsed();
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    let deadline = Instant::now() + Duration::from_secs(30);
                    loop {
                        if let Some(total) = *result_slot.lock().unwrap() {
                            break total;
                        }
                        assert!(Instant::now() < deadline, "prime wakeup loop timed out");
                        std::thread::yield_now();
                    }
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_reactor_rtt_breakdown(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reactor_rtt_breakdown");
    configure_attribution_group(&mut group);

    for payload_len in [1, 64, 4096] {
        group.throughput(Throughput::Bytes((payload_len * 2) as u64));

        for metric in RttBreakdownMetric::ALL {
            group.bench_with_input(
                BenchmarkId::new(format!("tokio_{}", metric.label()), payload_len),
                &payload_len,
                |bencher, &payload_len| {
                    let counters = Arc::new(EchoCounters::default());
                    let addr = spawn_measured_echo_peer(counters.clone());
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_io()
                        .build()
                        .expect("tokio runtime");

                    bencher.iter_custom(|iters| {
                        runtime.block_on(async {
                            use tokio::io::{AsyncReadExt, AsyncWriteExt};

                            let mut stream = tokio::net::TcpStream::connect(addr)
                                .await
                                .expect("tokio connect");
                            stream.set_nodelay(true).expect("nodelay");

                            let payload = vec![0x6d; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            stream.write_all(&payload).await.expect("warm write");
                            stream.read_exact(&mut buf).await.expect("warm read");
                            counters.reset();

                            let mut client_total = Duration::ZERO;
                            let mut client_write = Duration::ZERO;
                            let mut client_read_after_write = Duration::ZERO;
                            for _ in 0..iters {
                                let total_start = Instant::now();
                                let write_start = Instant::now();
                                stream.write_all(&payload).await.expect("write");
                                let after_write = Instant::now();
                                stream.read_exact(&mut buf).await.expect("read");
                                let after_read = Instant::now();

                                client_write += after_write.duration_since(write_start);
                                client_read_after_write += after_read.duration_since(after_write);
                                client_total += after_read.duration_since(total_start);
                            }
                            assert_eq!(buf, payload);
                            metric.select(
                                &counters,
                                client_total,
                                client_write,
                                client_read_after_write,
                            )
                        })
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("prime_{}", metric.label()), payload_len),
                &payload_len,
                |bencher, &payload_len| {
                    let counters = Arc::new(EchoCounters::default());
                    let addr = spawn_measured_echo_peer(counters.clone());
                    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                        .expect("prime shard launch");

                    bencher.iter_custom(|iters| {
                        let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                        let result_for_task = result_slot.clone();
                        let counters_for_task = counters.clone();

                        handle
                            .dispatch_send(Box::pin(async move {
                                use futures::io::{AsyncReadExt, AsyncWriteExt};

                                let mut stream = prime::os::net::TcpStream::connect(addr)
                                    .await
                                    .expect("prime connect");

                                let payload = vec![0x6d; payload_len];
                                let mut buf = vec![0u8; payload_len];
                                stream.write_all(&payload).await.expect("warm write");
                                stream.read_exact(&mut buf).await.expect("warm read");
                                counters_for_task.reset();

                                let mut client_total = Duration::ZERO;
                                let mut client_write = Duration::ZERO;
                                let mut client_read_after_write = Duration::ZERO;
                                for _ in 0..iters {
                                    let total_start = Instant::now();
                                    let write_start = Instant::now();
                                    stream.write_all(&payload).await.expect("write");
                                    let after_write = Instant::now();
                                    stream.read_exact(&mut buf).await.expect("read");
                                    let after_read = Instant::now();

                                    client_write += after_write.duration_since(write_start);
                                    client_read_after_write +=
                                        after_read.duration_since(after_write);
                                    client_total += after_read.duration_since(total_start);
                                }
                                assert_eq!(buf, payload);
                                *result_for_task.lock().unwrap() = Some(metric.select(
                                    &counters_for_task,
                                    client_total,
                                    client_write,
                                    client_read_after_write,
                                ));
                            }))
                            .expect("dispatch_send");

                        wait_for_duration(&result_slot, "prime rtt breakdown loop")
                    });

                    handle.shutdown_and_join().expect("prime shard shutdown");
                },
            );
        }
    }

    group.finish();
}

fn bench_reactor_rtt_trace(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reactor_rtt_trace");
    configure_attribution_group(&mut group);

    for payload_len in RTT_TRACE_PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes((payload_len * 2) as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let counters = Arc::new(EchoCounters::default());
                let addr = spawn_measured_echo_peer(counters.clone());
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                if should_report_rtt_trace("tokio", payload_len) {
                    let (totals, counts) = runtime.block_on(run_tokio_rtt_trace(
                        addr,
                        counters.clone(),
                        payload_len,
                        RTT_TRACE_DIAGNOSTIC_ITERS,
                    ));
                    report_rtt_trace(
                        "tokio",
                        payload_len,
                        RTT_TRACE_DIAGNOSTIC_ITERS,
                        &counters,
                        totals,
                        counts,
                    );
                }

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        let (totals, _) =
                            run_tokio_rtt_trace(addr, counters.clone(), payload_len, iters).await;
                        totals.client_total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_thread", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let counters = Arc::new(EchoCounters::default());
                let addr = spawn_measured_echo_peer(counters.clone());

                if should_report_rtt_trace("tokio_thread", payload_len) {
                    let (totals, counts) = run_tokio_rtt_trace_on_thread(
                        addr,
                        counters.clone(),
                        payload_len,
                        RTT_TRACE_DIAGNOSTIC_ITERS,
                    );
                    report_rtt_trace(
                        "tokio_thread",
                        payload_len,
                        RTT_TRACE_DIAGNOSTIC_ITERS,
                        &counters,
                        totals,
                        counts,
                    );
                }

                bencher.iter_custom(|iters| {
                    let (totals, _) =
                        run_tokio_rtt_trace_on_thread(addr, counters.clone(), payload_len, iters);
                    totals.client_total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let counters = Arc::new(EchoCounters::default());
                let addr = spawn_measured_echo_peer(counters.clone());
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                if should_report_rtt_trace("prime", payload_len) {
                    let diagnostic_slot: Arc<Mutex<Option<(RttTraceTotals, ReadPollCounts)>>> =
                        Arc::new(Mutex::new(None));
                    let diagnostic_for_task = diagnostic_slot.clone();
                    let counters_for_task = counters.clone();
                    handle
                        .dispatch_send(Box::pin(async move {
                            let result = run_prime_rtt_trace(
                                addr,
                                counters_for_task,
                                payload_len,
                                RTT_TRACE_DIAGNOSTIC_ITERS,
                            )
                            .await;
                            *diagnostic_for_task.lock().unwrap() = Some(result);
                        }))
                        .expect("dispatch_send");
                    let (totals, counts) =
                        wait_for_rtt_trace_result(&diagnostic_slot, "prime rtt trace diagnostic");
                    report_rtt_trace(
                        "prime",
                        payload_len,
                        RTT_TRACE_DIAGNOSTIC_ITERS,
                        &counters,
                        totals,
                        counts,
                    );
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    report_prime_internal_trace(payload_len, prime::trace::snapshot());
                }

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<(RttTraceTotals, ReadPollCounts)>>> =
                        Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();
                    let counters_for_task = counters.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            let result =
                                run_prime_rtt_trace(addr, counters_for_task, payload_len, iters)
                                    .await;
                            *result_for_task.lock().unwrap() = Some(result);
                        }))
                        .expect("dispatch_send");

                    let (totals, _) =
                        wait_for_rtt_trace_result(&result_slot, "prime rtt trace loop");
                    totals.client_total
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_stream_ready_io(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("stream_ready_io");
    configure_attribution_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        use tokio::io::AsyncReadExt;

                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0xa5; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        stream.read_exact(&mut buf).await.expect("warm read");

                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            peer.write_all(&payload).expect("peer write");
                            let start = Instant::now();
                            stream.read_exact(&mut buf).await.expect("read");
                            total += start.elapsed();
                        }
                        assert_eq!(buf, payload);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_write_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        use tokio::io::AsyncWriteExt;

                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0xa5; payload_len];
                        let mut drain = vec![0u8; payload_len];
                        stream.write_all(&payload).await.expect("warm write");
                        peer.read_exact(&mut drain).expect("warm peer read");

                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let start = Instant::now();
                            stream.write_all(&payload).await.expect("write");
                            total += start.elapsed();
                            peer.read_exact(&mut drain).expect("peer read");
                        }
                        assert_eq!(drain, payload);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            use futures::io::AsyncReadExt;

                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0xa5; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            peer.write_all(&payload).expect("warm peer write");
                            stream.read_exact(&mut buf).await.expect("warm read");

                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                peer.write_all(&payload).expect("peer write");
                                let start = Instant::now();
                                stream.read_exact(&mut buf).await.expect("read");
                                total += start.elapsed();
                            }
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    wait_for_duration(&result_slot, "prime ready-read loop")
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_write_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            use futures::io::AsyncWriteExt;

                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0xa5; payload_len];
                            let mut drain = vec![0u8; payload_len];
                            stream.write_all(&payload).await.expect("warm write");
                            peer.read_exact(&mut drain).expect("warm peer read");

                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let start = Instant::now();
                                stream.write_all(&payload).await.expect("write");
                                total += start.elapsed();
                                peer.read_exact(&mut drain).expect("peer read");
                            }
                            assert_eq!(drain, payload);
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    wait_for_duration(&result_slot, "prime ready-write loop")
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_read_exact_poll_count_probe(criterion: &mut Criterion) {
    const DIAGNOSTIC_ITERS: u64 = 10_000;

    let mut group = criterion.benchmark_group("read_exact_poll_count_probe");
    configure_attribution_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio_counted_read_exact", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                let (_, diagnostic_counts) = runtime.block_on(async {
                    let (listener, addr) = bind_loopback_listener();
                    let mut stream = tokio::net::TcpStream::connect(addr)
                        .await
                        .expect("tokio connect");
                    let (mut peer, _) = listener.accept().expect("accept peer");
                    stream.set_nodelay(true).expect("nodelay");
                    peer.set_nodelay(true).ok();

                    let payload = vec![0xb7; payload_len];
                    let mut buf = vec![0u8; payload_len];
                    peer.write_all(&payload).expect("warm peer write");
                    let mut warm_counts = ReadPollCounts::default();
                    tokio_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                    let mut counts = ReadPollCounts::default();
                    let mut total = Duration::ZERO;
                    for _ in 0..DIAGNOSTIC_ITERS {
                        peer.write_all(&payload).expect("peer write");
                        let start = Instant::now();
                        tokio_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                        total += start.elapsed();
                    }
                    assert_eq!(buf, payload);
                    (total, counts)
                });
                report_counted_read(
                    "tokio_counted_read_exact",
                    payload_len,
                    DIAGNOSTIC_ITERS,
                    diagnostic_counts,
                );

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0xb7; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        let mut warm_counts = ReadPollCounts::default();
                        tokio_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                        let mut counts = ReadPollCounts::default();
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            peer.write_all(&payload).expect("peer write");
                            let start = Instant::now();
                            tokio_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                            total += start.elapsed();
                        }
                        assert_eq!(buf, payload);
                        black_box(counts.polls);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_counted_read_exact", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                let diagnostic_slot: Arc<Mutex<Option<(Duration, ReadPollCounts)>>> =
                    Arc::new(Mutex::new(None));
                let diagnostic_for_task = diagnostic_slot.clone();
                handle
                    .dispatch_send(Box::pin(async move {
                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = prime::os::net::TcpStream::connect(addr)
                            .await
                            .expect("prime connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0xb7; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        let mut warm_counts = ReadPollCounts::default();
                        prime_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                        let mut counts = ReadPollCounts::default();
                        let mut total = Duration::ZERO;
                        for _ in 0..DIAGNOSTIC_ITERS {
                            peer.write_all(&payload).expect("peer write");
                            let start = Instant::now();
                            prime_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                            total += start.elapsed();
                        }
                        assert_eq!(buf, payload);
                        *diagnostic_for_task.lock().unwrap() = Some((total, counts));
                    }))
                    .expect("dispatch_send");
                let (_, diagnostic_counts) =
                    wait_for_counted_read_result(&diagnostic_slot, "prime counted diagnostic");
                report_counted_read(
                    "prime_counted_read_exact",
                    payload_len,
                    DIAGNOSTIC_ITERS,
                    diagnostic_counts,
                );

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<(Duration, ReadPollCounts)>>> =
                        Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0xb7; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            peer.write_all(&payload).expect("warm peer write");
                            let mut warm_counts = ReadPollCounts::default();
                            prime_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                            let mut counts = ReadPollCounts::default();
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                peer.write_all(&payload).expect("peer write");
                                let start = Instant::now();
                                prime_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                                total += start.elapsed();
                            }
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some((total, counts));
                        }))
                        .expect("dispatch_send");

                    let (total, counts) =
                        wait_for_counted_read_result(&result_slot, "prime counted read loop");
                    black_box(counts.polls);
                    total
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_read_exact_arrival_slack_probe(criterion: &mut Criterion) {
    const DIAGNOSTIC_ITERS: u64 = 10_000;

    let mut group = criterion.benchmark_group("read_exact_arrival_slack_probe");
    configure_attribution_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio_yield_counted_read_exact", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                let (_, diagnostic_counts) = runtime.block_on(async {
                    let (listener, addr) = bind_loopback_listener();
                    let mut stream = tokio::net::TcpStream::connect(addr)
                        .await
                        .expect("tokio connect");
                    let (mut peer, _) = listener.accept().expect("accept peer");
                    stream.set_nodelay(true).expect("nodelay");
                    peer.set_nodelay(true).ok();

                    let payload = vec![0x9e; payload_len];
                    let mut buf = vec![0u8; payload_len];
                    peer.write_all(&payload).expect("warm peer write");
                    std::thread::yield_now();
                    let mut warm_counts = ReadPollCounts::default();
                    tokio_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                    let mut counts = ReadPollCounts::default();
                    let mut total = Duration::ZERO;
                    for _ in 0..DIAGNOSTIC_ITERS {
                        peer.write_all(&payload).expect("peer write");
                        std::thread::yield_now();
                        let start = Instant::now();
                        tokio_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                        total += start.elapsed();
                    }
                    assert_eq!(buf, payload);
                    (total, counts)
                });
                report_counted_read(
                    "tokio_yield_counted_read_exact",
                    payload_len,
                    DIAGNOSTIC_ITERS,
                    diagnostic_counts,
                );

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0x9e; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        std::thread::yield_now();
                        let mut warm_counts = ReadPollCounts::default();
                        tokio_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                        let mut counts = ReadPollCounts::default();
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            peer.write_all(&payload).expect("peer write");
                            std::thread::yield_now();
                            let start = Instant::now();
                            tokio_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                            total += start.elapsed();
                        }
                        assert_eq!(buf, payload);
                        black_box(counts.polls);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_yield_counted_read_exact", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                let diagnostic_slot: Arc<Mutex<Option<(Duration, ReadPollCounts)>>> =
                    Arc::new(Mutex::new(None));
                let diagnostic_for_task = diagnostic_slot.clone();
                handle
                    .dispatch_send(Box::pin(async move {
                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = prime::os::net::TcpStream::connect(addr)
                            .await
                            .expect("prime connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0x9e; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        std::thread::yield_now();
                        let mut warm_counts = ReadPollCounts::default();
                        prime_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                        let mut counts = ReadPollCounts::default();
                        let mut total = Duration::ZERO;
                        for _ in 0..DIAGNOSTIC_ITERS {
                            peer.write_all(&payload).expect("peer write");
                            std::thread::yield_now();
                            let start = Instant::now();
                            prime_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                            total += start.elapsed();
                        }
                        assert_eq!(buf, payload);
                        *diagnostic_for_task.lock().unwrap() = Some((total, counts));
                    }))
                    .expect("dispatch_send");
                let (_, diagnostic_counts) =
                    wait_for_counted_read_result(&diagnostic_slot, "prime yield diagnostic");
                report_counted_read(
                    "prime_yield_counted_read_exact",
                    payload_len,
                    DIAGNOSTIC_ITERS,
                    diagnostic_counts,
                );

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<(Duration, ReadPollCounts)>>> =
                        Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0x9e; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            peer.write_all(&payload).expect("warm peer write");
                            std::thread::yield_now();
                            let mut warm_counts = ReadPollCounts::default();
                            prime_read_exact_counted(&mut stream, &mut buf, &mut warm_counts).await;

                            let mut counts = ReadPollCounts::default();
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                peer.write_all(&payload).expect("peer write");
                                std::thread::yield_now();
                                let start = Instant::now();
                                prime_read_exact_counted(&mut stream, &mut buf, &mut counts).await;
                                total += start.elapsed();
                            }
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some((total, counts));
                        }))
                        .expect("dispatch_send");

                    let (total, counts) =
                        wait_for_counted_read_result(&result_slot, "prime yield read loop");
                    black_box(counts.polls);
                    total
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_read_exact_combinator_probe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("read_exact_combinator_probe");
    configure_attribution_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("manual_poll_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                bencher.iter_custom(|iters| {
                    let waker = futures::task::noop_waker();
                    let mut context = Context::from_waker(&waker);
                    let mut reader = AlwaysReadyRead::new();
                    let mut buf = vec![0u8; payload_len];
                    let mut checksum = 0u8;
                    let mut total = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        match FuturesAsyncRead::poll_read(
                            Pin::new(&mut reader),
                            &mut context,
                            &mut buf,
                        ) {
                            Poll::Ready(Ok(n)) => assert_eq!(n, payload_len),
                            Poll::Ready(Err(err)) => panic!("always-ready poll_read failed: {err}"),
                            Poll::Pending => panic!("always-ready poll_read returned Pending"),
                        }
                        total += start.elapsed();
                        checksum ^= black_box(buf[0]);
                    }
                    black_box(checksum);
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("futures_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                bencher.iter_custom(|iters| {
                    let waker = futures::task::noop_waker();
                    let mut context = Context::from_waker(&waker);
                    let mut reader = AlwaysReadyRead::new();
                    let mut buf = vec![0u8; payload_len];
                    let mut checksum = 0u8;
                    let mut total = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let mut future = futures::io::AsyncReadExt::read(&mut reader, &mut buf);
                        match Pin::new(&mut future).poll(&mut context) {
                            Poll::Ready(Ok(n)) => assert_eq!(n, payload_len),
                            Poll::Ready(Err(err)) => panic!("futures read failed: {err}"),
                            Poll::Pending => panic!("futures read returned Pending"),
                        }
                        total += start.elapsed();
                        checksum ^= black_box(buf[0]);
                    }
                    black_box(checksum);
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("futures_read_exact_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                bencher.iter_custom(|iters| {
                    let waker = futures::task::noop_waker();
                    let mut context = Context::from_waker(&waker);
                    let mut reader = AlwaysReadyRead::new();
                    let mut buf = vec![0u8; payload_len];
                    let mut checksum = 0u8;
                    let mut total = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let mut future =
                            futures::io::AsyncReadExt::read_exact(&mut reader, &mut buf);
                        match Pin::new(&mut future).poll(&mut context) {
                            Poll::Ready(Ok(())) => {}
                            Poll::Ready(Err(err)) => panic!("futures read_exact failed: {err}"),
                            Poll::Pending => panic!("futures read_exact returned Pending"),
                        }
                        total += start.elapsed();
                        checksum ^= black_box(buf[0]);
                    }
                    black_box(checksum);
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_read_syscall_probe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("read_syscall_probe");
    configure_attribution_group(&mut group);
    group.throughput(Throughput::Bytes(1));

    group.bench_function("std_read_ready_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (mut stream, mut peer) = connected_std_pair();
            let mut buf = [0u8; 1];

            peer.write_all(&[0x4d]).expect("warm peer write");
            std_read_ready_once(&mut stream, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                peer.write_all(&[0x4d]).expect("peer write");
                total += std_read_ready_once(&mut stream, &mut buf);
            }
            total
        });
    });

    group.bench_function("socket2_recv_ready_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (stream, mut peer) = connected_std_pair();
            let socket = socket2::Socket::from(stream);
            let mut buf = [0u8; 1];

            peer.write_all(&[0x4d]).expect("warm peer write");
            socket2_recv_ready_once(&socket, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                peer.write_all(&[0x4d]).expect("peer write");
                total += socket2_recv_ready_once(&socket, &mut buf);
            }
            total
        });
    });

    group.bench_function("libc_recv_ready_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (stream, mut peer) = connected_std_pair();
            let mut buf = [0u8; 1];

            peer.write_all(&[0x4d]).expect("warm peer write");
            libc_recv_ready_once(&stream, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                peer.write_all(&[0x4d]).expect("peer write");
                total += libc_recv_ready_once(&stream, &mut buf);
            }
            total
        });
    });

    group.bench_function("libc_read_ready_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (stream, mut peer) = connected_std_pair();
            let mut buf = [0u8; 1];

            peer.write_all(&[0x4d]).expect("warm peer write");
            libc_read_ready_once(&stream, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                peer.write_all(&[0x4d]).expect("peer write");
                total += libc_read_ready_once(&stream, &mut buf);
            }
            total
        });
    });

    group.bench_function("socket2_recv_wouldblock_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (stream, _peer) = connected_std_pair();
            let socket = socket2::Socket::from(stream);
            let mut buf = [0u8; 1];

            socket2_recv_wouldblock_once(&socket, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += socket2_recv_wouldblock_once(&socket, &mut buf);
            }
            total
        });
    });

    group.bench_function("libc_read_wouldblock_1", |bencher| {
        bencher.iter_custom(|iters| {
            let (stream, _peer) = connected_std_pair();
            let mut buf = [0u8; 1];

            libc_read_wouldblock_once(&stream, &mut buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += libc_read_wouldblock_once(&stream, &mut buf);
            }
            total
        });
    });

    group.finish();
}

fn bench_read_poll_probe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("read_poll_probe");
    configure_attribution_group(&mut group);

    for payload_len in PAYLOAD_SIZES {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("tokio_try_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        use tokio::io::AsyncReadExt;

                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (mut peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let payload = vec![0xc3; payload_len];
                        let mut buf = vec![0u8; payload_len];
                        peer.write_all(&payload).expect("warm peer write");
                        let (_, n) =
                            tokio_try_read_ready_once(&stream, &mut buf, payload_len).await;
                        if n < payload_len {
                            stream.read_exact(&mut buf[n..]).await.expect("warm drain");
                        }

                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            peer.write_all(&payload).expect("peer write");
                            let (elapsed, n) =
                                tokio_try_read_ready_once(&stream, &mut buf, payload_len).await;
                            total += elapsed;
                            if n < payload_len {
                                stream.read_exact(&mut buf[n..]).await.expect("drain read");
                            }
                        }
                        assert_eq!(buf, payload);
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_poll_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            use futures::io::AsyncReadExt;

                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0xc3; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            peer.write_all(&payload).expect("warm peer write");
                            let (_, n) =
                                prime_poll_read_ready_once(&mut stream, &mut buf, payload_len)
                                    .await;
                            if n < payload_len {
                                stream.read_exact(&mut buf[n..]).await.expect("warm drain");
                            }

                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                peer.write_all(&payload).expect("peer write");
                                let (elapsed, n) =
                                    prime_poll_read_ready_once(&mut stream, &mut buf, payload_len)
                                        .await;
                                total += elapsed;
                                if n < payload_len {
                                    stream.read_exact(&mut buf[n..]).await.expect("drain read");
                                }
                            }
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    wait_for_duration(&result_slot, "prime poll-read ready loop")
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_try_read_ready", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            use futures::io::AsyncReadExt;

                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (mut peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let payload = vec![0xc3; payload_len];
                            let mut buf = vec![0u8; payload_len];
                            peer.write_all(&payload).expect("warm peer write");
                            let (_, n) =
                                prime_try_read_ready_once(&mut stream, &mut buf, payload_len);
                            if n < payload_len {
                                stream.read_exact(&mut buf[n..]).await.expect("warm drain");
                            }

                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                peer.write_all(&payload).expect("peer write");
                                let (elapsed, n) =
                                    prime_try_read_ready_once(&mut stream, &mut buf, payload_len);
                                total += elapsed;
                                if n < payload_len {
                                    stream.read_exact(&mut buf[n..]).await.expect("drain read");
                                }
                            }
                            assert_eq!(buf, payload);
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    wait_for_duration(&result_slot, "prime try-read ready loop")
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_poll_read_pending_steady", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()
                    .expect("tokio runtime");

                bencher.iter_custom(|iters| {
                    runtime.block_on(async {
                        let (listener, addr) = bind_loopback_listener();
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        let (peer, _) = listener.accept().expect("accept peer");
                        stream.set_nodelay(true).expect("nodelay");
                        peer.set_nodelay(true).ok();

                        let mut buf = vec![0u8; payload_len];
                        tokio_poll_read_pending_once(&mut stream, &mut buf).await;

                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            total += tokio_poll_read_pending_once(&mut stream, &mut buf).await;
                        }
                        total
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prime_poll_read_pending_cached", payload_len),
            &payload_len,
            |bencher, &payload_len| {
                let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                    .expect("prime shard launch");

                bencher.iter_custom(|iters| {
                    let result_slot: Arc<Mutex<Option<Duration>>> = Arc::new(Mutex::new(None));
                    let result_for_task = result_slot.clone();

                    handle
                        .dispatch_send(Box::pin(async move {
                            let (listener, addr) = bind_loopback_listener();
                            let mut stream = prime::os::net::TcpStream::connect(addr)
                                .await
                                .expect("prime connect");
                            let (peer, _) = listener.accept().expect("accept peer");
                            peer.set_nodelay(true).ok();

                            let mut buf = vec![0u8; payload_len];
                            prime_poll_read_pending_once(&mut stream, &mut buf).await;

                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                total += prime_poll_read_pending_once(&mut stream, &mut buf).await;
                            }
                            *result_for_task.lock().unwrap() = Some(total);
                        }))
                        .expect("dispatch_send");

                    wait_for_duration(&result_slot, "prime poll-read pending loop")
                });

                handle.shutdown_and_join().expect("prime shard shutdown");
            },
        );
    }

    group.finish();
}

fn bench_parked_read_wake_probe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("parked_read_wake_probe");
    configure_attribution_group(&mut group);
    group.throughput(Throughput::Elements(1));

    group.bench_function("tokio_write_to_done", |bencher| {
        bencher.iter_custom(|iters| {
            let state = Arc::new(ReadWakeState::new());
            let state_for_task = state.clone();
            let (listener, addr) = bind_loopback_listener();

            let runtime_thread = std::thread::Builder::new()
                .name("tokio-parked-read".into())
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_io()
                        .build()
                        .expect("tokio runtime");
                    runtime.block_on(async move {
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        stream.set_nodelay(true).expect("nodelay");
                        for seq in 1..=iters {
                            TokioReadWake {
                                stream: &mut stream,
                                state: state_for_task.clone(),
                                seq,
                                phase: ReadWakePhase::Arm,
                                buf: [0; 1],
                            }
                            .await;
                            state_for_task.done_seq.store(seq, Ordering::Release);
                        }
                    });
                })
                .expect("spawn tokio parked-read runtime");

            let (mut peer, _) = listener.accept().expect("accept peer");
            peer.set_nodelay(true).ok();

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total +=
                    read_wake_one_iteration(&mut peer, &state, seq, ReadWakeTiming::WriteToDone);
            }

            runtime_thread.join().expect("tokio parked-read join");
            total
        });
    });

    group.bench_function("prime_write_to_done", |bencher| {
        let handle =
            core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("prime shard launch");

        bencher.iter_custom(|iters| {
            let state = Arc::new(ReadWakeState::new());
            let state_for_task = state.clone();
            let result_slot: Arc<Mutex<Option<()>>> = Arc::new(Mutex::new(None));
            let result_for_task = result_slot.clone();
            let (listener, addr) = bind_loopback_listener();

            handle
                .dispatch_send(Box::pin(async move {
                    let mut stream = prime::os::net::TcpStream::connect(addr)
                        .await
                        .expect("prime connect");
                    for seq in 1..=iters {
                        PrimeReadWake {
                            stream: &mut stream,
                            state: state_for_task.clone(),
                            seq,
                            phase: ReadWakePhase::Arm,
                            buf: [0; 1],
                        }
                        .await;
                        state_for_task.done_seq.store(seq, Ordering::Release);
                    }
                    *result_for_task.lock().unwrap() = Some(());
                }))
                .expect("dispatch_send");

            let (mut peer, _) = listener.accept().expect("accept peer");
            peer.set_nodelay(true).ok();

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total +=
                    read_wake_one_iteration(&mut peer, &state, seq, ReadWakeTiming::WriteToDone);
            }

            let deadline = Instant::now() + Duration::from_secs(30);
            while result_slot.lock().unwrap().is_none() {
                assert!(
                    Instant::now() < deadline,
                    "prime parked-read task timed out"
                );
                std::thread::yield_now();
            }
            total
        });

        handle.shutdown_and_join().expect("prime shard shutdown");
    });

    group.bench_function("tokio_post_write_to_done", |bencher| {
        bencher.iter_custom(|iters| {
            let state = Arc::new(ReadWakeState::new());
            let state_for_task = state.clone();
            let (listener, addr) = bind_loopback_listener();

            let runtime_thread = std::thread::Builder::new()
                .name("tokio-parked-read".into())
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_io()
                        .build()
                        .expect("tokio runtime");
                    runtime.block_on(async move {
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .expect("tokio connect");
                        stream.set_nodelay(true).expect("nodelay");
                        for seq in 1..=iters {
                            TokioReadWake {
                                stream: &mut stream,
                                state: state_for_task.clone(),
                                seq,
                                phase: ReadWakePhase::Arm,
                                buf: [0; 1],
                            }
                            .await;
                            state_for_task.done_seq.store(seq, Ordering::Release);
                        }
                    });
                })
                .expect("spawn tokio parked-read runtime");

            let (mut peer, _) = listener.accept().expect("accept peer");
            peer.set_nodelay(true).ok();

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total += read_wake_one_iteration(
                    &mut peer,
                    &state,
                    seq,
                    ReadWakeTiming::PostWriteToDone,
                );
            }

            runtime_thread.join().expect("tokio parked-read join");
            total
        });
    });

    group.bench_function("prime_post_write_to_done", |bencher| {
        let handle =
            core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("prime shard launch");

        bencher.iter_custom(|iters| {
            let state = Arc::new(ReadWakeState::new());
            let state_for_task = state.clone();
            let result_slot: Arc<Mutex<Option<()>>> = Arc::new(Mutex::new(None));
            let result_for_task = result_slot.clone();
            let (listener, addr) = bind_loopback_listener();

            handle
                .dispatch_send(Box::pin(async move {
                    let mut stream = prime::os::net::TcpStream::connect(addr)
                        .await
                        .expect("prime connect");
                    for seq in 1..=iters {
                        PrimeReadWake {
                            stream: &mut stream,
                            state: state_for_task.clone(),
                            seq,
                            phase: ReadWakePhase::Arm,
                            buf: [0; 1],
                        }
                        .await;
                        state_for_task.done_seq.store(seq, Ordering::Release);
                    }
                    *result_for_task.lock().unwrap() = Some(());
                }))
                .expect("dispatch_send");

            let (mut peer, _) = listener.accept().expect("accept peer");
            peer.set_nodelay(true).ok();

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total += read_wake_one_iteration(
                    &mut peer,
                    &state,
                    seq,
                    ReadWakeTiming::PostWriteToDone,
                );
            }

            let deadline = Instant::now() + Duration::from_secs(30);
            while result_slot.lock().unwrap().is_none() {
                assert!(
                    Instant::now() < deadline,
                    "prime parked-read task timed out"
                );
                std::thread::yield_now();
            }
            total
        });

        handle.shutdown_and_join().expect("prime shard shutdown");
    });

    group.finish();
}

fn bench_task_wake_repoll(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("task_wake_repoll");
    configure_attribution_group(&mut group);
    group.throughput(Throughput::Elements(1));

    group.bench_function("tokio_current_thread", |bencher| {
        bencher.iter_custom(|iters| {
            let state = Arc::new(WakeState::new());
            let state_for_task = state.clone();
            let runtime_thread = std::thread::Builder::new()
                .name("tokio-wake-repoll".into())
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_io()
                        .build()
                        .expect("tokio runtime");
                    runtime.block_on(async move {
                        for seq in 1..=iters {
                            ExternalWake {
                                state: state_for_task.clone(),
                                seq,
                            }
                            .await;
                            state_for_task.done_seq.store(seq, Ordering::Release);
                        }
                    });
                })
                .expect("spawn tokio runtime thread");

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total += wake_one_iteration(&state, seq);
            }
            runtime_thread.join().expect("tokio runtime join");
            total
        });
    });

    group.bench_function("prime_core_shard", |bencher| {
        let handle =
            core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("prime shard launch");

        bencher.iter_custom(|iters| {
            let state = Arc::new(WakeState::new());
            let state_for_task = state.clone();
            let result_slot: Arc<Mutex<Option<()>>> = Arc::new(Mutex::new(None));
            let result_for_task = result_slot.clone();

            handle
                .dispatch_send(Box::pin(async move {
                    for seq in 1..=iters {
                        ExternalWake {
                            state: state_for_task.clone(),
                            seq,
                        }
                        .await;
                        state_for_task.done_seq.store(seq, Ordering::Release);
                    }
                    *result_for_task.lock().unwrap() = Some(());
                }))
                .expect("dispatch_send");

            let mut total = Duration::ZERO;
            for seq in 1..=iters {
                total += wake_one_iteration(&state, seq);
            }

            let deadline = Instant::now() + Duration::from_secs(30);
            while result_slot.lock().unwrap().is_none() {
                assert!(Instant::now() < deadline, "prime wake task timed out");
                std::thread::yield_now();
            }
            total
        });

        handle.shutdown_and_join().expect("prime shard shutdown");
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reactor_wakeup,
    bench_reactor_rtt_breakdown,
    bench_reactor_rtt_trace,
    bench_stream_ready_io,
    bench_read_exact_poll_count_probe,
    bench_read_exact_arrival_slack_probe,
    bench_read_exact_combinator_probe,
    bench_read_syscall_probe,
    bench_read_poll_probe,
    bench_parked_read_wake_probe,
    bench_task_wake_repoll
);
criterion_main!(benches);
