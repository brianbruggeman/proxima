//! the monomorphic load driver.
//!
//! rekt defines no types and no traits, and it judges nothing. Every driver
//! here is a generic function over two things:
//!
//! - **the connection is a pipe** (`Conn: Pipe`) — `In` is whatever the
//!   protocol sends, `Out` is whatever came back. It has to be a trait: its
//!   future borrows the connection across an await, and no `Fn` bound expresses
//!   `for<'a> Fn(&'a Conn) -> impl Future + 'a` without boxing. That is what
//!   `Pipe` is for, so there is nothing here to invent.
//! - **opening is a function** (`Fn() -> Result<(Conn, Conn::In), Error>`) — it
//!   hands back an owned pair, so there is no borrow to carry and no reason for
//!   a type.
//!
//! There is no third thing. An instrument reports what happened, so `Out` goes
//! into the report WHOLE and buckets there — 200, 403, 500, 501 — and the `Err`
//! side buckets beside it. Every earlier shape reduced the reply first (an
//! `ok: bool`, a `Verdict`, an `accept` predicate), and each one had to be
//! reconstructed downstream by something guessing at what the target had
//! already said plainly.
//!
//! Which is also what makes this not request/response. The envelope was HTTP's
//! instantiation of the load model and pinning it into the engine is what
//! stopped rekt driving a tunnel, a redis session, or a centauri handshake. The
//! cure is not a load-generator abstraction beside the algebra: HTTP is
//! [`open_h1`], one function, over `H1ClientUpstream`'s own base-tier `Pipe`.
//!
//! `Pipe` rather than `SendPipe`: the base tier is h1's byte path
//! (`In = Bytes`, `Out = u16`), where the cross-core tier is the envelope
//! (`Request<Bytes>`). A base-tier future is not `Send`, so every loop here is
//! built ON the prime core it runs on.
//!
//! rekt holds the *concrete* pipe — never a type-erased `PipeHandle`
//! (`Arc<dyn DynPipe>`), whose blanket `DynPipe::call_dyn` boxes a fresh future
//! on every call (`Box::pin(SendPipe::call(..))`, proxima-pipe `pipe.rs:303`).
//! Monomorphized over `Conn: Pipe`, driven on a prime core, and awaited inline,
//! the hot path allocates zero futures. `call` takes `In` by value so the item
//! is cloned per send — chosen to be `Bytes` on the raw path, where a clone is
//! a refcount bump rather than an allocation.
//!
//! proxima supplies the primitives (the concrete pipes, `SendPipe::call`, the
//! prime runtime); rekt only composes them into the loop.
//!
//! Arrivals are paced against an injected `Clock` on an ABSOLUTE grid: arrival
//! `k` is due at `stage_start + k * interval`, never at `previous_response +
//! interval`. After a stall the arrivals that came due during it fire
//! immediately rather than being silently dropped from the offered rate. The
//! clock is injected so the pacing is testable in virtual time.
//!
//! NOT yet open-loop: the staged path awaits each send before pacing the next,
//! so a slow reply still delays subsequent arrivals on that connection. Closing
//! that needs a connection pool (which `adaptive_core` already has) plus an
//! in-flight bound; until then the grid caps the rate but a stalling target can
//! still pull the achieved rate below it.

use core::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use proxima::SendPipe;
use proxima::pipe::Pipe;
use proxima::pipe::capabilities::Clock;
use proxima::pipe::clock::TimeClock;
use proxima::pipe::fan_in::Exhausted;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime};
use proxima::{H1ClientUpstream, PrimeTcpUpstream};
use proxima_recording::pipe::RecordingSink;
use proxima_runtime::concurrency::{Concurrency, ConcurrencyController, Sample};
use proxima_telemetry::Metrics;

use crate::error::Error;
use crate::report;
use crate::scenario::{Arrival, Dump, LoadPlan, PayloadSpec, Stage};

/// Encode the scenario's payload once, into the bytes that go on the wire.
///
/// The whole item, not a struct that becomes one per send: there is no
/// `Request` here to clone, no header map to walk, no builder to run. `Bytes`
/// so the clone `SendPipe::call`'s by-value `In` requires is a refcount bump.
fn encode_payload(spec: &PayloadSpec, authority: &str) -> Bytes {
    let query = spec
        .query
        .iter()
        .enumerate()
        .fold(String::new(), |mut acc, (index, (name, value))| {
            acc.push(if index == 0 { '?' } else { '&' });
            acc.push_str(name);
            acc.push('=');
            acc.push_str(value);
            acc
        });

    let body = spec.body.as_deref().unwrap_or_default();
    let mut wire = format!("{} {}{} HTTP/1.1\r\nHost: {authority}\r\nConnection: keep-alive\r\n", spec.method, spec.path, query);
    for (name, value) in &spec.headers {
        wire.push_str(&format!("{name}: {value}\r\n"));
    }
    if !body.is_empty() {
        wire.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    wire.push_str("\r\n");
    wire.push_str(body);

    Bytes::from(wire)
}

// one send: hand the item to the pipe, time the round trip, hand back what came
// back. The pipe.s `Result` passes through untouched -- the engine judges
// nothing, and there is no outcome type between here and the report because
// there is nothing to put in one.
async fn fire<Conn: Pipe>(pipe: &Conn, item: Conn::In) -> (Duration, Result<Conn::Out, Conn::Err>) {
    let started = Instant::now();
    let reply = pipe.call(item).await;
    (started.elapsed(), reply)
}

/// The staged CLI path: fire each stage's planned arrival count against the
/// scenario's target, recording per-stage latencies.
///
/// Drives the same raw h1 pipe the throughput path does. It used to drive
/// `proxima::Client`, which reaches every registered protocol — but only
/// through `Request<Bytes>`/`Response<Bytes>`, so a redis or pgwire scenario
/// was HTTP-shaped load wearing another protocol's name. Reaching those for
/// real means a pipe that speaks the protocol, not an envelope that pretends.
/// The arrivals of one stage, as a source pipe: `() -> ()`, resolving when the
/// next one is due.
///
/// A schedule is a stream with a clock on it. Everything about *when* to fire
/// lives here — the absolute grid, the catch-up after a stall, the stage
/// deadline — so the drive loop below is just "pull, fire, record" with no
/// timing arithmetic in it at all.
///
/// `Err = Exhausted` is proxima's source convention (`fan_in`, `SignalSource`,
/// `PollSourceExt`), so this ends when the stage's duration does, and a caller
/// who wants it as a `futures::Stream` has the bridge already.
///
/// `Cell` rather than atomics: a stage's arrivals belong to one connection on
/// one core, so there is no contention to arbitrate — the same single-threaded
/// borrow `proxima-centauri`'s `Session` uses for the same reason.
pub struct Arrivals<Clk> {
    clock: Clk,
    arrival: Arrival,
    mean_nanos: u64,
    deadline_nanos: u64,
    due_nanos: Cell<u64>,
    index: Cell<u64>,
}

impl<Clk: Clock> Arrivals<Clk> {
    /// The arrivals `stage` asks for, starting now on `clock`.
    ///
    /// A stage that names no rate gets a mean gap of zero — every arrival is
    /// immediately due, which is flat out through the very same source.
    #[must_use]
    pub fn of(stage: &Stage, clock: Clk) -> Self {
        let started = clock.now_nanos();
        let mean_nanos = stage
            .rate_per_sec
            .filter(|rate| *rate > 0.0)
            .map_or(0, |rate| (1_000_000_000.0 / rate) as u64);
        Self {
            clock,
            arrival: stage.arrival,
            mean_nanos,
            deadline_nanos: started.saturating_add(u64::try_from(stage.duration.as_nanos()).unwrap_or(u64::MAX)),
            due_nanos: Cell::new(started),
            index: Cell::new(0),
        }
    }
}

impl<Clk: Clock> Pipe for Arrivals<Clk> {
    type In = ();
    type Out = ();
    type Err = Exhausted;

    async fn call(&self, (): ()) -> Result<(), Exhausted> {
        if self.clock.now_nanos() >= self.deadline_nanos {
            return Err(Exhausted);
        }

        let due = self.due_nanos.get();
        let now = self.clock.now_nanos();
        if due > now {
            self.clock
                .delay(Duration::from_nanos(due - now))
                .await;
        }

        // the NEXT due accumulates off this one, never off `now` — that is what
        // keeps the grid absolute, so arrivals swallowed by a stall are still
        // offered rather than silently lowering the rate.
        let index = self.index.get() + 1;
        self.index.set(index);
        self.due_nanos
            .set(due.saturating_add(self.arrival.gap_nanos(self.mean_nanos, index)));
        Ok(())
    }
}

pub fn run(plan: &LoadPlan) -> Result<Arc<Metrics>, Error> {
    let metrics = Arc::new(report::store());
    for scenario in &plan.scenarios {
        let (pipe, item) = open_h1(&scenario.url, &scenario.payload)?;
        let workload = plan
            .dump
            .as_ref()
            .map(|spec| (spec, item.clone()));
        drive_stages(pipe, item, TimeClock, &scenario.name, &scenario.stages, &metrics, workload)?;
    }
    Ok(metrics)
}

/// Drive one scenario's stages against a pipe, recording into `metrics`.
///
/// A stage that names a rate is paced on an absolute grid; a stage that names
/// none runs flat out. Both are bounded by the stage's own duration, which is
/// why this is one loop rather than two drivers — the only difference between a
/// closed-loop throughput bench and a paced open-loop run is whether the
/// interval is zero.
///
/// Bounding by duration is also the more honest count: the old paced path fired
/// exactly `rate * duration` arrivals however long that took, so a stalling
/// target stretched the stage past its stated window.
pub fn drive_stages<Conn, Clk>(pipe: Conn, item: Conn::In, clock: Clk, scenario: &str, stages: &[Stage], metrics: &Arc<Metrics>, dump: Option<(&Dump, Bytes)>) -> Result<(), Error>
where
    Conn: Pipe + Send + 'static,
    Conn::In: Clone + Send + 'static,
    Conn::Out: core::fmt::Debug + Send + 'static,
    Clk: Clock + Clone + Send + 'static,
{
    // the per-core factory rather than `proxima::runtime::run`: a base-tier
    // `Pipe` future is not `Send`, so it is built ON the core it runs on.
    let runtime = PrimeRuntime::new(1).map_err(|err| Error::Engine(err.to_string()))?;
    // the dump arm offloads its blocking writes onto a runtime; the one driving
    // this run is it. Built BEFORE the factory so a bad dump config fails the
    // run up front rather than halfway through it.
    let dump = match dump {
        Some((spec, workload)) => {
            let offload = Arc::new(PrimeRuntime::new(1).map_err(|err| Error::Engine(err.to_string()))?) as Arc<dyn Runtime>;
            Some((report::dump_sink(spec, offload, metrics)?, workload))
        }
        None => None,
    };
    let (sender, receiver) = mpsc::channel();
    let plan: Vec<Stage> = stages.to_vec();
    let scenario = scenario.to_string();
    let metrics = Arc::clone(metrics);

    let factory = move || -> Pin<Box<dyn Future<Output = ()>>> {
        Box::pin(async move {
            let scenario: Arc<str> = Arc::from(scenario.as_str());
            let fan = match &dump {
                // the workload goes down once, then every arrival links to it
                Some((sink, sent)) => {
                    let workload = report::dump_workload(sink, &scenario, sent).await;
                    report::dumping_fan(&metrics, sink, workload)
                }
                None => report::series_fan(&metrics),
            };
            for (index, stage) in plan.iter().enumerate() {
                let arrivals = Arrivals::of(stage, clock.clone());
                while arrivals.call(()).await.is_ok() {
                    let (latency, reply) = fire(&pipe, item.clone()).await;
                    let _ = fan
                        .call(report::Observed::of(&scenario, index, latency, reply))
                        .await;
                }
            }
            // drain before the run is called done. Without this the bounded
            // queue still holds whatever the drain worker had not written and
            // the dump is silently truncated — a capture that quietly loses its
            // tail is worse than no capture, because it looks complete.
            if let Some((sink, _)) = &dump {
                let _ = sink.flush().await;
            }
            let _ = sender.send(());
        })
    };
    runtime
        .spawn_factory_on_core(CoreId(0), Box::new(factory))
        .map_err(|err| Error::Engine(format!("spawn on core 0: {err:?}")))?;

    receiver
        .recv()
        .map_err(|err| Error::Engine(err.to_string()))
}

/// Fan a per-core worker factory across `cores` prime cores: each core spawns
/// `per_core` concurrent instances of `make_worker(deadline)`, drives them to
/// completion via `FuturesUnordered`, and sums every instance's `(completed,
/// errors)` tally into one [`Throughput`]. `make_worker` is cloned once per
/// core (crossing the `Send` factory boundary — see `spawn_factory_on_core`'s
/// doc) and called `per_core` times inside that core's factory to build the
/// independent connection loops.
///
/// This is rekt's single definition of "the load-generation fan": every
/// `drive_*` in this crate (`drive_throughput` below, `h2load::drive_h2`,
/// `h3load::drive_h3`) composes this function instead of re-hand-rolling the
/// spawn-per-core / `FuturesUnordered` / mpsc-sum boilerplate that used to be
/// copy-pasted three times.
///
/// Deliberately **not** [`proxima_primitives::pipe::FanOut`] or
/// [`proxima_primitives::pipe::ScatterGather`], even though both live in
/// proxima-primitives' pipe algebra specifically to model "one thing fanned
/// out to N": neither's shape fits.
/// - `FanOut` broadcasts ONE input to N sink `SendPipe`s, `Out = ()`, and
///   awaits each sink SEQUENTIALLY inside `call` (`proxima-primitives/src/
///   pipe/fanout.rs`). rekt's arms are N *independent*, long-running
///   connection loops firing many requests until `deadline` — not one shared
///   item, and sequential awaiting would run one connection at a time
///   instead of concurrently, which is the opposite of a load generator.
/// - `ScatterGather` is the concurrent, gather-shaped sibling (`scatter_
///   gather.rs`) and structurally is the closer match — but its `call` drives
///   `futures::future::join_all`, which for <=30 sources rescans every
///   source's completion state on each wake (`futures-util`'s `JoinAll::
///   Small` variant), instead of `FuturesUnordered`'s O(1)-amortized
///   per-wake dispatch. Measured on a synthetic N-sockets/K-events harness
///   (round-robin readiness, no batching): leaf poll counts are identical
///   between the two (`MaybeDone` short-circuits completed sources), but
///   `join_all`'s per-wake O(N) rescan shows up as a real, if modest,
///   wall-time cost that grows with N (~5-9% slower at N=32-128 in an
///   all-CPU worst case). rekt is a throughput-*measurement* instrument
///   (`docs/rekt-h3-parity/discipline.md`'s binding CoV<5% bench discipline)
///   — its own fan-out overhead is not allowed to become part of what it
///   measures, so `FuturesUnordered` stays the mechanism.
pub(crate) fn drive_replicated<MakeWorker, Fut>(cores: usize, per_core: usize, duration: Duration, make_worker: MakeWorker) -> Result<Arc<Metrics>, Error>
where
    MakeWorker: Fn(Instant, Arc<Metrics>) -> Fut + Send + Clone + 'static,
    Fut: Future<Output = ()> + 'static,
{
    let cores = cores.max(1);
    let per_core = per_core.max(1);
    let runtime = PrimeRuntime::new(cores).map_err(|err| Error::Engine(err.to_string()))?;
    let metrics = Arc::new(report::store());
    let started = Instant::now();
    let deadline = started + duration;
    let (sender, receiver) = mpsc::channel::<()>();

    for core in 0..cores {
        let sender = sender.clone();
        let make_worker = make_worker.clone();
        let core_metrics = Arc::clone(&metrics);
        // a factory: the `Send` closure crosses to the target core and builds the
        // (?Send) per-connection drivers THERE, so each core's clients live on its
        // own reactor. FuturesUnordered polls only the workers whose socket woke.
        let factory = move || -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async move {
                let mut workers: FuturesUnordered<_> = (0..per_core)
                    .map(|_| make_worker(deadline, Arc::clone(&core_metrics)))
                    .collect();
                // nothing to sum: every worker records into the shared store,
                // whose histogram shards are per-thread and merged on read. The
                // channel now carries only "this core is done".
                while workers.next().await.is_some() {}
                let _ = sender.send(());
            })
        };
        runtime
            .spawn_factory_on_core(CoreId(core), Box::new(factory))
            .map_err(|err| Error::Engine(format!("spawn on core {core}: {err:?}")))?;
    }
    drop(sender);
    while receiver.recv().is_ok() {}

    report::stamp_run(&metrics, started.elapsed(), per_core * cores, cores);
    Ok(metrics)
}

/// closed-loop throughput drive against a real HTTP target over `cores` prime
/// cores. ONE `PrimeRuntime` pins one worker thread per distinct core; each core
/// then drives `connections_per_core` keep-alive clients firing `GET /` back to
/// back (build-once, clone-per-send) until the deadline. the analog of `wrk
/// -t<cores> -c<cores*connections_per_core>`. reports COMPLETED requests over the
/// measured wall-clock, never offered. composes [`drive_replicated`] — see its
/// doc for why this fans via `FuturesUnordered` and not the pipe-algebra
/// `FanOut`/`ScatterGather` family.
///
/// NB: a separate `PrimeRuntime::new(1)` per OS thread does NOT scale — prime pins
/// every one-core runtime's worker to the same physical core (core_affinity), so
/// on Linux (hard affinity) they pile onto core 0. one multi-core runtime spreads
/// across cores 0..`cores`, which is the only thing that actually parallelizes.
///
/// Workers float by default (the prime default) — the OS schedules them, which
/// dodges background contention on a shared box and naturally spreads off a
/// colocated server's busy cores. Pin explicitly for a dedicated box via the
/// prime affinity surface (`PrimeRuntime::builder().packed()/.affinity(..)`).
pub fn drive_throughput(url: &str, connections_per_core: usize, cores: usize, duration: Duration) -> Result<Arc<Metrics>, Error> {
    let url = url.to_string();
    drive_throughput_on(move || open_h1(&url, &PayloadSpec::default()), connections_per_core, cores, duration)
}

/// The same closed loop against any pipe.
///
/// `drive_throughput` is now this with the h1 opener passed in — the engine no
/// longer knows a URL from a socket path from a tunnel endpoint. `open` is a
/// plain function because opening hands back an owned pair; `accept` is a plain
/// function because judging a reply borrows nothing past the call. Only the
/// send is a pipe, because only the send has to hold the connection across an
/// await.
pub fn drive_throughput_on<Open, Conn>(open: Open, connections_per_core: usize, cores: usize, duration: Duration) -> Result<Arc<Metrics>, Error>
where
    Open: Fn() -> Result<(Conn, Conn::In), Error> + Send + Clone + 'static,
    Conn: Pipe + 'static,
    Conn::In: Clone + 'static,
    Conn::Out: core::fmt::Debug + 'static,
{
    drive_replicated(cores, connections_per_core, duration, move |deadline, metrics| {
        let open = open.clone();
        async move { worker(&open, deadline, &metrics).await }
    })
}

// one connection's hot loop: clone the item, call, tally, until the deadline.
// a connect/build failure ends this worker (its tally is whatever it managed).
async fn worker<Open, Conn>(open: &Open, deadline: Instant, metrics: &Arc<Metrics>)
where
    Open: Fn() -> Result<(Conn, Conn::In), Error>,
    Conn: Pipe,
    Conn::In: Clone,
    Conn::Out: core::fmt::Debug,
{
    // one item, prepared once by `open` and cloned per send. For the h1 opener
    // that clone is a `Bytes` refcount bump, so there is no per-send envelope
    // and no dyn box — only the transport write and whatever minimal parse the
    // protocol needs. The connection reuses; the store counts.
    let fan = report::series_fan(metrics);
    let scenario: Arc<str> = Arc::from(report::RUN);
    let debug_errors = std::env::var_os("REKT_DEBUG_ERRORS").is_some();
    let (pipe, item) = match open() {
        Ok(pair) => pair,
        Err(err) => {
            if debug_errors {
                eprintln!("rekt worker setup error: {err}");
            }
            report::record_setup_failure(metrics, &err.to_string());
            return;
        }
    };
    let mut seen_error = false;
    while Instant::now() < deadline {
        let started = Instant::now();
        let reply = pipe.call(item.clone()).await;
        if debug_errors
            && !seen_error
            && let Err(err) = &reply
        {
            seen_error = true;
            eprintln!("rekt worker first send error: {err:?}");
        }
        // WHICH reply came back survives the trip now — it is a label on the
        // shared store rather than a pair of integers collapsed per worker and
        // summed over a channel. That collapse is the whole reason `Throughput`
        // existed, and why a target answering 500 to everything used to
        // benchmark clean.
        let _ = fan
            .call(report::Observed::of(&scenario, 0, started.elapsed(), reply))
            .await;
    }
}

/// Open the prime h1 connection + encode the payload for an
/// `http://host[:port]/` target. DNS is deferred to connect time
/// (`with_host`); the payload is encoded ONCE and re-sent every call.
///
/// A plain function, not a source type: opening returns an owned pair, so there
/// is no borrow to carry and nothing a `Fn` cannot say. Sending is the one that
/// has to be a pipe — its future borrows the connection across an await, which
/// no `Fn` bound expresses without boxing.
pub fn open_h1(url: &str, payload: &PayloadSpec) -> Result<(H1ClientUpstream<PrimeTcpUpstream>, Bytes), Error> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| Error::Engine("throughput target must be http://host[:port]/".into()))?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (
            host.to_string(),
            port.parse::<u16>()
                .map_err(|err| Error::Engine(format!("target port: {err}")))?,
        ),
        None => (authority.to_string(), 80),
    };
    let upstream = PrimeTcpUpstream::with_host(host, port);
    let pipe = H1ClientUpstream::new(upstream, authority, "rekt");
    Ok((pipe, encode_payload(payload, authority)))
}

// ── adaptive drive ──────────────────────────────────────────────────────────
//
// Same closed loop as `drive_throughput`, but the per-core in-flight count is no
// longer a fixed `connections_per_core`: a [`ConcurrencyController`] (hillclimb,
// maximising throughput) raises and lowers a per-core target each window, and the
// connections self-gate on it. The controller and its decision live in
// proxima-runtime; rekt only measures the window and applies the target.

/// Per-core search bound: explore up to 8x the seed (where the crest may sit
/// above a conservatively-configured `connections_per_core`), floored at 8.
fn adaptive_max(seed: usize) -> usize {
    seed.saturating_mul(8).max(8)
}

fn build_controller(seed: usize) -> Result<Concurrency, Error> {
    let max = adaptive_max(seed);
    Concurrency::builder()
        .hillclimb()
        .start(seed.clamp(1, max))
        .bounds(1, max)
        .build()
        .map_err(|err| Error::Engine(err.to_string()))
}

/// Closed-loop adaptive throughput drive: one `PrimeRuntime`, one hillclimb
/// controller per core, each driving a self-gating pool of keep-alive
/// connections toward the crest. Reports completed requests over the wall-clock.
pub fn drive_adaptive(url: &str, seed: usize, cores: usize, duration: Duration) -> Result<Arc<Metrics>, Error> {
    let url = url.to_string();
    drive_adaptive_on(move || open_h1(&url, &PayloadSpec::default()), seed, cores, duration)
}

/// The same adaptive loop against any pipe.
pub fn drive_adaptive_on<Open, Conn>(open: Open, seed: usize, cores: usize, duration: Duration) -> Result<Arc<Metrics>, Error>
where
    Open: Fn() -> Result<(Conn, Conn::In), Error> + Send + Clone + 'static,
    Conn: Pipe + 'static,
    Conn::In: Clone + 'static,
    Conn::Out: core::fmt::Debug + 'static,
{
    let cores = cores.max(1);
    let seed = seed.max(1);
    let runtime = PrimeRuntime::new(cores).map_err(|err| Error::Engine(err.to_string()))?;
    let metrics = Arc::new(report::store());
    let started = Instant::now();
    let deadline = started + duration;
    let (sender, receiver) = mpsc::channel::<()>();

    for core in 0..cores {
        let open = open.clone();
        let sender = sender.clone();
        let core_metrics = Arc::clone(&metrics);
        let factory = move || -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async move {
                adaptive_core(&open, seed, deadline, &core_metrics).await;
                let _ = sender.send(());
            })
        };
        runtime
            .spawn_factory_on_core(CoreId(core), Box::new(factory))
            .map_err(|err| Error::Engine(format!("spawn on core {core}: {err:?}")))?;
    }
    drop(sender);
    while receiver.recv().is_ok() {}

    report::stamp_run(&metrics, started.elapsed(), seed * cores, cores);
    Ok(metrics)
}

// one core's adaptive loop. Connections are opened ONCE into a persistent pool
// (no per-window reconnect — that throws away keep-alive and is the difference
// between beating wrk and losing to it); each window fires the first `target` of
// them flat-out for `window` via the proven `drive_throughput` tight loop
// (`while let Some = workers.next().await`), then the controller reads the
// window's throughput and picks the next target. The rest of the pool sits idle
// on its keep-alive socket, ready when the target grows.
async fn adaptive_core<Open, Conn>(open: &Open, seed: usize, deadline: Instant, metrics: &Arc<Metrics>)
where
    Open: Fn() -> Result<(Conn, Conn::In), Error>,
    Conn: Pipe,
    Conn::In: Clone,
    Conn::Out: core::fmt::Debug,
{
    let scenario: Arc<str> = Arc::from(report::RUN);
    let concurrency = match build_controller(seed) {
        Ok(concurrency) => concurrency,
        Err(err) => {
            report::record_setup_failure(metrics, &err.to_string());
            return;
        }
    };
    let mut controller = ConcurrencyController::new(concurrency);
    let window = controller
        .window()
        .unwrap_or(Duration::from_millis(150));
    let max = adaptive_max(seed);

    // persistent pool: built once, reused every window. `open_h1` defers the
    // socket connect to the first send, so unused entries cost nothing.
    let mut pool: Vec<(Conn, Conn::In)> = Vec::with_capacity(max);
    for _ in 0..max {
        match open() {
            Ok(pair) => pool.push(pair),
            Err(err) => {
                report::record_setup_failure(metrics, &err.to_string());
                return;
            }
        }
    }

    while Instant::now() < deadline {
        let in_flight = controller.target().clamp(1, max);
        let window_started = Instant::now();
        let window_deadline = (window_started + window).min(deadline);

        // a fresh store per window IS the window boundary: the controller wants
        // this window's rtt distribution, not the run's. `WindowStats` and
        // `WorkerTally` used to hand-roll the min/sum/max fold that
        // `histogram_summary` already computes, then approximate p50 with the
        // mean and p99 with the max. The histogram reports the real ones.
        // a fresh window store per window IS the window boundary; the fan hands
        // each observation to both it and the run's report in one call.
        let window = Arc::new(report::store());
        let fan = report::windowed_fan(metrics, &window);
        let completed = run_window(&pool[..in_flight], &fan, &scenario, window_deadline).await;

        let elapsed = window_started.elapsed().as_secs_f64();
        let rtt = report::window_rtt(&window);
        let micros = |value: f64| Duration::from_secs_f64((value / 1_000_000.0).max(0.0));
        controller.observe(Sample {
            concurrency: in_flight,
            throughput: if elapsed > 0.0 { completed as f64 / elapsed } else { 0.0 },
            cov: 0.0,
            rtt_min: micros(rtt.as_ref().map_or(0.0, |summary| summary.min)),
            rtt_p50: micros(rtt.as_ref().map_or(0.0, |summary| summary.p50)),
            rtt_p99: micros(rtt.as_ref().map_or(0.0, |summary| summary.p99)),
            util: 0.0,
        });
    }
}

// fire the given persistent connections flat-out until `deadline`, drained by the
// same tight `workers.next().await` loop `drive_throughput` uses. Latencies land
// in `window`; the return is just the completed/errored counts the caller sums.
// Connections are borrowed, not created — keep-alive survives the window.
async fn run_window<Conn>(connections: &[(Conn, Conn::In)], fan: &report::Fan, scenario: &Arc<str>, deadline: Instant) -> u64
where
    Conn: Pipe,
    Conn::In: Clone,
    Conn::Out: core::fmt::Debug,
{
    let mut workers: FuturesUnordered<_> = connections
        .iter()
        .map(|(pipe, item)| fire_connection(pipe, item, fan, scenario, deadline))
        .collect();
    let mut completed = 0u64;
    while let Some(replies) = workers.next().await {
        completed += replies;
    }
    completed
}

// fire the workload item back-to-back on an EXISTING keep-alive connection until
// the window deadline. Identical hot path to `drive_throughput`'s `worker` — no
// connect here, the pool owns the connection across windows.
//
// A declined reply's latency is deliberately NOT recorded: the round trip
// happened, but mixing refusal latency into the rtt distribution would let a
// target that fails fast look faster to the controller than one that works.
async fn fire_connection<Conn>(pipe: &Conn, item: &Conn::In, fan: &report::Fan, scenario: &Arc<str>, deadline: Instant) -> u64
where
    Conn: Pipe,
    Conn::In: Clone,
    Conn::Out: core::fmt::Debug,
{
    let mut completed = 0u64;
    while Instant::now() < deadline {
        let send_started = Instant::now();
        let reply = pipe.call(item.clone()).await;
        let arrived = reply.is_ok();
        // ONE call, every store. Which arms exist and what each does with the
        // observation is the fan's business, not this loop's — it used to write
        // to two stores by name, which is a fan-out spelled out longhand.
        let _ = fan
            .call(report::Observed::of(scenario, 0, send_started.elapsed(), reply))
            .await;
        if arrived {
            completed += 1;
        }
    }
    completed
}

#[cfg(test)]
mod tests {
    // tests assert on known tallies; unwrap/expect are the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use proxima_primitives::pipe::clock::testing::RecordingClock;

    /// A pipe that knows nothing about HTTP — its `In` is raw bytes, the shape a
    /// tunnel, a redis session, or a centauri handshake presents.
    ///
    /// This is the test the lift exists for: before it, the engine could not
    /// name this type at all, because it demanded a `Request<Bytes>`. It needs
    /// no adapter to get here — it is a pipe, which is the whole contract.
    struct BytesEcho;

    impl Pipe for BytesEcho {
        type In = Bytes;
        type Out = usize;
        type Err = Error;

        async fn call(&self, item: Bytes) -> Result<usize, Error> {
            Ok(item.len())
        }
    }

    // plain #[test]: the staged drive runs on its own prime core, so the test
    // must run OFF a worker (mirrors proxima's own client-on-prime tests).
    #[test]
    fn the_engine_drives_a_pipe_that_is_not_http() {
        let metrics = Arc::new(report::store());
        drive_stages(BytesEcho, Bytes::from_static(b"not a request"), TimeClock, "echo", &flat_out(), &metrics, None).expect("drive");

        assert!(report::arrivals(&metrics, "echo", 0) > 0);
        assert_eq!(
            report::replies(&metrics, "echo", 0, "13"),
            report::arrivals(&metrics, "echo", 0),
            "every arrival echoed the same 13-byte payload"
        );
    }

    #[test]
    fn distinct_replies_land_in_distinct_buckets() {
        // the engine judges nothing: two different replies are two buckets, and
        // which of them is "good" is a question the report leaves to whoever
        // reads it.
        let short = Arc::new(report::store());
        drive_stages(BytesEcho, Bytes::from_static(b"ab"), TimeClock, "short", &flat_out(), &short, None).expect("drive");
        let long = Arc::new(report::store());
        drive_stages(BytesEcho, Bytes::from_static(b"abcd"), TimeClock, "long", &flat_out(), &long, None).expect("drive");

        assert_eq!(report::replies(&short, "short", 0, "2"), report::arrivals(&short, "short", 0));
        assert_eq!(report::replies(&long, "long", 0, "4"), report::arrivals(&long, "long", 0));
    }

    #[test]
    fn the_engine_drives_a_unit_payload() {
        // the degenerate case, and a real one: a pipe whose input carries no
        // payload at all — a poll, a tick, a keepalive.
        struct Tick;

        impl Pipe for Tick {
            type In = ();
            type Out = ();
            type Err = Error;

            async fn call(&self, (): ()) -> Result<(), Error> {
                Ok(())
            }
        }

        let metrics = Arc::new(report::store());
        drive_stages(Tick, (), TimeClock, "tick", &flat_out(), &metrics, None).expect("drive");

        let fired = report::arrivals(&metrics, "tick", 0);
        assert!(fired > 0);
        assert_eq!(report::replies(&metrics, "tick", 0, "()"), fired);
    }

    // the invariant the deleted `sched/` module claimed in its doc comment and
    // never wired to anything: "arrivals are scheduled against absolute time,
    // never against when the target happened to answer, so a slow target
    // produces a catch-up burst rather than a silently slipped rate."
    /// Dependency-free executor: the arrival source's future only ever awaits a
    /// `RecordingClock` delay, which is already `Ready`.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    /// One flat-out stage, short enough to keep the suite fast.
    fn flat_out() -> Vec<Stage> {
        vec![Stage {
            rate_per_sec: None,
            duration: Duration::from_millis(20),
            arrival: Arrival::Even,
        }]
    }

    #[test]
    fn a_paced_stage_honours_its_rate() {
        // 200/s for 100ms is ~20 arrivals. Flat out against an in-process echo
        // it would be orders of magnitude more, which is the whole point: the
        // rate in the file is now load-bearing.
        let metrics = Arc::new(report::store());
        let paced = vec![Stage {
            rate_per_sec: Some(200.0),
            duration: Duration::from_millis(100),
            arrival: Arrival::Even,
        }];
        drive_stages(BytesEcho, Bytes::from_static(b"x"), TimeClock, "paced", &paced, &metrics, None).expect("drive");

        let fired = report::arrivals(&metrics, "paced", 0);
        assert!((5..=40).contains(&fired), "expected ~20 paced arrivals, got {fired}");
    }

    #[test]
    fn a_stall_produces_catch_up_not_drift() {
        // the invariant the deleted `sched/` module claimed in a doc comment and
        // never wired to anything: arrivals are scheduled against absolute time,
        // so a slow target produces a catch-up burst rather than a silently
        // slipped rate. Driven on a controllable clock, so this asserts on the
        // schedule rather than on wall-time luck.
        let clock = RecordingClock::at(0);
        let stage = Stage {
            rate_per_sec: Some(1000.0), // 1ms apart
            duration: Duration::from_millis(100),
            arrival: Arrival::Even,
        };
        let arrivals = Arrivals::of(&stage, clock.clone());

        // first is due immediately
        block_on(arrivals.call(())).expect("first arrival");
        assert!(clock.delays().is_empty(), "arrival 0 is due at once");

        // the target stalls 5ms: the five arrivals that came due during it are
        // still offered, back to back, with no wait
        clock.advance(Duration::from_millis(5));
        for missed in 1..=5u64 {
            block_on(arrivals.call(())).expect("caught-up arrival");
            assert!(clock.delays().is_empty(), "arrival {missed} came due during the stall and must not be waited on");
        }

        // and the grid is still the ORIGINAL one, not rebased on the stall
        block_on(arrivals.call(())).expect("arrival 6");
        assert_eq!(clock.delays(), vec![Duration::from_millis(1)], "back on the 1ms grid");
    }

    #[test]
    fn the_source_exhausts_at_the_stage_deadline() {
        // the stage's duration ends the stream — the drive loop has no deadline
        // check of its own, it just pulls until the source says stop.
        let clock = RecordingClock::at(0);
        let stage = Stage {
            rate_per_sec: None,
            duration: Duration::from_millis(10),
            arrival: Arrival::Even,
        };
        let arrivals = Arrivals::of(&stage, clock.clone());

        assert!(block_on(arrivals.call(())).is_ok());
        clock.advance(Duration::from_millis(10));
        assert!(matches!(block_on(arrivals.call(())), Err(Exhausted)), "past the stage duration the source is exhausted");
    }

    #[test]
    fn no_rate_means_every_arrival_is_already_due() {
        let clock = RecordingClock::at(0);
        let stage = Stage {
            rate_per_sec: None,
            duration: Duration::from_secs(1),
            arrival: Arrival::Even,
        };
        let arrivals = Arrivals::of(&stage, clock.clone());

        for _ in 0..100 {
            block_on(arrivals.call(())).expect("flat out");
        }
        assert!(clock.delays().is_empty(), "flat out never sleeps");
    }

    #[test]
    fn poisson_is_bursty_but_reproducible() {
        let mean = 1_000_000u64; // 1ms
        let poisson = Arrival::Poisson { seed: 42 };

        let gaps: Vec<u64> = (0..1000)
            .map(|k| poisson.gap_nanos(mean, k))
            .collect();
        let again: Vec<u64> = (0..1000)
            .map(|k| poisson.gap_nanos(mean, k))
            .collect();
        assert_eq!(gaps, again, "same seed, same arrival pattern — on any machine");

        // even is a flat line; poisson is not
        let spread = gaps.iter().copied().max().unwrap() - gaps.iter().copied().min().unwrap();
        assert!(spread > mean, "expected clustering, got a spread of {spread}ns");

        // but the MEAN still lands on the requested rate
        let average = gaps.iter().sum::<u64>() / gaps.len() as u64;
        assert!((mean / 2..mean * 2).contains(&average), "mean gap {average}ns should be near the requested {mean}ns");

        // a different seed is a different pattern
        let other: Vec<u64> = (0..1000)
            .map(|k| Arrival::Poisson { seed: 7 }.gap_nanos(mean, k))
            .collect();
        assert_ne!(gaps, other);
    }

    #[test]
    fn a_dump_writes_a_readable_log() {
        // the whole point of the dump: a run leaves an artifact you can read
        // back. Asserted by actually reading it, not by the write compiling.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("run.jsonl");
        let spec = Dump {
            path: path.to_string_lossy().into_owned(),
            format: "json".to_string(),
            capacity: 1024,
            on_full: "fail_closed".to_string(),
            batch: 1,
        };

        let metrics = Arc::new(report::store());
        let sent = Bytes::from_static(b"GET / HTTP/1.1\r\n\r\n");
        drive_stages(BytesEcho, sent.clone(), TimeClock, "dumped", &flat_out(), &metrics, Some((&spec, sent))).expect("drive");

        let text = std::fs::read_to_string(&path).expect("dump exists and is readable");
        assert!(text.contains("rekt.workload"), "the payload is recorded once, as its own event");
        assert!(text.contains("rekt.arrival"), "and every arrival is recorded");
        assert!(text.contains("GET / HTTP/1.1"), "what we SENT is in the log, not just that we sent something");

        // arrivals link back to the workload rather than repeating its bytes
        let dumped = text.matches("rekt.arrival").count() as u64;
        let workloads = text.matches("rekt.workload").count();
        assert_eq!(workloads, 1, "one workload event, however many arrivals");
        assert!(dumped > 1, "got {dumped} arrivals");

        // the load-bearing invariant: a BOUNDED dump is allowed to shed under
        // pressure, but never silently. What reached the log plus what the drop
        // counter admits must account for every arrival the run measured.
        let measured = report::arrivals(&metrics, "dumped", 0);
        let dropped = report::dump_dropped(&metrics);
        assert_eq!(dumped + dropped, measured, "dumped {dumped} + dropped {dropped} must account for all {measured} arrivals");
    }

    #[test]
    fn stages_are_recorded_separately() {
        let metrics = Arc::new(report::store());
        let two = vec![
            Stage {
                rate_per_sec: None,
                duration: Duration::from_millis(20),
                arrival: Arrival::Even,
            },
            Stage {
                rate_per_sec: None,
                duration: Duration::from_millis(20),
                arrival: Arrival::Even,
            },
        ];
        drive_stages(BytesEcho, Bytes::from_static(b"x"), TimeClock, "two", &two, &metrics, None).expect("drive");

        assert!(report::arrivals(&metrics, "two", 0) > 0, "stage 0 recorded");
        assert!(report::arrivals(&metrics, "two", 1) > 0, "stage 1 recorded separately");
    }

    #[test]
    fn the_payload_is_encoded_once_into_wire_bytes() {
        // the whole `[request]` table becomes one byte string at connect time,
        // so nothing in the send loop assembles anything
        let payload = PayloadSpec {
            method: "POST".to_string(),
            path: "/submit".to_string(),
            body: Some("hi".to_string()),
            headers: [("x-test".to_string(), "yes".to_string())]
                .into_iter()
                .collect(),
            query: [("a".to_string(), "1".to_string())]
                .into_iter()
                .collect(),
        };

        let wire = encode_payload(&payload, "example.test:8080");

        assert_eq!(
            wire,
            Bytes::from_static(
                b"POST /submit?a=1 HTTP/1.1\r\n\
                  Host: example.test:8080\r\n\
                  Connection: keep-alive\r\n\
                  x-test: yes\r\n\
                  Content-Length: 2\r\n\
                  \r\n\
                  hi"
            )
        );
    }

    #[test]
    fn the_default_payload_is_the_benchmark_workload() {
        let wire = encode_payload(&PayloadSpec::default(), "127.0.0.1:8080");

        assert_eq!(
            wire,
            Bytes::from_static(b"GET / HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: keep-alive\r\n\r\n"),
            "the pre-encoded GET the throughput path has always sent"
        );
    }

    #[test]
    fn a_plain_opener_feeds_the_throughput_drivers() {
        // opening is a function, so a non-http target needs no type at all —
        // this closure IS the whole integration. Compile-time claim, driven
        // briefly to prove it also runs.
        let metrics = drive_throughput_on(|| Ok((BytesEcho, Bytes::from_static(b"tick"))), 2, 1, Duration::from_millis(20)).expect("drive");

        assert!(report::completed(&metrics) > 0, "a non-http opener drove real completions");
        assert_eq!(report::failed(&metrics), 0);
        assert!(report::per_sec(&metrics) > 0.0, "reqs/sec reads back out of the store");
        // the reply survives the cross-core trip now: 4 bytes echoed, bucketed
        assert_eq!(report::replies(&metrics, report::RUN, 0, "4"), report::completed(&metrics));
    }
}
