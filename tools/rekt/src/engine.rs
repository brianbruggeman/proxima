//! the monomorphic load driver.
//!
//! rekt holds the *concrete* pipe `P` — never a type-erased `PipeHandle`
//! (`Arc<dyn DynPipe>`), whose blanket `DynPipe::call_dyn` boxes a fresh future
//! on every call (`Box::pin(SendPipe::call(..))`, proxima-pipe `pipe.rs:303`).
//! Held as `Arc<P>` the unboxed path survives: `impl<Inner: SendPipe> SendPipe
//! for Arc<Inner>` delegates straight to `Inner::call`, so `SendPipe::call(&p,
//! req)` is a monomorphized `impl Future` with no per-send box. Driven on a
//! prime core (via `proxima::runtime::run`) and awaited inline, the send
//! hot path allocates zero futures.
//!
//! proxima supplies the primitives (the concrete pipes, `SendPipe::call`, the
//! prime runtime); rekt only composes them into the loop.
//!
//! Scope of this layer: the scenario runner drives the generic
//! `proxima::Client` surface, while the `rekt_load` benchmark path drives the
//! concrete raw H1 client. Open-loop concurrency is modeled by the scheduler
//! primitives; the staged CLI still fires each planned stage count sequentially.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use proxima::client::Client;
use proxima::request::Request;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime};
use proxima::{H1ClientUpstream, PrimeTcpUpstream, SendPipe};
use proxima_runtime::concurrency::{Concurrency, ConcurrencyController, Sample};

use crate::error::Error;
use crate::outcome::Outcome;
use crate::report::Recorder;
use crate::scenario::{RequestSpec, Scenario};

/// a monomorphic load source over the concrete pipe `P`.
pub struct Load<P: SendPipe<In = Request<Bytes>>> {
    pipe: Arc<P>,
}

impl<P: SendPipe<In = Request<Bytes>>> Load<P> {
    #[must_use]
    pub fn new(pipe: P) -> Self {
        Self { pipe: Arc::new(pipe) }
    }

    #[must_use]
    pub fn from_arc(pipe: Arc<P>) -> Self {
        Self { pipe }
    }

    /// fire `count` sends into `stage`, sequentially, awaited inline on a prime
    /// core. each send is `SendPipe::call(&p, req)` — unboxed, monomorphized.
    pub fn drive(&self, stage: usize, count: u64) -> Result<Recorder, Error> {
        let pipe = Arc::clone(&self.pipe);
        let template = default_request()?;
        proxima::runtime::run(async move {
            let mut recorder = Recorder::new();
            for _ in 0..count {
                recorder.record(stage, fire(&pipe, template.clone()).await);
            }
            recorder
        })
        .map_err(|err| Error::Engine(err.to_string()))
    }
}

/// the per-stage request template, built once and cloned per send. `from_static`
/// keeps method/path off the heap; the `Request: Clone` reuse keeps the whole
/// `RequestContext` assembly out of the hot loop.
fn default_request() -> Result<Request<Bytes>, Error> {
    build_request(&RequestSpec {
        method: "GET".to_string(),
        path: "/".to_string(),
        body: None,
        headers: Default::default(),
        query: Default::default(),
    })
}

fn build_request(spec: &RequestSpec) -> Result<Request<Bytes>, Error> {
    let mut builder = Request::builder()
        .method(Bytes::from(spec.method.clone()))
        .path(Bytes::from(spec.path.clone()));
    for (name, value) in &spec.headers {
        builder = builder.header(name.clone(), value.clone());
    }
    for (name, value) in &spec.query {
        builder = builder.query_param(name.clone(), value.clone());
    }
    if let Some(body) = &spec.body {
        builder = builder.body(Bytes::from(body.clone()));
    }
    builder
        .build()
        .map_err(|err| Error::Engine(err.to_string()))
}

// one send: call the concrete pipe inline with a pre-cloned request, time it.
// takes the request by value — `&Request` can't cross the await (`Request` is
// `!Sync`: its streamed-body field is `Send` but not `Sync`), so the caller
// clones the template per send and hands ownership in.
async fn fire<P: SendPipe<In = Request<Bytes>>>(pipe: &Arc<P>, request: Request<Bytes>) -> Outcome {
    let started = Instant::now();
    let ok = SendPipe::call(pipe, request).await.is_ok();
    Outcome {
        latency: started.elapsed(),
        ok,
        timed_out: false,
    }
}

pub fn run(scenario: &Scenario) -> Result<Recorder, Error> {
    // Staged CLI path drives the generic proxima Client surface, not an HTTP
    // switch. The target spec can be {"http": ...}, {"grpc": ...},
    // {"type":"redis", ...}, {"type":"pgwire", ...}, {"type":"h3-native", ...},
    // synth/replay/fs/process/etc., or any future protocol registered with
    // Client. rekt supplies only the request shape and timing.
    let client = Client::from_value(scenario.client_spec.clone()).map_err(|err| Error::Engine(err.to_string()))?;
    let pipe = Arc::new(client);
    let template = build_request(&scenario.request)?;
    // planned arrivals per stage, owned so the drive future stays 'static.
    let plan: Vec<u64> = scenario
        .stages
        .iter()
        .map(|stage| {
            (stage.rate_per_sec * stage.duration.as_secs_f64())
                .round()
                .max(0.0) as u64
        })
        .collect();

    proxima::runtime::run(async move {
        let mut recorder = Recorder::new();
        for (idx, count) in plan.into_iter().enumerate() {
            for _ in 0..count {
                recorder.record(idx, fire(&pipe, template.clone()).await);
            }
        }
        recorder
    })
    .map_err(|err| Error::Engine(err.to_string()))
}

/// throughput of a closed-loop run: each connection keeps one request in flight,
/// fires back-to-back for `duration`, completions summed across connections.
#[derive(Debug, Clone, Copy)]
pub struct Throughput {
    pub completed: u64,
    pub errors: u64,
    pub connections: usize,
    pub cores: usize,
    pub elapsed: Duration,
}

impl Throughput {
    #[must_use]
    pub fn per_sec(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds > 0.0 { self.completed as f64 / seconds } else { 0.0 }
    }
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
pub(crate) fn drive_replicated<MakeWorker, Fut>(
    cores: usize,
    per_core: usize,
    duration: Duration,
    make_worker: MakeWorker,
) -> Result<Throughput, Error>
where
    MakeWorker: Fn(Instant) -> Fut + Send + Clone + 'static,
    Fut: Future<Output = (u64, u64)> + 'static,
{
    let cores = cores.max(1);
    let per_core = per_core.max(1);
    let runtime = PrimeRuntime::new(cores).map_err(|err| Error::Engine(err.to_string()))?;
    let started = Instant::now();
    let deadline = started + duration;
    let (sender, receiver) = mpsc::channel::<(u64, u64)>();

    for core in 0..cores {
        let sender = sender.clone();
        let make_worker = make_worker.clone();
        // a factory: the `Send` closure crosses to the target core and builds the
        // (?Send) per-connection drivers THERE, so each core's clients live on its
        // own reactor. FuturesUnordered polls only the workers whose socket woke.
        let factory = move || -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async move {
                let mut workers: FuturesUnordered<_> =
                    (0..per_core).map(|_| make_worker(deadline)).collect();
                let mut completed = 0u64;
                let mut errors = 0u64;
                while let Some((ok, bad)) = workers.next().await {
                    completed += ok;
                    errors += bad;
                }
                let _ = sender.send((completed, errors));
            })
        };
        runtime
            .spawn_factory_on_core(CoreId(core), Box::new(factory))
            .map_err(|err| Error::Engine(format!("spawn on core {core}: {err:?}")))?;
    }
    drop(sender);

    let mut completed = 0u64;
    let mut errors = 0u64;
    for _ in 0..cores {
        match receiver.recv() {
            Ok((ok, bad)) => {
                completed += ok;
                errors += bad;
            }
            Err(_) => break,
        }
    }
    Ok(Throughput {
        completed,
        errors,
        connections: per_core * cores,
        cores,
        elapsed: started.elapsed(),
    })
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
pub fn drive_throughput(url: &str, connections_per_core: usize, cores: usize, duration: Duration) -> Result<Throughput, Error> {
    let url = url.to_string();
    drive_replicated(cores, connections_per_core, duration, move |deadline| {
        let url = url.clone();
        async move { worker(&url, deadline).await }
    })
}

// one connection's hot loop: clone the template, send, tally, until the deadline.
// a connect/build failure ends this worker (its tally is whatever it managed).
async fn worker(url: &str, deadline: Instant) -> (u64, u64) {
    // the bytes-in/status-out fast path: drive the CONCRETE prime h1 client over
    // `send_raw`, handing it ONE pre-encoded request buffer reused every send.
    // This skips the whole `Request`/`Response` envelope the `Pipe` call builds
    // per request — no per-send `Request` clone, no `Response`/header alloc, no
    // dyn box — leaving only the transport write + minimal head parse + body
    // drain. The connection keep-alive-reuses; the load gen counts completions.
    let debug_errors = std::env::var_os("REKT_DEBUG_ERRORS").is_some();
    let (pipe, request) = match raw_pipe(url) {
        Ok(pair) => pair,
        Err(err) => {
            if debug_errors {
                eprintln!("rekt worker setup error: {err}");
            }
            return (0, 1);
        }
    };
    let mut completed = 0u64;
    let mut errors = 0u64;
    while Instant::now() < deadline {
        match pipe.send_raw(&request).await {
            Ok(_status) => completed += 1,
            Err(err) => {
                if debug_errors && errors == 0 {
                    eprintln!("rekt worker first send error: {err}");
                }
                errors += 1;
            }
        }
    }
    (completed, errors)
}

/// Build the concrete prime h1 client + the pre-encoded `GET /` request bytes
/// for an `http://host[:port]/` target. DNS is deferred to connect time
/// (`with_host`); the request is encoded ONCE and re-sent every call.
fn raw_pipe(url: &str) -> Result<(H1ClientUpstream<PrimeTcpUpstream>, Vec<u8>), Error> {
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
    let request = format!("GET / HTTP/1.1\r\nHost: {authority}\r\nConnection: keep-alive\r\n\r\n").into_bytes();
    Ok((pipe, request))
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
pub fn drive_adaptive(url: &str, seed: usize, cores: usize, duration: Duration) -> Result<Throughput, Error> {
    let cores = cores.max(1);
    let seed = seed.max(1);
    let runtime = PrimeRuntime::new(cores).map_err(|err| Error::Engine(err.to_string()))?;
    let started = Instant::now();
    let deadline = started + duration;
    let (sender, receiver) = mpsc::channel::<(u64, u64)>();

    for core in 0..cores {
        let url = url.to_string();
        let sender = sender.clone();
        let factory = move || -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async move {
                let tally = adaptive_core(&url, seed, deadline).await;
                let _ = sender.send(tally);
            })
        };
        runtime
            .spawn_factory_on_core(CoreId(core), Box::new(factory))
            .map_err(|err| Error::Engine(format!("spawn on core {core}: {err:?}")))?;
    }
    drop(sender);

    let mut completed = 0u64;
    let mut errors = 0u64;
    for _ in 0..cores {
        match receiver.recv() {
            Ok((ok, bad)) => {
                completed += ok;
                errors += bad;
            }
            Err(_) => break,
        }
    }
    Ok(Throughput {
        completed,
        errors,
        connections: seed * cores,
        cores,
        elapsed: started.elapsed(),
    })
}

// one core's adaptive loop. Connections are opened ONCE into a persistent pool
// (no per-window reconnect — that throws away keep-alive and is the difference
// between beating wrk and losing to it); each window fires the first `target` of
// them flat-out for `window` via the proven `drive_throughput` tight loop
// (`while let Some = workers.next().await`), then the controller reads the
// window's throughput and picks the next target. The rest of the pool sits idle
// on its keep-alive socket, ready when the target grows.
async fn adaptive_core(url: &str, seed: usize, deadline: Instant) -> (u64, u64) {
    let concurrency = match build_controller(seed) {
        Ok(concurrency) => concurrency,
        Err(_) => return (0, 1),
    };
    let mut controller = ConcurrencyController::new(concurrency);
    let window = controller
        .window()
        .unwrap_or(Duration::from_millis(150));
    let max = adaptive_max(seed);

    // persistent pool: built once, reused every window. `raw_pipe` defers the
    // socket connect to the first send, so unused entries cost nothing.
    let mut pool: Vec<(H1ClientUpstream<PrimeTcpUpstream>, Vec<u8>)> = Vec::with_capacity(max);
    for _ in 0..max {
        match raw_pipe(url) {
            Ok(pair) => pool.push(pair),
            Err(_) => return (0, 1),
        }
    }

    let mut completed_total = 0u64;
    let mut errors_total = 0u64;
    while Instant::now() < deadline {
        let target = controller.target().clamp(1, max);
        let window_started = Instant::now();
        let window_deadline = (window_started + window).min(deadline);
        let window_stats = run_window(&pool[..target], window_deadline).await;
        completed_total += window_stats.completed;
        errors_total += window_stats.errors;

        let elapsed = window_started.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 { window_stats.completed as f64 / elapsed } else { 0.0 };
        let sample = Sample {
            concurrency: target,
            throughput,
            cov: 0.0,
            rtt_min: window_stats.rtt_min,
            rtt_p50: window_stats.rtt_mean,
            rtt_p99: window_stats.rtt_max,
            util: 0.0,
        };
        controller.observe(sample);
    }
    (completed_total, errors_total)
}

/// One window's aggregate across its connections.
struct WindowStats {
    completed: u64,
    errors: u64,
    rtt_min: Duration,
    rtt_mean: Duration,
    rtt_max: Duration,
}

// fire the given persistent connections flat-out until `deadline`, drained by the
// same tight `workers.next().await` loop `drive_throughput` uses. Returns the
// window's completions, errors, and rtt summary (min/mean/window-max as a p99
// proxy). Connections are borrowed, not created — keep-alive survives the window.
async fn run_window(connections: &[(H1ClientUpstream<PrimeTcpUpstream>, Vec<u8>)], deadline: Instant) -> WindowStats {
    let mut workers: FuturesUnordered<_> = connections
        .iter()
        .map(|(pipe, request)| fire_connection(pipe, request, deadline))
        .collect();
    let (mut completed, mut errors) = (0u64, 0u64);
    let (mut rtt_min, mut rtt_sum, mut rtt_max) = (u64::MAX, 0u64, 0u64);
    while let Some(worker) = workers.next().await {
        completed += worker.completed;
        errors += worker.errors;
        rtt_sum += worker.rtt_sum_ns;
        if worker.completed > 0 && worker.rtt_min_ns < rtt_min {
            rtt_min = worker.rtt_min_ns;
        }
        if worker.rtt_max_ns > rtt_max {
            rtt_max = worker.rtt_max_ns;
        }
    }
    let rtt_mean = rtt_sum.checked_div(completed).unwrap_or(0);
    WindowStats {
        completed,
        errors,
        rtt_min: Duration::from_nanos(if rtt_min == u64::MAX { 0 } else { rtt_min }),
        rtt_mean: Duration::from_nanos(rtt_mean),
        rtt_max: Duration::from_nanos(rtt_max),
    }
}

/// One connection's tally for a window.
struct WorkerTally {
    completed: u64,
    errors: u64,
    rtt_min_ns: u64,
    rtt_sum_ns: u64,
    rtt_max_ns: u64,
}

// fire `GET /` back-to-back on an EXISTING keep-alive connection until the window
// deadline, tally completions + rtts. Identical hot path to `drive_throughput`'s
// `worker` — no connect here, the pool owns the socket across windows.
async fn fire_connection(pipe: &H1ClientUpstream<PrimeTcpUpstream>, request: &[u8], deadline: Instant) -> WorkerTally {
    let (mut completed, mut errors) = (0u64, 0u64);
    let (mut rtt_min, mut rtt_sum, mut rtt_max) = (u64::MAX, 0u64, 0u64);
    while Instant::now() < deadline {
        let send_started = Instant::now();
        match pipe.send_raw(request).await {
            Ok(_status) => {
                let elapsed_ns = send_started.elapsed().as_nanos() as u64;
                completed += 1;
                rtt_sum += elapsed_ns;
                if elapsed_ns < rtt_min {
                    rtt_min = elapsed_ns;
                }
                if elapsed_ns > rtt_max {
                    rtt_max = elapsed_ns;
                }
            }
            Err(_) => errors += 1,
        }
    }
    WorkerTally {
        completed,
        errors,
        rtt_min_ns: if rtt_min == u64::MAX { 0 } else { rtt_min },
        rtt_sum_ns: rtt_sum,
        rtt_max_ns: rtt_max,
    }
}

#[cfg(test)]
mod tests {
    // tests assert on known tallies; unwrap/expect are the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::scenario::Thresholds;

    // plain #[test]: `run` drives the loop on its own prime core, so the
    // test must run OFF a worker (mirrors proxima's own client-on-prime tests).
    fn synth_load() -> Load<proxima::upstreams::SynthUpstream> {
        Load::new(proxima::upstreams::SynthUpstream::new("synth", 200, "ok".to_string()))
    }

    fn open() -> Thresholds {
        Thresholds { p99: None, error_rate: None }
    }

    #[test]
    fn monomorphic_send_drives_synth() {
        let recorder = synth_load().drive(0, 8).expect("drive");
        let report = recorder.report(&open());
        assert_eq!(report.stages[0].count, 8);
        assert_eq!(report.stages[0].errors, 0);
    }
}
