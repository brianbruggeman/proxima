# rekt

command your rest api to get wrecked.

a load and soak tester built on [proxima](../..): single binary,
config-first, multi-protocol, and native-fast. the goal is wrk-class throughput
with k6-class ergonomics and neither one's compromise.

status: the staged `rek` CLI parses a scenario, builds a generic
`proxima::Client` from the target spec, drives stage counts, and reports
thresholds. The raw H1 `rekt_load` path is a specialized closed-loop benchmark
for rekt-vs-wrk work; H2/H3 bench examples live beside it.

```bash
cargo run -p rekt --features scheduler --bin rek -- tools/rekt/rek.toml
```

## what it competes with

two tiers, and rekt wants to sit above both.

- the fast tier: wrk (http/1.1), h2load (http/2 and 3), trex (raw l4). these win
  on speed because they run a native loop and interpret nothing per request.
  they're also narrow: one or two protocols, little scripting.
- the ergonomic tier: k6, locust, gatling, vegeta, hurl. nice to write, but
  slower (a language runs per request) or http-only.

the bet: native speed *and* multi-protocol *and* scriptable, in one binary. match
the fast tier within a small factor, beat the ergonomic tier on speed, and beat
the fast tier on breadth (grpc, redis, pgwire, raw tcp/udp, stateful flows).
being "as fast as wrk but also does everything else" beats "5% faster than wrk at
the one thing wrk already does".

## the one principle: compile once, interpret never

the slow tools run a language on every request (k6's js vm, locust's python).
the interpreter, not the network, becomes the bottleneck. the fast tools run a
native loop and interpret nothing.

rekt is the second kind. a scenario is parsed once into native proxima pipes plus
a small state machine, and the hot loop just fires pipes. nothing is parsed,
eval'd, or interpreted per request. every design choice below exists to protect
that property.

## the model: config compiles to pipes

the scenario is first-class: a typed, serializable IR — steps, captures, guards,
load profile. that IR is the real api. it compiles to a set of proxima pipes and
an fsm, and the engine runs over them at rate.

pipes are generic, not http-bound:

```rust
trait Pipe { type In; type Out; async fn call(&self, input: In) -> Result<Out>; }
```

raw tcp is `Pipe<In = Bytes, Out = Bytes>`; udp is a datagram in and out; http is
`Request`/`Response`; redis is `Command`/`Reply`; pgwire is `Query`/`RowSet`. a
step adapts the shared session context to whatever its pipe's `In`/`Out` are:

```rust
struct ProtoStep<P: Pipe> {
    pipe:    P,                      // built once: host, pool, tls, timeouts
    render:  fn(&Ctx) -> P::In,      // ctx -> typed input (splice captured vars)
    capture: fn(&P::Out, &mut Ctx),  // typed output -> ctx (json:.token, etc)
    guard:   fn(&P::Out) -> bool,
}
```

a mixed-protocol chain (http login, then a redis check, then a tcp blast) has a
different `In`/`Out` per step, so steps are type-erased behind `fire(&mut Ctx)`
and the fsm walks `Vec<Box<dyn Step>>`. the **session context is the one
protocol-agnostic thing** that crosses steps — a small var bag the pipes never
see. data goes into a pipe as the input you build, comes back out of its output
through capture, and threads forward in the context.

the pipe rekt drives is `proxima::Client`, the runtime's single client entry
point, which itself implements `Pipe`. one front door, never hand-wired upstreams.
protocol is a spec key (`http`, `grpc`, `synth`, `replay`, `fs`, `process`,
`type = "redis"`, `type = "pgwire"`, `type = "h3-native"`, ...); feature-gated
protocols become available when the corresponding proxima feature registers the
factory. Use `--features scheduler,redis-client`, `--features
scheduler,pgwire-client`, or `--features all-client-protocols` to compile the
extra client factories into the `rek` binary. every protocol is one `Client`;
rekt supplies the scenario target, request shape, and timing.

`[target] url = "http://..."` is HTTP shorthand. Use `[target.client]` for any
full `Client::from_value` spec:

```toml
[target.client]
synth = { status = 200, body = "ok" }

[request]
method = "GET"
path = "/"
```

Byte-request protocols can be driven directly. Redis uses the request method as
the command and the body as NUL-delimited arguments:

```toml
[target.client]
type = "redis"
dsn = "redis://127.0.0.1:6379"

[request]
method = "GET"
body = "proxima:e2e"
```

Typed protocols still enter through `[target.client]`, but useful scenario
frontends may need protocol-specific request adapters on top of the generic
method/path/body fields. Pgwire currently maps `method = "QUERY"` plus body SQL
to a simple query and returns a byte summary of the typed reply.

the speed guard: you build each client once (the expensive part — dns, tls,
pooling) and feed it a fresh lightweight input per fire. construction is amortized
to zero; per fire you only build the input and call.

## fsm

state lives in a state machine where it earns its place. two uses:

- run controller: idle → ramp → hold/soak → drain → report, transitioning on
  elapsed time or a tripped threshold. governs the offered rate.
- chained scenario: states are steps, edges carry captured values and a guard.
  this is where login → token → reuse lives.

a plain constant-rate blast needs no machine, so it doesn't get one.

## a step is the same shape in any protocol

build the pipe's input from context, send, extract from the output, check a
guard. only the input and output types change, and those live in the protocol's
module. the core IR knows `pipe`, `to`, what to send, `capture`, and `expect`,
nothing else.

raw tcp, write bytes and match the reply:

```toml
[[step]]
pipe = "tcp"
to   = "10.0.0.5:6379"
send = "PING\r\n"
expect = "contains PONG"
```

udp, fire a datagram:

```toml
[[step]]
pipe = "udp"
to   = "10.0.0.5:9999"
send = "{{payload}}"
```

http, one shape among many. `method`, `path`, `status`, `json:` are http-pipe
vocabulary, scoped to this pipe, never in the engine. this chain logs in and
carries the captured token forward:

```toml
[[step]]
name = "login"
pipe = "http"          # transport "h3" for http/3 over quic
to   = "https://api/login"
method = "POST"
body = '{"user":"u","pass":"p"}'
capture.token = "json:.access_token"
expect = "status == 200"

[[step]]
name = "me"
pipe = "http"
to   = "https://api/me"
header.authorization = "Bearer {{token}}"
expect = "status == 200"

[load]
setup = ["login"]   # once per session: log in, open the connection
loop  = ["me"]      # then hammer at the target rate
rate  = "1000/s"
duration = "5m"
```

redis, pgwire, grpc are more shapes reached through the same `proxima::Client`.
adding one doesn't touch the engine.

## open-loop scheduler

closed-loop tools quietly drop the offered rate when the target slows, hiding the
tail. that's coordinated omission. rekt's scheduler primitives use an absolute
arrival grid so a late poll catches up instead of sliding the schedule. The
staged CLI still runs the planned count sequentially; wiring that scheduler into
the scenario runner is the next runtime step. closed-loop is there when you want
fixed concurrency, and the `rekt_load` benchmark uses that shape deliberately.

## proxima-shaped: inherited vs owned

every step drives a `proxima::Client` pipe, so resilience and capture are
middleware you wrap around it, not code rekt writes. `Retry::new(into_handle(client))`
and friends compose straight onto it. what comes free:

- retry — `Retry::new(pipe).with_max_attempts(n).with_base_delay(..)`, with a
  budget, jittered backoff, and idempotency gating. models how real clients retry.
- rate limit — `RateLimit` token bucket. this *is* the driver's offered-rate
  control, not a bolt-on.
- record — `RecordUpstream` wraps the pipe and captures every call to disk
  (zstd/postcard or jsonl). gives a deterministic offline target for testing
  scenarios, and a way to capture real traffic.
- replay — `ReplayUpstream::from_jsonl(..)` serves captured traffic back, matched
  on method+path+query. two uses: a fake target for ci, and recorded traffic as
  the load model.

what proxima does *not* hand you as a wrapper, so rekt owns it deliberately:

- in-flight bound — there's no concurrency-limit middleware, only stream-level
  backpressure. and that's correct here: in open loop you must not backpressure
  the arrival schedule, or you've rebuilt coordinated omission and your "offered
  rate" is a lie. rekt bounds in-flight calls itself, and when a slow target
  blows the bound it counts them as timeouts instead of slowing arrivals. the
  policy is load-tester-specific; a generic one would mislead.
- timeout — no standalone timeout middleware; it's a knob on the leaf (the
  client/upstream config). rekt sets it per step.
- circuit / outlier — exists only at the pool-selection layer (`OutlierPolicy`
  on `UpstreamRef`), which is a feature, not a gap: spread load across N targets
  and eject unhealthy ones. per-single-target breaking isn't needed for load gen.

so retry, rate-limit, record, and replay are config rather than code, which thins
rekt further. the arrival scheduler and the in-flight policy were always going to
be the core, and they're exactly the parts that must stay load-tester-correct
instead of inherited.

## many faces, one engine

config-first means the IR is the api, so frontends just emit it. none of them
touch the hot path.

- toml and the `.http` file format — the default, no toolchain.
- python via pyo3 — build-time authoring only. python composes the plan with
  loops and logic, then hands the finished IR to the engine. it never runs per
  request; crossing the pyo3 boundary holds the gil and would serialize the whole
  load generator (locust's mistake). python builds plans, it does not fire.
- wasm — two jobs: the sandboxed per-request hook for the rare custom step (sign
  this, derive that) at near-native speed with no gil, and a portable authoring
  target for any language that compiles to it.

the rule is one line: no interpreted language on the hot path. a builtin
vocabulary (`capture`, `expect`, `now()`, `uuid()`, `sign_hmac_sha256()`,
feeders) keeps the wasm hook rare. the moment a frontend runs per request, you've
rebuilt k6 and thrown the speed away.

## reporting

real latency distribution per stage: p50/p90/p99/p999, throughput, error rate.
thresholds become the exit code, so it gates ci. The scheduler module contains
the open-loop grid pacer; the staged CLI currently uses stage rate × duration as
a planned count and fires that count sequentially.

## same basics as proxima

- edition 2024, warnings denied, no unwrap/expect/panic in the tree.
- clippy and fmt clean; thin crates; sans-io parsing where it fits.
- deterministic offline tests via record/replay.
- benches with checked-in baselines. "competes with wrk" is a hypothesis until a
  client-side load-gen bench proves it against wrk/h2load/trex on the same box.

## roadmap

1. ~~skeleton: scenario → fsm → driver → report on a mock target~~ (done)
2. ~~generic proxima client target: any registered `Client::from_value` protocol~~ (done)
3. real open-loop scenario pacing over the scheduler primitives.
4. the scenario IR + stepped scenarios: `Ctx`, `ProtoStep`, capture/bind/guard,
   the jwt chain as a test.
5. resilience as config: wrap steps with proxima `Retry` / `RateLimit` / record;
   plus rekt's own in-flight bound and per-step timeout.
6. ramp stages, soak, closed-loop mode.
7. replay captures as the load model.
8. more protocol-specific request helpers on top of the generic client door.
9. frontends: pyo3 (build-time), wasm (per-request hook + authoring).
10. the bench harness and baselines vs wrk / h2load / trex.
11. distributed runs.

## layout

```
rekt/
├─ rek.toml          a scenario
├─ examples/
│  ├─ rekt_load.rs         raw H1 closed-loop rekt-vs-wrk benchmark
│  ├─ rekt_h2.rs           raw H2 closed-loop benchmark
│  ├─ rekt_h3.rs           raw H3 (native QUIC) closed-loop benchmark
│  ├─ rekt_plan.rs         LoadPlan-driven run
│  ├─ bench_server.rs      H1 target server for the above
│  ├─ bench_server_h2.rs   H2 target server
│  ├─ bench_server_h3.rs   H3 target server
│  ├─ bench_server_axum.rs axum target server (incumbent-side comparison)
│  └─ bench_target.rs      shared target-handler helpers
└─ src/
   ├─ main.rs        cli: rek tools/rekt/rek.toml
   ├─ lib.rs         library surface — benches/future frontends reach the engine directly
   ├─ error.rs       the crate's Error enum
   ├─ outcome.rs     per-arrival outcome (success / error / timed-out)
   ├─ scenario.rs    the IR: steps, captures, guards, load
   ├─ report.rs      histograms -> summary + slo exit
   ├─ fsm.rs         run controller + scenario state machines (mock path, `scheduler` off)
   ├─ driver.rs      mock planned-count driver (`scheduler` off)
   ├─ engine.rs      scheduler-feature client runner + raw H1 throughput driver
   ├─ plan.rs        LoadPlan: the throughput load as first-class config + fluent builder
   ├─ h2load.rs      multiplexed HTTP/2 load over proxima's native h2 `Connection`
   ├─ h3load.rs      HTTP/3 load over proxima's native QUIC (H3NativeUpstream)
   └─ sched/         open-loop pacer and in-flight gate primitives
```

`engine.rs`/`plan.rs`/`h2load.rs`/`h3load.rs`/`sched/` compile only under the
`scheduler` feature; `driver.rs`/`fsm.rs` are the mock path used when it's off
(see `src/lib.rs`'s `#[cfg(feature = "scheduler")]` gating). The `[[example]]`
targets above (`tools/rekt/Cargo.toml`) all require `--features scheduler`.

## migrating off rek

about 90% of the old `rek` crate is replaced. the `arc/` archive and the
edition-2018 hyper client are gone; proxima is the client. the `.http` grammar
survives as a frontend, reworked off `unwrap`/`expect`. the engine, protocols,
histograms, and replay come from proxima, so the job is teardown plus the IR,
fsm, and report layers on top.
