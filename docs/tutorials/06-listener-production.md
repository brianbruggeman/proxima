# Listener on-ramp, part 3: growing it into production

**Prerequisites:** [part 1: hello](./04-listener-hello.md) and
[part 2: the universal listener](./05-listener-universal.md). You should be
comfortable with `Listener::builder()...serve()` and `.any()`/`.accept(name)`.

**You will:** take the toy listener from parts 1–2 and grow it into a
production shape, one real concern at a time: telemetry, an accept/deny
allowlist with a DoS blacklist, request-level admission that actually
sheds load under pressure, a resilient client, config-driven tuning, and
the same-port-vs-separate-port decision.

**New concepts (in order):** `Recorder`/`Exporter` (telemetry) ·
`.deny(name, literal)` / `.denies([...])` · `.blacklist(config)` ·
`DenySignature` · `max_in_flight_requests` · `ConnAdmission` / `ShedReason` ·
client-side `RateLimit` + `Retry`/`Backoff` · `BlacklistConfig::layered()` ·
same-port vs. separate-port.

Every code block below is real, cited by `file:line`, and backed by a
runnable example that compiles and runs green on this repository as of the
commit this page was written against (`86c9302f`):

- `examples/any_listener_production.rs` (§1–§4, §7)
- `examples/any_listener_client_resilience.rs` (§5)
- `examples/any_listener_conflag.rs` (§6)

Run them yourself: `cargo run --example any_listener_production --features
http1-native`, and likewise for the other two.

## 1. Telemetry: console + file, plus a real counter

Every production listener needs to be observable from `day 0`, not bolted
on later. proxima's house rule: telemetry export is never OTLP-only —
console and file sinks are wired side by side (`~/.claude/rules/rust.md`).
There is no single `.export()` call that fans to two sinks yet
(`proxima-telemetry/src/export.rs:277-282`'s own doc: "fan-out over
multiple exporters lands with the OTLP slice's `FanOut` stage" — not built
today), so two sinks side by side means building each as its own
`FormatterPipe` and combining them with `fan_exporters` — the same
combinator `examples/logs/main.rs`'s own fan-out section uses:

```rust
let stdout_handle = into_telemetry_handle(FormatterPipe::new(std::io::stdout(), LogFormat::Human));
let file_handle = into_telemetry_handle(FormatterPipe::new(
    std::fs::File::create(&log_path).expect("create log file"),
    LogFormat::Human,
));
let fanned = fan_exporters(vec![stdout_handle, file_handle]);
let recorder = Recorder::builder()
    .pipe(fanned)
    .core_count(1)
    .install()
    .expect("recorder installs as the process default");
```

`.install()` registers this as the process-default recorder, so
`proxima::telemetry::info!`/`warn!`/`error!` callsites anywhere in the
process find it automatically (`proxima-telemetry/src/export.rs:285-291`).
A counter is the direct-instrument fast path — one `AtomicU64::fetch_add`
per call, no ring, no allocation (`proxima-telemetry/src/recorder/mod.rs:1541`):

```rust
let requests_total = recorder.counter("proxima.any_listener.requests_total");
// inside your handler: requests_total.add(1, &[]);
```

This is what "metrics on" means concretely for this on-ramp — a real
counter you bump per request, not a separate toggle. One gotcha worth
naming here rather than discovering it later: `install_emit_filter` (from
`proxima::telemetry::emit::global::install`) has to run *before* your
process's first `info!`/`debug!` call — the emit filter is read lazily on
first use and cached per callsite, matching `examples/logs/main.rs`'s own
comment.

## 2/3. Accept + deny + blacklist — the whole simple form

This IS the simple form — nothing below is elided for the "production"
framing. `.accept("h1")` selects exactly the legit candidate this service
speaks; `.deny(name, literal)` registers a fixed malicious/scanner byte
literal ALONGSIDE it, reviewed by the same classifier every other candidate
is — never instead of the legit ones
(`src/listener/handle.rs:342-352`'s own doc). A match records a strike and
drops the connection, no handler dispatch, ever
(`proxima-listen/src/any/deny.rs`'s `DenySignature::drive`).
`.blacklist(config)` turns strikes into a real ban:

```rust
let service = Listener::builder()
    .bind(service_bind)
    .accept("h1")
    .deny("scanner", SCANNER_LITERAL.to_vec())
    .blacklist(
        BlacklistConfig::layered()
            .with_deny_strike_threshold(1)
            .build(),
    )
    .handle(into_handle(CountingOk { requests_total: requests_total.clone() }))
    .serve()
    .await?;
```

A `DenySignature` candidate is registered at priority `60000`
(`ANY_DENY_PRIORITY_DEFAULT`, `proxima-listen/proxima-listen.toml:44`) —
deliberately far above the default `100` every legit candidate gets, so a
positively-identified malicious literal is never held back waiting for a
legit candidate to lose (`proxima-listen/src/any/deny.rs`'s own doc on
why). `BlacklistConfig`'s default `deny_strike_threshold` is already `1`
(`proxima-listen/src/admission/blacklist.rs:66-73`) — a signature match is
not ambiguous noise, it bans on the first hit; the `.with_deny_strike_threshold(1)`
call above is explicit for teaching, not strictly required.

Running `examples/any_listener_production.rs`'s §2/§3 against a real
scanner literal and a real legit client produces exactly this:

```
§2/3: legit h1 request served, counter now at 1
§2/3: scanner literal dropped, no HTTP response, peer now banned
§2/3: same peer's next connection (legit payload!) dropped — banned pre-classify
```

That third line is the point of a *blacklist* rather than a one-shot deny:
the SAME peer's next connection — even carrying a perfectly legitimate
request — is dropped before the classifier ever inspects it, because the
peer itself is now banned (`ShedReason::Blacklisted`,
`proxima-listen/src/admission/state.rs:81-88`), checked before
`ListenerCore::admit` on every accepted connection.

## 4. Request-level admission: a real `ShedReason` on the wire

Connections aren't the only thing worth capping — a single h2 connection
can multiplex many concurrent streams, and you may want to cap *those*
too. `max_in_flight_requests` is a raw spec key, reached through the
`.spec(key, value)` escape hatch every `ListenerBuilder` chain has
(`src/listener/handle.rs:427-431`) — there is no typed
`.max_in_flight_requests(n)` builder method today:

```rust
let admission_service = Listener::builder()
    .bind(admission_bind)
    .accept("h2")
    .spec("max_in_flight_requests", json!(1))
    .handle(into_handle(SlowOk))
    .serve()
    .await?;
```

`proxima-http/src/any_listener.rs:574-578` reads that spec key and builds a
listener-wide [`ConnAdmission::new(n)`](../../proxima-listen/src/admission/request.rs)
— cloned into every accepted connection so the cap is shared across all of
them, not per-connection. h2 is the candidate that actually enforces it:
`proxima-http/src/http2/server.rs:307-320` calls
`admission.request_admit()` at its own per-STREAM boundary and renders a
real in-band 503 + `retry-after: 1` on `RequestAdmit::Shed`:

```rust
match admission.request_admit() {
    RequestAdmit::Admit => spawn_handler(stream_id, request, &dispatch, &mut handlers),
    RequestAdmit::Shed { reason } => {
        let response = Response::new(503)
            .with_body(Bytes::from_static(b"service unavailable"))
            .with_header("retry-after", "1");
        // ... render it on THIS stream only; the connection and every
        // other live stream keep running.
    }
}
```

Two real, concurrent h2 requests against a cap of 1, driven by
`examples/any_listener_production.rs`'s §4:

```
§4: two concurrent h2 requests against max_in_flight_requests=1 -> [200, 503]
§4: the shed request's body is the listener's real 503 rendering, not a stub
```

### A real defect this on-ramp found, and routed around

While verifying this section, a body-carrying (POST-with-bytes) request
that gets shed on this exact path produced a hard `RST_STREAM
(INTERNAL_ERROR)` instead of the documented in-band 503 — reproducible,
deterministic, isolated by direct instrumentation of
`proxima-http/src/http2/server.rs` (not left in the source; this
paragraph is the record of it). Root cause: the shed branch never
dispatches the built `Request` to a handler, so its embedded body-stream
receiver (`body_senders.insert(stream_id, tx)`, `proxima-http/src/http2/
server.rs:295`, unconditional whenever a request carries a body) is
dropped with the `Request`; the client's own DATA frame then arrives to a
closed channel, and the `BodyData` handler's `Some(Err(_))` arm resets the
stream instead (`proxima-http/src/http2/server.rs:357-360`). A **bodyless**
request never opens that channel, so it never hits this path — every shed
demonstration in this on-ramp uses a bodyless GET for exactly that reason.
This is filed as a real, reproducible gap for the owner to fix (not fixed
here — out of scope for a docs task); h1's own request loop has a
DIFFERENT gap worth knowing too: it never calls `request_admit()`/
`request_release()` at all (it only bridges the raw `in_flight`/`quiescing`
atomics for the quiesce/drain bookkeeping, `proxima-http/src/http1/
serve.rs`), so `max_in_flight_requests` is silently NOT enforced on the h1
candidate — h2 is the one this section demonstrates because it is the one
that actually works.

## 5. Client-side resilience: composed, not a new type

The listener sheds load; a well-behaved client should absorb that
gracefully instead of surfacing a raw 503 to whoever called it. The
question worth asking before reaching for a new type (`~/.claude/rules/
rust.md`'s binary pipe question): can this be expressed with pipes that
already exist? Yes — three of them, composed:

- **`RateLimit`** (`proxima_primitives::pipe::RateLimit`, the identical
  token bucket `docs/tutorials`'s own `rate_limit` example teaches
  server-side) self-throttles the OUTBOUND send rate. A client held to a
  quota is the same primitive, just wrapping a dial instead of a handler.
- **`Retry` + `Backoff` + `Jitter`** (`proxima_primitives::pipe::resilience`,
  the same primitives the `backoff` example teaches) retry a transient 503
  with exponential backoff, driven by the REAL production clock
  (`proxima_primitives::pipe::clock::TimeClock`) — real sleeps, proven
  against a real shedding listener, not a fake clock standing in for one.

```rust
let resilient = Retry::new(
    AsPipe(client),                 // the real H2ClientUpstream, bridged (see below)
    RetryController {
        rules: RetryRules::default(),   // retries 502/503/504 + real transport errors
        backoff: Backoff::Exponential {
            initial: Duration::from_millis(80),
            factor: 2,
            max: Duration::from_millis(400),
        },
        jitter: Jitter::None,
        max_attempts: 5,
        deadline: None,
    },
    TimeClock,
    7,
);
```

Running `examples/any_listener_client_resilience.rs`'s §2 against a real
listener capped at `max_in_flight_requests = 1`, with a second call
occupying the one slot for 150ms:

```
occupier (no retry wrapper): status 503
resilient client: Response { status: 200, ... } (the caller-visible outcome is always a
  clean 200, whether or not it needed a retry underneath)
```

The un-wrapped `occupier` call surfaces the 503 directly, exactly as it
should — it isn't wrapped in anything. The `resilient` client's
caller-visible result is a clean 200 either way: if it loses the race for
the one slot, its first attempt is shed and `Retry` recovers once the
occupier releases; if it wins the race instead, `Retry` is a no-op on a
non-retryable 200. (Which one wins that race is a genuine timing race, not
a controllable outcome — that's *why* the client needs its own resilience
in the first place.)

### The one bridge this composition needed, and why

`H2ClientUpstream` and `RateLimit` are both `SendPipe`-only — the
cross-core, `Send`-future tier. `Retry`'s generic bound is the plain `Pipe`
tier, which carries no `Send` requirement at all. This is not an
oversight: `SendPipe` is a genuinely SEPARATE trait, not `SendPipe: Pipe`,
because an RPITIT (return-position-impl-Trait-in-trait) method's
`Send`-ness cannot be strengthened by a subtrait on stable Rust —
`proxima-primitives/src/pipe/primitives.rs:104-106` says so directly in
its own doc comment. (The deeper reason this split exists at all: `Pipe`'s
non-`Send` future is what lets a per-core, shared-nothing runtime like
prime avoid a `Send` bound it doesn't need; `SendPipe` is the *additive*
form for the work-stealing case where a future genuinely needs to cross
cores.) One line closes the gap — a `Send` future trivially satisfies a
bound that only asks for "any future":

```rust
struct AsPipe<T>(T);

impl<T: SendPipe> Pipe for AsPipe<T> {
    type In = T::In;
    type Out = T::Out;
    type Err = T::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        self.0.call(input)
    }
}
```

This is a per-example bridge, not a library addition — exactly the kind of
small, local adapter the house rules explicitly bless ("structures in
`examples/` are FINE").

### What is deliberately NOT here: `FanIn`

You might expect `fanin!`/`FanIn` here, racing several backend candidates
and taking whichever answers first. It is deliberately absent, and the
reason is worth teaching rather than hand-waving: `FanIn`'s own module doc
is explicit that it does *not* race concurrent sources — "Scan, don't
race... [the merge] does not drive N sources concurrently and take a
winner... A source whose `call(())` is not yet ready is polled once, found
`Pending`, and its in-flight future is then DROPPED"
(`proxima-primitives/src/pipe/fan_in.rs`'s module doc). A live TCP dial +
h2 handshake will return `Pending` at least once — a `FanIn` source
wrapping one would have its dial restarted every single scan and never
complete. The shipped codebase agrees with this reading: the one tutorial
that actually forwards a real request to a real backend pool,
`examples/load-balance/main.rs`, hand-rolls its own round-robin cursor
(`select_backend`, `load-balance/main.rs:92-102`) instead of reaching for
`FanIn`; the one place `FanIn` DOES appear next to a gate
(`examples/gate/main.rs`'s BALANCE section) merges pre-populated
`VecDeque` queues, not a live dial. This on-ramp follows that same, real
precedent — a hand-rolled round-robin-over-healthy pool, proven directly
in `examples/any_listener_client_resilience.rs`'s §3:

```
4 requests round-robinned: 2 to backend A, 2 to backend B
```

## 6. Conflaguration as first-class

Every knob above (`.blacklist(config)`) is also a config surface, not just
a fluent call — house pattern P4: `Thing::builder()` for the mutable
fluent surface, a layered loader for config. `BlacklistConfig` (`proxima-
listen/src/admission/blacklist.rs`) follows it exactly:

```rust
let config = BlacklistConfig::layered()
    .from_path(&toml_path)
    .expect("a well-formed file loads")
    .build();
```

with `toml_path` pointing at a flat TOML file:

```toml
deny_strike_threshold = 1
unclassifiable_strike_threshold = 5
strike_window_ms = 60000
ban_duration_ms = 300000
```

### A real gotcha this on-ramp verified directly

There are TWO genuinely different `[admission...]` TOML shapes in this
codebase, and they are NOT interchangeable. The build-time SIZING TOML
(`proxima-listen/proxima-listen-core.toml`, read by `build.rs`, baked into
the no_std+no_alloc floor's `sized::` consts) nests its data under
`[admission]` / `[admission.blacklist]` table headers. The RUNTIME layered
loader shown above — the one `.from_path()` actually calls — is FLAT:
`conflaguration::from_file` deserializes the file straight into
`BlacklistConfigPartial` (`proxima-listen/src/admission/blacklist.rs:152`),
which has no field literally named `admission`. Handing it a
`[admission.blacklist]`-nested file (matching the *other* TOML's shape by
habit) does not error — it's syntactically valid TOML — but it silently
changes nothing, because the partial never sees those nested keys.
`examples/any_listener_conflag.rs`'s §2 proves this directly rather than
asserting it from documentation:

```
§2: a [admission.blacklist]-nested TOML loads WITHOUT error but changes nothing
   (deny_strike_threshold stayed at the default 1, not the file's 99) — the runtime
   loader wants a FLAT file, unlike the build-time sizing TOML
```

If you're tuning this in production and a config change silently doesn't
take effect, this is the first thing to check.

## 7. Same port vs. separate port

The question this on-ramp set out to answer directly, both sides shown
side by side, on the same running process (`examples/any_listener_production.rs`'s §7):

**SAME port — `.any()`:**

```rust
let server = Listener::builder()
    .bind(same_port_bind)
    .any()
    .handle(into_handle(handler))
    .serve()
    .await?;
```

One socket. Every registered candidate reachable at one address. Use this
when every candidate SHOULD be reachable behind one firewall rule, one DNS
name, or one reverse proxy in front of it — the common case for "this
service speaks HTTP, I don't care which version a given client uses."

**SEPARATE ports — N independent `.accept(name)` binds:**

```rust
let h1_only = Listener::builder().bind(h1_only_bind).accept("h1").handle(...).serve().await?;
let h2_only = Listener::builder().bind(h2_only_bind).accept("h2").handle(...).serve().await?;
```

N sockets, each pinned to exactly one wire, each with its OWN
`Listener`/`Server` handle. Use this when candidates need independent
lifecycle (restart or drain one without touching the other), independent
firewalling (an internal-only h2 port vs. a public h1 port), or when a
wire has a conventional dedicated port you're expected to honor (h2 on
8443, a metrics-only port on 9090). Both binds ran in the same process in
this on-ramp's own example, proving the two shapes genuinely coexist:

```
§7: SAME port 127.0.0.1:54809 answers both h1 and h2 (.any())
§7: SEPARATE ports — h1 only on 127.0.0.1:54811, h2 only on 127.0.0.1:54812 — each with
   its own bind, its own lifecycle
```

There is no third option hiding here, and no hand-waving: pick same-port
when the candidates share a lifecycle and an address is meant to be
protocol-agnostic; pick separate-port the moment any candidate needs its
own bind, its own firewall rule, or its own restart schedule.

## What you built

Starting from part 1's three-line hello: telemetry wired from the start,
an accept/deny allowlist backed by a real DoS blacklist, a request-level
cap that renders a real `ShedReason` on the wire, a client that survives
that shed gracefully by composing existing primitives (never inventing a
new one), the same tuning driven from a config file instead of hardcoded
calls, and — the question this whole on-ramp exists to answer — a clear,
tested rule for when one port is right and when N ports are right. Nothing
here required leaving the shape parts 1–2 taught; every section only added
one more real call onto the same `Listener::builder()...serve()` chain.

## Where to go next

- [`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) — the
  full builder story: `.tcp()`/`.udp()`/`.quic()`/`.tls()`/`.grpc()`/
  `.pgwire(query)`, why TLS composes as a decorator instead of a spec field.
- [Part 4: composing the sugar](./07-sugar-composition.md) — every axis
  from part 3 (`.accept`/`.deny`/`.blacklist`) still composes with the
  transport/security/protocol axes this page adds next.
- [`docs/tutorials/03-native-runtime.md`](./03-native-runtime.md) — the
  `Runtime` trait, `http1` vs `http1-native`, and the ambient-runtime
  adoption rule `#[proxima::main]` relies on.
- [Build a load balancer](./build-a-load-balancer.md) — the real,
  hand-rolled round-robin-over-healthy pattern §5 above pointed at, in
  full.
- [Foundations: the Pipe](./00-foundations.md) — the base algebra every
  primitive in this page (`RateLimit`, `Retry`, `FanIn`) is built from.
