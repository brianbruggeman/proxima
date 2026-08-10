# The native runtime: serving real HTTP with zero tokio

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md), sections 1–7 and 13. You should already know: what `Pipe`/`SendPipe` are; the free-function and stateful-`impl` forms of `#[proxima::piped]`; and that `App::mount` attaches a handler at a path, `into_handle` holds any `Handler` behind one uniform `PipeHandle`, and *something* — Foundations calls it only "the engine that actually drives async work" — sits underneath `App` and actually runs the futures a pipe's `call` returns. This document names that thing and shows you how to control it.

**You will learn:** that proxima serves real HTTP with **zero tokio anywhere in the build** — not "tokio hidden behind a feature flag," but genuinely absent from the dependency graph, provably so with one `cargo tree` command — and that this is the *default*, not a stripped-down alternative. You will also learn the one non-obvious rule that trips up every multi-`App` program: booting one runtime for `main` silently becomes booting one runtime for *every* `App` you build inside it, unless you explicitly opt out.

**New concepts (in order):** the `Runtime` trait · `http1` vs. `http1-native` (tokio-coupled vs. tokio-free h1) · `#[proxima::main(cores = N)]`'s ambient-runtime publication · `App::with_runtime` / `App::with_acceptor_factory` (and the matched-bundle `RuntimeSelection` / `AppBuilder::runtime(selection)` surface promoted over them) · the `AcceptorFactory` trait · `ShutdownBarrier` · `deferred_runtime`/`DeferredRuntime` (a runtime handed to a component on purpose, not adopted by accident).

Every code block below is copied verbatim from a real file in this repository, cited by `file:line`, or is a command this tutorial's author actually ran — every transcript shown is real output, captured the day this document was written. Several blocks cite code that cannot stand alone as an external doctest — an excerpt mid-way through a real `main`, a trimmed trait/struct definition missing the source file's own `use` block — and are marked `` ```rust,ignore `` for exactly that reason, with a one-line comment naming it: the citation is real and (for the `main`-excerpts) independently verified live via the `cargo run --example` transcript that follows it in the same section; `scripts/tutorials-gate.sh` compile-checks every other block against this repository's current source. Where the current repository state disagrees with a claim in an older document, this tutorial says so explicitly rather than repeating it. (This document was originally checked against commit `238229cd`, re-verified against `0ac7a565`, and has now been fully re-verified against `c507563eb` — roughly fifty commits later, dominated by the runtime-selection-by-value arc: `InstalledRuntime`'s three loose fields were promoted to a named, `Validate`-checked `RuntimeSelection` value tagged with a `RuntimeBackend`, `#[proxima::main]`'s installed runtime is now wrapped in an extra-worker `AdoptedRuntime`, and `App::with_runtime`/`with_acceptor_factory` — the pair this tutorial's five examples still use — are now flagged a HAZARD in their own doc comments next to the promoted `AppBuilder::runtime(selection)`/`App::with_runtime_selection(selection)` surface. Section 4 and every citation below were re-checked line-for-line against this commit; every `cargo run`/`cargo tree`/`cargo build` transcript was re-captured live. None of the five migrated examples' own source changed in that window beyond a harmless import reorder.)

## Contents

1. A pipe never knows who is running it: the `Runtime` trait
2. Two h1 features, one listener: `http1` vs. `http1-native`
3. `proxy`: the minimal shape, proven tokio-free
4. The ambient-runtime seam — the centerpiece
5. `gateway`: policy composition is orthogonal to the runtime choice
6. `load-balance`: four independent runtimes, one process
7. `integration`: a runtime you build, and a runtime you deliberately share
8. `distributed_trace`: trace context survives a real TCP hop, still zero tokio
9. When tokio *is* the right answer: `multi_runtime` and `runtime_select`
10. Where to go next

## 1. A pipe never knows who is running it: the `Runtime` trait

Foundations §13 built `hello` on `App::new()` and never named what actually executes the `Future` a `Pipe::call` returns. Here is the answer, copied (doc comments trimmed) from `proxima-runtime/src/lib.rs:276`:

```rust,ignore
// doc comments trimmed and the tail methods elided — the real trait has
// the imports (`SpawnError`, `CoreId`, ...) this excerpt does not repeat.
pub trait Runtime: Send + Sync + 'static {
    fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>);
    fn spawn_on_core(
        &self,
        core_id: CoreId,
        future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError>;
    // ...spawn_factory_on_core, spawn_background_blocking, timer_at, num_cores, current_core
}
```

A `Runtime` is the engine: something that owns OS threads (or, on `no_std`, a fixed set of cores) and knows how to run a `Future` to completion on one of them. `App` never spawns a raw OS thread itself — it holds an `Arc<dyn Runtime>` and asks it to spawn. Nothing about `Pipe`, `SendPipe`, or `Handler` (Foundations §2–13) mentions a runtime at all, and that is the point: a pipe's `call` is just "eventually produces `Result<Out, Err>`" — *who* drives that future to completion is a separate, swappable concern, injected from outside.

This tutorial's first five examples (sections 3, 5–8) use exactly one implementation of this trait, `PrimeRuntime` (`proxima::prime::PrimeRuntime`, re-exported at `src/lib.rs:264` from `prime::os::runtime::PrimeRuntime`, defined at `prime/src/os/runtime.rs:28`) — prime's own per-core executor, no tokio underneath it anywhere. Section 9 introduces a second implementation, `TokioPerCoreRuntime`, to show the trait is genuinely open — but you do not need it before then.

## 2. Two h1 features, one listener: `http1` vs. `http1-native`

Before wiring a runtime to a listener, one more piece of vocabulary: the HTTP/1.1 listener itself comes in two feature-gated flavors, and which one you link determines whether tokio enters the build at all. From `proxima-http/Cargo.toml:102–115`:

```toml
# `http1-native` is the tokio-free base: the sans-IO codec
# (proxima-protocols::http1_codec) + the futures-io connection driver
# (`http1::serve` — `serve_connection`/`serve_h1_connection`), mirroring
# `http2-native`. `http1` layers the legacy hyper/tokio client stack...
http1-native = [
    "proxima-protocols/http1_codec-alloc",
]
```

(The feature value itself is `http1_codec-alloc`, not the bare `http1_codec` the comment names in prose — the alloc-tier split landed after this comment was last worded; same codec, one more named tier.)

and the umbrella listener feature that turns this codec into something `App` can actually bind a socket to, `proxima-http/Cargo.toml:215–219`:

```toml
http-listener = [
    "http1-native",
    "proxima-core/io-async-compat",
    "proxima-protocols/proxy_protocol-std",
]
```

The umbrella `proxima` crate (this repository's top-level package, what `examples/*/main.rs` depend on) re-exposes that pairing as its own `http1-native` feature, `Cargo.toml:518–519`:

```toml
http1-native = ["any-listener"]
http1 = ["tokio", "http1-native", "proxima-http/http1"]
```

(`http1-native` reaches `proxima-http/http-listener` one hop indirectly now,
through `any-listener = ["proxima-http/http-listener"]`, `Cargo.toml:504` —
same transitive dependency, one more named feature in between.)

Read those two lines carefully: `http1-native` pulls in the sans-IO codec — "sans-IO" meaning the HTTP/1.1 framing/parsing logic itself never touches a socket or an async runtime; it only turns bytes into requests/responses and back, so it can be driven by *any* I/O source, prime's or tokio's or a plain in-memory buffer in a test — and the `AcceptorFactory`-driven accept path (`listener::serve_via_factory`) — no `dep:tokio` anywhere in that chain. `http1` is `http1-native` *plus* `dep:tokio` plus hyper's legacy tokio-coupled client/accept-loop machinery, kept around for callers who have not migrated. **`http1-native` is not a stripped-down `http1`** — it is the tokio-free base that `http1` is built on top of, not the other way around.

One more precision worth having exact, since it corrects a claim in an older tutorial: `http1-native`'s connection driver (`serve_h1_connection`/`serve_connection`, `proxima-http/src/http1/serve.rs:107,166`) is generic over `Stream: futures::io::AsyncRead + futures::io::AsyncWrite + Unpin + Send` (imported at `proxima-http/src/http1/serve.rs:26` — the `futures` crate's I/O traits, not tokio's). The h1 *protocol driver* has always been sans-IO once `http1-native` exists; what still varies by backend is one layer below it — who opens the listening socket and hands back an accepted connection that implements those traits. That is the `AcceptorFactory` trait, `proxima-primitives/src/stream/mod.rs:197–199`:

```rust,ignore
// excerpted without the crate's own `use` block — `SocketAddr`/
// `TcpBindOptions`/`TcpAcceptor`/`io` resolve inside that file, not here.
pub trait AcceptorFactory: Send + Sync + 'static {
    fn bind(&self, addr: SocketAddr, options: TcpBindOptions) -> io::Result<Box<dyn TcpAcceptor>>;
}
```

`proxima_net::prime::PrimeAcceptorFactory` (`proxima-net/src/prime/mod.rs:123`) binds the socket through prime's own reactor and hands back a connection (`PrimeTcpConnection`, implementing `futures::io::{AsyncRead, AsyncWrite}` at `proxima-net/src/prime/mod.rs:86,96`) that the tokio-free h1 driver can drive directly. A `TokioAcceptorFactory` exists too (`proxima_net::tokio::TokioAcceptorFactory`, used in section 9) — same trait, tokio underneath instead. **Runtime and acceptor factory are always a matched pair**: the runtime is what runs the connection's future; the acceptor factory is what produced the socket that future reads and writes. Mismatching them (a tokio socket handed to a task spawned on a prime worker with no tokio reactor registered) is exactly the kind of bug `App::with_runtime`/`App::with_acceptor_factory` (section 4) are designed to be called together to prevent.

Finally, the umbrella `Cargo.toml`'s own default-feature list states the header claim plainly (comments elided for the middle section — the full block runs `Cargo.toml:434–456`):

```toml
default = [
    # `serve-prime` makes PrimeRuntime the default serve+chain runtime —
    # tokio is NOT in the default dependency graph at all (verify with
    # `cargo tree -e normal -i tokio`). `http2`/`http3` resolve to the
    # native, tokio-free drivers (`http2-native`/`http3-native`). Opt into
    # the tokio-backed capability set (sister-tokio serve runtime, hyper,
    # quinn-compat h3, legacy h1 client+listener) with `--features tokio`;
    # `http1` layers that legacy hyper/tokio h1 stack on top of
    # `http1-native`, which is itself the tokio-free sans-IO h1 driver
    # (`serve_connection`/`serve_h1_connection`, generic over
    # `futures::io::AsyncRead`/`AsyncWrite`) — see `hello`'s doc comment
    # below for the tokio-free flagship built on it.
    "serve-prime",
    "http2", "http3",
    "histogram", "macros",
    "http-prime-deps",
]
```

**Update since this document's first pass:** the comment above used to read "`http1` specifically needs `tokio` because its connection driver has no sans-IO implementation yet (h2/h3 do)" — the same stale claim `00-foundations.md:778` and `hello`'s own doc comment carried (see §10's link list), false since `http1-native`'s `serve_connection`/`serve_h1_connection` (this section, above) landed. That comment has been corrected directly, as part of landing this document, to say precisely what §2 above already teaches: `http1-native` is the tokio-free base, `http1` layers the legacy hyper/tokio stack on top of it. Section 3 still proves the tokio-free claim for real rather than trusting any comment.

## 3. `proxy`: the minimal shape, proven tokio-free

`examples/proxy/main.rs` is the smallest of the five: one pipe, `ProxyPipe`, whose entire `call` body is handing the inbound request to a `Client` and returning what comes back (`proxy/main.rs:57–63`) — `proxima::Client` is itself a `SendPipe<In = Request<Bytes>, Out = Response<Bytes>>`, so forwarding is composition, not new machinery. That is not this tutorial's subject (`00-foundations.md` and the `build-a-*` project tutorials cover the pipe side in depth); this tutorial is about the handful of lines around it that decide *what runs it*.

Its `Cargo.toml` entry, `Cargo.toml:1819–1822`:

```toml
[[example]]
name = "proxy"
path = "examples/proxy/main.rs"
required-features = ["runtime-prime-executor", "runtime-prime-inbox-alloc", "runtime-prime-reactor", "runtime-prime-bgpool", "http-prime-deps", "http1-native", "macros"]
```

Note what is *absent*: no `"tokio"`, no `"runtime-tokio"`. Prove it yourself — the minimal feature set that actually builds cleanly standalone is `serve-prime` (the umbrella bundle that includes all four `runtime-prime-*` features plus `http-prime-deps`, and additionally arms this crate's own test harness — see the callout at the end of this section for why the literal list above needs that addition) plus `http1-native` plus `macros`:

```
$ cargo tree --no-default-features --features "serve-prime,http1-native,macros" -e normal -i tokio
warning: nothing to print.
```

Empty output *is* the proof — `cargo tree -i <crate>` prints every path from the root to a matching dependency, and here there is none. Now run it for real, with the exact same features:

```
$ cargo run --example proxy --no-default-features --features "serve-prime,http1-native,macros"
origin listening on 127.0.0.1:8081
proxy  listening on 127.0.0.1:8080, forwards to 127.0.0.1:8081

client -> proxy raw response:
HTTP/1.1 201 Created
x-origin: proxima-origin
traceparent: 00-4afced3d50e38e99a43c32862276e721-c03a100d8f0e7ea9-01
content-length: 21

origin response body


PASS: forward-to-upstream is composition — the proxy pipe added no bytes, dropped none.
proxy  drained: cores_acked=1 hooks_drained=0
origin drained: cores_acked=1 hooks_drained=0
```

A real HTTP/1.1 response, over a real `TcpStream` (`proxy/main.rs`'s own client is a hand-rolled blocking socket, deliberately not another proxima pipe — see `blocking_get`, `proxy/main.rs:152–159`), served by a build with no tokio in it anywhere. That is the whole headline claim, demonstrated rather than asserted.

Now the two Apps that make it happen, `proxy/main.rs:81–99`:

```rust,ignore
// excerpted from `proxy/main.rs`'s `main` — `origin_pipe`/`origin_bind` are
// defined earlier in that function, not repeated in this excerpt.
let origin_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(Arc::new(PrimeRuntime::new(1)?))
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
origin_app.mount("/", origin_pipe)?;

let origin_listener = origin_app.build_listener(ListenerSpec::http(origin_bind))?;
// ...
let proxy_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(Arc::new(PrimeRuntime::new(1)?))
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
```

Piece by piece, each grounded in source:

- **`App::builder()`** (`src/app.rs:882`) returns an `AppBuilder` (`src/app_builder.rs:56`) — the mutable, fluent construction surface. `.with_defaults()` (`app_builder.rs:106`) registers the built-in listen protocols, upstream factories, and codecs; `.build()` (`app_builder.rs:330`) consumes the builder and returns a plain `Result<App, ProximaError>`.
- **`.with_runtime(Arc::new(PrimeRuntime::new(1)?))`** (`App::with_runtime`, `src/app.rs:413`) replaces the `App`'s runtime with a freshly built, one-core `PrimeRuntime` (`PrimeRuntime::new`, `prime/src/os/runtime.rs:48`). Note it is called *after* `.build()`, on the already-constructed `App` — section 4 explains exactly why that ordering matters, and why the source itself now flags this setter a HAZARD next to the promoted alternative.
- **`.with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))`** (`App::with_acceptor_factory`, `src/app.rs:431`) pairs that runtime with the matching prime-backed socket opener from section 2. Both setters are `#[must_use] fn(self) -> Self` — plain builder methods, not `Result`, so no `?` after either.
- **`origin_app.build_listener(ListenerSpec::http(origin_bind))`** (`App::build_listener`, `src/app.rs:1114`; `ListenerSpec::http`, `proxima-listen/src/handle.rs:80`) binds and starts accepting *before returning* — it blocks the calling thread only until the accept lane has acked ready, never polling or sleeping to find out.

At the end, instead of `Foundations`'s `server.run_until_signal()` (which blocks forever waiting for `SIGINT`/`SIGTERM` — right for a long-running server, wrong for a demo/test process that needs to prove something and then exit), `proxy` drains deterministically with `ShutdownBarrier` (`proxima_primitives::sync::shutdown::ShutdownBarrier`, re-exported as `proxima::shutdown::ShutdownBarrier`, `src/lib.rs:189`):

```rust,ignore
// excerpted from `proxy/main.rs`'s `main` — `proxy_runtime` and the
// `ShutdownBarrier` import are defined/brought in earlier in that file.
let proxy_report = ShutdownBarrier::new(proxy_runtime).broadcast_drop().await;
println!(
    "proxy  drained: cores_acked={} hooks_drained={}",
    proxy_report.cores_acked, proxy_report.hooks_drained
);
```

`ShutdownBarrier::new(runtime)` (`proxima-primitives/src/sync/shutdown.rs:151`) and `.broadcast_drop()` (same file, returning a `ShutdownReport { cores_acked, hooks_drained }` at line 213–217) broadcast a stop signal to every worker on *that one runtime* and wait for every core to acknowledge — a report you print and assert on, not a signal you have to send yourself from another shell. Two `App`s, two independent runtimes, two independent drains: nothing here waits on the other.

**A verified gap, flagged rather than hidden:** the *literal* `required-features` list quoted above (without `serve-prime`) still fails to build standalone — `cargo build --example proxy --no-default-features --features "runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http-prime-deps,http1-native,macros"` exits 101 — but the cause has changed. It used to stop in `proxima-test` (16 `deny(warnings)` dead-code errors, e.g. `function 'report_from' is never used`), because that crate's driver half was ungated and only `serve-prime` forwards `proxima-test/test-prime`; that is fixed as of 2026-08-05 (the driver half is now behind `any(tokio-driver, test-prime)`, gated by `scripts/proxima-test-gate.sh`). What remains is one dead-code error in the `proxima` crate itself at this cell: `struct 'AdoptedRuntime' is never constructed`, `src/runtime.rs:632`. Still pre-existing — every one of the five migrated examples' `Cargo.toml` entries has the same gap — not something this tutorial introduces. The commands shown above (`serve-prime` instead of the four sub-features) are the verified-working substitute; `cargo build/run --example proxy --features http1-native` (default features plus `http1-native`, no `--no-default-features`) also builds and runs cleanly, since `default` already includes `serve-prime`.

## 4. The ambient-runtime seam — the centerpiece

Every one of the five migrated examples repeats the same four-line idiom — `.build()?.with_runtime(Arc::new(PrimeRuntime::new(N)?)).with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))` — on *every* `App` it builds, even when there is only one `App` in the whole program. That repetition is not boilerplate for its own sake. It exists to opt out of something `#[proxima::main]` does automatically, and skipping it produces a real, silent bug. This section is that bug, and the mechanism behind it, in full.

### What `#[proxima::main(cores = N)]` actually does

`#[proxima::main]` (`proxima-macros/src/lib.rs:106`) turns `async fn main() -> R` into a synchronous `fn main() -> R` that boots a runtime and drives your body to completion on it. Its own module doc states the mechanism plainly (`proxima-macros/src/main_attr.rs:28–31`):

> The booted runtime is published via `proxima::runtime::install_runtime` so `App::new()` called from `main`'s body adopts it instead of building an independent second one — one `#[proxima::main(cores = N)]` now means one N-core runtime, not two runtimes with contradictory core counts.

`install_runtime` (`src/runtime.rs:252–254`) and its reader `installed_runtime` (`src/runtime.rs:260–263`) are a process-wide, set-once cell holding one `RuntimeSelection` — the matched bundle a runtime travels with everywhere now:

```rust,ignore
// excerpted without the crate's own `use` block — `RuntimeBackend`/
// `Runtime`/`AcceptorFactory`/`DatagramFactory`/`UnixUpstreamFactory`/
// `PacketListenerFactory` resolve inside that file, not repeated here.
static INSTALLED_RUNTIME: OnceLock<RuntimeSelection> = OnceLock::new();

pub struct RuntimeSelection {
    pub backend: RuntimeBackend,
    pub runtime: Arc<dyn Runtime>,
    pub acceptor_factory: Arc<dyn AcceptorFactory>,
    pub datagram_factory: Option<Arc<dyn DatagramFactory>>,
    pub unix_upstream_factory: Option<Arc<dyn UnixUpstreamFactory>>,
    pub packet_listener_factory: Option<Arc<dyn PacketListenerFactory>>,
}

pub fn install_runtime(selection: RuntimeSelection) {
    let _ = INSTALLED_RUNTIME.set(selection);
}

pub fn installed_runtime() -> Option<RuntimeSelection> {
    INSTALLED_RUNTIME.get().cloned()
}
```
(`src/runtime.rs:60,137–145,252–263`)

**Update since this document's first pass:** the seam originally described here as a bare `InstalledRuntime { runtime, acceptor_factory, datagram_factory }` struct plus three free-standing `install_runtime` parameters has been promoted to the named value type above, `RuntimeSelection`, tagged with a `RuntimeBackend` (`Prime`/`Tokio`/`Other(&'static str)` for an out-of-tree backend, `src/runtime.rs:85–92`) and grown two more matched factories — `unix_upstream_factory`/`packet_listener_factory`, siblings of `acceptor_factory` for the unix-socket upstream and UDP packet-listener dispatch sites. The *mechanism* this section teaches — an ambiently-published value an `App` adopts unless it opts out — is unchanged; only its shape got a name and two matched-bundle constructors, `RuntimeSelection::from_prime`/`from_tokio` (`src/runtime.rs:198–247`), that make hand-assembling a mismatched bundle (a prime runtime paired with a tokio acceptor) something you have to go out of your way to do. Every field stays `pub`, but `AppBuilder::build()` now calls `.validate()` (`Validate` impl, `src/runtime.rs:147–178`) on whatever `RuntimeSelection` it resolves, rejecting a mismatch with a named error before the first opaque poll failure deep inside the wrong reactor — see `app_builder.rs`'s `build_rejects_a_hand_assembled_selection_with_a_mismatched_datagram_factory` test for that failure mode proven directly. This tutorial's five examples are all TCP/h1, so `datagram_factory`/`unix_upstream_factory`/`packet_listener_factory` stay populated-but-unused throughout and never come up again below.

(An implementation aside, not a behavior change: the runtime `#[proxima::main(cores = N)]` installs today is not a bare `PrimeRuntime` but an `AdoptedRuntime` wrapper (`src/runtime.rs:632–687`) that internally boots one *extra*, invisible worker to drive `main`'s own body — without it, `main`'s readiness-wait for a listener on the SAME core it also runs on would deadlock that core's one OS thread. `AdoptedRuntime::num_cores()` still reports exactly `N`, so every claim below about sizing and collapse is unaffected.)

`#[proxima::main(cores = 1)]` calls this once, at startup, with the one-core runtime it just booted (wrapped in `AdoptedRuntime`, per the aside above). And here is the seam that matters: `App::builder()...build()`'s internals check this cell **first**, before considering anything else — `resolve_runtime_selection`, `src/app.rs:139–150`:

```rust,ignore
// excerpted without the crate's own `use` block — `RuntimeSelection`/
// `ProximaError`/`resolve_default_runtime_selection` resolve inside that
// file, not repeated here.
fn resolve_runtime_selection(
    explicit: Option<RuntimeSelection>,
    cores_override: Option<usize>,
) -> Result<Option<RuntimeSelection>, ProximaError> {
    if let Some(selection) = explicit {
        return Ok(Some(selection));
    }
    if let Some(installed) = crate::runtime::installed_runtime() {
        return Ok(Some(installed));
    }
    resolve_default_runtime_selection(cores_override)
}
```

Three tiers, checked in that exact order: (1) an EXPLICIT `RuntimeSelection` the caller passed — via `AppBuilder::runtime(selection)` or `App::with_runtime_selection(selection)`, the promoted surface the "fix" subsection below teaches — always wins; (2) the ambient selection `#[proxima::main]` (or any other `run*` driver) already booted and published; (3) only then the documented fallback cascade, `resolve_default_runtime_selection` (`src/app.rs:193–254`) — prime-first-if-linked, else tokio, else no runtime at all.

Put those two together: **every `App::builder()...build()` call inside a `#[proxima::main(cores = N)]`-driven `main`, with no explicit `.runtime(...)` override, adopts the exact same `RuntimeSelection` — the same `Arc<dyn Runtime>` inside it, not an equivalent one.** Two `App`s built this way do not get "two 1-core runtimes"; they get one 1-core runtime, shared.

### The collapse, proven

Here is a minimal, three-line repro — not one of this repository's shipped `examples/`, written and run once for this tutorial to verify the claim empirically rather than assert it:

```rust
#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    let app_one = App::builder().with_defaults()?.build()?;
    let app_two = App::builder().with_defaults()?.build()?;

    let runtime_one = app_one.runtime().expect("app_one has a runtime");
    let runtime_two = app_two.runtime().expect("app_two has a runtime");

    println!("app_one cores = {}", runtime_one.num_cores());
    println!("app_two cores = {}", runtime_two.num_cores());
    println!("same runtime instance (Arc::ptr_eq) = {}", Arc::ptr_eq(&runtime_one, &runtime_two));
    Ok(())
}
```

Real, captured output (this file was added temporarily as a throwaway `[[example]]`, run once, and removed — it is not part of this repository's `examples/` today, so treat the code above as a verified transcript, not a command you can re-run as-is):

```
app_one cores = 1
app_two cores = 1
same runtime instance (Arc::ptr_eq) = true
```

`Arc::ptr_eq` compares pointer identity, not just equal values — `true` here means `app_one` and `app_two` are not "two 1-core runtimes that happen to agree," they are the *literal same* runtime object. Neither `App` passed an explicit `RuntimeSelection`, so both fell through to `resolve_runtime_selection`'s tier-2 ambient-adoption branch and got back the identical `Arc` `#[proxima::main]` installed. Scale this to `gateway`'s three `App`s or `load-balance`'s four (sections 5–6): without the override, all of them would collapse onto the one runtime `#[proxima::main(cores = 1)]` booted — one shared, one-core executor serving every listener in the whole program, not the N independent ones each example's own doc comment says it wants. Every one of the five migrated examples' `main` functions carries this exact explanation inline, e.g. `proxy/main.rs:65–73`:

> `#[proxima::main(cores = 1)]` boots a throwaway 1-core prime runtime just to give `main` an async context to `.await` on (no tokio anywhere in the build...). That boot publishes an AMBIENT runtime (`crate::runtime::install_runtime`), which `App::builder().build()` would otherwise silently adopt — collapsing the two apps below onto ONE shared runtime instead of each having its own. Each app opts back OUT of that adoption with an explicit `.with_runtime(...)` + `.with_acceptor_factory(...)`.

### The near-miss: `AppBuilder::with_runtime_cores`

There is a method that *looks* like the right tool and is not, once you are inside `#[proxima::main]`: `AppBuilder::with_runtime_cores(usize)` (`src/app_builder.rs:289–296`):

```rust
/// Sugar for `.with_runtime_config(RuntimeConfig::builder().cores(cores).build())`.
#[must_use]
pub fn with_runtime_cores(self, cores: usize) -> Self {
    self.with_runtime_config(
        crate::app_config::RuntimeConfig::builder()
            .cores(cores)
            .build(),
    )
}
```

It is real, public API, called *before* `.build()` — which reads as "size this App's runtime to `cores` cores." Follow it through: `AppBuilder::build()` (`app_builder.rs:330–416`) resolves a `cores_override` from `self.runtime_config` (`.resolved_cores()`), and separately resolves a `runtime_selection` value that is `self.runtime_selection` (set only by `.runtime(...)`, the promoted surface below) when present, else `self.runtime_config.resolve_selection()` (`src/app_config.rs:210–212`). That second lookup is the trap: `RuntimeConfig::resolve_selection` returns `None` whenever `backend` is `RuntimeBackendSelection::Auto` (`src/app_config.rs:100–106`) — the default, and exactly what `.with_runtime_cores`'s sugar leaves it at, since it only ever sets `.cores(...)`. So `runtime_selection` stays `None`; `App::with_components` (call site `app_builder.rs:409–415`, definition `423–439`) passes `cores_override = Some(N)` and `runtime_selection = None` into `App::__internal_assemble` (`src/app.rs:809–858`), which calls exactly the `resolve_runtime_selection(None, Some(N))` shown above — tier 2, the ambient-adoption check, fires and returns before `cores_override` is ever consulted. Inside any `#[proxima::main]`-driven binary something is always installed at tier 2, so `with_runtime_cores`'s value is silently never read. It is not broken — it does exactly what its doc comment says, sizing a *fallback* — it is just the wrong tool for "give this `App` its own runtime" once an ambient one already exists, and nothing about the call site tells you that.

**A sharper version of the same trap, and the one escape hatch inside `.with_runtime_config` that `.with_runtime_cores` cannot reach:** build a `RuntimeConfig` with an EXPLICIT `backend` (`RuntimeBackendSelection::Prime`/`Tokio`, not `Auto` — no sugar method sets this; you construct the `RuntimeConfig` yourself) and `resolve_selection()` returns `Some(RuntimeSelection)`, which *does* flow into `runtime_selection` as tier-1 "explicit" and *does* win over the ambient install. `.with_runtime_cores` can never trigger this path — it never touches `backend` — which is exactly why it is the one that reads as a fix and silently isn't.

### The fix, and why the order matters

`App::with_runtime` (`src/app.rs:413`) is different in kind from `with_runtime_cores`, not just in name: it runs **after** `.build()`, directly on the already-constructed `App`, and it unconditionally overwrites `self.runtime` — no ambient check, no fallback branch, no way for it to be silently skipped:

```rust
#[must_use]
pub fn with_runtime(mut self, runtime: Arc<dyn crate::runtime::Runtime>) -> Self {
    self.runtime = Some(runtime);
    self
}
```

That is the mechanism every one of this tutorial's five examples uses — paired with `App::with_acceptor_factory` (`src/app.rs:431`), exactly as walked in section 3. But read `with_runtime`'s own doc comment today (`src/app.rs:402–411`) and it flags itself, in the source, a **HAZARD**: it, `with_acceptor_factory`, and `with_datagram_factory` are three INDEPENDENT setters — nothing stops calling only one, or pairing a prime `runtime` with a tokio `acceptor_factory`, and an `App` built that way has its chain dispatch and its socket accept disagreeing about which backend is live. The promoted fix for new code — one this tutorial's examples predate — is the matched-bundle setter `App::with_runtime_selection(selection)` (`src/app.rs:473–481`), or, before `.build()`, `AppBuilder::runtime(selection)` (`app_builder.rs:324–328`). Both take one `RuntimeSelection` (section 4's `RuntimeSelection::prime(cores)`/`::tokio(cores)`, `src/runtime.rs:214–216,244–246`) and set every matched field atomically, so the runtime/acceptor/datagram/unix-upstream/packet-listener quintet can never disagree:

```rust,ignore
// verified compiling+running standalone (`use proxima::runtime::
// RuntimeSelection;` plus a `#[proxima::main]`-driven `async fn main() ->
// Result<(), ProximaError>`) against this repository's current source —
// this excerpt omits that wrapping for brevity, matching the style above.
let app = App::builder()
    .runtime(RuntimeSelection::prime(1)?)
    .with_defaults()?
    .build()?;
```

is the same override as this tutorial's `.build()?.with_runtime(Arc::new(PrimeRuntime::new(1)?)).with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))` idiom, minus the ability to mismatch the two pieces — verified compiling and running (`app.runtime().unwrap().num_cores() == 1`) against this repository's current source. This tutorial keeps teaching the older trio through section 8 because that is the real, unmodified code in every one of `proxy`/`gateway`/`load-balance`/`integration`/`distributed_trace` today — but write new code against `.runtime(selection)`.

Either surface teaches the same rule this section exists for: **inside `#[proxima::main(cores = N)]`, give every `App` that needs its own runtime an explicit override applied *after* `.build()` (`.with_runtime`/`.with_runtime_selection`) or *as part of* the builder chain (`AppBuilder::runtime`) — never `.with_runtime_cores(M)` alone before `.build()`, and never rely on the default.** `#[proxima::main]`'s adoption is the right behavior for the common case (one `App`, one process, one runtime, no ceremony) — every one of this tutorial's five examples is the *uncommon* case, deliberately.

## 5. `gateway`: policy composition is orthogonal to the runtime choice

`gateway/main.rs` builds **three** independent `App`s: the gateway itself, and two upstream origins (`api`, `web`), each behind the identical idiom (`gateway/main.rs:120–124`, and `spawn_origin`, `312–316`):

```rust,ignore
// excerpted from `gateway/main.rs`'s `main` — real, unmodified running code
// (verified live via `cargo run --example gateway` below), not a
// standalone program.
let gateway_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(Arc::new(PrimeRuntime::new(1)?))
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
```

Everything the gateway *does* — `Auth` (401 on a missing/wrong bearer token), `RoutingPipe` (path-prefix dispatch to one of the two upstreams), `RateLimit` (429 once a per-upstream token bucket is exhausted), each wrapping a `ForwardPipe` that is exactly `proxy`'s one-line forward — is ordinary pipe composition, already taught by [Build an API gateway](./build-an-api-gateway.md) and Foundations' filter/gate sections. Nothing about *that* composition changes because there are now three `App`s instead of one; that is the point of this section. The runtime idiom from section 4 does not know or care what pipes are mounted on the `App` it is attached to — a rejected request never even reaches routing (`gateway/main.rs`'s `run_scenarios`, verified live below), and the runtime wiring around it is identical whether the pipe chain behind it is one line (`proxy`) or four policies deep (`gateway`).

```
$ cargo run --example gateway --no-default-features --features "serve-prime,http1-native,macros"
...
rate-limit: a third call exceeds the budget (429), origin never hit
HTTP/1.1 429 Too Many Requests
retry-after: 1
...
PASS: auth rejects before route, route sends each prefix to its own upstream, rate-limit throttles per upstream before the forward — three composed policies, no bytes copied by hand.
gateway    drained: cores_acked=1 hooks_drained=0
origin api drained: cores_acked=1 hooks_drained=0
origin web drained: cores_acked=1 hooks_drained=0
```

Three `cores_acked=1` lines, one per `App`, each drained independently — three separate one-core `PrimeRuntime`s, proven by the fact that shutting down the gateway's runtime does not touch the two origins', and vice versa.

## 6. `load-balance`: four independent runtimes, one process

`load-balance/main.rs` scales the same idiom to **four** `App`s: three origin backends (`origin-a` healthy, `origin-b` deliberately marked unhealthy, `origin-c` healthy) plus the load balancer itself, each built by `spin_up_origin` (`load-balance/main.rs:202–206`) or `spin_up_load_balancer` (`254–255`) — the identical `.build()?.with_runtime(Arc::new(PrimeRuntime::new(1)?)).with_acceptor_factory(...)` call, four times over. `LoadBalancerPipe::select_backend` (`92–102`) round-robins over only the backends flagged healthy; each origin's own `Arc<AtomicU32>` hit counter (owned by that origin's `OriginPipe`, not shared with any other `App`) is the ground truth `assert_distribution` (`275–302`) checks against, not the load balancer's own bookkeeping.

```
$ cargo run --example load-balance --no-default-features --features "serve-prime,http1-native,macros"
...
per-backend counts: origin-a=6 origin-b=0 origin-c=6
PASS: distributed across healthy backends only, unhealthy backend saw zero requests.
load balancer drained: cores_acked=1 hooks_drained=0
origin-a drained: cores_acked=1 hooks_drained=0
origin-b drained: cores_acked=1 hooks_drained=0
origin-c drained: cores_acked=1 hooks_drained=0
```

Twelve real HTTP requests, routed across four independently-runtimed `App`s in one OS process, land exactly where the round-robin-over-healthy policy says they should — `origin-b`, unhealthy, sees zero of them, and `origin-a`/`origin-c` split the rest exactly in half. Nothing here is special-cased for "four" — it is section 4's one-`App`-one-`with_runtime`-call idiom, repeated as many times as you have `App`s. That is what "coexisting cleanly" means concretely: the pattern does not get more complicated as the process grows more listeners, because each `App`'s runtime is a value you hand it once, independent of how many siblings it has.

## 7. `integration`: a runtime you build, and a runtime you deliberately share

`integration/main.rs` runs two phases: **LIVE**, where a real edge fronts a (stand-in) third-party vendor and records every response to a cassette, and **REPLAY**, where the vendor `App` is fully drained — gone — and a second edge serves the exact same bytes straight off disk. Every `App` in both phases still gets the section-4 idiom: `origin_app` (the vendor, `72–76`), `edge_live_app` (`86–90`), and later `edge_fake_app` (`160–164`) each call `.build()?.with_runtime(Arc::new(PrimeRuntime::new(1)?)).with_acceptor_factory(...)` independently — three more instances of the same override, nothing new there.

What *is* new: a component that is not an `App` at all, but still needs a runtime to drive its own background work — `RecordUpstream`'s durable sink-drain. Rather than handing it a **fourth** independent runtime, the example deliberately *shares* `edge_live_app`'s own (`integration/main.rs:91–112`):

```rust,ignore
// excerpted from `integration/main.rs`'s `main` — real, unmodified running
// code (verified live via `cargo run --example integration` below).
let edge_runtime = edge_live_app.runtime().expect("builder installs a runtime");
let spigot = deferred_runtime();
spigot.set(Arc::clone(&edge_runtime)).ok();
// ...
let recorder = RecordUpstream::new("live-front", client, sink, "third-party").with_runtime(spigot);
```

`deferred_runtime()` (`proxima_recording::pipe::lazy::deferred_runtime`, re-exported at `src/lib.rs:419`) returns a `DeferredRuntime` — `Arc<OnceLock<Arc<dyn Runtime>>>` (`proxima-recording/src/pipe/lazy.rs:31,36`) — a runtime *cell* you can build a component around before the actual runtime exists, then fill in once (`spigot.set(...)`) with whichever `Arc<dyn Runtime>` you choose. `RecordUpstream::with_runtime` (`src/upstreams/record.rs:179`) accepts exactly that cell. The result: the recorder's background drain and `edge_live_app`'s own listener run on the *same* `PrimeRuntime` — one core doing both jobs, on purpose, spelled out explicitly at the call site.

Contrast the two mechanisms precisely, because they look similar and are not: section 4's bug is an *App* silently adopting a runtime it never asked for, through a global, ambient cell (`install_runtime`/`installed_runtime`) it has no reference to. This is a *component* being handed a specific, named runtime (`Arc::clone(&edge_runtime)`) through an explicit, local cell (`spigot`) its constructor takes as a parameter. Both are "sharing a runtime" — the difference is entirely whether the sharing is visible at the call site. One is a trap; the other is a design choice, spelled out in the same three lines that make it happen.

Phase 2 adds one more wrinkle: a runtime that is never attached to an `App` at all. `recorded_response_body` (`integration/main.rs:232–245`) reads the cassette back off disk to compute the ground truth `replay` is checked against, using `JsonlSource::new(path, runtime)` — a sans-IO component one level below `App`, needing only *a* `Runtime` to offload its blocking file read, built and thrown away in a few lines (`integration/main.rs:154–155`):

```rust,ignore
// excerpted from `integration/main.rs`'s `main` — `cassette_path` is
// defined earlier in that function, not repeated in this excerpt.
let cassette_runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1)?);
let recorded_body = recorded_response_body(&cassette_path, cassette_runtime).await?;
```

Runtimes compose at whatever level actually needs one — an `App`, a `RecordUpstream`, a bare `JsonlSource` — not only at `App::builder()` call sites.

```
$ cargo run --example integration --no-default-features --features "serve-prime,http1-native,macros"
...
vendor drained: cores_acked=1 hooks_drained=0 -- the vendor is now GONE

phase 2: REPLAY — serve the capture, no vendor required

cassette loaded, known match keys: ["GET /?"]
in-process proof: 32 bytes recorded == 32 bytes replayed, no vendor call made
...
PASS: acme-quotes-api was fronted live, recorded, and replayed byte-identical with the vendor removed.
```

The vendor's `App` — and its runtime — are fully torn down (`ShutdownBarrier::broadcast_drop`) before phase 2 even starts building its own. That the fake edge still answers, byte-identical, is only possible because its runtime was never entangled with the vendor's in the first place.

## 8. `distributed_trace`: trace context survives a real TCP hop, still zero tokio

`distributed_trace/main.rs` is the capstone: two `App`s, `front` (instance A) and `origin` (instance B), each on its own two-core `PrimeRuntime` (`distributed_trace/main.rs:180–191`):

```rust,ignore
// excerpted from `distributed_trace/main.rs`'s `main` — real, unmodified
// running code (verified live via `cargo run --example distributed_trace`
// below).
let origin_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(Arc::new(PrimeRuntime::new(2)?))
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
// ...
let front_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(Arc::new(PrimeRuntime::new(2)?))
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
```

— the identical section-4 idiom, sized to two cores each this time instead of one, purely because this example chose to; nothing about the mechanism changes with the count. The interesting question this example answers is not about runtimes at all: a real client hits `front` over a plain blocking `TcpStream`; `front` forwards to `origin` over a *second*, hand-rolled blocking TCP request (deliberately not `proxima::Client`, so the proof does not depend on a client stack); do the two instances' spans land in the same trace, or two disconnected ones?

```
$ cargo run --example distributed_trace --no-default-features --features "serve-prime,http1-native,macros"
...
W3C header layer (RequestContext.trace_id via inject_propagation/establish_trace_context):
  front  traceparent = 00-5a136770aded5d0538a2e18b23a8e5ef-ffca11584ca5a4c5-01
  origin traceparent = 00-5a136770aded5d0538a2e18b23a8e5ef-ffca11584ca5a4c5-01
  -> CONNECTED: same trace_id crossed the A -> B hop
...
PASS: distributed tracing across two proxima instances lands in ONE trace.
      Both layers agree: the header layer via inject_propagation/establish_trace_context,
      the span layer via #[instrument(parent = request.context.traceparent())] routing
      to `Recorder::span_from_traceparent` instead of a fresh root.
      The literal parent_span_id chain crosses the wire hop too: establish_trace_context
      preserves the inbound span-id instead of discarding it.
```

(**Update since this document's first pass:** the example now also validates a second and third layer beyond the W3C header — the `#[proxima::telemetry::instrument(parent = ...)]` span on each pipe, and the literal `parent_span_id` byte chain — elided above with `...` for the same reason the header layer's own leading transcript is; the headline claim this section teaches, one trace crossing a real TCP hop, is unchanged and still the first thing printed under "-> CONNECTED".)

The runtime story here is entirely mundane — two `App`s, two independent `.with_runtime(...)` calls, exactly as taught in section 4 — and that is the point of placing it last: by now the pattern is load-bearing enough to disappear into the background of a much more interesting proof (W3C trace propagation across a real network hop), instead of being the thing under test. And it is still true: `cargo tree --no-default-features --features "serve-prime,http1-native,macros" -e normal -i tokio` prints nothing for this example's own required-features either (`Cargo.toml:1538`) — two real proxima server instances, a real TCP hop between them, still zero tokio.

## 9. When tokio *is* the right answer: `multi_runtime` and `runtime_select`

None of this makes tokio forbidden — it makes it **opt-in**. Two sibling examples exist specifically to prove `Runtime` is a genuinely open trait, and both reach for tokio on purpose, as the second, contrasting implementation:

`runtime_select/main.rs` serves the *identical* pipe twice, sequentially — once on prime, once on tokio (`runtime_select/main.rs:54–61`):

```
$ cargo run --example runtime_select --features "runtime-tokio,tokio,http1"
--- pass 1: the SAME pipe served on prime ---
...
HTTP/1.1 200 OK
...
hello from whichever runtime is listening
prime drained: cores_acked=1 hooks_drained=0

--- pass 2: the SAME pipe served on tokio ---
listening on 127.0.0.1:8084 (tokio runtime, 1 core)
...
tokio drained: cores_acked=1 hooks_drained=0

same Pipe, two runtimes, identical response both times.
```

`multi_runtime/main.rs` goes further — prime and tokio serving **concurrently**, in the same process, dispatching into the *same* `Arc<AtomicU64>`-backed pipe from two independently-scheduled runtimes at once (`multi_runtime/main.rs:76–91`):

```rust,ignore
// excerpted from `multi_runtime/main.rs`'s `main` — real, unmodified
// running code (verified live via `cargo run --example multi_runtime`
// below).
let prime_runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(2)?);
let prime_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(prime_runtime.clone())
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
// ...
let tokio_runtime: Arc<dyn Runtime> = Arc::new(TokioPerCoreRuntime::new(2)?);
let tokio_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(tokio_runtime.clone())
    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
```

```
$ cargo run --example multi_runtime --features "runtime-tokio,tokio,http1"
prime listener on 127.0.0.1:8081 (prime runtime, 2 cores)
tokio listener on 127.0.0.1:8082 (tokio runtime, 2 cores)
GET http://127.0.0.1:8081/ (prime) -> shared_total=2
GET http://127.0.0.1:8081/ (prime) -> shared_total=4
GET http://127.0.0.1:8081/ (prime) -> shared_total=6
GET http://127.0.0.1:8082/ (tokio) -> shared_total=1
GET http://127.0.0.1:8082/ (tokio) -> shared_total=3
GET http://127.0.0.1:8082/ (tokio) -> shared_total=5
observed totals across both listeners (sorted): [1, 2, 3, 4, 5, 6]
prime drained: cores_acked=2 hooks_drained=0
tokio drained: cores_acked=2 hooks_drained=0
both runtimes shut down cleanly; final shared total = 6
```

**Update since this document's first pass:** the client side is now a genuine race, not a round-robin — two OS threads (`std::thread::spawn`, `multi_runtime/main.rs:111–119`) each fire `REQUESTS_PER_LISTENER` blocking GETs at their own listener concurrently, so the two listed sequences (`shared_total=2,4,6` on prime; `1,3,5` on tokio) show each listener's own requests landing on non-contiguous slots of the one shared counter, interleaved with the other listener's — proof neither runtime serializes behind the other. `observed totals across both listeners (sorted): [1, 2, 3, 4, 5, 6]` is the assertion this teaches: every counter value from 1 to `REQUESTS_PER_LISTENER * 2` appears exactly once, no lost update and no double count, regardless of interleaving order (a fresh run reorders which listener gets the even vs. odd slots).

tokio and glommio and monoio are process-singletons by convention — one runtime per process is the norm everywhere else. `Runtime` here is just an interface (section 1); `multi_runtime` is the smallest proof that can't be faked that any number of implementations coexist in one process, side by side, sharing state safely across the boundary. Note the acceptor factory changes to match, exactly as section 2 said it must: `PrimeAcceptorFactory` for the prime-backed app, `proxima_net::tokio::TokioAcceptorFactory` for the tokio-backed one — the runtime and the socket-opener are still always a matched pair, even when there are two of each in the same process.

**Update since this document's first pass — a gap this document itself flagged, now closed:** at the time this document was first written, neither example's `Cargo.toml` `required-features` (`multi_runtime`, `Cargo.toml:1495`; `runtime_select`, `:1519`) listed `http1-native` or `http1`, so the bare commands they printed (`cargo run --example multi_runtime --features "runtime-tokio tokio"`) failed with `Registry("no listen protocol named 'http'")` — no h1 listener is registered without one of those two features. Both entries have since been fixed to require `http1` directly (`required-features = [..., "http1"]`), so `cargo build --example multi_runtime --features "runtime-tokio,tokio,http1-native"` (the workaround this document originally suggested) now fails a DIFFERENT way — `error: target 'multi_runtime' ... requires the features: ... 'http1'` — because `http1-native` alone no longer satisfies the declared requirement; `http1` does (and pulls in `http1-native` underneath it, per §2). The commands shown above (`http1`, not `http1-native`) are the current verified-working form for both examples.

And to be precise about what "opt-in" means at the dependency level — `runtime-tokio,tokio` genuinely does pull tokio into the graph, unlike everything in sections 3–8:

```
$ cargo tree --features "runtime-tokio,tokio" -e normal -i tokio
tokio v1.53.1
├── h2 v0.4.15
│   └── proxima v0.1.0 (...)
├── prime v0.1.0 (...)
...
└── tokio-util v0.7.19
    ├── h2 v0.4.15 (*)
    ├── proxima v0.1.0 (...)
    └── proxima-net v0.1.0 (...)
```

## 10. Where to go next

You now know the runtime dimension of proxima's HTTP surface: the `Runtime` trait itself (an interface, not a singleton); `http1-native` as the tokio-free base `http1` builds on, with the `AcceptorFactory` trait as the one backend-specific piece below it; `#[proxima::main]`'s ambient-runtime publication and the one rule it demands — every `App` that needs its own runtime calls `.build()?.with_runtime(...).with_acceptor_factory(...)` (or, in new code, the mismatch-proof `.runtime(RuntimeSelection::prime(N)?)`), never `.with_runtime_cores(...)` alone before `.build()`; and the difference between an accidental collapse (ambient adoption) and a deliberate share (`deferred_runtime`/`DeferredRuntime`).

- [Build an API gateway](./build-an-api-gateway.md), [Build a load balancer](./build-a-load-balancer.md), and [Build a record/replay harness](./build-a-record-replay-harness.md) teach the *pipe* side of `gateway`, `load-balance`, and `integration` respectively — `Auth`/`RoutingPipe`/`RateLimit`, backend selection, and the record/replay chain — in depth. This tutorial deliberately did not re-teach that; read those for the composition this document treated as a black box. **Drift this document originally flagged for a follow-up, now fixed as part of landing it:** [Build an API gateway](./build-an-api-gateway.md)'s own code citation used to show the pre-migration `.with_runtime_cores(1)` / `#[proxima::main(runtime = "tokio")]` idiom against `gateway/main.rs:114–123`; it now cites the current `.with_runtime(...).with_acceptor_factory(...)` idiom against `gateway/main.rs:120–130`, matching §4 of this document. `00-foundations.md`'s §7 citation of `examples/proxy/main.rs` (the struct sits at `53–63`, not `51–61`) is also corrected. `00-foundations.md:778` and `FEATURES.md`'s feature-flags section, which both stated "h1 has no sans-IO driver yet," now say what section 2 above teaches: `http1-native`'s `serve_connection`/`serve_h1_connection` is exactly that driver, and `http1` layers the legacy tokio stack on top of it.
- `examples/README.md` and `ai_docs/examples-index.jsonl` are the agent-facing map of every combinator to its module; this document is the human-facing narrative for one axis of it (the runtime, not the algebra).
- [Build a multi-runtime service](./build-a-multi-runtime-service.md) is the project-tutorial companion to section 9 above — it walks `multi_runtime`'s `Runtime` trait story as a standalone build, if you want the deeper version of "prime and tokio, concurrently, one process."
