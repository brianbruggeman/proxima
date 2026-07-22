# The native runtime: serving real HTTP with zero tokio

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md), sections 1–7 and 13. You should already know: what `Pipe`/`SendPipe` are; the free-function and stateful-`impl` forms of `#[proxima::piped]`; and that `App::mount` attaches a handler at a path, `into_handle` holds any `Handler` behind one uniform `PipeHandle`, and *something* — Foundations calls it only "the engine that actually drives async work" — sits underneath `App` and actually runs the futures a pipe's `call` returns. This document names that thing and shows you how to control it.

**You will learn:** that proxima serves real HTTP with **zero tokio anywhere in the build** — not "tokio hidden behind a feature flag," but genuinely absent from the dependency graph, provably so with one `cargo tree` command — and that this is the *default*, not a stripped-down alternative. You will also learn the one non-obvious rule that trips up every multi-`App` program: booting one runtime for `main` silently becomes booting one runtime for *every* `App` you build inside it, unless you explicitly opt out.

**New concepts (in order):** the `Runtime` trait · `http1` vs. `http1-native` (tokio-coupled vs. tokio-free h1) · `#[proxima::main(cores = N)]`'s ambient-runtime publication · `App::with_runtime` / `App::with_acceptor_factory` · the `AcceptorFactory` trait · `ShutdownBarrier` · `deferred_runtime`/`DeferredRuntime` (a runtime handed to a component on purpose, not adopted by accident).

Every code block below is copied verbatim from a real file in this repository, cited by `file:line`, or is a command this tutorial's author actually ran — every transcript shown is real output, captured the day this document was written. Where the current repository state disagrees with a claim in an older document, this tutorial says so explicitly rather than repeating it. (This document was originally checked against commit `238229cd`; it has since been re-verified against `0ac7a565`, one commit later, which touched `Cargo.toml`, `src/app.rs`, `src/app_builder.rs`, and `src/runtime.rs` — the h2/h3-native/pgwire listener land added a third, optional `datagram_factory` alongside every `Runtime`/`AcceptorFactory` pair this document teaches. Sections 4 and the citations below reflect that; none of the five migrated examples this document walks through were touched by that commit.)

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

Foundations §13 built `hello` on `App::new()` and never named what actually executes the `Future` a `Pipe::call` returns. Here is the answer, copied (doc comments trimmed) from `proxima-runtime/src/lib.rs:274`:

```rust
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

This tutorial's first five examples (sections 3, 5–8) use exactly one implementation of this trait, `PrimeRuntime` (`proxima::prime::PrimeRuntime`, re-exported at `src/lib.rs:184` from `prime::os::runtime::PrimeRuntime`, defined at `prime/src/os/runtime.rs:51`) — prime's own per-core executor, no tokio underneath it anywhere. Section 9 introduces a second implementation, `TokioPerCoreRuntime`, to show the trait is genuinely open — but you do not need it before then.

## 2. Two h1 features, one listener: `http1` vs. `http1-native`

Before wiring a runtime to a listener, one more piece of vocabulary: the HTTP/1.1 listener itself comes in two feature-gated flavors, and which one you link determines whether tokio enters the build at all. From `proxima-http/Cargo.toml:90–103`:

```toml
# `http1-native` is the tokio-free base: the sans-IO codec
# (proxima-protocols::http1_codec) + the futures-io connection driver
# (`http1::serve` — `serve_connection`/`serve_h1_connection`), mirroring
# `http2-native`. `http1` layers the legacy hyper/tokio client stack...
http1-native = [
    "proxima-protocols/http1_codec",
]
```

and the umbrella listener feature that turns this codec into something `App` can actually bind a socket to, `proxima-http/Cargo.toml:191–195`:

```toml
http-listener = [
    "http1-native",
    "proxima-core/io-async-compat",
    "proxima-protocols/proxy_protocol-std",
]
```

The umbrella `proxima` crate (this repository's top-level package, what `examples/*/main.rs` depend on) re-exposes that pairing as its own `http1-native` feature, `Cargo.toml:465–466`:

```toml
http1-native = ["proxima-http/http-listener"]
http1 = ["tokio", "http1-native", "proxima-http/http1"]
```

Read those two lines carefully: `http1-native` pulls in the sans-IO codec — "sans-IO" meaning the HTTP/1.1 framing/parsing logic itself never touches a socket or an async runtime; it only turns bytes into requests/responses and back, so it can be driven by *any* I/O source, prime's or tokio's or a plain in-memory buffer in a test — and the `AcceptorFactory`-driven accept path (`listener::serve_via_factory`) — no `dep:tokio` anywhere in that chain. `http1` is `http1-native` *plus* `dep:tokio` plus hyper's legacy tokio-coupled client/accept-loop machinery, kept around for callers who have not migrated. **`http1-native` is not a stripped-down `http1`** — it is the tokio-free base that `http1` is built on top of, not the other way around.

One more precision worth having exact, since it corrects a claim in an older tutorial: `http1-native`'s connection driver (`serve_connection`/`serve_h1_connection`, `proxima-http/src/http1/serve.rs:70,126`) is generic over `Stream: futures::io::AsyncRead + futures::io::AsyncWrite + Unpin + Send` (imported at `proxima-http/src/http1/serve.rs:27` — the `futures` crate's I/O traits, not tokio's). The h1 *protocol driver* has always been sans-IO once `http1-native` exists; what still varies by backend is one layer below it — who opens the listening socket and hands back an accepted connection that implements those traits. That is the `AcceptorFactory` trait, `proxima-primitives/src/stream/mod.rs:174–176`:

```rust
pub trait AcceptorFactory: Send + Sync + 'static {
    fn bind(&self, addr: SocketAddr, options: TcpBindOptions) -> io::Result<Box<dyn TcpAcceptor>>;
}
```

`proxima_net::prime::PrimeAcceptorFactory` (`proxima-net/src/prime/mod.rs:115`) binds the socket through prime's own reactor and hands back a connection (`PrimeTcpConnection`, implementing `futures::io::{AsyncRead, AsyncWrite}` at `proxima-net/src/prime/mod.rs:78,88`) that the tokio-free h1 driver can drive directly. A `TokioAcceptorFactory` exists too (`proxima_net::tokio::TokioAcceptorFactory`, used in section 9) — same trait, tokio underneath instead. **Runtime and acceptor factory are always a matched pair**: the runtime is what runs the connection's future; the acceptor factory is what produced the socket that future reads and writes. Mismatching them (a tokio socket handed to a task spawned on a prime worker with no tokio reactor registered) is exactly the kind of bug `App::with_runtime`/`App::with_acceptor_factory` (section 4) are designed to be called together to prevent.

Finally, the umbrella `Cargo.toml`'s own default-feature list states the header claim plainly (comments elided for the middle section — the full block runs `Cargo.toml:403–425`):

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

Its `Cargo.toml` entry, `Cargo.toml:1514–1516`:

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

```rust
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

- **`App::builder()`** (`src/app.rs:712`) returns an `AppBuilder` (`src/app_builder.rs:57`) — the mutable, fluent construction surface. `.with_defaults()` (`app_builder.rs:105`) registers the built-in listen protocols, upstream factories, and codecs; `.build()` (`app_builder.rs:281`) consumes the builder and returns a plain `Result<App, ProximaError>`.
- **`.with_runtime(Arc::new(PrimeRuntime::new(1)?))`** (`App::with_runtime`, `src/app.rs:305`) replaces the `App`'s runtime with a freshly built, one-core `PrimeRuntime` (`PrimeRuntime::new`, `prime/src/os/runtime.rs:51`). Note it is called *after* `.build()`, on the already-constructed `App` — section 4 explains exactly why that ordering matters.
- **`.with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))`** (`App::with_acceptor_factory`, `src/app.rs:321`) pairs that runtime with the matching prime-backed socket opener from section 2. Both setters are `#[must_use] fn(self) -> Self` — plain builder methods, not `Result`, so no `?` after either.
- **`origin_app.build_listener(ListenerSpec::http(origin_bind))`** (`App::build_listener`, `src/app.rs:902`; `ListenerSpec::http`, `proxima-listen/src/handle.rs:81`) binds and starts accepting *before returning* — it blocks the calling thread only until the accept lane has acked ready, never polling or sleeping to find out.

At the end, instead of `Foundations`'s `server.run_until_signal()` (which blocks forever waiting for `SIGINT`/`SIGTERM` — right for a long-running server, wrong for a demo/test process that needs to prove something and then exit), `proxy` drains deterministically with `ShutdownBarrier` (`proxima_primitives::sync::shutdown::ShutdownBarrier`, re-exported as `proxima::shutdown::ShutdownBarrier`, `src/lib.rs:113`):

```rust
let proxy_report = ShutdownBarrier::new(proxy_runtime).broadcast_drop().await;
println!(
    "proxy  drained: cores_acked={} hooks_drained={}",
    proxy_report.cores_acked, proxy_report.hooks_drained
);
```

`ShutdownBarrier::new(runtime)` (`proxima-primitives/src/sync/shutdown.rs:151`) and `.broadcast_drop()` (same file, returning a `ShutdownReport { cores_acked, hooks_drained }` at line 213–217) broadcast a stop signal to every worker on *that one runtime* and wait for every core to acknowledge — a report you print and assert on, not a signal you have to send yourself from another shell. Two `App`s, two independent runtimes, two independent drains: nothing here waits on the other.

**A verified gap, flagged rather than hidden:** the *literal* `required-features` list quoted above (without `serve-prime`) currently fails to build standalone — `cargo build --example proxy --no-default-features --features "runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http-prime-deps,http1-native,macros"` errors compiling `proxima-test` (16 `deny(warnings)` dead-code errors, e.g. `function 'report_from' is never used`, `proxima-test/src/lib.rs:328`) because `proxima-test` is an unconditional dependency of the `proxima` crate (`Cargo.toml:226`) whose test-driving code is only reachable once its own `test-prime` feature is on, and only `serve-prime` (not the four bare `runtime-prime-*` features) forwards that (`Cargo.toml:747`, `"proxima-test/test-prime"`). This is pre-existing — every one of the five migrated examples' `Cargo.toml` entries has the same gap — not something this tutorial introduces. The commands shown above (`serve-prime` instead of the four sub-features) are the verified-working substitute; `cargo build/run --example proxy --features http1-native` (default features plus `http1-native`, no `--no-default-features`) also builds and runs cleanly, since `default` already includes `serve-prime`.

## 4. The ambient-runtime seam — the centerpiece

Every one of the five migrated examples repeats the same four-line idiom — `.build()?.with_runtime(Arc::new(PrimeRuntime::new(N)?)).with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))` — on *every* `App` it builds, even when there is only one `App` in the whole program. That repetition is not boilerplate for its own sake. It exists to opt out of something `#[proxima::main]` does automatically, and skipping it produces a real, silent bug. This section is that bug, and the mechanism behind it, in full.

### What `#[proxima::main(cores = N)]` actually does

`#[proxima::main]` (`proxima-macros/src/lib.rs:87`) turns `async fn main() -> R` into a synchronous `fn main() -> R` that boots a runtime and drives your body to completion on it. Its own module doc states the mechanism plainly (`proxima-macros/src/main_attr.rs:28–31`):

> The booted runtime is published via `proxima::runtime::install_runtime` so `App::new()` called from `main`'s body adopts it instead of building an independent second one — one `#[proxima::main(cores = N)]` now means one N-core runtime, not two runtimes with contradictory core counts.

`install_runtime` (`src/runtime.rs:79–89`) and its reader `installed_runtime` (`src/runtime.rs:95–98`) are a process-wide, set-once cell. **Update since this document's first pass:** both `InstalledRuntime` and `install_runtime` grew a third field/parameter, `datagram_factory` — the h2/h3-native/pgwire listener land's UDP counterpart to `acceptor_factory`, `None` for backends with no matching `DatagramFactory` impl (tokio, today). It composes and adopts exactly like `acceptor_factory` does; this tutorial's five examples are all TCP/h1, so it stays `None` throughout and never comes up again below:

```rust
static INSTALLED_RUNTIME: OnceLock<InstalledRuntime> = OnceLock::new();

pub struct InstalledRuntime {
    pub runtime: Arc<dyn Runtime>,
    pub acceptor_factory: Arc<dyn AcceptorFactory>,
    pub datagram_factory: Option<Arc<dyn DatagramFactory>>,
}

pub fn install_runtime(
    runtime: Arc<dyn Runtime>,
    acceptor_factory: Arc<dyn AcceptorFactory>,
    datagram_factory: Option<Arc<dyn DatagramFactory>>,
) {
    let _ = INSTALLED_RUNTIME.set(InstalledRuntime {
        runtime,
        acceptor_factory,
        datagram_factory,
    });
}

pub fn installed_runtime() -> Option<InstalledRuntime> {
    INSTALLED_RUNTIME.get().cloned()
}
```
(`src/runtime.rs:60–98`)

`#[proxima::main(cores = 1)]` calls this once, at startup, with the one-core `PrimeRuntime` it just booted. And here is the seam that matters: `App::builder()...build()`'s internals check this cell **first**, before considering anything else — `default_runtime`, `src/app.rs:112–127` (also grown the matching third tuple element since the first pass):

```rust
fn default_runtime(cores_override: Option<usize>) -> Result<RuntimeAndFactory, ProximaError> {
    // `#[proxima::main]` (or any other `block_on*` driver) may have already
    // booted a runtime sized by its own `runtime = ...` / `cores = ...` /
    // `affinity = ...` args and published it — adopt that instead of
    // building an independent second one. ...
    if let Some(installed) = crate::runtime::installed_runtime() {
        return Ok((
            Some(installed.runtime),
            Some(installed.acceptor_factory),
            installed.datagram_factory,
        ));
    }
    // ...only reached if nothing is installed yet
}
```

Put those two together: **every `App::builder()...build()` call inside a `#[proxima::main(cores = N)]`-driven `main`, with no override, adopts the exact same `Arc<dyn Runtime>` — the same one, not an equivalent one.** Two `App`s built this way do not get "two 1-core runtimes"; they get one 1-core runtime, shared.

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

`Arc::ptr_eq` compares pointer identity, not just equal values — `true` here means `app_one` and `app_two` are not "two 1-core runtimes that happen to agree," they are the *literal same* runtime object. Neither `App` called `.with_runtime(...)`, so both fell through to `default_runtime`'s ambient-adoption branch and got back the identical `Arc` `#[proxima::main]` installed. Scale this to `gateway`'s three `App`s or `load-balance`'s four (sections 5–6): without the override, all of them would collapse onto the one runtime `#[proxima::main(cores = 1)]` booted — one shared, one-core executor serving every listener in the whole program, not the N independent ones each example's own doc comment says it wants. Every one of the five migrated examples' `main` functions carries this exact explanation inline, e.g. `proxy/main.rs:65–73`:

> `#[proxima::main(cores = 1)]` boots a throwaway 1-core prime runtime just to give `main` an async context to `.await` on (no tokio anywhere in the build...). That boot publishes an AMBIENT runtime (`crate::runtime::install_runtime`), which `App::builder().build()` would otherwise silently adopt — collapsing the two apps below onto ONE shared runtime instead of each having its own. Each app opts back OUT of that adoption with an explicit `.with_runtime(...)` + `.with_acceptor_factory(...)`.

### The near-miss: `AppBuilder::with_runtime_cores`

There is a method that *looks* like the right tool and is not, once you are inside `#[proxima::main]`: `AppBuilder::with_runtime_cores(usize)` (`src/app_builder.rs:276–279`):

```rust
/// Sugar for `.with_runtime_config(RuntimeConfig::builder().cores(cores).build())`.
#[must_use]
pub fn with_runtime_cores(self, cores: usize) -> Self {
    self.with_runtime_config(crate::app_config::RuntimeConfig::builder().cores(cores).build())
}
```

It is real, public API, called *before* `.build()` — which reads as "size this App's runtime to `cores` cores." Follow it through: `AppBuilder::build()` (`app_builder.rs:281`) resolves `cores_override` from `self.runtime_config` and hands it to `App::with_components` (`app_builder.rs:330–336`), a thin public wrapper (`app_builder.rs:344–352`) around the same `App::__internal_assemble` — which calls exactly the same `default_runtime(cores_override)` shown above (`src/app.rs:668`). And `default_runtime`'s **first** line, again, is the ambient-adoption check — `cores_override` is only consulted in the branches reached *after* that check finds nothing installed. Inside any `#[proxima::main]`-driven binary, something is always installed, so `with_runtime_cores`'s value is silently never read. It is not broken — it does exactly what its doc comment says, sizing a *fallback* — it is just the wrong tool for "give this `App` its own runtime" once an ambient one already exists, and nothing about the call site tells you that.

### The fix, and why the order matters

`App::with_runtime` (`src/app.rs:305`) is different in kind, not just in name: it runs **after** `.build()`, directly on the already-constructed `App`, and it unconditionally overwrites `self.runtime` — no ambient check, no fallback branch, no way for it to be silently skipped:

```rust
#[must_use]
pub fn with_runtime(mut self, runtime: Arc<dyn crate::runtime::Runtime>) -> Self {
    self.runtime = Some(runtime);
    self
}
```

That is the entire rule this section exists to teach: **inside `#[proxima::main(cores = N)]`, give every `App` that needs its own runtime an explicit `.build()?.with_runtime(Arc::new(PrimeRuntime::new(M)?)).with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))` — never `.with_runtime_cores(M)` before `.build()`, and never rely on the default.** `#[proxima::main]`'s adoption is the right behavior for the common case (one `App`, one process, one runtime, no ceremony) — every one of this tutorial's five examples is the *uncommon* case, deliberately.

## 5. `gateway`: policy composition is orthogonal to the runtime choice

`gateway/main.rs` builds **three** independent `App`s: the gateway itself, and two upstream origins (`api`, `web`), each behind the identical idiom (`gateway/main.rs:120–124`, and `spawn_origin`, `309–313`):

```rust
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

`load-balance/main.rs` scales the same idiom to **four** `App`s: three origin backends (`origin-a` healthy, `origin-b` deliberately marked unhealthy, `origin-c` healthy) plus the load balancer itself, each built by `spin_up_origin` (`load-balance/main.rs:204–208`) or `spin_up_load_balancer` (`256–257`) — the identical `.build()?.with_runtime(Arc::new(PrimeRuntime::new(1)?)).with_acceptor_factory(...)` call, four times over. `LoadBalancerPipe::select_backend` (`77–104`) round-robins over only the backends flagged healthy; each origin's own `Arc<AtomicU32>` hit counter (owned by that origin's `OriginPipe`, not shared with any other `App`) is the ground truth `assert_distribution` (`277–304`) checks against, not the load balancer's own bookkeeping.

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

`integration/main.rs` runs two phases: **LIVE**, where a real edge fronts a (stand-in) third-party vendor and records every response to a cassette, and **REPLAY**, where the vendor `App` is fully drained — gone — and a second edge serves the exact same bytes straight off disk. Every `App` in both phases still gets the section-4 idiom: `origin_app` (the vendor, `73–77`), `edge_live_app` (`87–91`), and later `edge_fake_app` (`161–165`) each call `.build()?.with_runtime(Arc::new(PrimeRuntime::new(1)?)).with_acceptor_factory(...)` independently — three more instances of the same override, nothing new there.

What *is* new: a component that is not an `App` at all, but still needs a runtime to drive its own background work — `RecordUpstream`'s durable sink-drain. Rather than handing it a **fourth** independent runtime, the example deliberately *shares* `edge_live_app`'s own (`integration/main.rs:92–113`):

```rust
let edge_runtime = edge_live_app.runtime().expect("builder installs a runtime");
let spigot = deferred_runtime();
spigot.set(Arc::clone(&edge_runtime)).ok();
// ...
let recorder = RecordUpstream::new("live-front", client, sink, "third-party").with_runtime(spigot);
```

`deferred_runtime()` (`proxima_recording::pipe::lazy::deferred_runtime`, re-exported at `src/lib.rs:334`) returns a `DeferredRuntime` — `Arc<OnceLock<Arc<dyn Runtime>>>` (`proxima-recording/src/pipe/lazy.rs:31,36`) — a runtime *cell* you can build a component around before the actual runtime exists, then fill in once (`spigot.set(...)`) with whichever `Arc<dyn Runtime>` you choose. `RecordUpstream::with_runtime` (`src/upstreams/record.rs:94`) accepts exactly that cell. The result: the recorder's background drain and `edge_live_app`'s own listener run on the *same* `PrimeRuntime` — one core doing both jobs, on purpose, spelled out explicitly at the call site.

Contrast the two mechanisms precisely, because they look similar and are not: section 4's bug is an *App* silently adopting a runtime it never asked for, through a global, ambient cell (`install_runtime`/`installed_runtime`) it has no reference to. This is a *component* being handed a specific, named runtime (`Arc::clone(&edge_runtime)`) through an explicit, local cell (`spigot`) its constructor takes as a parameter. Both are "sharing a runtime" — the difference is entirely whether the sharing is visible at the call site. One is a trap; the other is a design choice, spelled out in the same three lines that make it happen.

Phase 2 adds one more wrinkle: a runtime that is never attached to an `App` at all. `recorded_response_body` (`integration/main.rs:231–244`) reads the cassette back off disk to compute the ground truth `replay` is checked against, using `JsonlSource::new(path, runtime)` — a sans-IO component one level below `App`, needing only *a* `Runtime` to offload its blocking file read, built and thrown away in a few lines (`integration/main.rs:155–156`):

```rust
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

`distributed_trace/main.rs` is the capstone: two `App`s, `front` (instance A) and `origin` (instance B), each on its own two-core `PrimeRuntime` (`distributed_trace/main.rs:182–193`):

```rust
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
  front  traceparent = 00-c7ea59b8f54722e4174a6404654d9b51-a15859da4451158f-01
  origin traceparent = 00-c7ea59b8f54722e4174a6404654d9b51-a15859da4451158f-01
  -> CONNECTED: same trace_id crossed the A -> B hop
...
PASS: distributed tracing across two proxima instances lands in ONE trace.
```

The runtime story here is entirely mundane — two `App`s, two independent `.with_runtime(...)` calls, exactly as taught in section 4 — and that is the point of placing it last: by now the pattern is load-bearing enough to disappear into the background of a much more interesting proof (W3C trace propagation across a real network hop), instead of being the thing under test. And it is still true: `cargo tree --no-default-features --features "serve-prime,http1-native,macros" -e normal -i tokio` prints nothing for this example's own required-features either (`Cargo.toml:1326`) — two real proxima server instances, a real TCP hop between them, still zero tokio.

## 9. When tokio *is* the right answer: `multi_runtime` and `runtime_select`

None of this makes tokio forbidden — it makes it **opt-in**. Two sibling examples exist specifically to prove `Runtime` is a genuinely open trait, and both reach for tokio on purpose, as the second, contrasting implementation:

`runtime_select/main.rs` serves the *identical* pipe twice, sequentially — once on prime, once on tokio (`runtime_select/main.rs:53–61`):

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

`multi_runtime/main.rs` goes further — prime and tokio serving **concurrently**, in the same process, dispatching into the *same* `Arc<AtomicU64>`-backed pipe from two independently-scheduled runtimes at once (`multi_runtime/main.rs:77–92`):

```rust
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
GET http://127.0.0.1:8081/ (prime) -> shared_total=1
GET http://127.0.0.1:8082/ (tokio) -> shared_total=2
...
both runtimes shut down cleanly; final shared total = 6
```

tokio and glommio and monoio are process-singletons by convention — one runtime per process is the norm everywhere else. `Runtime` here is just an interface (section 1); `multi_runtime` is the smallest proof that can't be faked that any number of implementations coexist in one process, side by side, sharing state safely across the boundary. Note the acceptor factory changes to match, exactly as section 2 said it must: `PrimeAcceptorFactory` for the prime-backed app, `proxima_net::tokio::TokioAcceptorFactory` for the tokio-backed one — the runtime and the socket-opener are still always a matched pair, even when there are two of each in the same process.

**Update since this document's first pass — a gap this document itself flagged, now closed:** at the time this document was first written, neither example's `Cargo.toml` `required-features` (`multi_runtime`, `Cargo.toml:1283`; `runtime_select`, `:1307`) listed `http1-native` or `http1`, so the bare commands they printed (`cargo run --example multi_runtime --features "runtime-tokio tokio"`) failed with `Registry("no listen protocol named 'http'")` — no h1 listener is registered without one of those two features. Both entries have since been fixed to require `http1` directly (`required-features = [..., "http1"]`), so `cargo build --example multi_runtime --features "runtime-tokio,tokio,http1-native"` (the workaround this document originally suggested) now fails a DIFFERENT way — `error: target 'multi_runtime' ... requires the features: ... 'http1'` — because `http1-native` alone no longer satisfies the declared requirement; `http1` does (and pulls in `http1-native` underneath it, per §2). The commands shown above (`http1`, not `http1-native`) are the current verified-working form for both examples.

And to be precise about what "opt-in" means at the dependency level — `runtime-tokio,tokio` genuinely does pull tokio into the graph, unlike everything in sections 3–8:

```
$ cargo tree --features "runtime-tokio,tokio" -e normal -i tokio
...
└── tokio-util v0.7.18
    ├── h2 v0.4.15 (*)
    ├── proxima v0.1.0 (...)
    └── proxima-net v0.1.0 (...)
```

## 10. Where to go next

You now know the runtime dimension of proxima's HTTP surface: the `Runtime` trait itself (an interface, not a singleton); `http1-native` as the tokio-free base `http1` builds on, with the `AcceptorFactory` trait as the one backend-specific piece below it; `#[proxima::main]`'s ambient-runtime publication and the one rule it demands — every `App` that needs its own runtime calls `.build()?.with_runtime(...).with_acceptor_factory(...)`, never `.with_runtime_cores(...)` before `.build()`; and the difference between an accidental collapse (ambient adoption) and a deliberate share (`deferred_runtime`/`DeferredRuntime`).

- [Build an API gateway](./build-an-api-gateway.md), [Build a load balancer](./build-a-load-balancer.md), and [Build a record/replay harness](./build-a-record-replay-harness.md) teach the *pipe* side of `gateway`, `load-balance`, and `integration` respectively — `Auth`/`RoutingPipe`/`RateLimit`, backend selection, and the record/replay chain — in depth. This tutorial deliberately did not re-teach that; read those for the composition this document treated as a black box. **Drift this document originally flagged for a follow-up, now fixed as part of landing it:** [Build an API gateway](./build-an-api-gateway.md)'s own code citation used to show the pre-migration `.with_runtime_cores(1)` / `#[proxima::main(runtime = "tokio")]` idiom against `gateway/main.rs:114–123`; it now cites the current `.with_runtime(...).with_acceptor_factory(...)` idiom against `gateway/main.rs:120–130`, matching §4 of this document. `00-foundations.md`'s §7 citation of `examples/proxy/main.rs` (the struct sits at `53–63`, not `51–61`) is also corrected. `00-foundations.md:778` and `FEATURES.md`'s feature-flags section, which both stated "h1 has no sans-IO driver yet," now say what section 2 above teaches: `http1-native`'s `serve_connection`/`serve_h1_connection` is exactly that driver, and `http1` layers the legacy tokio stack on top of it.
- `examples/README.md` and `ai_docs/examples-index.jsonl` are the agent-facing map of every combinator to its module; this document is the human-facing narrative for one axis of it (the runtime, not the algebra).
- [Build a multi-runtime service](./build-a-multi-runtime-service.md) is the project-tutorial companion to section 9 above — it walks `multi_runtime`'s `Runtime` trait story as a standalone build, if you want the deeper version of "prime and tokio, concurrently, one process."
