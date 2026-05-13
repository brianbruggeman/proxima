# Runtime Coupling — Where Proxima Still Touches Tokio

> **Superseded framing, corrected (2026-07).** This doc originally
> described tokio as always-linked-but-fenced off the public API. That is
> no longer the shape: `tokio` is now an **opt-in** dependency of the
> `proxima` umbrella crate (`Cargo.toml`: `tokio = { ..., optional = true }`).
> The default `proxima` build links **zero tokio** in its dependency graph;
> tokio (and the tokio-wired `h2`-crate listener) come back only under
> `--features tokio` (or the narrower `runtime-tokio` feature on the
> individual crates it composes). [docs/tokio-optional/discipline.md](tokio-optional/discipline.md)
> is the current source of truth for that migration (landed on `main`,
> commit `764917bc`, 2026-07-19) — read it first for the up-to-date gate
> and status. The categories below still describe *where* tokio touches
> the library internally when the `tokio` feature is on; read "the library
> aims to be runtime-agnostic" as "the default build has no tokio to be
> coupled to in the first place," not "tokio is always present but fenced."

The proxima **library** aims to be runtime-agnostic: a consumer should be
able to build proxima against tokio, prime, or a future third runtime
without source changes. The tokio-elimination plan (P0/P1/P2) closed
every leak in the library's **public-API surface** and the macro paths
under `src/`. The verification gate
`tests/units/no_tokio_in_public_api.rs` (part of the `units` integration
test binary) prevents regression.

This doc records the tokio coupling that **remains** in the library when
the `tokio` feature is enabled, why it's acceptable, and the contract for
changing it. If you find a tokio call site not covered here, treat it as
a leak and either fix it or add it to this doc with justification.

## What does not count as a leak

`tokio_util::compat::*` — the futures-io ↔ tokio-io bridge. Used at the
edge between proxima's public surface (`futures::io`) and the listener
internals. The whole point of P0 was to make this bridge possible; it's
the boundary, not the leak.

`tokio_rustls::*` — TLS termination. Stays tokio-coupled by design. The
prime serve_https path bridges through tokio_util::compat into
tokio_rustls and back. No production-ready pure-prime TLS substrate
exists yet; when it does, this category moves.

`tokio_uring::*` — only inside the io_uring backend gated by the
`io-uring` feature. The `prime` crate's io_uring backend at
`prime/src/os/io_uring/` is the prime-native alternative; both
exist behind the same feature flag and dispatch in `io_uring_compat.rs`.

## Acceptable residual coupling (by category)

### 1. `tokio::spawn` / `tokio::task::spawn_local` inside listener accept loops

The listener implementations (`listeners/http.rs`, `listeners/h1.rs`,
`listeners/h2.rs`, `listeners/h3.rs`, `listeners/mcp.rs`,
`listeners/http_uring.rs`, etc.) call `tokio::spawn` to handle accepted
connections. These run only when the active runtime is tokio — chosen at
runtime initialization by `Runtime::spawn_on_core` selecting the
`TokioPerCoreRuntime` impl. Equivalent prime-native paths exist at
`runtime/prime/os/runtime.rs::serve_http` / `serve_https`.

**Boundary:** private impl detail. The public surface is
`Runtime::spawn_on_core(handle, future)`; what the impl does underneath
is its business.

**Contract for change:** any new listener spawn site goes through the
`Runtime` trait, not bare `tokio::spawn`. Existing sites stay until a
larger listener refactor decides on a unified accept-loop abstraction.

### 2. `tokio::sync::*` primitives in private impl detail

`Notify`, `Mutex`, `mpsc::*`, `watch`, `broadcast`, `OnceCell` appear in
~30 sites across listener / pipeline / control-plane modules. None of
these types appear in any public-API signature (verified by the gate
test). They live inside types like:

- `listener.rs:152` — `Arc<tokio::sync::Notify>` for shutdown bridging
  between `futures::channel::oneshot` and the listener's drain signal
- `client/handle.rs:5` — `tokio::sync::OnceCell` for lazy client init
- pipeline executor channels — `tokio::sync::mpsc` for fan-in to the
  control-plane queue
- recording drainer tasks — `tokio::sync::mpsc::unbounded_channel` in
  `upstreams/record.rs:134,275`

**Boundary:** these are constructed inside the module that owns them
and never exposed. The `proxima::sync::*` re-export surface (`Mutex`,
`Notify`, `mpsc`, etc., backed by `futures::*` / `async-lock` /
`event-listener`) is what proxima **users** see.

**Contract for change:** new code in these modules can keep using
`tokio::sync::*` if it's not in a public signature. New public API
must use `proxima::sync::*`. The gate test enforces this.

### 3. `tokio::time::*` inside listener / drainer paths

`tokio::time::sleep_until`, `tokio::time::interval`, `tokio::time::Instant`
appear inside listener accept loops and recording drainer tasks. Same
runtime-selection logic as #1: these only run when tokio is the active
runtime.

Examples:

- `listeners/http_uring.rs:162-165` — quiescent-window deadline using
  `tokio::time::Instant::now()` + `tokio::time::sleep_until(deadline)`
- `upstreams/record.rs` — drainer task uses `tokio::sync::mpsc` recv
  which composes with tokio's reactor

Hot-path middleware (P1.1 targets — `middlewares/isolate.rs`,
`middlewares/retry.rs`, `upstreams/http.rs`, `framing/reconnect.rs`)
have already been migrated to `Runtime::timer_at` for portability.

**Contract for change:** new time calls in the listener/drainer paths
can keep `tokio::time::*`. New time calls in middleware or upstream
business logic route through `Runtime::timer_at` or
`proxima::time::{sleep, timeout, interval}`.

### 4. `tokio::io::*` traits inside listener bodies

`tokio::io::{AsyncRead, AsyncWrite, AsyncBufReadExt, AsyncWriteExt,
BufReader}` appear in listener bodies and CONNECT-upgrade plumbing.
These types compose with `tokio::net::TcpStream` / `UnixStream` which
the listener owns end-to-end.

**Public-API note:** the user-facing `HijackStream` trait in
`upgrade.rs` was migrated in P0.1 to `futures::io::AsyncRead +
AsyncWrite`. Listener internals that hand a tokio-io stream into the
hijack pipeline bridge through `tokio_util::compat::TokioAsyncReadCompatExt`.

**Contract for change:** any new `pub` trait or function that takes a
stream-shaped parameter must use `futures::io::*`. Internal use of
`tokio::io::*` is fine.

### 5. `tokio::net::*` listeners

`TcpListener`, `TcpStream`, `UnixListener`, `UnixStream` appear in
listener implementations. The `io_uring_compat.rs:31-55` module this
section originally cited no longer exists under that name in this
worktree — the h1/h2/h3 listeners now live in the `proxima-http` crate
(`proxima-http/src/http1/listener.rs`, etc., re-exported via
`src/listeners/mod.rs`) and the tokio-vs-prime io_uring dispatch has
moved with them. The shape described (dispatch behind the `io-uring`
feature × active runtime, result never re-exported publicly) is still
the intent; verify the current dispatch site before citing a file:line.

**Contract for change:** listener code uses the proxima-internal
typedef from `io_uring_compat`, never `tokio::net::*` directly. Any new
listener backend goes into the dispatch matrix.

### 6. `tokio::signal::*` in `server.rs::wait_for_signal`

Unix signal handling (`SIGTERM`, `SIGINT`) uses `tokio::signal::unix`.
This is tied to tokio's signal driver and is only invoked from the
`server.rs::Server::run_until_signal*` entry points.

**Boundary:** the function is `async` and called from a tokio runtime
context (the proxima `Server` is built on the `Runtime` handle, and
when the handle is `TokioPerCoreRuntime`, signal driver is available).

**Contract for change:** if a prime-native equivalent
`run_until_signal` is needed, add it behind a `cfg(feature =
"runtime-prime-full")` gate using the prime signal-handling
infrastructure when that exists. Don't replace the tokio version —
keep both.

### 7. `tokio::process::*` in `upstreams/process.rs`

Child-process lifecycle (`Command`, `Child`, `ChildStderr`,
`ChildStdout`) uses tokio's async process API. The process upstream
runs the user-supplied binary and pipes its stdio.

**Why not portable:** there's no widely-used runtime-agnostic async
process crate. `std::process` is sync, `async_process` exists but is
single-crate-maintainer status. Tokio's is the de facto standard.

**Contract for change:** if proxima ever needs to spawn a process from
a non-tokio runtime, the process upstream becomes a leaf-runtime
adapter (the upstream selects its runtime, and proxima dispatches
through the `Runtime` trait). Until then, this stays tokio.

### 8. `tokio::runtime::Handle::try_current()` in cold-path detection

One site uses `tokio::runtime::Handle::try_current()` to detect
whether the caller is on a tokio runtime. This is a no-op when there
is no tokio runtime — it returns an error rather than panicking — so
it composes safely on prime.

**Contract for change:** none. Acceptable as-is. If a prime-only build
ever drops the tokio dep entirely, this site becomes a `cfg(feature)`
no-op stub returning the error variant.

### 9. `spawn_local` in test harnesses and per-core scheduler tasks

`tokio::task::spawn_local` appears inside `TokioPerCoreRuntime`'s
worker loops and in some test harnesses. These are per-core executor
boundaries that the runtime owns — outside the listener / pipeline /
middleware code that the rest of proxima cares about.

**Contract for change:** none. These are tokio impl detail and stay
inside `runtime/tokio_per_core.rs` and test files.

### 10. Dual-reactor in `prime-tokio-compat` mode

When the `prime-tokio-compat` feature is on, each prime worker spawns
a `tokio::runtime::Builder::new_current_thread()` runtime to host
tokio-API code. Tokio's mio reactor runs alongside prime's reactor
(epoll or io_uring) — they don't merge. See
`docs/runtime-prime/compat-mode-tradeoffs.md` for the full
discussion.

**Why:** tokio 1.52 stable does not expose
`Builder::with_io_driver`. There's no way to wire prime's reactor as
tokio's I/O backend without forking tokio.

**Contract for change:** when tokio exposes a custom-driver API, wire
prime's reactor in and collapse to a single reactor per core. Until
then, dual-reactor is the cost of running tokio code on prime.

## How to add a new tokio call site

Decision tree:

1. Is the call inside a `pub` signature (return type, parameter,
   trait bound)? **No** — gate test enforces this. Use
   `proxima::sync::*`, `proxima::time::*`, `futures::io::*`, or the
   `Runtime` trait.
2. Is the call inside a hot-path middleware / upstream / framing
   module? Route through the `Runtime` trait or `proxima::sync` /
   `proxima::time`. Don't add new direct `tokio::*` here.
3. Is the call inside a listener body, drainer task, signal handler,
   process upstream, or the tokio runtime impl itself? Direct
   `tokio::*` is fine. Add a one-line note to the relevant section
   above if the call introduces a new pattern (not just a new
   instance of an existing pattern).
4. Is the call a `tokio::select!` / `tokio::join!` / `tokio::pin!`
   macro path? **No.** Use `futures::select!` / `futures::join!` /
   `futures::pin_mut!`. The gate test enforces this.

## Verification

- `tests/units/no_tokio_in_public_api.rs` (module `no_tokio_in_public_api`
  inside the `units` integration test binary — `cargo nextest run -p
  proxima --test units -E 'test(no_tokio_in_public_api)'`) — two tests,
  both must pass. Macro path test (P1.4 gate) + public-API signature
  test.
- `tests/units/dx_audit.rs` (module `dx_audit`, same `units` binary) —
  earlier-stage doc and source-level invariants from the audit.
- For the tokio-optional migration's own gate (default build has zero
  tokio, `--features tokio` restores it, both runtimes proven), see
  [docs/tokio-optional/discipline.md](tokio-optional/discipline.md).

If either test fails, the work doesn't ship.
