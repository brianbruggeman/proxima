# Runtime Drivers — Which Verb Drives Your Future

You have a `Future`. Something has to poll it to completion. Proxima gives
you exactly three verbs for that job, and each means one thing wherever you
see it:

- **`block_on`** — drive a future to completion. Nothing about booting or
  owning a runtime.
- **`run` / `run_prime` / `run_tokio`** — boot a runtime, *then* `block_on`
  the body. The edge of a binary; the one place a backend is named.
- **`run_until_signal`** — serve until a shutdown signal. Not a future-driver
  at all (see the last section) — it lives here only so you never reach for
  it by mistake.

If you know tokio's `Runtime::block_on`, you already know the shape. The rest
of this doc is *which* `block_on` to call, and the two gotchas that decide it.

## Pick by situation

| you have… | call | why |
|---|---|---|
| a future, no runtime at all (no_std, a sync boundary, a bench that never parks) | `proxima_primitives::block_on(fut)` | a bare poll loop on the calling thread |
| a future and an `Arc<dyn Runtime>` (or a concrete `PrimeRuntime`/`TokioPerCoreRuntime`) you already built | `proxima_runtime::block_on(&rt, fut)` or `rt.block_on(fut)` | drives on *that* runtime, whichever backend it is |
| a `fn main` | `#[proxima::main]` | boots the runtime and drives the body for you |
| to boot a runtime inline for one job (a tool, a test) | `run(fut)` — or `run_prime` / `run_tokio` to force a backend | boot, drive, tear down |

Each row has a macro twin: `block_on!(fut)`, `block_on!(rt, fut)`,
`run!(fut)`, `run!(prime, fut)`, `run!(tokio, fut)`
(`src/lib.rs:359`, `src/lib.rs:377`). The macro is sugar that routes to the
same functions by what you pass it — `block_on!` picks the no-runtime vs
on-a-runtime form by arity.

## `block_on` with no runtime

`proxima_primitives::block_on(fut)` (`proxima-primitives/src/driver.rs`) is a
`core`-only `Waker::noop` poll loop — no runtime, no reactor, no allocator, so
it compiles `no_std` + no-alloc. It is the floor every other `block_on` points
down to.

```rust
let occupancy = proxima_primitives::block_on(store.call(frame))?;
```

**Gotcha — it busy-loops if the future parks.** Nothing wakes a noop waker, so
a future that genuinely suspends (waits on I/O, a timer, a channel) spins the
calling thread forever. Use it only where the future resolves without parking:
a sync boundary, a bare-metal caller, a bench. Never inside async code.

Re-exported as `proxima::block_on` (`src/lib.rs:30`) for the umbrella build.

## `block_on` on a runtime you hold

`proxima_runtime::block_on(&dyn Runtime, fut)`
(`proxima-runtime/src/lib.rs:230`) drives `fut` on whatever runtime you hand
it — prime, tokio, a mix, or a backend added next year. That is the whole
answer to "how do I stay runtime-agnostic": you don't pick a backend, you
pass the one you hold.

```rust
let runtime: Arc<dyn Runtime> = /* prime, tokio, whatever */;
let summary = proxima_runtime::block_on(&*runtime, drive_workload(pipe))?;
```

For method ergonomics the concrete backends carry the same verb inherently —
`PrimeRuntime::block_on` (`prime/src/os/runtime.rs:288`),
`TokioPerCoreRuntime::block_on` (`proxima-runtime/src/tokio/mod.rs:125`),
and the low-level `LocalExecutor::block_on`
(`prime/src/core/local_executor.rs:581`) — so `rt.block_on(fut)` reads the
same whichever `rt` is.

### Why isn't `block_on` a method on the `Runtime` trait?

Because the trait is consumed as `Arc<dyn Runtime>` — that trait object *is*
the runtime-agnostic (including mixed-runtime) handle. A method
`fn block_on<F>(&self, f: F) -> F::Output` is generic over the *return* type,
and a generic return can't be erased behind `dyn`: adding it would make
`Runtime` no longer object-safe. So the agnostic form is a **free function
over `&dyn Runtime`**, and the concrete backends add the inherent method on
top. Same verb, same meaning, both call shapes.

### Gotcha — call it from outside the runtime's workers

`proxima_runtime::block_on` spawns the future on a runtime worker and blocks
the *calling* thread until the result comes back. If the calling thread is
itself a worker of that runtime, it deadlocks — it would have to both block
and drive the work it's blocking on. This is the same rule as tokio's
`Runtime::block_on`. Call it from a foreign thread (a test, a tool, a sync
FFI boundary). The edge driver `run_prime` sidesteps this by booting a
dedicated driver core for the body; the bare primitive does not.

## `run` — boot, then drive (the edge)

`run` / `run_prime` / `run_tokio` (`src/runtime.rs:604`, `:290`, `:558`) each
build a runtime, `block_on` the body on it, and tear it down. This is the
**only** place a backend is named, and it belongs at the edge of a binary:

- `run(fut)` — adaptive: prime when the prime runtime is compiled in, else
  tokio. Matches what `#[proxima::test]` picks for the same build.
- `run_prime(fut)` / `run_tokio(true, None, fut)` — force a backend. Reserve
  these for a bench or a proof-test that must pin one runtime.

Almost always you want `#[proxima::main]` instead of calling these by hand —
it expands to `run*`, publishes the runtime so `App::new()` inside the body
adopts it, and preserves your `main`'s return type. A `clippy.toml` guardrail
bans direct `run_prime` / `run_tokio` outside sanctioned sites for exactly
this reason: keep app and library code backend-agnostic; name a backend only
when you deliberately mean to.

## `run_until_signal` is not a driver

`run_until_signal` (`src/app.rs`, `src/server.rs:94`) runs on a *different
axis*: it is an `async fn` — already inside a runtime — that serves until a
shutdown signal (SIGINT) arrives. It does not drive an arbitrary future to
completion; it awaits a *lifecycle*. Do not confuse "run until a signal" with
"drive until a future resolves." The future-driver is `block_on`; the
serve-loop is `run_until_signal`.

## One picture

```
no runtime ........ proxima_primitives::block_on(fut)        // poll loop, no backend
hold a runtime .... proxima_runtime::block_on(&rt, fut)      // drive on rt, any backend
                    rt.block_on(fut)                         // same, method form
the edge .......... #[proxima::main] / run(fut)              // boot + block_on the body
                    run_prime / run_tokio                    // ...forcing a backend
serve loop ........ app.run_until_signal()                   // until SIGINT, not a driver
```

`block_on` drives, `run` boots-then-drives, `run_until_signal` serves. Learn
the three verbs once; they mean the same thing at every tier.
