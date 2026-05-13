# runtime_select

The same `Pipe`, served first on prime, then again on tokio. Nothing about
the pipe changes between the two passes — only which `Runtime` +
`AcceptorFactory` pair `App` is built with.

## Builds on

[hello](../hello/README.md) — the same `App::new().with_runtime(...)` wiring,
run twice with two different runtimes instead of once.

## What it demonstrates

`hello` picks one `Runtime` and stops there. This example factors that
choice out into a parameter: `serve_and_check` takes a `Runtime` +
`AcceptorFactory` pair and a bind address, and is called twice — once with
`PrimeRuntime` + `PrimeAcceptorFactory`, once with `TokioPerCoreRuntime` +
`TokioAcceptorFactory` — against the exact same `PipeHandle`, cloned, not
rebuilt.

That's only possible because `select_pipe` is sans-IO: `SendPipe::call` never
names a runtime, a reactor, or a socket. `App` is the only thing that knows
which executor is driving it, and `App` is rebuilt fresh on each pass. The
pipe itself is oblivious — proof that "which runtime" is a deployment
choice, not something baked into how you write the service.

This is the *sequential* half of the runtime-composability story:
`multi-runtime` is the next rung — prime and tokio serving CONCURRENTLY, in
the same process, sharing state across the boundary. Here the two passes
run one after the other and never overlap; there's nothing to share because
only one runtime is live at a time.

## Run

```
cargo run --example runtime_select --features "runtime-tokio tokio"
```

(`runtime-tokio` is required — `TokioPerCoreRuntime` is opt-in; `tokio` is
also required — the tokio-backed `AcceptorFactory` lives behind the full
`tokio` feature, not the narrower marker. The prime pass ships in
`serve-prime`, which is on by default.)

## What you'll see

```
--- pass 1: the SAME pipe served on prime ---
listening on 127.0.0.1:8083 (prime runtime, 1 core)
GET http://127.0.0.1:8083/ (prime) ->
HTTP/1.1 200 OK
...
hello from whichever runtime is listening

prime drained: cores_acked=1 hooks_drained=0

--- pass 2: the SAME pipe served on tokio ---
listening on 127.0.0.1:8084 (tokio runtime, 1 core)
GET http://127.0.0.1:8084/ (tokio) ->
HTTP/1.1 200 OK
...
hello from whichever runtime is listening

tokio drained: cores_acked=1 hooks_drained=0

same Pipe, two runtimes, identical response both times.
```

Both responses carry the identical body — the pipe did not need to be
rewritten, only re-wired, to move from one runtime to the other.
