# Build a record/replay harness

**Prerequisites:** [Foundations](./00-foundations.md) — the **transform** and **observe** roles.
**You will:** record live traffic through any `Pipe` to a cassette, then replay it byte-identical for tests — no upstream call. The recorder is just another `Pipe` wrapped around yours; replay serves the cassette back.
**New concepts (in order):** record (`RecordUpstream` + a sink chain) · a fire-once completion `Signal` (durability await) · replay (`ReplayUpstream`, match key, typed miss).
**Answer keys:** [`examples/record/main.rs`](../../examples/record/main.rs), [`examples/replay/main.rs`](../../examples/replay/main.rs) — `cargo run --example record` / `--example replay`.

## Part A — Record: the tee is the pipe

Wrap any `Pipe` in `RecordUpstream`; every (request, response) tees to a cassette as it flows. There is no "record mode" — the recorder **is** a `Pipe`, `In -> Out` unchanged (`record/main.rs:3-6, 35, 56, 65`):

```rust
let inner = into_handle(SynthUpstream::new("echo", 200, "hello from the wire")); // any Pipe
let recorder = RecordUpstream::new("recorded", inner, sink, "echo");
let response = SendPipe::call(&recorder, request).await?; // flows through, and tees
```

`SynthUpstream::new(name, status, body)` is a canned stand-in for a real upstream — a fixed status code and body, no network call — so the demo has nothing external to depend on. The trailing `"echo"` on `RecordUpstream::new` is a separate label: the pipe name `RecordUpstream` writes into every event it records, distinct from `SynthUpstream`'s own `"echo"` name above.

The `sink` is a small chain that writes the cassette (`record/main.rs:42-56`):

- `LazyFanOut` — the drainer — writes to a JSONL cassette file (`SinkSpec::new(path, FormatKind::Json)`).
- `AccumulatingSink` batches the interaction's events.
- `TerminalSignal` wraps the sink and fires once the terminal `Ended` event has been appended **and** flushed.

Because the drainer appends off the hot path (a background task, not the `call`), the cassette is not durable the instant `call` returns. Instead of polling the file, **await the Signal** (`record/main.rs:71-77`):

```rust
terminal.drained().await;   // parked, not polled — no loop, no retry count, no sleep
```

`terminal` is the `TerminalSignal` built above, wrapped around the sink chain — `sink` was cloned from it, so awaiting `terminal` waits on the same chain `recorder` writes to.

A `Signal` is a fire-once, awaitable completion (see [`examples/signal`](../../examples/signal)) — the **observe** idea from Foundations plus a completion you can wait on. Read the cassette back as a stream of `RecordingEvent`s (`JsonlSource::new(path, runtime).events()`); the example asserts served bytes == captured bytes (`record/main.rs:99-108`).

## Part B — Replay: serve the cassette byte-identical

`ReplayUpstream` loads a cassette and serves it back — same status, headers, and chunk framing — with **no** upstream call (`replay/main.rs:1-8, 56-65`):

```rust
let replay = ReplayUpstream::from_jsonl(&cassette_path, "chat-replay", runtime).await?;
let response = SendPipe::call(&replay, request).await?;   // straight off disk
```

`"chat-replay"` labels this `ReplayUpstream` pipe — the replay-side counterpart to the `"recorded"` label on `RecordUpstream::new` in Part A, not part of the cassette file itself.

Replay matches a request by method + path (+ query/body); `replay.known_keys()` lists what is captured. The response is byte-identical, chunk boundaries included (`replay/main.rs:74-99`).

Replay never guesses. A request that was never captured is a **typed miss**, not a wrong-body 200 (`replay/main.rs:109-125`):

```rust
match SendPipe::call(&replay, unrecorded).await {
    Err(ProximaError::ReplayMiss { fingerprint }) => { /* clean, typed miss */ }
    Err(other) => return Err(other),
    Ok(_) => { /* would be a bug: replay must not invent a response */ }
}
```

The cassette `record` writes and `replay` reads is the same event-log format record wrote (`replay/main.rs:130-198`).

## What you built

- **record** — `RecordUpstream` wraps any `Pipe`; the tee is the pipe; a `TerminalSignal` tells you when the cassette is durable (await, don't poll).
- **replay** — `ReplayUpstream` serves the cassette back byte-identical, and misses cleanly (`ReplayMiss`) on anything never captured.

Front a third-party API with `RecordUpstream` in one run, then swap in `ReplayUpstream` for your tests — same cassette, byte-identical, no network. Both are ordinary `Pipe`s; the harness is composition, not a mock framework.
