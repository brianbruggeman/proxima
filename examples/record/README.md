# record

Wrap a `Pipe` in `RecordUpstream` and every interaction it serves tees
itself onto disk as it flows.

## Builds on

- [transform](../transform/README.md) — recording is a pipe over your traffic:
  `RecordUpstream<Inner>` has the exact same `In -> Out` shape as `Inner`, it
  just observes on the way through.
- [signal](../signal/README.md) — `TerminalSignal` is the same fire-once
  completion pattern, applied here to real drain durability instead of a toy
  stream.

## What it demonstrates

`RecordUpstream::new(label, inner, sink, pipe_label)` sits in front of any
`Pipe`. Each call tees a `RecordingEvent` stream — `Started`, request/response
chunks, `Ended` — onto a background drainer that appends them to a
`DynRecordingSink`. The sink here is the real production stack:

- `LazyFanOut` — the durable terminal. Disarmed by default (a config-load-time
  pipe graph shouldn't open files before serve); it only opens once its
  `DeferredRuntime` spigot is armed with a `Runtime`.
- `AccumulatingSink` — coalesces per-event appends into batches before handing
  them to the terminal, so the block codec earns its ratio.
- `SinkSpec` + `FormatKind::Json` — one destination, JSON-lines format. This
  example picks `Json` for readability; the production default is
  `FormatKind::Bin` (zstd-compressed postcard frames), so a real cassette on
  disk is compact and binary unless something asks for `Json` explicitly.

The example arms the spigot, drives one request through the recorder, then
reads the cassette back with `JsonlSource` (the same reader `replay` and
a downstream consumer's ingestion use) and diffs the captured response bytes against what
the client actually received. That diff is the substrate every fake/replay
example downstream depends on — if capture isn't faithful, replay can't be
either.

Draining happens on a background task, off the request's hot path, so the
cassette isn't durable the instant the call returns. `TerminalSignal` (see the
`signal` example) wraps the durable sink and fires a `Signal` once the
interaction's terminal `Ended` event has been appended AND flushed —
`terminal.drained().await` waits on that instead of polling the cassette from
outside. No loop, no retry count, no sleep: the read that follows is a single
pass, because `drained()` already guarantees durability.

## Run

```
cargo run --example record
```

## What you'll see

```
--- capture: live traffic through RecordUpstream ---
served: 200 "hello from the wire"
--- replay: reading the cassette back ---
  awaiting terminal.drained() (parked, not polled)...
  Started:  POST /v1/chat
  ResponseStarted: 200
  ResponseChunk: 19 bytes
  Ended
--- proof: 19 bytes served == 19 bytes captured ---
```

`Started` carries the request method/path recorded at call time.
`ResponseChunk` carries the actual response bytes — concatenating every chunk
and comparing against what `response.collect_body()` returned is the proof:
the cassette holds exactly what was served, byte for byte.
