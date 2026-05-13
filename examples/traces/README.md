# traces

A span observes an operation's *scope*, and that scope survives an async boundary — and, given the
context explicitly, a plain same-scope call too.

## Builds on

[transform](../transform/README.md) — a span is the observe form over an operation's scope, the same way an observe pipe is that form over a single value.

## What it demonstrates

A span is `Pipe`'s observe form (`In = Out`) aimed at *duration and place in
the request tree* instead of a single value: it wraps `start..end` around an
operation and records where that operation sits relative to its caller.

proxima keeps that "where it sits" fact as **explicit data** — a W3C
`traceparent` (`trace_id` + `span_id`), captured with
`RecorderSpanGuard::inject` and reopened on the other side with
`Recorder::span_from_traceparent`. There is no ambient/thread-local "current
span" the way `tracing` has; nothing reads a parent for you. That's a
deliberate design choice, not just a gap: proxima's own use case is many
concurrently-interleaved requests sharing one executor thread, where a naive
thread-local push/pop span stack would corrupt across tasks (task A opens a
span, awaits and yields the thread, task B runs on the same thread and would
see A's still-pushed span as its own parent).

The example proves both directions of that rule on one call:

- `handle_request` opens its span by hand — `recorder.span(name)...start()`,
  the exact `Recorder::span`/`RecorderSpanGuard` chain `#[instrument]` itself
  expands to — captures its own context into a `HeaderList`, hands the
  captured bytes across a `tokio::spawn` boundary (a different task, possibly
  a different OS thread), and only closes after the spawned `run_query`
  returns. `run_query` reopens its span from the handed-in `traceparent` via
  `span_from_traceparent` — same `trace_id`, `parent_span_id` set to
  `handle_request`'s `span_id`. The context survived the boundary because it
  was carried, not because it was ambient.
- `validate_request` is instrumented with the `#[proxima::telemetry::instrument]`
  sugar and an explicit `parent = parent` argument (`parent: Option<&[u8]>`),
  called TWICE from inside `handle_request`'s scope: once with `parent = None`
  (nothing carried) — its own fresh `trace_id`, no auto-parent, proving the
  sugar form still never nests on its own — and once with the parent's own
  captured `traceparent` bytes threaded through explicitly — same `trace_id`
  as `handle_request`, `parent_span_id` set. Nesting the sugar form in
  wall-clock scope does nothing by itself; passing `parent = <expr>` is what
  connects it — and when it resolves to `Some`, the macro expands to the SAME
  `recorder.span_from_traceparent(name, bytes)...start()` call `run_query`
  makes by hand above. The sugar and the hand-written path are one mechanism,
  not two.

Both facts are asserted against the spans an `InMemoryPipe` actually
captured, not just printed.

## Run

```
cargo run --example traces
```

## What you'll see

```
proxima traces: spans across an async boundary

handle_request: opened span Some(SpanId([...]))
  run_query:      opened span Some(SpanId([...])), carried in from handle_request

handle_request -> 42

span tree (4 spans captured):
  name=validate_request  trace_id=02eb0d43...  span_id=e2e58541...  parent_span_id=-
  name=validate_request  trace_id=603f9a63...  span_id=6d258398...  parent_span_id=89bda28a...
  name=run_query         trace_id=603f9a63...  span_id=2a3349ea...  parent_span_id=89bda28a...
  name=handle_request    trace_id=603f9a63...  span_id=89bda28a...  parent_span_id=-

-> run_query IS a child of handle_request: same trace_id, parent_span_id set
-> validate_request(parent = None) is its OWN root: no context carried, no auto-parent
-> validate_request(parent = Some(..)) IS a child: the explicit seam connects it

PASS: span context survives both the tokio::spawn boundary and a same-scope call,
      via explicit data in both cases -- never ambient.
```

`run_query` and `handle_request` share one `trace_id`, and `run_query`'s
`parent_span_id` equals `handle_request`'s `span_id` — proof the tree
survived `tokio::spawn`. The first `validate_request` call (`parent = None`)
carries a different `trace_id` and no parent at all — proof the
`#[instrument]` sugar never auto-nests on its own, even when called from
inside another span's live scope. The second `validate_request` call
(`parent = Some(traceparent)`) shares `handle_request`'s `trace_id` and
records its `span_id` as parent — proof the explicit `parent = <expr>`
argument is what connects it.
