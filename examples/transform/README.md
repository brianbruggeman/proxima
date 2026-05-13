# transform

Write one `Pipe`. That's the whole lesson.

## Builds on

[hello](../hello/README.md) — the pipe you put behind a listener; here you write your own.

## What it demonstrates

Proxima's algebra has one primitive shape: a typed `In -> Out` step, async,
called through a shared reference (no exclusive access needed — any state a
step keeps lives behind interior mutability). Every composed piece of the
algebra — the chain that joins steps into a pipeline, filters, fan-out,
fan-in, backpressure, listeners — is built by wrapping or chaining something
shaped exactly like this.

That one shape does not name "source", "sink", or "observe" anywhere. Those
are not separate machinery — they're the same shape with `In`/`Out` chosen
to be degenerate:

| form | shape | meaning |
|---|---|---|
| source | `() -> Out` | produces without consuming |
| transform | `In -> Out` | the general map |
| sink | `In -> ()` | consumes, produces nothing |
| observe | `In -> In` | passes through, side-effect only |

The example implements all four forms and drives them from `main` in that
order, so the same value flows source → transform → observe → sink: a
source pipe produces a number, a transform doubles it, an observe
watches it go by, and a sink consumes it and reports nothing back. Nothing
here is chained together through the algebra's chain primitive — each form
is called directly, one at a time, so the forms themselves stay in view.

## Run

```
cargo run --example transform
```

## What you'll see

```
--- round 0: one Pipe trait, four roles chosen by type ---
source    (In=(),    Out=u64):  () -> 0
transform (In=u64,   Out=u64):  0 -> 0
  echo: observed 0, call #1
observe   (In=Out=u64):         0 -> 0 (unchanged)
  sink: final value 0
--- round 1: one Pipe trait, four roles chosen by type ---
source    (In=(),    Out=u64):  () -> 1
transform (In=u64,   Out=u64):  1 -> 2
  echo: observed 2, call #2
observe   (In=Out=u64):         2 -> 2 (unchanged)
  sink: final value 2
--- round 2: one Pipe trait, four roles chosen by type ---
source    (In=(),    Out=u64):  () -> 2
transform (In=u64,   Out=u64):  2 -> 4
  echo: observed 4, call #3
observe   (In=Out=u64):         4 -> 4 (unchanged)
  sink: final value 4
```

The source pipe advances its own state on every call (`0, 1, 2, ...`) even
though its input is `()` — proof that a source still fits the one-shape
contract, it just has nothing to read from `In`. The observe's output
always equals its input — proof that observe is a call used for its side
effect, not its return value. The sink returns `()` — proof that a sink's
contract is satisfied by doing something and reporting nothing back.

## In algebra terms

- One shape, four forms: a source (`() -> Out`), a transform (`In -> Out`),
  a sink (`In -> ()`), and an observe (`In -> In`) are the same
  primitive with `In`/`Out` chosen to be degenerate.
- The example calls each form directly rather than joining them through the
  chain — the chain, and the fan-out/fan-in/filter primitives built on top
  of it, are a separate lesson from the forms themselves.
- Patterns and strategies — composed behaviors and their policy dials — are
  built on top of these forms; this example is the algebra's floor, not its
  ceiling.
