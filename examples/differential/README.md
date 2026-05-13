# differential — run against a reference oracle

**Builds on:** `proxima-protocols`' sans-IO request-head parser
(`proxima-protocols/src/http1_codec/h1.rs`) — the pipe under test.
`proxima-http`'s H1 connection state machine (the actual serving pipe) sits
on top of it.

## The one concept

Differential testing feeds **identical inputs** to two **independently
implemented** parsers and asserts they agree. A self-consistent test can
only catch a bug proxima's own author already imagined; an independent
oracle — written by someone else, with different assumptions, fuzzed and
audited on its own timeline — catches what proxima's tests can't. The
value is entirely in the oracle's independence: the moment the "oracle"
is just proxima calling itself twice, the test proves nothing new.

`proxima_protocols::http1_codec::h1` openly delegates its grammar parsing to
[`httparse`](https://docs.rs/httparse) — the de-facto independent
Rust HTTP/1 head parser (used by `hyper`, `reqwest`, `actix`). So this
example doesn't diff proxima against a black box; it diffs proxima
against **the same `httparse` call, constructed independently**, and
targets the part that *is* proxima's own code: `ParserLimits` — a
resource budget (max method / path / header-line bytes, max header
count) that `httparse`'s grammar has no notion of at all.

## What agreement looks like, and where it doesn't

| input class | proxima | httparse (oracle) | agree? |
|---|---|---|---|
| well-formed GET/POST/OPTIONS, HTTP/1.0 and 1.1, mixed header casing | complete | complete | yes |
| partial request line / partial headers (no terminator) | partial | partial | yes |
| bad version, control bytes in method/header-name | rejected | rejected | yes |
| header count over a **shared** header-slot cap | rejected | rejected | yes |
| method token over proxima's 16-byte budget, still valid grammar | rejected | complete | **documented divergence** |
| request-target over a tightened path budget, still valid grammar | rejected | complete | **documented divergence** |
| header line over a tightened line-length budget, still valid grammar | rejected | complete | **documented divergence** |

12 of 15 inputs agree outright. The 3 that don't are not bugs — they're
`ParserLimits` doing exactly its job. `httparse` will parse a request
line of any length as long as the buffer holds it; proxima additionally
enforces a byte budget per field as a DoS guard. Differential testing
*surfaces* that gap for a human to look at; it doesn't paper over it by
asserting blind equality. The "shared header-slot cap" row is the
control case that proves the divergence isn't a testing artifact: when
both parsers are handed the *same* header-array capacity, they reject
the same input the same way — the divergence only appears where
proxima checks something `httparse` was never asked to check.

## Run

```sh
cargo run --example differential
```

## What you'll see

```
differential: proxima's h1 parser vs httparse, same bytes, independent parse

input                                                      proxima    oracle     agree?
GET root, no headers                                       complete   complete   yes
GET path+query, two headers                                complete   complete   yes
POST with content-length, body left unconsumed             complete   complete   yes
OPTIONS asterisk-form request-target                       complete   complete   yes
HTTP/1.0 request                                           complete   complete   yes
header name case preserved verbatim                        complete   complete   yes
partial request line                                       partial    partial    yes
partial headers, no blank-line terminator                  partial    partial    yes
unsupported HTTP/2.0 version                                rejected   rejected   yes
control byte in method token                                rejected   rejected   yes
control byte in header name                                 rejected   rejected   yes
header count exceeds a shared 2-slot cap                    rejected   rejected   yes
method token longer than proxima's default 16-byte budget   rejected   complete   documented-no
  documented divergence: method token longer than proxima's default 16-byte budget — httparse has no maximum method length; proxima's ParserLimits does
request-target within grammar, over a tightened path budget rejected   complete   documented-no
  documented divergence: request-target within grammar, over a tightened path budget — httparse accepts any request-target length; proxima's budget is a DoS guard, not grammar
header line within grammar, over a tightened line budget    rejected   complete   documented-no
  documented divergence: header line within grammar, over a tightened line budget — httparse has no per-header-line length cap; proxima's budget is a DoS guard, not grammar

12 of 15 inputs: proxima and httparse agree on the parsed result (or both reject)
3 documented divergence(s): proxima's resource-budget limits reject inputs httparse's bare grammar accepts — a human-adjudicated tradeoff, not a bug
```

Each row is backed by an `assert!` in `differential.rs` — an `Agree` case
panics if the two parsers disagree, and a `Diverges` case panics if they
*stop* disagreeing (the moment `httparse` grows its own length caps, or
proxima's defaults change, this test starts failing until the docs and
the case list are updated to match).

## Why this pairing, not TCP-vs-smoltcp

`smoltcp` is the heavier oracle: standing up two independent TCP state
machines against the same packet trace is a real integration, not a
unit-scale differential test. `proxima_protocols::http1_codec::h1`'s head
parser already has a known-independent oracle one `httparse::Request::parse`
call away, with no new dependency to add — `httparse` is already a direct,
non-optional dependency of the `proxima` crate, and `proxima-protocols` is
already a dev-dependency (via the `http1_codec-codec-trait` feature). Zero
ceremony, same teaching point: an independent oracle catches what a
self-consistent test can't, and when it disagrees, that disagreement is
either a bug to fix or a tradeoff to name — never something to ignore.
