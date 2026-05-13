# fuzz

**Builds on:** [`proxima-codec`](../proxima-codec/src/lib.rs)'s `LengthDelimitedCodec` — the `[u32 BE len][payload]` framer.

## The one concept

Fuzzing a wire codec is not "run it with random data and see if it breaks."
It is asserting two invariants hold for *every* input a generator can produce:

1. **no-panic** — a parser fed arbitrary, hostile, malformed bytes must
   **reject, never crash**. It returns a `Result` (`Ok` or `Err`) on every
   input; it never panics, never aborts, never overruns. A network parser
   that panics on a garbage packet is a denial-of-service, not a bug.
2. **round-trip** — for every *valid* value `x`, `decode(encode(x)) == x`.
   Encoding and decoding are inverses; a frame the codec builds is a frame
   the codec reads back byte-for-byte.

The codec under test is `LengthDelimitedCodec`: a 4-byte big-endian length
prefix followed by that many payload bytes — the shape an HTTP/gRPC/JSON-RPC
listener hands raw, untrusted socket bytes to via the `FrameCodec` trait.

## The two invariants

| invariant   | generator                                                                                             | what it catches                                                                                       |
|-------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| no-panic    | 10 hand-picked edge cases + 4096 seeded-random byte strings (length 0–256) fed to `parse_frame`       | slice-index overruns, unchecked length arithmetic, `consumed > input.len()`, integer-overflow on the prefix |
| round-trip  | 4 boundary payload lengths (0/1/4/4096) + 2048 seeded-random payloads (length 0–512) through `encode_frame` → `parse_frame` | encode/decode drift, off-by-one framing, a payload that survives encode but not decode                 |

The edge cases are chosen to land on each named branch of the parser: empty
buffer, 1–3 byte truncated header, complete header with no payload, a declared
length past the buffer (`Incomplete`), a `0xffffffff` length prefix
(`FrameTooLarge`), a zero-length frame under a reject-zero policy
(`ZeroLength`), a length exactly at the cap boundary, and a non-utf-8 payload.

The random draws come from a fixed-seed xorshift64\* generator written inline
(`proptest` is a dev-dependency of a few leaf crates but is not available to
`examples/`). Seeding, not real randomness, is the whole point: every run
reproduces the same 6158 inputs byte-for-byte, so a failure is a failure you
can re-hit, not a lottery.

Because a real panic aborts the process before `main` can print, **the sweep
completing is itself the no-panic proof** — the counts only print if every one
of the 4106 inputs returned a `Result` without crashing.

## Run

```sh
cargo run --example fuzz
```

## What you'll see

Deterministic — same seed, same counts, every run:

```
fuzz: codec under test = proxima-codec::LengthDelimitedCodec (parse_frame / encode_frame)
fuzz: seed = 0x5eedc0de12345678 (deterministic, no real randomness)
fuzz: no-panic   — 4106 garbage/edge-case inputs fed to parse_frame, 0 panics
fuzz: round-trip — 2052 valid frames through encode_frame -> parse_frame, 0 mismatches
```

Each count is backed by `assert!`/`assert_eq!` in `fuzz.rs`: the no-panic
sweep asserts every planned draw ran and `consumed <= input.len()` on every
`Ok`; the round-trip sweep asserts `decode(encode(payload)) == payload` and
that `parse_frame` consumes exactly what `encode_frame` wrote.

## Required features

None — `proxima-codec` is a default (non-optional) dependency, so this example
builds with `--no-default-features`-safe defaults; no extra `--features` flag
is needed.
