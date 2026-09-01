# fuzz

Malformed-input smoke harness for wire parsers, protocol FSMs, lowering, and
persistence-boundary readers.

## Tooling path: fallback, not `cargo-fuzz`

`which cargo-fuzz` found nothing on this box, and `rustup toolchain list`
shows only `stable-*` beyond the pre-existing `nightly-aarch64-apple-darwin`
/ `nightly-x86_64-apple-darwin` entries -- `cargo-fuzz` itself (the crate,
not the toolchain) is not installed, and per the task directive no toolchain
install happens here. So this crate follows the honest fallback already
established at `examples/fuzz/main.rs`: a fixed-seed xorshift64\* generator
(`src/lib.rs::Xorshift64`), not libFuzzer/AFL coverage-guided search. It
proves the same **no-panic** contract that generator was built for, just
without coverage feedback -- deterministic, seed-driven, reproducible
byte-for-byte on every run.

If `cargo-fuzz` + nightly become available later, each `fuzz_*` bin's inner
`feed` closure is already isolated per target and drops straight into a
`fuzz_target!` macro with no change to the parser call sites.

## The one concept

For every target: **no-panic**. A parser fed arbitrary, hostile, malformed,
or truncated bytes must reject or parse -- it returns `Result`/`Option` on
every input, never panics, never aborts, never overruns. A real panic aborts
the process before `main` (or the `#[test]`) returns, so a sweep completing
at all, with every planned draw accounted for by the `assert_eq!` in
`run_no_panic_sweep`, is itself the proof (same reasoning as
`examples/fuzz/main.rs`'s doc).

## One bin per target

| bin | target | entry point |
|---|---|---|
| `fuzz_inet` | Ethernet/IPv4/UDP/TCP frame decode | `proxima_protocols::inet::{ethernet,ipv4,tcp,udp}::*Header::parse` / `EthernetFrame::parse` |
| `fuzz_h1` | HTTP/1.x request + response head parse | `proxima_protocols::http1_codec::h1::parse_head`, `h1_client::parse_response_head` |
| `fuzz_h2` | HTTP/2 frame header + payload parse | `proxima_protocols::http2_codec::frame::{FrameHeader::parse,parse_payload}` |
| `fuzz_quic` | QUIC long/short packet header, incl. version negotiation | `proxima_protocols::quic::packet::header::{parse_long,parse_short}` |
| `fuzz_gguf` | GGUF file parse (persistence boundary) | `proxima_gguf::parse_complete` |
| `fuzz_onnx` | ONNX protobuf parse + graph lowering | `proxima_onnx::{parse_complete,lower_graph}` |
| `fuzz_safetensors` | safetensors header/manifest load | `proxima_safetensors::parse_complete` |

Each bin runs a "pure random bytes" sweep plus at least one
structure-aware sweep (a known-good magic/prefix/header followed by random
bytes, or truncated prefixes of a well-formed message) so draws land past
the parser's cheap early rejection and into the interior decode logic the
target exists to stress.

## Run

```sh
cargo run --release --bin fuzz_inet     # or fuzz_h1 / fuzz_h2 / fuzz_quic / fuzz_gguf / fuzz_onnx / fuzz_safetensors
```

150,000 iterations per sweep (well past the 100k floor), seed printed in
each report line so any run reproduces byte-for-byte:

```
fuzz_inet: proxima-protocols::inet::{ethernet,ipv4,tcp,udp} no-panic sweep
fuzz: target=inet::ethernet seed=0x1e7a123456789abc iterations=150000 accepted=133818 rejected=16182 panics=0
...
```

## CI smoke (nextest)

Every bin also carries a `#[cfg(test)] mod tests` with a 3,000-iteration
smoke, small enough to stay in a CI budget while still nextest-visible per
target:

```sh
cargo nextest run
```

## Excluded from the main workspace

This crate has its own `[workspace]` table (same precedent as
`scripts/burn_reference`) and is listed in the root `Cargo.toml`'s
`exclude`, so `cargo check --workspace` from the repo root never builds it
and its dev-only path deps never enter the main lockfile.

## Findings

None as of the run recorded in this README's companion commit: 0 panics
across 7 bins / 16 sweeps / >=150,000 iterations each (2 bins additionally
run bounded truncation sweeps over a fixed-length well-formed message,
correctly reporting fewer iterations for those).
