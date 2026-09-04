# Third-Party Notices

This repository is dual-licensed under MIT OR Apache-2.0. Cargo dependencies are
declared in the workspace manifests and resolved through `Cargo.lock`; each
dependency remains under its own license.

This file tracks third-party source or externally derived artifacts that are
checked into this repository directly.

## Vendored Source Code

### h2 0.4.14 HPACK benchmark baseline

Location:

- `proxima-protocols/benches/vendored_h2/`

Source:

- `h2` crate version 0.4.14
- Original package: https://crates.io/crates/h2/0.4.14
- License: MIT
- Copyright notice retained in the vendored files.

Purpose:

- Benchmark-only incumbent comparison for HPACK integer, static table, and
  Huffman behavior.
- Not part of Proxima's production protocol path.

### ggml/llama.cpp q4_K Metal kernel port

Location:

- `omega/src/msl.rs` — `push_q4k_ggml_port_body`

Source:

- `ggml` (vendored inside `llama.cpp`), `kernel_mul_mv_q4_K_f32_impl<4,2,32>`
- Original file: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal
  (lines 5086-5193 at the time of porting; no upstream commit hash is named in
  the in-tree comment)
- License: MIT — https://github.com/ggml-org/llama.cpp/blob/master/LICENSE
- Copyright notice retained in the ported file (see comment above
  `push_q4k_ggml_port_body`).

Purpose:

- `metal-q4k-ggml-port` (default-off): a verbatim transcription of ggml's
  per-thread q4_K matvec math, used as a home-turf comparison arm against this
  crate's own `q4k_header`/`q4k_run8` re-derivation of the same technique.

### tracing-subscriber benchmark shape

Location:

- `proxima-telemetry/benches/bench_emit_vs_tracing.rs`

Source:

- A minimal enabled subscriber shape copied/adapted from `tracing` /
  `tracing-subscriber` benchmark code.
- License: MIT.

Purpose:

- Benchmark-only home-turf comparison against `tracing-subscriber::EnvFilter`.
- Not part of Proxima's production telemetry implementation.

## Compatibility References

### tracing-subscriber EnvFilter semantics

Location:

- `proxima-telemetry/src/emit/env_filter.rs`
- `proxima-telemetry/tests/integration/envfilter_parity.rs`

Reference:

- `tracing-subscriber` 0.3.x `EnvFilter` behavior.
- No upstream source file or parity vector is vendored here; the integration
  test compares Proxima's parser against the real dependency at test runtime.

Purpose:

- Preserve compatibility with common `RUST_LOG` / `EnvFilter` directive strings.

## Fixture Data

The repository contains captured or generated protocol fixtures used for offline
regression tests:

- `spec/examples/*.jsonl`
- `proxima-protocols/tests/fixtures/realpg/*.bin`
- `proxima-protocols/tests/fixtures/realredis/*.bin`

These files are test data, not vendored source code. They should contain only
sanitized or synthetic protocol traffic. Do not commit real credentials,
cookies, bearer tokens, API keys, account identifiers, customer data, private
prompts, or proprietary production captures.

If a fixture was generated from a real service, keep enough provenance in the
test or documentation to regenerate it without relying on private data.

## Adding New Third-Party Material

Before checking in third-party source, generated artifacts, or captured traffic:

1. Confirm the license permits redistribution.
2. Preserve required copyright and license notices.
3. Keep benchmark-only code out of production paths.
4. Redact secrets and private data from fixtures.
5. Update this file with the source, license, location, and purpose.
