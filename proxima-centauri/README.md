# proxima-centauri

Sans-IO, no-alloc secure-channel state machines: IKE-style key agreement with mutual authentication, rekey and stateless cookies, plus an ESP child SA with AEAD and anti-replay.

One crate, not a family: each secure protocol is a compile-time feature-gated module. Everything compiles with no operating system, no allocator and no runtime, so a secret has one address for its whole life, a wire length cannot allocate, and there is no OOM abort path. Entropy and time are `Pipe`-shaped inputs passed per step, never capabilities the machines hold.

## What it composes

`proxima-primitives`, `proxima-clock`, plus RustCrypto primitives (`aes-gcm`, `chacha20poly1305`, `x25519-dalek`, `blake3`, `subtle`).

## Features

`std` (default), `config`, `aead-aes-gcm`, `aead-chacha20poly1305`, `loom`; see `[features]` in `Cargo.toml`. Sizing constants come from `proxima-centauri.toml` via `build.rs`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
