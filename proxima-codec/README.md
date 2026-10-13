# proxima-codec

Pluggable codecs for proxima: a family of codec traits for request/response bodies, length-delimited frames, stateful wire formats and datagrams, with JSON and bytes-passthrough defaults and a name-keyed registry.

Add it with `cargo add proxima-codec`.

## What it provides

Traits (each lives in `proxima_codec`):

- `MessageCodec`: schema-driven full-message codec with an owned `Input` / `Output` pair (`decode_input`, `encode_output`).
- `FrameCodec`: stateless, borrow-only parse and append-to-`Vec<u8>` encode for length-delimited wire formats; no allocation on the inner loop.
- `StatefulCodec`: separate per-session encoder and decoder for formats that track wire state.
- `WireCodec`: a field-at-a-time iterator for formats like protobuf.
- `Datagram`, plus `Addressed<M>`: a message paired with the peer `SocketAddr` for connectionless transports.

Also `ShareBuf` (in `share_buf`): a trait for handing back a cheap, independently owned sub-slice of the buffer a codec parsed from, so a borrowed frame can be re-owned without copying.

Types: `BytesPassthrough`, `LengthDelimitedCodec`, `DelimiterCodec` (with `DelimiterFraming`), `FrameLimits`, `FrameError`, and, under `std`, `JsonCodec<Input, Output>` (serde with simd-json decode).

Registry (feature `std`, in `factory`): `CodecRegistry`, `CodecFactory` / `DynCodecFactory`, and the stock factories `JsonCodecFactory` and `BytesPassthroughCodecFactory`.

## What it composes

- `proxima-core`: `ProximaError` is the error type of the message-level traits, and the registry is built on `proxima_core::FactoryRegistry` (enabled through `proxima-core/registry`).
- External: `bytes`, `memchr` (delimiter scans), `thiserror`; `serde`, `serde_json`, `simd-json` and `base64` under `std` only.

## Features and tiers

Default feature: `std`.

- `alloc`: no_std + alloc. The codec traits, `Addressed`, `FrameLimits`, `LengthDelimitedCodec`, `DelimiterCodec` and `BytesPassthrough`. Per the `Cargo.toml` comments, this tier is checked on `thumbv7m-none-eabi` by the `proxima-codec` cell of `scripts/_floor-crate-matrix.sh` in the repository.
- `std` (default): adds `JsonCodec` and the registry. `serde` is std-only on purpose, so a bare-metal consumer does not pay for serde or its derive proc-macro.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
