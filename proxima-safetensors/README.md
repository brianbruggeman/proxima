# proxima-safetensors

Sans-IO safetensors reader: a header and tensor-directory state machine over chunked bytes, with no dequantization. It never opens, reads, writes or mmaps a file; callers feed bytes in any chunking through `SafetensorsParser` and read back `Manifest` / `TensorEntry` records (names, dtypes, shapes, byte ranges). `writer::write_complete` is the dual, turning an in-memory model into owned bytes.

The format is checked against the published safetensors specification and the reference implementation's `Dtype` enum and offset validation.

## What it composes

`proxima-codec`, `proxima-primitives`, `proxima-tensor`, `serde_json`.

## Features

`alloc`, `std` (default); see `[features]` in `Cargo.toml`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
