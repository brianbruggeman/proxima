# proxima-mil

Sans-IO writer for Core ML ML Programs: the MIL protobuf encoder, weight blob writer and package assembler that `proxima-omega`'s door plugs in as its Emit and Package stages.

The crate is `no_std`; `alloc` and `std` are additive features. `wire` and `weights` need only `core`; `census` and `writer` need `alloc`. Tunable axes are keys of `proxima-mil.toml`, resolved by `build.rs`.

## What it composes

`proxima-tensor` (the program being written out), `proxima-primitives`, `half`.

## Features

`alloc`, `std`, `alloc-count`; see `[features]` in `Cargo.toml`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
