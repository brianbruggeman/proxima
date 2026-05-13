{{#include ../../../examples/no-std/README.md}}

## `src/lib.rs` — the pipe, `#![no_std]`, no allocator

```rust
{{#include ../../../examples/no-std/src/lib.rs}}
```

## `src/main.rs` — the `std`-feature demo driver

Compiled only with `--features std`; the default `#![no_std]` build has no
`main` entry point and no `println!` to link against.

```rust
{{#include ../../../examples/no-std/src/main.rs}}
```
