# proxima-main — `#[proxima::main]`

Reference, not a `Pipe`/listener lesson: this is the macro's own proof.
`#[proxima::main]` boots the prime or tokio runtime and drives the async
`main` body to completion — one attribute, two runtimes, same body.

## `proxima_main_demo.rs` — prime

```rust
{{#include ../../../examples/proxima_main_demo.rs}}
```

## `proxima_main_tokio_demo.rs` — tokio

```rust
{{#include ../../../examples/proxima_main_tokio_demo.rs}}
```
