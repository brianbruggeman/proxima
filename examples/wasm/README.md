# wasm

proxima at the edge.

## Builds on

[transform](../transform/README.md) — the same `Pipe`, this time compiled for a
target with no OS underneath it.

## What it demonstrates

`transform` showed `proxima_primitives::pipe::primitives::Pipe` — no_std, no-alloc,
no `Send` bound. Nothing in that contract names a socket, a thread, or an
allocator, so nothing stops it from targeting `wasm32-unknown-unknown`. This
example is that claim, proven: the identical `Double` pipe from `transform`,
unchanged, compiled for a target with no OS at all.

```rust
pub struct Double;

impl Pipe for Double {
    type In = u64;
    type Out = u64;
    type Err = Overflow;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Overflow>> {
        async move { input.checked_mul(2).ok_or(Overflow) }
    }
}
```

Three things make this crate wasm32-clean:

- `#![no_std]`, unconditionally — no feature toggle needed, because
  `primitives::Pipe` never needed `std` or `alloc` to begin with.
- `block_on` is `proxima_primitives::block_on` — the workspace's `core`-only
  `Waker::noop()` poll loop (stable since 1.85), not an executor. Driving a
  `Pipe`'s future to completion needs no reactor, no thread pool, no OS wakeup
  source, so the same drive verb compiles for wasm as for a bare-metal caller.
- the host entry point, `double_at_the_edge`, is a bare
  `#[unsafe(no_mangle)] extern "C" fn` — no `wasm-bindgen` glue. It mirrors
  the hand-rolled host-import ABI `proxima-time`'s `driver-wasm` and
  `proxima-net-wasm` already use elsewhere in this repo: a wasm host calls
  the exported symbol directly.

## Build

```
cargo build -p proxima-example-wasm --target wasm32-unknown-unknown
```

wasm examples don't run on the host — `wasm32-unknown-unknown` has no OS to
run a binary against. The build succeeding *is* the proof: rustc accepted a
crate that touches no thread, no filesystem, no socket, no allocator, for a
target that has none of those to offer.

The `Pipe` logic itself is still exercised normally — same source, no
`#[cfg]` branch needed — with the ordinary native test suite:

```
cargo test -p proxima-example-wasm
```

## What you'll see

```
$ cargo test -p proxima-example-wasm
running 2 tests
test tests::doubles_a_regular_input ... ok
test tests::overflow_is_reported_not_wrapped ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo build -p proxima-example-wasm --target wasm32-unknown-unknown
   Compiling proxima-example-wasm v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s)
```

Same source, two targets: one runs and asserts on the host, the other only
needs to compile — there's no host on the other side to run it against yet.
