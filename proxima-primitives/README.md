# proxima-primitives

The pipe algebra itself. `Pipe` is one async trait, `In -> Result<Out, Err>`, that becomes a transform, a source, a sink or an observe by your choice of `In` and `Out`. Everything else in proxima composes what is defined here.

Add it with `cargo add proxima-primitives`.

## The one trait

| form | shape | what it is |
|---|---|---|
| transform | `In -> Out` | turns one thing into another |
| source | `() -> Out` | takes nothing, produces something |
| sink | `In -> ()` | takes something, produces nothing |
| observe | `In -> In` | hands back its input; acts on the side |

From the `Pipe` doc comment (it is compiled by `cargo test` there):

```rust
use core::convert::Infallible;
use core::future::Future;
use proxima_primitives::pipe::Pipe;

// transform: In -> Out
struct Double;
impl Pipe for Double {
    type In = u64;
    type Out = u64;
    type Err = Infallible;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(input * 2) }
    }
}
```

`Pipe` has no `Send` bound, so it runs on a per-core, shared-nothing worker. The additive tiers are standalone traits: `SendPipe` (pipe and future are `Send`), `UnpinPipe` (the returned future is `Unpin`) and `UnpinSendPipe`. `AndThen` is the composition law, reached through `PipeExt` (`use proxima_primitives::PipeExt`).

## What is in it

Four modules, folded together from what were separate crates:

- `pipe`: the `Pipe` family; `Handler` (request to response); `SourcePipe` (`Signal -> ()`); `HeaderList`; the payload types `Carry`, `ResponseStream` and `RequestStream`; and combinators and plugin surface (fan-in and fan-out, filters, retry, rate limiting, routing, and more).
- `stream`: sans-IO byte-stream traits (`StreamConnection`, `StreamListener`, `StreamUpstream`, datagram sockets) over `futures::io`, shared by the wire-protocol crates.
- `sync`: runtime-agnostic concurrency primitives shaped like `tokio::sync` (`Mutex`, `Notify`, `mpsc`, `oneshot`, `watch`, `broadcast`, `blocking`, `task`, `shutdown`, and others).
- `transport`: `Replay` (record-and-replay) and the `GenericStream` seam, below `pipe`.

Also at the root: `block_on` (drive a future to completion with no runtime and no allocator), `Codec` (a payload-less tag for a packed byte layout), and the `float_layout` / `block_layout` modules (`FloatLayout`, `BlockLayout`: data descriptions of packed numeric formats).

## What it composes

- `proxima-core`: markers, `ProximaError`, time, and the registry and live-swap cells, enabled per tier.
- `proxima-runtime` (optional, from `alloc` up): the runtime traits.
- `proxima-config` (optional, pulled in by `std` with its `schema-std` feature).
- External: `bytes`, `futures`, `serde`, `heapless`, `crossbeam-queue`, `portable-atomic`; `event-listener` and `async-lock` from `alloc`; `arc-swap`, `parking_lot`, `tracing`, `bon` and `conflaguration` under `std`.

Proxima's own telemetry, clock, codec and runtime crates all sit on top of this one.

## Features and tiers

Default feature: `std`.

- no_std, no alloc: the `pipe::primitives` module (`Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe`, `AndThen`) allocates nothing and touches no `std` (its module doc says it compiles on a bare `#![no_std]` target).
- `alloc`: no_std + alloc floor. The pipe forms and combinators that need a heap, `sync::notify`, `sync::oneshot`, `sync::shutdown` (`ResourceRegistry`) and the async semaphore.
- `std` (default): implies `alloc`; adds `sync::{mpsc, watch, broadcast, rwlock, once_cell}` and `Mutex`, the `stream` traits and `transport::Replay` (these use `futures::io` and std `Mutex`; the `alloc` tier carries only `stream`'s `BindAddr` / `PeerInfo` re-export and `TcpBindOptions`), and the config mirror.

Opt-in features: `fan-concurrent` (`Race` / `ScatterGather`), `config` (layered conflaguration loader for the filter registry), `part-source`, `atomic-permits`, `async-mutex` (waker-based async mutex that reaches bare metal), `blocking` (futex `sync::blocking::Mutex` for no_std + OS), `runtime-shaped-mutex`, `runtime-tokio` (`sync::task::JoinSet`), `io-async` and `io-bridge` (ring sources and sinks as `AsyncRead` / `AsyncWrite`), `test-support` (`MockClock` / `RecordingClock` doubles), `bon`, `conflaguration`. `runtime-prime-full` is empty and kept for feature forwarding.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
