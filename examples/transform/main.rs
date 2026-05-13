//! Write one `Pipe`: the atom of the whole model. Typed `In -> Out`, `&self`,
//! an async `call` returning `Result<Out, Err>` — that is the entire
//! contract. Everything composed in proxima (`AndThen`, `Filter`, `FanIn`,
//! backpressure, listeners) is built from something shaped exactly like the
//! four pipes below.
//!
//! One trait, four roles — chosen entirely by `In`/`Out`, not by separate
//! machinery:
//!   - source:    `Pipe<In = (), Out = T>` — produces without consuming.
//!   - transform: `Pipe<In = T,  Out = U>` — the general map.
//!   - sink:      `Pipe<In = T,  Out = ()>` — consumes, produces nothing.
//!   - observe:   `Pipe<In = T,  Out = T>` — passes through, side-effect only.
//!
//! Run: `cargo run --example transform`

use core::cell::Cell;
use core::convert::Infallible;
use core::future::Future;

use proxima_primitives::pipe::Pipe;

/// source: `() -> u64`. No input to consume, so `In = ()`; the produced
/// state lives in the pipe itself (`Cell`, since `call` takes `&self`).
struct Counter {
    next: Cell<u64>,
}

impl Pipe for Counter {
    type In = ();
    type Out = u64;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<u64, Infallible>> {
        let value = self.next.get();
        self.next.set(value + 1);
        async move { Ok(value) }
    }
}

/// transform: `u64 -> u64`. The general map — `In` and `Out` happen to share
/// a type here, but the value changes; this is the shape most pipes take.
struct Double;

impl Pipe for Double {
    type In = u64;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(input * 2) }
    }
}

/// observe: `u64 -> u64`, value unchanged. `In = Out` down to the identity —
/// the call exists for its side effect (counting), not its return value.
struct Echo {
    calls: Cell<u64>,
}

impl Pipe for Echo {
    type In = u64;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        self.calls.set(self.calls.get() + 1);
        println!("  echo: observed {input}, call #{}", self.calls.get());
        async move { Ok(input) }
    }
}

/// sink: `u64 -> ()`. Consumes, produces nothing — `Out = ()` is the type
/// system's way of saying "this is the end of the line".
struct Print;

impl Pipe for Print {
    type In = u64;
    type Out = ();
    type Err = Infallible;

    fn call(&self, input: u64) -> impl Future<Output = Result<(), Infallible>> {
        async move {
            println!("  sink: final value {input}");
            Ok(())
        }
    }
}

async fn call_pipe<PipeImpl: Pipe<Err = Infallible>>(
    pipe: &PipeImpl,
    input: PipeImpl::In,
) -> PipeImpl::Out {
    pipe.call(input)
        .await
        .expect("Err = Infallible, so this can never fail")
}

// `Cell`-backed pipes are !Send; current-thread tokio drives the body
// in-place, so it never needs `Send + 'static`.
#[proxima::main(runtime = "tokio")]
async fn main() {
    let counter = Counter { next: Cell::new(0) };
    let double = Double;
    let echo = Echo {
        calls: Cell::new(0),
    };
    let print = Print;

    for round in 0..3 {
        println!("--- round {round}: one Pipe trait, four roles chosen by type ---");

        let sourced = call_pipe(&counter, ()).await;
        println!("source    (In=(),    Out=u64):  () -> {sourced}");

        let transformed = call_pipe(&double, sourced).await;
        println!("transform (In=u64,   Out=u64):  {sourced} -> {transformed}");

        let observed = call_pipe(&echo, transformed).await;
        println!("observe   (In=Out=u64):         {transformed} -> {observed} (unchanged)");

        call_pipe(&print, observed).await;
    }
}
