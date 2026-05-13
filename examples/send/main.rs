//! One pipe, two flavors. `Pipe` is the root form: typed `In -> Out`, no
//! `Send` bound anywhere, so it can BORROW — cheap, zero-copy, no allocation.
//! `SendPipe` is the additive constraint you climb to only when a pipe must
//! leave the thread that made it: it owns its data and is `Send + Sync +
//! 'static`, top to bottom.
//!
//! The bounds form a ladder, each rung strictly additive:
//!
//!   `Pipe`            -- borrow, `!Send`, the permissive root (this file's `Borrows`)
//!   `Pipe + 'static`  -- own it, so it can be erased into a `DynPipe`/spawned
//!   `SendPipe`        -- `+ Send + Sync`, so it can cross a core (this file's `Summarize`)
//!
//! `SendPipe` is a STANDALONE trait, not `Pipe + Send` — on stable Rust an
//! RPITIT future's `Send`-ness cannot be strengthened by a subtrait bound, so
//! `primitives.rs` writes the whole contract twice with the `Send` future
//! baked into the second copy's `call`. A blanket impl bridges them where a
//! type implements both; that bridge is not needed here.
//!
//! Run: `cargo run --example send`

use core::cell::Cell;
use core::convert::Infallible;
use core::future::Future;

use proxima_primitives::pipe::{Pipe, SendPipe};

/// State a borrowing pipe can reach into. `Cell` is not `Sync`, so `&Ledger`
/// is not `Send` either — a pipe that borrows this is stuck on this thread by
/// construction, not by convention.
struct Ledger {
    total: Cell<u64>,
}

/// The root form: holds a borrow, never an owned copy. `Pipe` puts no `Send`
/// or `'static` bound on `Self`, so this compiles and runs even though it can
/// never outlive the stack frame that holds `ledger`.
struct Borrows<'a>(&'a Ledger);

impl Pipe for Borrows<'_> {
    type In = u64;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        let total = self.0.total.get() + input;
        self.0.total.set(total);
        async move { Ok(total) }
    }
}

/// Owns its data outright and satisfies `Send + Sync + 'static` on `Self` —
/// the pipe itself, not just its future, has to clear that bar before
/// `SendPipe` is even implementable.
struct Summarize {
    label: String,
}

impl SendPipe for Summarize {
    type In = Vec<u64>;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: Vec<u64>) -> impl Future<Output = Result<u64, Infallible>> + Send {
        let label = self.label.clone();
        async move {
            let sum: u64 = input.iter().sum();
            println!(
                "  {label}: summed {} values on a worker thread -> {sum}",
                input.len()
            );
            Ok(sum)
        }
    }
}

async fn call_pipe<PipeImpl: Pipe<Err = Infallible>>(
    pipe: &PipeImpl,
    input: PipeImpl::In,
) -> PipeImpl::Out {
    // Err = Infallible has zero variants, so matching it out is exhaustive —
    // no panic path exists, unlike `.expect()` on a Result that could fail.
    match pipe.call(input).await {
        Ok(output) => output,
        Err(never) => match never {},
    }
}

/// Spawn a `SendPipe` onto a real OS thread and join the result — the proof
/// that `Send + Sync + 'static` is load-bearing here, not decorative. The
/// worker thread drives the future on its own proxima-backed executor
/// (`run_tokio`), not proxima's main runtime — that is the point: a
/// `SendPipe` must be pollable from ANY thread, not just the one it was
/// built on.
// deliberately names the tokio backend to prove any-thread pollability; the
// edge-driver guardrail is not for this demonstration.
#[allow(clippy::disallowed_methods)]
fn call_send_pipe_on_thread<PipeImpl>(pipe: PipeImpl, input: PipeImpl::In) -> PipeImpl::Out
where
    PipeImpl: SendPipe<Err = Infallible> + 'static,
    PipeImpl::In: Send + 'static,
    PipeImpl::Out: Send + 'static,
{
    let handle = std::thread::spawn(move || {
        match proxima::runtime::run_tokio(false, None, pipe.call(input)) {
            Ok(result) => result,
            Err(error) => panic!("worker thread failed to build its runtime: {error}"),
        }
    });
    // join() fails only if the worker panicked; resume that panic on this
    // thread instead of swallowing or unwrap()-ing it away.
    match handle
        .join()
        .unwrap_or_else(|panic_payload| std::panic::resume_unwind(panic_payload))
    {
        Ok(output) => output,
        Err(never) => match never {},
    }
}

// `Borrows` holds a `Cell`-backed borrow, so this body is !Send; current-thread
// tokio drives it in-place without demanding `Send + 'static`.
#[proxima::main(runtime = "tokio")]
async fn main() {
    println!("--- local: Pipe borrows, stays on this thread, no allocation ---");
    let ledger = Ledger {
        total: Cell::new(0),
    };
    let borrows = Borrows(&ledger);
    for input in [10, 20, 30] {
        let total = call_pipe(&borrows, input).await;
        println!("  Borrows::call({input}) -> running total {total}");
    }
    println!(
        "  final ledger total, read straight through the borrow: {}",
        ledger.total.get()
    );

    println!();
    println!("--- cross-thread: SendPipe owns its data, spawned, joined ---");
    let summarize = Summarize {
        label: "worker".to_string(),
    };
    let values = vec![1, 2, 3, 4, 5];
    println!("  main thread: handing {values:?} to a spawned worker");
    let sum = call_send_pipe_on_thread(summarize, values);
    println!("  main thread: joined the worker, sum = {sum}");

    // The ladder only climbs, never descends: `Borrows<'_>` cannot be erased
    // into a `'static` `DynPipe` or spawned like `Summarize` was above. Both
    // moves need `Self: 'static`; `Borrows<'a>` is generic over a non-'static
    // lifetime by construction, so neither line below compiles:
    //
    //   proxima_primitives::pipe::alloc_tier::into_local_handle(borrows);
    //   // error[E0310]: the parameter type `Borrows<'a>` may not live long
    //   // enough -- `into_local_handle` requires `P: Pipe + 'static`
    //
    //   std::thread::spawn(move || call_pipe(&borrows, 1));
    //   // error: closure may outlive the current function, but it borrows
    //   // `ledger` -- `thread::spawn`'s closure bound requires `'static`
    //
    // Borrowing is the permissive default: zero-copy, no allocation, no
    // thread hop. `'static` (own it, erase it) and `Send` (cross a core) are
    // costs you climb into only when the use case demands them.
    println!();
    println!("--- the ladder (additive, one-directional) ---");
    println!("  Pipe            (borrow, !Send)  Borrows above -- the root form");
    println!("  Pipe + 'static  (own it, erase)   what into_local_handle/DynPipe require");
    println!("  SendPipe        (+Send, +Sync)    Summarize above -- crosses a thread");
}
