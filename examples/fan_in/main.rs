//! `FanIn<S, Strategy, const N>`: `N` sources merged into one item stream, round-robin
//! fair, pull-based. A call only ever returns an item from a source that has
//! one ready *right now*; a source with nothing ready is skipped for that
//! pass, not treated as failed or drained. No gate, no priority — the plain
//! merge. `gate`'s BALANCE shape wraps each source in a `DemandGate` on top
//! of exactly this.
//!
//! `FanIn` is itself a `Pipe`/`UnpinPipe` (source form: `In = ()`); its
//! merged sources are `UnpinPipe<In = (), Err = Exhausted>` too — each
//! merged source is backed by a real `Pipe<In = (), Out = u32>` (the source
//! role `transform` taught: no input to consume, produces on demand).
//! `FanIn` never inspects what produced an item, only whether the source's
//! `call` has one ready.
//!
//! Run: `cargo run --example fan_in`

use core::cell::{Cell, RefCell};
use core::convert::Infallible;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::collections::VecDeque;

use proxima_core::markers::DropSafe;
use proxima_primitives::pipe::{Exhausted, FanIn, FanInStrategy, Pipe, Select, UnpinPipe};

fn main() {
    println!("fan-in: merge 3 upstreams, pull only the ready");

    let orders = Upstream::new("orders", [false, true, true], Counter::new(1, 1));
    let payments = Upstream::new("payments", [false, true, true], Counter::new(10, 10));
    let shipping = Upstream::new("shipping", [false, true], Counter::new(100, 100));

    let drained = drain_merged(FanIn::new([orders, payments, shipping], Select::RoundRobin));

    let total = drained.len();
    let orders_count = drained
        .iter()
        .filter(|(label, _)| *label == "orders")
        .count();
    let payments_count = drained
        .iter()
        .filter(|(label, _)| *label == "payments")
        .count();
    let shipping_count = drained
        .iter()
        .filter(|(label, _)| *label == "shipping")
        .count();
    println!(
        "drained {total} items total: {orders_count} orders, {payments_count} payments, {shipping_count} shipping"
    );

    assert_eq!(
        drained.len(),
        5,
        "every ready item from every upstream is drained"
    );
    assert_eq!(orders_count, 2, "orders' whole queue drained");
    assert_eq!(payments_count, 2, "payments' whole queue drained");
    assert_eq!(shipping_count, 1, "shipping's whole queue drained");
    assert_eq!(
        drained,
        vec![
            ("orders", 1),
            ("payments", 10),
            ("shipping", 100),
            ("orders", 2),
            ("payments", 20),
        ],
        "round-robin cursor advances past whoever just emitted, so drain order is deterministic"
    );
}

// drives a fan-in to completion, printing each call's outcome as it goes.
fn drain_merged<S, Strategy, const N: usize>(fan_in: FanIn<S, Strategy, N>) -> Vec<(&'static str, u32)>
where
    S: UnpinPipe<In = (), Out = (&'static str, u32), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut drained = Vec::new();
    let mut poll_number = 0_u32;

    loop {
        poll_number += 1;
        let mut call = Pipe::call(&fan_in, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(item)) => {
                println!("poll {poll_number}: drained {item:?}");
                drained.push(item);
            }
            Poll::Ready(Err(Exhausted)) => {
                println!("poll {poll_number}: all upstreams drained");
                break;
            }
            Poll::Pending => {
                println!("poll {poll_number}: nothing ready yet, all live upstreams pending");
            }
        }
    }

    drained
}

// every Counter future resolves on its first poll (no real I/O), so a
// one-shot poll is a legitimate block_on for driving it from call().
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("fan_in example futures resolve on first poll"),
    }
}

// ── the inner Pipe: the source role transform taught ────────────────────────

// source: `Pipe<In = (), Out = u32>` — no input to consume, so each call
// produces the next value in an arithmetic sequence (`start`, `start+step`, ...).
struct Counter {
    next: Cell<u32>,
    step: u32,
}

impl Counter {
    fn new(start: u32, step: u32) -> Self {
        Self {
            next: Cell::new(start),
            step,
        }
    }
}

impl Pipe for Counter {
    type In = ();
    type Out = u32;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<u32, Infallible>> {
        let value = self.next.get();
        self.next.set(value + self.step);
        async move { Ok(value) }
    }
}

// ── the fan-in source: readiness schedule over a Pipe ───────────────────────

// resolves immediately to a fixed outcome — the hand-written poll struct an
// `UnpinPipe::call` needs in place of an `!Unpin` async block.
struct UpstreamCall(Poll<Result<(&'static str, u32), Exhausted>>);

impl Future for UpstreamCall {
    type Output = Result<(&'static str, u32), Exhausted>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0
    }
}

// an `UnpinPipe` source that plays back a fixed readiness schedule, calling
// `inner` for the item on every `true`: this is what fronts a real upstream
// (a queue, a socket, a `Pipe`'s output) with pull-based readiness for
// `FanIn`. `schedule` is a `RefCell` because `call` takes `&self`.
struct Upstream<P> {
    label: &'static str,
    schedule: RefCell<VecDeque<bool>>,
    inner: P,
}

impl<P> Upstream<P> {
    fn new(label: &'static str, schedule: impl IntoIterator<Item = bool>, inner: P) -> Self {
        Self {
            label,
            schedule: RefCell::new(schedule.into_iter().collect()),
            inner,
        }
    }
}

impl<P> DropSafe for Upstream<P> {}

impl<P> UnpinPipe for Upstream<P>
where
    P: Pipe<In = (), Out = u32, Err = Infallible>,
{
    type In = ();
    type Out = (&'static str, u32);
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        match self.schedule.borrow_mut().pop_front() {
            Some(true) => {
                let value = block_on_ready(self.inner.call(()))
                    .expect("Err = Infallible, so this can never fail");
                UpstreamCall(Poll::Ready(Ok((self.label, value))))
            }
            Some(false) => UpstreamCall(Poll::Pending),
            None => UpstreamCall(Poll::Ready(Err(Exhausted))),
        }
    }
}
