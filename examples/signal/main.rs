#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `signal` — fully-async fire-once completion: no polls, no waits, no sleeps.
//!
//! `filter` teaches the terminal condition as a decision pipe (`In ->
//! Result<In, Err>`); `gate` teaches readiness as composition instead of a
//! `poll_ready` method. This example is the completion sibling of `gate`:
//! instead of "is it ready right now", the question is "has it finished
//! yet" — answered once, for good.
//!
//! The pattern, four stages: a producer drives a stream of items through an
//! **observe** tap (watch each item go by), into a **filter** that recognises
//! the terminal item and admits only it, whose inner pipe **fires** a
//! [`proxima_core::signal::Signal`] — and only the terminal item ever reaches
//! it, everything else is rejected before the fire pipe is called. A consumer
//! task, parked on `signal.fired()` since before the producer started, wakes
//! exactly once. No loop, no timeout, no re-check on a timer: the consumer
//! task is polled twice total — once to register, once to resolve — proven
//! inline by a poll-counting wrapper around the await.
//!
//! Run: `cargo run --example signal`

use core::convert::Infallible;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{Context, Poll};
use std::sync::Arc;

use proxima_core::signal::Signal;
use proxima_primitives::pipe::{Pipe, PipeExt, SendPipe};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("signal: fire-once completion, no polls\n");

    let signal = Signal::new();
    let consumer_polls = Arc::new(AtomicUsize::new(0));

    // the consumer parks on signal.fired() before the producer has emitted a
    // single item — proving it is genuinely woken, not polled to completion.
    let consumer = tokio::spawn(consumer_task(signal.clone(), Arc::clone(&consumer_polls)));
    tokio::task::yield_now().await;

    let watch = Watch::new();
    let stack = is_terminal.and_then(FireOnTerminal::new(signal.clone()));

    let stream = [
        StreamItem {
            seq: 1,
            terminal: false,
        },
        StreamItem {
            seq: 2,
            terminal: false,
        },
        StreamItem {
            seq: 3,
            terminal: false,
        },
        StreamItem {
            seq: 4,
            terminal: true,
        },
    ];

    println!("--- producer: observe -> filter(terminal) -> fire ---");
    for item in stream {
        let watched = SendPipe::call(&watch, item)
            .await
            .expect("observe never errors");
        // both channels of the decision carry `Observed` — admit reaches
        // `FireOnTerminal` and comes back `Ok(Watched)`; reject never
        // reaches it and comes back `Err(Skipped)` directly from the gate.
        let outcome = match SendPipe::call(&stack, watched).await {
            Ok(observed) | Err(observed) => observed,
        };
        match outcome {
            Observed::Skipped { seq } => {
                println!("  item {seq}: observed, not terminal, dropped before the fire pipe")
            }
            Observed::Watched { seq } => println!("  item {seq}: terminal -> Signal::fire()"),
        }
    }

    consumer.await.expect("consumer task");

    let polls = consumer_polls.load(Ordering::Relaxed);
    println!("\n--- proof: the consumer never polled in a loop ---");
    println!("consumer's await point was polled {polls} time(s): once to park, once to wake");
    assert_eq!(
        polls, 2,
        "park (Pending, registers a waker) + wake (Ready) is the whole story; \
         a busy-poll loop would have called poll() far more than twice"
    );
    assert!(signal.is_fired(), "the level is sticky: it stays fired");

    // a late observer (subscribing after the fire) resolves on its very first
    // poll — the sticky level, not a fresh wait, is what makes that legal.
    let late_polls = Arc::new(AtomicUsize::new(0));
    CountingFuture {
        inner: signal.fired(),
        polls: Arc::clone(&late_polls),
    }
    .await;
    let late = late_polls.load(Ordering::Relaxed);
    println!("late observer (subscribes after fire): resolved after {late} poll");
    assert_eq!(
        late, 1,
        "an already-fired signal resolves on the first poll, no registration wait"
    );
}

async fn consumer_task(signal: Signal, polls: Arc<AtomicUsize>) {
    println!("consumer: parked on signal.fired() (no poll loop, no timeout)");
    CountingFuture {
        inner: signal.fired(),
        polls,
    }
    .await;
    println!("consumer: woken by fire() -> proceeding");
}

// wraps any Unpin future and counts how many times it is polled — the proof
// instrument for "parked, not spinning". Fired is Unpin (its fields are
// Arc/Option<Arc<..>>), so this needs no unsafe pin projection.
struct CountingFuture<InnerFuture> {
    inner: InnerFuture,
    polls: Arc<AtomicUsize>,
}

impl<InnerFuture: Future<Output = ()> + Unpin> Future for CountingFuture<InnerFuture> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        Pin::new(&mut self.inner).poll(cx)
    }
}

// ── the stream ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct StreamItem {
    seq: u64,
    terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Observed {
    Watched { seq: u64 },
    Skipped { seq: u64 },
}

// ── observe: a tap, side-effect only, In = Out ──────────────────────────────

#[derive(Clone)]
struct Watch {
    seen: Arc<AtomicUsize>,
}

impl Watch {
    fn new() -> Self {
        Self {
            seen: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl SendPipe for Watch {
    type In = StreamItem;
    type Out = StreamItem;
    type Err = Infallible;

    fn call(
        &self,
        item: StreamItem,
    ) -> impl Future<Output = Result<StreamItem, Infallible>> + Send {
        let count = self.seen.fetch_add(1, Ordering::Relaxed) + 1;
        println!(
            "  watch: item {} passing by (seen {count} so far)",
            item.seq
        );
        async move { Ok(item) }
    }
}

// ── filter: the terminal condition, a decision pipe ─────────────────────────

// `In = Out = StreamItem` (the item survives on admit), `Err = Observed` (the
// rejection reason on reject) — reusing `Observed` as the reject payload
// needs no separate sentinel type, and it is what makes
// `is_terminal.and_then(FireOnTerminal)` type-check. Stateless, so
// `#[proxima::piped]` writes the `SendPipe` impl.
#[proxima::piped(send)]
async fn is_terminal(item: StreamItem) -> Result<StreamItem, Observed> {
    if item.terminal {
        Ok(item)
    } else {
        Err(Observed::Skipped { seq: item.seq })
    }
}

// ── fire: the inner pipe the terminal item alone reaches ───────────────────

#[derive(Clone)]
struct FireOnTerminal {
    signal: Signal,
}

impl FireOnTerminal {
    fn new(signal: Signal) -> Self {
        Self { signal }
    }
}

impl SendPipe for FireOnTerminal {
    type In = StreamItem;
    type Out = Observed;
    type Err = Observed;

    fn call(&self, item: StreamItem) -> impl Future<Output = Result<Observed, Observed>> + Send {
        self.signal.fire();
        async move { Ok(Observed::Watched { seq: item.seq }) }
    }
}

// base-tier mirror, delegating straight through — every pipe implements the
// root `Pipe` too, which is what lets `PipeExt::and_then` reach it.
impl Pipe for FireOnTerminal {
    type In = StreamItem;
    type Out = Observed;
    type Err = Observed;

    fn call(&self, item: StreamItem) -> impl Future<Output = Result<Observed, Observed>> {
        SendPipe::call(self, item)
    }
}
