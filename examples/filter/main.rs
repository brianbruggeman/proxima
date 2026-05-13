#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `filter` — pass or drop by a decision pipe.
//!
//! `transform` teaches you to write one `Pipe`: `In -> Out`. A filter is the
//! same shape turned into a gate: a decision pipe, `In -> Result<In, Err>`
//! (`Ok` = admit, the item survives; `Err` = reject), composed in front of
//! the inner pipe with `.and_then(inner)`. `AndThen`'s own `?` already
//! short-circuits on a first-stage `Err` — only the admitted items ever
//! reach the inner pipe.
//!
//! Run: `cargo run --example filter`

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_primitives::pipe::SendPipe;

fn main() {
    println!("filter: pass or drop by a predicate\n");

    let (ledger, calls) = Ledger::new();
    let stack = MinAmount {
        threshold_cents: 2_000,
    }
    .and_then(ledger);

    let orders = [
        Order {
            id: 1,
            amount_cents: 1_200,
        },
        Order {
            id: 2,
            amount_cents: 4_500,
        },
        Order {
            id: 3,
            amount_cents: 9_900,
        },
        Order {
            id: 4,
            amount_cents: 300,
        },
        Order {
            id: 5,
            amount_cents: 5_000,
        },
    ];

    let mut processed = Vec::new();
    let mut dropped = Vec::new();

    for order in orders {
        // the decision's `Ok`/`Err` channels both carry `Outcome` — admit
        // reaches the ledger and comes back `Ok(Processed)`; reject never
        // reaches it and comes back `Err(Dropped)` directly from the gate.
        let outcome = match block_on_ready(SendPipe::call(&stack, order)) {
            Ok(outcome) | Err(outcome) => outcome,
        };
        match outcome {
            Outcome::Processed { id } => {
                println!("order {id}: passed");
                processed.push(id);
            }
            Outcome::Dropped { id } => {
                println!("order {id}: dropped (below threshold)");
                dropped.push(id);
            }
        }
    }

    assert_eq!(
        processed,
        vec![2, 3, 5],
        "orders at or above the threshold reach the ledger"
    );
    assert_eq!(
        dropped,
        vec![1, 4],
        "orders below the threshold are dropped, never reach the ledger"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        processed.len(),
        "the ledger's own call counter proves the gate runs before the inner pipe: \
         dropped orders never increment it"
    );

    println!("\nprocessed: {processed:?}");
    println!("dropped:   {dropped:?}");
    println!(
        "ledger called {} times for {} orders ({} dropped before reaching it)",
        calls.load(Ordering::Relaxed),
        orders.len(),
        dropped.len()
    );
}

// every future here resolves on its first poll (no real I/O), so a one-shot
// poll is a legitimate `block_on` — no executor dependency needed to prove
// the pattern.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("filter example futures resolve on first poll"),
    }
}

// ── the inner Pipe: what transform taught you to write ─────────────────────

#[derive(Clone, Copy, Debug)]
struct Order {
    id: u32,
    amount_cents: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Processed { id: u32 },
    Dropped { id: u32 },
}

#[derive(Clone)]
struct Ledger {
    calls: Arc<AtomicUsize>,
}

impl Ledger {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                calls: calls.clone(),
            },
            calls,
        )
    }
}

// the ledger's `Err` is `Outcome`, not `Infallible` — it must match the
// gate's `Err` so `MinAmount.and_then(ledger)` type-checks (`AndThen`
// requires `Second::Err: From<First::Err>`; here both sides are the same
// type, so the identity `From` applies).
impl SendPipe for Ledger {
    type In = Order;
    type Out = Outcome;
    type Err = Outcome;

    fn call(&self, order: Order) -> impl Future<Output = Result<Outcome, Outcome>> + Send {
        self.calls.fetch_add(1, Ordering::Relaxed);
        async move {
            println!(
                "  ledger processing order {} (${:.2})",
                order.id,
                f64::from(order.amount_cents) / 100.0
            );
            Ok(Outcome::Processed { id: order.id })
        }
    }
}

// ── the gate: the whole filter surface, a decision pipe ────────────────────

#[derive(Clone)]
struct MinAmount {
    threshold_cents: u32,
}

// `In = Order`, `Out = Order` (the item survives on admit), `Err = Outcome`
// (the rejection reason on reject) — reusing `Outcome` as the reject payload
// needs no separate sentinel type.
impl SendPipe for MinAmount {
    type In = Order;
    type Out = Order;
    type Err = Outcome;

    fn call(&self, order: Order) -> impl Future<Output = Result<Order, Outcome>> + Send {
        let admits = order.amount_cents >= self.threshold_cents;
        async move {
            if admits {
                Ok(order)
            } else {
                Err(Outcome::Dropped { id: order.id })
            }
        }
    }
}
