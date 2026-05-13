//! `delivery` — at-most-once, at-least-once, exactly-once. None of them is a
//! built-in mode. Each is a small pipeline assembled from the same two
//! choices: how many times does the sender try, and does the sink remember
//! what it has already seen.
//!
//! - **at-most-once** — send through the wire once. If the wire drops it,
//!   it's gone; nothing retries. Fire-and-forget.
//! - **at-least-once** — keep sending until the wire's ack is observed,
//!   giving up only when an attempt budget is spent. The wire can deliver a
//!   message and still lose the ack on the way back, so the sender retries a
//!   message the sink already has: a duplicate.
//! - **exactly-once** — the SAME retry-until-ack loop as at-least-once, plus
//!   one more stage: the sink remembers every id it has already recorded and
//!   rejects a repeat instead of counting it twice.
//!
//! The retry-until-ack loop's give-up boundary is a [`Signal`] fired by an
//! attempt-budget check — the same fire-then-checkpoint shape `deadline`
//! uses to fire a `Signal` from a clock, read the same way `cancellation`'s
//! cooperative mode reads `Signal::is_fired()` before doing another unit of
//! work. The wire's loss/ack-loss behavior is a fixed, per-message script
//! rather than a shared `BoundedQueue` (`backpressure`'s primitive): a
//! queue's `FailMode` explains why an item is dropped, but not why an ack is
//! dropped after the item already landed — that is its own event, so this
//! rung composes a minimal one instead of forcing the queue to say something
//! it doesn't model.
//!
//! Builds on: backpressure, cancellation.
//!
//! Run: `cargo run --example delivery`

use core::cell::Cell;
use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::Infallible;
use std::rc::Rc;

use proxima_core::signal::Signal;
use proxima_primitives::pipe::Pipe;

/// What the wire did with one send attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireEvent {
    /// The item never reached the sink; nothing was recorded.
    Dropped,
    /// The item reached the sink, but the ack back to the sender was lost —
    /// the sender sees a failure and, under a retry strategy, sends again
    /// even though the sink already has it.
    DeliveredAckLost,
    /// The item reached the sink and the sender observed the ack.
    DeliveredAckOk,
}

/// The result of one send attempt for one message.
#[derive(Clone, Copy, Debug)]
struct SendOutcome {
    id: u32,
    event: WireEvent,
}

impl SendOutcome {
    fn is_success(self) -> bool {
        self.event == WireEvent::DeliveredAckOk
    }
}

/// Where a landed item is recorded — the only thing that differs between
/// at-least-once and exactly-once. `Raw` records every landing, duplicates
/// included. `Dedup` keeps a set of ids already seen and counts a repeat as
/// rejected instead of recording it again.
#[derive(Clone)]
enum SinkLedger {
    Raw(Rc<RefCell<Vec<u32>>>),
    Dedup(Rc<RefCell<HashSet<u32>>>, Rc<Cell<u32>>),
}

impl SinkLedger {
    fn record(&self, id: u32) {
        match self {
            SinkLedger::Raw(landed) => landed.borrow_mut().push(id),
            SinkLedger::Dedup(seen, rejected) => {
                if !seen.borrow_mut().insert(id) {
                    rejected.set(rejected.get() + 1);
                }
            }
        }
    }
}

/// A `Pipe` over an unreliable wire: each call consumes the next scripted
/// [`WireEvent`] for this message and, if it landed, records it in the
/// shared ledger. `Err = Infallible` — the wire never errors, it just may
/// not deliver; that failure is expressed in `Out`, not `Err`.
struct UnreliableSink {
    id: u32,
    script: &'static [WireEvent],
    attempt: Cell<usize>,
    ledger: SinkLedger,
}

impl UnreliableSink {
    fn new(id: u32, script: &'static [WireEvent], ledger: SinkLedger) -> Self {
        Self {
            id,
            script,
            attempt: Cell::new(0),
            ledger,
        }
    }
}

impl Pipe for UnreliableSink {
    type In = ();
    type Out = SendOutcome;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<SendOutcome, Infallible>> {
        let index = self.attempt.get();
        self.attempt.set(index + 1);
        // every script in MESSAGE_SCRIPTS terminates in DeliveredAckOk well
        // inside its own length, so this index is always in bounds.
        let event = self.script[index];
        if matches!(
            event,
            WireEvent::DeliveredAckLost | WireEvent::DeliveredAckOk
        ) {
            self.ledger.record(self.id);
        }
        let outcome = SendOutcome { id: self.id, event };
        async move { Ok(outcome) }
    }
}

/// Polls a future once against a no-op waker — every future here resolves
/// on its first poll, so this is a legitimate `block_on` with no executor
/// and no sleep, the same shape `cancellation` and `retry` use.
fn block_on_ready<Fut: Future>(future: Fut) -> Fut::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("delivery example futures resolve on first poll"),
    }
}

/// How many attempts a sender is willing to spend on one message before it
/// gives up — the retry-until-ack loop's boundary condition.
struct AttemptBudget {
    max_attempts: u32,
}

impl AttemptBudget {
    fn exhausted(&self, attempts_made: u32) -> bool {
        attempts_made >= self.max_attempts
    }
}

/// Retry-until-ack: call the sink, stop on success. Otherwise fire the
/// give-up `Signal` once the budget is spent, and stop at the next
/// checkpoint — the same fire-then-check shape `deadline` uses, aimed at a
/// send loop instead of arbitrary work.
fn send_until_acked(sink: &UnreliableSink, budget: &AttemptBudget) -> (SendOutcome, u32) {
    let give_up = Signal::new();
    let mut attempts = 0_u32;
    loop {
        attempts += 1;
        let outcome =
            block_on_ready(sink.call(())).unwrap_or_else(|never: Infallible| match never {});
        if outcome.is_success() {
            return (outcome, attempts);
        }
        if budget.exhausted(attempts) {
            give_up.fire();
        }
        if give_up.is_fired() {
            return (outcome, attempts);
        }
    }
}

/// Five messages, each with its own wire script. Every script terminates in
/// `DeliveredAckOk` so at-least-once/exactly-once never lose one — the only
/// thing under test is how many times each lands and who notices.
const MESSAGE_SCRIPTS: [(u32, &[WireEvent]); 5] = [
    (1, &[WireEvent::Dropped, WireEvent::DeliveredAckOk]),
    (2, &[WireEvent::DeliveredAckOk]),
    (3, &[WireEvent::DeliveredAckLost, WireEvent::DeliveredAckOk]),
    (
        4,
        &[
            WireEvent::Dropped,
            WireEvent::DeliveredAckLost,
            WireEvent::DeliveredAckOk,
        ],
    ),
    (5, &[WireEvent::DeliveredAckOk]),
];

// AT-MOST-ONCE: one attempt, no retry. A first-attempt drop is a permanent
// loss; a first-attempt ack-loss still counts as delivered, because nobody
// here is watching for the ack.
fn at_most_once_strategy() {
    let landed = Rc::new(RefCell::new(Vec::new()));
    for (id, script) in MESSAGE_SCRIPTS {
        let sink = UnreliableSink::new(id, script, SinkLedger::Raw(Rc::clone(&landed)));
        block_on_ready(sink.call(())).unwrap_or_else(|never: Infallible| match never {});
    }

    let landed = landed.borrow();
    let delivered: HashSet<u32> = landed.iter().copied().collect();
    let lost = MESSAGE_SCRIPTS.len() - delivered.len();
    let duplicates = landed.len() - delivered.len();

    assert!(
        lost > 0,
        "the unreliable wire must cost at-most-once a message to prove the guarantee is real"
    );
    assert_eq!(duplicates, 0, "one attempt per message can never duplicate");

    println!(
        "at-most-once:  sent {}  delivered {}  lost {lost}  duplicates {duplicates}",
        MESSAGE_SCRIPTS.len(),
        delivered.len()
    );
}

// AT-LEAST-ONCE: retry until ack. Never loses a message, but an ack-lost
// attempt plus its follow-up both land at the sink — a duplicate.
fn at_least_once_strategy() {
    let landed = Rc::new(RefCell::new(Vec::new()));
    let budget = AttemptBudget { max_attempts: 5 };
    for (id, script) in MESSAGE_SCRIPTS {
        let sink = UnreliableSink::new(id, script, SinkLedger::Raw(Rc::clone(&landed)));
        let (outcome, _attempts) = send_until_acked(&sink, &budget);
        assert!(
            outcome.is_success() && outcome.id == id,
            "message {id} must eventually be acked under its own id"
        );
    }

    let landed = landed.borrow();
    let delivered: HashSet<u32> = landed.iter().copied().collect();
    let duplicates = landed.len() - delivered.len();

    assert_eq!(
        delivered.len(),
        MESSAGE_SCRIPTS.len(),
        "every message landed at least once"
    );
    assert!(
        duplicates > 0,
        "retry-until-ack must show a duplicate to prove the tradeoff is real"
    );

    println!(
        "at-least-once: sent {}  delivered {} (raw {})  lost 0  duplicates {duplicates}",
        MESSAGE_SCRIPTS.len(),
        delivered.len(),
        landed.len()
    );
}

// EXACTLY-ONCE: the same retry-until-ack loop, plus a dedup key at the sink.
// What the sink ends up with must equal what was sent, and every duplicate
// the retry loop produced must have been rejected, not counted.
fn exactly_once_strategy() {
    let seen = Rc::new(RefCell::new(HashSet::new()));
    let rejected = Rc::new(Cell::new(0_u32));
    let budget = AttemptBudget { max_attempts: 5 };
    for (id, script) in MESSAGE_SCRIPTS {
        let ledger = SinkLedger::Dedup(Rc::clone(&seen), Rc::clone(&rejected));
        let sink = UnreliableSink::new(id, script, ledger);
        let (outcome, _attempts) = send_until_acked(&sink, &budget);
        assert!(
            outcome.is_success() && outcome.id == id,
            "message {id} must eventually be acked under its own id"
        );
    }

    let delivered: HashSet<u32> = seen.borrow().clone();
    let sent: HashSet<u32> = MESSAGE_SCRIPTS.iter().map(|&(id, _)| id).collect();

    assert_eq!(
        delivered, sent,
        "delivered set equals sent set: nothing lost, nothing double-counted"
    );
    assert!(
        rejected.get() > 0,
        "dedup must have actually rejected a duplicate to prove it ran"
    );

    println!(
        "exactly-once:  sent {}  delivered {}  lost 0  duplicates rejected {}",
        MESSAGE_SCRIPTS.len(),
        delivered.len(),
        rejected.get()
    );
}

fn main() {
    println!("delivery: at-most-once / at-least-once / exactly-once, composed\n");

    at_most_once_strategy();
    at_least_once_strategy();
    exactly_once_strategy();

    println!("\nsame wire, same messages: only the per-stage strategy choice changed the outcome.");
}
