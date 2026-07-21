#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The "gate pipe" pattern: readiness/backpressure as composition, not a
//! `poll_ready` trait method. `proxima_primitives::pipe::SendPipe` has no such
//! method — every gate shape below is composed from existing primitives
//! instead of being baked into the trait.
//!
//! Three shapes, one gate vocabulary (`DemandGate` / `AtomicGate`):
//!
//! 1. SHED    — a decision pipe reading a `DemandGate`, composed in front of
//!    the inner pipe with `.and_then(inner)`: reject (shed) per call while
//!    closed, admit while open.
//! 2. WAIT    — `Demand<S, G>`: the wrapped pipe goes dormant (a no-op `Ok`,
//!    no inner await) while closed, resumes once armed.
//! 3. BALANCE — `FanIn` over gated `UnpinPipe` sources: a closed gate makes
//!    its source `Pending` for that call, so the round-robin merge skips it
//!    and drains whichever backend is ready — `poll_ready` across backends,
//!    without a `poll_ready` method anywhere.
//!
//! Run: `cargo run --example gate`

use core::convert::Infallible;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_core::markers::DropSafe;
use proxima_macros::piped;
use proxima_primitives::pipe::{
    Admission, AlwaysArmed, AtomicGate, Demand, DemandGate, DropReason,
};
use proxima_primitives::pipe::{Exhausted, FanIn, Pipe, PipeExt, Select, SendPipe, UnpinPipe};

fn main() {
    println!("shed: a filter reading the gate");
    run_shed_gate();

    println!("\nwait: dormant while the gate is closed");
    run_wait_gate();

    println!("\nbalance: the merge skips a gated backend, drains the ready one");
    run_balance_gate();
}

// ── shared driver ───────────────────────────────────────────────────────────

// every future in this example resolves on its first poll (atomic checks
// only, no real I/O), so a one-shot poll is a legitimate `block_on` — no
// executor dependency needed to prove the pattern.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("gate example futures resolve on first poll"),
    }
}

// ── 1. SHED: a decision pipe reading the gate ───────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Job {
    id: u32,
}

// the ledger's `Err` is `Admission`, matching the gate's reject payload — the
// same vocabulary a SinkFront would use to report a shed item — so
// `Gated(gate).and_then(ingest)` type-checks. Stateless, so `#[proxima::piped]`
// writes the `SendPipe` impl.
#[proxima::piped(send)]
async fn ingest(job: Job) -> Result<Admission, Admission> {
    println!("  ingest processing job {}", job.id);
    Ok(Admission::Accepted)
}

// a decision pipe that ignores the item entirely and answers from the gate —
// this is the whole shim that turns a DemandGate into a filter stage: `In ->
// Result<In, Admission>`, admitting the job unchanged or rejecting with the
// same `Admission::Dropped` a SinkFront would report.
struct Gated<G> {
    gate: Arc<G>,
}

impl<G> Gated<G> {
    fn new(gate: G) -> Self {
        Self {
            gate: Arc::new(gate),
        }
    }
}

// manual, not derived: #[derive(Clone)] would add a `G: Clone` bound that
// AtomicGate does not satisfy. Arc<G> clones regardless.
impl<G> Clone for Gated<G> {
    fn clone(&self) -> Self {
        Self {
            gate: Arc::clone(&self.gate),
        }
    }
}

impl<G: DemandGate + Send + Sync + 'static> SendPipe for Gated<G> {
    type In = Job;
    type Out = Job;
    type Err = Admission;

    fn call(&self, job: Job) -> impl Future<Output = Result<Job, Admission>> + Send {
        let armed = self.gate.is_armed();
        async move {
            if armed {
                Ok(job)
            } else {
                Err(Admission::Dropped(DropReason::Refused))
            }
        }
    }
}

// base-tier mirror, delegating straight through — every pipe implements the
// root `Pipe` too, which is what lets `PipeExt::and_then` reach it.
impl<G: DemandGate + Send + Sync + 'static> Pipe for Gated<G> {
    type In = Job;
    type Out = Job;
    type Err = Admission;

    fn call(&self, job: Job) -> impl Future<Output = Result<Job, Admission>> {
        SendPipe::call(self, job)
    }
}

fn run_shed_gate() {
    let (gate, controller) = AtomicGate::pair(true);
    let stack = Gated::new(gate).and_then(ingest);

    let admitted = admission(block_on_ready(SendPipe::call(&stack, Job { id: 1 })));
    print_admission(1, admitted);
    assert_eq!(admitted, Admission::Accepted, "armed gate admits");

    controller.disarm();
    let shed = admission(block_on_ready(SendPipe::call(&stack, Job { id: 2 })));
    print_admission(2, shed);
    assert_eq!(
        shed,
        Admission::Dropped(DropReason::Refused),
        "disarmed gate sheds"
    );

    controller.arm();
    let readmitted = admission(block_on_ready(SendPipe::call(&stack, Job { id: 3 })));
    print_admission(3, readmitted);
    assert_eq!(
        readmitted,
        Admission::Accepted,
        "re-armed gate admits again"
    );
}

// both channels of the decision carry `Admission` — collapse them to the one
// value the caller cares about, same as `filter`'s `Ok(x) | Err(x) => x`.
fn admission(outcome: Result<Admission, Admission>) -> Admission {
    match outcome {
        Ok(admission) | Err(admission) => admission,
    }
}

fn print_admission(id: u32, outcome: Admission) {
    match outcome {
        Admission::Accepted => println!("job {id}: accepted"),
        Admission::Dropped(reason) => println!("job {id}: shed ({reason:?})"),
        Admission::Dormant => println!("job {id}: dormant"),
    }
}

// ── 2. WAIT: Demand<S, G> ───────────────────────────────────────────────────

struct CountingSink {
    calls: Arc<AtomicUsize>,
}

impl SendPipe for CountingSink {
    type In = u32;
    type Out = ();
    type Err = Infallible;

    fn call(&self, _item: u32) -> impl Future<Output = Result<(), Infallible>> + Send {
        let calls = Arc::clone(&self.calls);
        async move {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
}

fn run_wait_gate() {
    let baseline_calls = Arc::new(AtomicUsize::new(0));
    // no gate at all: AlwaysArmed is zero-sized, the optimiser deletes the
    // check. This is the "readiness is composition" baseline — an ungated
    // pipe is not a special case, it is just not wrapped.
    let ungated = Demand::new(
        CountingSink {
            calls: Arc::clone(&baseline_calls),
        },
        AlwaysArmed,
    );
    block_on_ready(SendPipe::call(&ungated, 0)).expect("demand never errors");
    println!(
        "ungated (AlwaysArmed): {} dispatched",
        baseline_calls.load(Ordering::Relaxed)
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let (gate, controller) = AtomicGate::pair(false);
    let production = Demand::new(
        CountingSink {
            calls: Arc::clone(&calls),
        },
        gate,
    );

    for item in 0..3 {
        block_on_ready(SendPipe::call(&production, item)).expect("demand never errors");
    }
    let dormant_count = calls.load(Ordering::Relaxed);
    println!("closed gate: {dormant_count} dispatched (dormant, no-op)");
    assert_eq!(
        dormant_count, 0,
        "a closed gate never reaches the inner pipe"
    );

    controller.arm();
    for item in 3..6 {
        block_on_ready(SendPipe::call(&production, item)).expect("demand never errors");
    }
    let armed_count = calls.load(Ordering::Relaxed);
    println!("armed gate: {armed_count} dispatched (resumed)");
    assert_eq!(armed_count, 3, "an armed gate dispatches every call");

    controller.disarm();
    block_on_ready(SendPipe::call(&production, 6)).expect("demand never errors");
    let final_count = calls.load(Ordering::Relaxed);
    println!("disarmed again: {final_count} dispatched (dormant again)");
    assert_eq!(final_count, armed_count, "disarming stops dispatch again");
}

// ── 3. BALANCE: FanIn over gated UnpinPipe sources ──────────────────────────

struct BackendQueue {
    label: &'static str,
    items: RefCell<VecDeque<u32>>,
}

impl DropSafe for BackendQueue {}

#[piped]
impl BackendQueue {
    fn call(&self, (): ()) -> impl Future<Output = Result<(&'static str, u32), Exhausted>> + Unpin {
        match self.items.borrow_mut().pop_front() {
            Some(value) => core::future::ready(Ok((self.label, value))),
            None => core::future::ready(Err(Exhausted)),
        }
    }
}

// wraps any `UnpinPipe` source with a `DemandGate` standing in for backend
// health or capacity: a closed gate answers Pending (not ready right now),
// never `Exhausted` (permanently drained) — FanIn's round-robin scan just
// moves on to the next source within the same call. This is `poll_ready`
// across backends, composed from a gate + a merge, with no `poll_ready`
// method.
struct GatedSource<S, G> {
    inner: S,
    gate: G,
}

impl<S, G> GatedSource<S, G> {
    fn new(inner: S, gate: G) -> Self {
        Self { inner, gate }
    }
}

impl<S: DropSafe, G> DropSafe for GatedSource<S, G> {}

// the call future: either the gate is closed (Pending, the inner source is
// never even asked) or open (forward to the inner source's own call future).
enum GatedCall<F> {
    Closed,
    Open(F),
}

impl<F: Future + Unpin> Future for GatedCall<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            GatedCall::Closed => Poll::Pending,
            GatedCall::Open(inner) => Pin::new(inner).poll(cx),
        }
    }
}

impl<S, G> UnpinPipe for GatedSource<S, G>
where
    S: UnpinPipe<In = (), Err = Exhausted>,
    G: DemandGate,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        if self.gate.is_armed() {
            GatedCall::Open(self.inner.call(()))
        } else {
            GatedCall::Closed
        }
    }
}

fn run_balance_gate() {
    let backend_a = BackendQueue {
        label: "a",
        items: RefCell::new(VecDeque::from([1, 2, 3, 4])),
    };
    let backend_b = BackendQueue {
        label: "b",
        items: RefCell::new(VecDeque::from([10, 20, 30, 40])),
    };

    let (gate_a, controller_a) = AtomicGate::pair(true);
    let (gate_b, controller_b) = AtomicGate::pair(true);

    let fan_in = FanIn::new(
        [
            GatedSource::new(backend_a, gate_a),
            GatedSource::new(backend_b, gate_b),
        ],
        Select::RoundRobin,
    );

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut drained_from_a = 0_u32;
    let mut drained_from_b = 0_u32;

    for step in 0..20_u32 {
        match step {
            2 => {
                controller_a.disarm();
                println!("-- backend a disarmed (unhealthy) --");
            }
            5 => {
                controller_a.arm();
                controller_b.disarm();
                println!("-- backend a recovered, backend b disarmed (over capacity) --");
            }
            9 => {
                controller_b.arm();
                println!("-- backend b recovered --");
            }
            _ => {}
        }

        let mut call = Pipe::call(&fan_in, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok((label, value))) => {
                println!("step {step}: drained ({label}, {value})");
                if label == "a" {
                    drained_from_a += 1;
                } else {
                    drained_from_b += 1;
                }
            }
            Poll::Ready(Err(Exhausted)) => {
                println!("step {step}: all backends drained");
                break;
            }
            Poll::Pending => {
                println!(
                    "step {step}: nothing ready this poll (closed backend skipped, not failed)"
                );
            }
        }
    }

    println!("drained {drained_from_a} from a, {drained_from_b} from b");
    assert_eq!(
        drained_from_a, 4,
        "backend a's whole queue is eventually drained"
    );
    assert_eq!(
        drained_from_b, 4,
        "backend b's whole queue is eventually drained"
    );
}
