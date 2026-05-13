#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `retry` — re-run a pipe on a `Retryable` decision.
//!
//! `filter` teaches the gate shape: a decision pipe (`In -> Result<In, Err>`)
//! looks at the INPUT and decides whether the inner pipe runs at all. Retry
//! is the same shape turned around: a decision looks at the OUTCOME of an attempt and
//! decides whether the inner pipe runs AGAIN. `RetryController::on_outcome`
//! is that decision as a pure function — no I/O, no clock, no sleeping. It
//! takes the last outcome (via `Retryable`) plus attempt/deadline state and
//! returns a `RetryAction`: `Retry { after }`, `Done`, or `Exhausted`. The
//! caller drives the loop; this example's `retry_call` is that caller.
//!
//! Three jobs run through the SAME `retry_call` loop below, each landing on
//! a different `RetryAction`: a retryable status that eventually gets `Done`
//! via success, a retryable status that never succeeds and hits `Exhausted`
//! (the attempt cap), and a status the default `RetryRules` doesn't consider
//! retryable at all — `on_outcome` returns `Done` on the very first attempt,
//! proving the decision gates the re-run rather than the loop just guessing.
//!
//! `proxima_primitives::pipe::retry::Retry<Inner>` is the ready-made `Pipe` wrapper for
//! HTTP/event pipelines built on this same idea; it inlines its own copy of
//! the decision against `RetryRules` directly (not `RetryController`) and
//! awaits a real backoff between attempts. This example never sleeps, so it
//! stays on the pure controller instead.
//!
//! Run: `cargo run --example retry`

use core::future::Future;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use proxima_primitives::pipe::{
    Backoff, Jitter, Pipe, RetryAction, RetryController, RetryRules, Retryable,
};

fn main() {
    println!("retry: re-run a pipe on a RetryController decision\n");

    println!("-- retryable, succeeds before the cap --");
    let (flaky_worker, flaky_calls) = Worker::new(Script::FlakyThenSucceeds { fail_until: 3 });
    let controller = controller_with_max_attempts(5);
    let (result, attempts) = retry_call(&flaky_worker, Job { id: 1 }, &controller);
    let result = result.expect("flaky call");
    assert_eq!(result.status, 200, "the third attempt lands 200");
    assert_eq!(
        flaky_calls.load(Ordering::SeqCst),
        3,
        "two 503s, then success"
    );
    println!(
        "job 1: status {} after {attempts} attempts\n",
        result.status
    );

    println!("-- retryable, exhausts the cap --");
    let (busy_worker, busy_calls) = Worker::new(Script::AlwaysBusy);
    let controller = controller_with_max_attempts(3);
    let (result, attempts) = retry_call(&busy_worker, Job { id: 2 }, &controller);
    let result = result.expect("busy call");
    assert_eq!(
        result.status, 503,
        "still busy — the cap stopped it, not success"
    );
    assert_eq!(
        busy_calls.load(Ordering::SeqCst),
        3,
        "exactly max_attempts tries, then on_outcome returns Exhausted"
    );
    println!(
        "job 2: status {} after {attempts} attempts (cap reached)\n",
        result.status
    );

    println!("-- not retryable, passes straight through --");
    let (rejected_worker, rejected_calls) =
        Worker::new(Script::PermanentlyRejected { status: 422 });
    let controller = controller_with_max_attempts(5);
    let (result, attempts) = retry_call(&rejected_worker, Job { id: 3 }, &controller);
    let result = result.expect("rejected call");
    assert_eq!(result.status, 422);
    assert_eq!(
        rejected_calls.load(Ordering::SeqCst),
        1,
        "422 is outside the default retry_on_status set — on_outcome returns Done \
         on the first attempt, so the worker is never called again"
    );
    println!(
        "job 3: status {} after {attempts} attempt (not retryable)\n",
        result.status
    );

    println!(
        "all three jobs ran through the same retry_call loop; \
         only RetryController::on_outcome's decision on each outcome changed the attempt count"
    );
}

fn controller_with_max_attempts(max_attempts: u32) -> RetryController {
    RetryController {
        rules: RetryRules::default(),
        backoff: Backoff::Exponential {
            initial: Duration::from_millis(50),
            factor: 2,
            max: Duration::from_millis(2000),
        },
        jitter: Jitter::None,
        max_attempts,
        deadline: None,
    }
}

// the caller-driven attempt loop `RetryController` expects: call the pipe,
// ask on_outcome what to do with the result, repeat until Done or Exhausted.
// no clock and no entropy are wired up (now_nanos/rand are both 0) because
// this demo sets no deadline and uses Jitter::None — a real caller threads a
// monotonic clock and real entropy through here instead.
fn retry_call<P>(pipe: &P, job: Job, controller: &RetryController) -> (Result<P::Out, P::Err>, u32)
where
    P: Pipe<In = Job>,
    P::Out: Retryable,
{
    let mut attempt = 0;
    let mut prev_delay = Duration::ZERO;
    let mut outcome = block_on_ready(pipe.call(job));
    loop {
        match controller.on_outcome(attempt, &outcome, 0, 0, prev_delay) {
            RetryAction::Done | RetryAction::Exhausted => return (outcome, attempt + 1),
            RetryAction::Retry { after } => {
                println!("  on_outcome: retry after {after:?}");
                prev_delay = after;
                attempt += 1;
                outcome = block_on_ready(pipe.call(job));
            }
        }
    }
}

// every future here resolves on its first poll — the worker's own future
// never yields, so a one-shot poll is a legitimate block_on with no executor
// and no real sleep (retry_call never awaits `after`; it only prints it).
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("retry example futures resolve on first poll"),
    }
}

// ── the inner Pipe: what retry_call wraps ───────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Job {
    id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JobResult {
    id: u32,
    status: u16,
}

// this is the whole retry surface: a status the default RetryRules
// recognizes (502/503/504) is retryable, anything else is terminal.
impl Retryable for JobResult {
    fn retry_status(&self) -> Option<u16> {
        Some(self.status)
    }

    fn is_success(&self) -> bool {
        self.status == 200
    }
}

#[derive(Clone, Copy)]
enum Script {
    FlakyThenSucceeds { fail_until: u32 },
    AlwaysBusy,
    PermanentlyRejected { status: u16 },
}

struct Worker {
    calls: Arc<AtomicU32>,
    script: Script,
}

impl Worker {
    fn new(script: Script) -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        (
            Self {
                calls: calls.clone(),
                script,
            },
            calls,
        )
    }
}

impl Pipe for Worker {
    type In = Job;
    type Out = JobResult;
    type Err = Infallible;

    fn call(&self, job: Job) -> impl Future<Output = Result<JobResult, Infallible>> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let script = self.script;
        async move {
            let status = match script {
                Script::FlakyThenSucceeds { fail_until } => {
                    if attempt < fail_until {
                        503
                    } else {
                        200
                    }
                }
                Script::AlwaysBusy => 503,
                Script::PermanentlyRejected { status } => status,
            };
            println!(
                "  worker attempt {attempt} for job {}: status {status}",
                job.id
            );
            Ok(JobResult { id: job.id, status })
        }
    }
}
