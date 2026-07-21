// integration tests for the function-like leaf-lift macros
// (`pipe!`/`filter!`/`fanout!`/`fanin!`): each one is exercised against
// the REAL `proxima` crate, not a mock of it — closures are actually
// called, awaited/polled, composed via `PipeExt`, and (for `pipe!`)
// mounted on a real `App` and dispatched through its router.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::convert::Infallible;
use std::future::Future;
use std::task::{Context, Poll, Waker};

use proxima::pipe::{Exhausted, Pipe, PipeExt, SendPipe, UnpinPipe};
use proxima::{App, Bytes, ProximaError, Request, Response};
use proxima_macros::{fanin, fanout, filter, pipe};

fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = std::pin::pin!(future);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        if let Poll::Ready(output) = pinned.as_mut().poll(&mut cx) {
            return output;
        }
    }
}

fn assert_send<Type: Send>(_: &Type) {}
fn assert_unpin<Type: Unpin>(_: &Type) {}

// ---- pipe!: sync closure reaches every tier ----

#[test]
fn sync_closure_lifts_into_a_working_unpin_pipe() {
    let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let mut call = UnpinPipe::call(&doubled, 21);
    assert_unpin(&call);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    match std::pin::Pin::new(&mut call).poll(&mut cx) {
        Poll::Ready(out) => assert_eq!(out, Ok(42)),
        Poll::Pending => panic!("core::future::ready is always immediately ready"),
    }
}

#[test]
fn sync_closure_with_send_reaches_send_and_unpin_send() {
    let tripled = pipe!(
        |input: u64| -> Result<u64, Infallible> { Ok(input * 3) },
        send
    );
    let future = SendPipe::call(&tripled, 14);
    assert_send(&future);
    assert_eq!(block_on(future), Ok(42));
}

// ---- pipe!: async closure reaches Pipe only (zero-box: no `unpin,
// boxed` escape hatch here — see pipe_bang.rs's module doc) ----

#[test]
fn async_closure_lifts_into_a_working_pipe() {
    let incremented =
        pipe!(async move |input: u64| -> Result<u64, Infallible> { Ok(input + 1) });
    let out = block_on(Pipe::call(&incremented, 41));
    assert_eq!(out, Ok(42));
}

// ---- pipe!: composes through PipeExt like any other pipe ----

#[test]
fn pipe_composes_via_pipe_ext_and_then() {
    let increment = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input + 1) });
    let double = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let chain = increment.and_then(double);
    let out = block_on(Pipe::call(&chain, 4));
    assert_eq!(out, Ok(10), "4 -> +1 -> 5 -> *2 -> 10");
}

// ---- pipe!: a pass-through pipe expression is unchanged ----

struct Existing;
impl Pipe for Existing {
    type In = u64;
    type Out = u64;
    type Err = Infallible;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(input) }
    }
}

#[test]
fn pipe_passes_a_non_closure_expression_through_unchanged() {
    let pipe = pipe!(Existing);
    let out = block_on(Pipe::call(&pipe, 7));
    assert_eq!(out, Ok(7));
}

// ---- pipe! mounts directly on a real App and dispatches ----

#[proxima::test]
async fn pipe_mounts_and_dispatches_through_the_real_router() {
    let app = App::new().expect("app");
    let handler = pipe!(
        |_request: Request<Bytes>| -> Result<Response<Bytes>, ProximaError> {
            Ok(Response::ok("hello from pipe!\n"))
        },
        send
    );
    app.mount("/", handler).expect("mount");

    let dispatch = app.router_handle();
    let request = Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("builder");
    let response = SendPipe::call(&dispatch, request)
        .await
        .expect("dispatch through the real router");
    let body = response.collect_body().await.expect("body");
    assert_eq!(&body[..], b"hello from pipe!\n");
}

// ---- filter!: decision-shaped closure gates a chain ----

#[test]
fn filter_gates_a_chain_admitting_and_rejecting() {
    let gate = filter!(|input: u64| -> Result<u64, &'static str> {
        if input < 100 { Ok(input) } else { Err("too big") }
    });
    let increment = pipe!(|input: u64| -> Result<u64, &'static str> { Ok(input + 1) });
    let stack = gate.and_then(increment);

    let admitted = block_on(Pipe::call(&stack, 10));
    assert_eq!(admitted, Ok(11), "10 passes the gate, then +1");

    let rejected = block_on(Pipe::call(&stack, 200));
    assert_eq!(rejected, Err("too big"), "200 is rejected before +1 runs");
}

// ---- fanout!: broadcasts a closure and a pass-through pipe arm ----

struct RecordingSink {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Pipe for RecordingSink {
    type In = u32;
    type Out = ();
    type Err = Infallible;
    fn call(&self, _input: u32) -> impl Future<Output = Result<(), Infallible>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        async move { Ok(()) }
    }
}

#[test]
fn fanout_broadcasts_to_a_closure_arm_and_a_pass_through_arm() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sink = RecordingSink {
        calls: std::sync::Arc::clone(&calls),
    };

    let fan = fanout!(
        sink,
        |input: u32| -> Result<(), Infallible> {
            assert_eq!(input, 9);
            Ok(())
        }
    );

    let outcome = block_on(Pipe::call(&fan, 9));
    assert_eq!(outcome, Ok(()));
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the pass-through arm ran too"
    );
}

// ---- fanin!: merges two closure sources round-robin ----

#[test]
fn fanin_merges_two_closure_sources() {
    // each source emits exactly once, then reports Exhausted — a real
    // source that ran forever would hang the drain loop below, same as any
    // hand-written FanIn source must eventually terminate.
    let emitted_one = std::cell::Cell::new(false);
    let emitted_two = std::cell::Cell::new(false);
    let fan = fanin!(
        move |(): ()| -> Result<u8, Exhausted> {
            if emitted_one.replace(true) {
                Err(Exhausted)
            } else {
                Ok(1)
            }
        },
        move |(): ()| -> Result<u8, Exhausted> {
            if emitted_two.replace(true) {
                Err(Exhausted)
            } else {
                Ok(2)
            }
        }
    );

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut merged = Vec::new();
    loop {
        let mut call = Pipe::call(&fan, ());
        match std::pin::Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => merged.push(value),
            Poll::Ready(Err(Exhausted)) => break,
            Poll::Pending => panic!("these sources never pend"),
        }
    }
    merged.sort_unstable();
    assert_eq!(merged, vec![1, 2]);
}

// ---- fanin!: a genuinely stateful closure source (not just a constant),
// proving the merge really calls through to each arm's own captured state
// rather than coincidentally matching on `core::future::ready`'s shape.

#[test]
fn fanin_drains_a_stateful_source_to_exhaustion() {
    let remaining = std::cell::Cell::new(2u8);
    let fan = fanin!(
        move |(): ()| -> Result<u8, Exhausted> {
            let value = remaining.get();
            if value == 0 {
                Err(Exhausted)
            } else {
                remaining.set(value - 1);
                Ok(value)
            }
        },
        |(): ()| -> Result<u8, Exhausted> { Err(Exhausted) }
    );

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut drained = Vec::new();
    loop {
        let mut call = Pipe::call(&fan, ());
        match std::pin::Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => drained.push(value),
            Poll::Ready(Err(Exhausted)) => break,
            Poll::Pending => panic!("these sources never pend"),
        }
    }
    assert_eq!(drained, vec![2, 1], "the stateful source drains in order");
}
