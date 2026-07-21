// integration tests for #[proxima::piped] macro expansion: each generated
// struct is actually called, awaited/polled, and asserted on — not just
// checked for the right tokens.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::Cell;
use std::convert::Infallible;
use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

use proxima::error::markers::DropSafe;
use proxima::pipe::{Exhausted, FanIn, Pipe, Select, SendPipe, UnpinPipe, UnpinSendPipe};
use proxima::{App, Bytes, ProximaError, Request, Response};
use proxima_macros::piped;

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

// ---- Pipe: async fn, no args required beyond the ladder's root ----

#[piped]
async fn double(input: u64) -> Result<u64, Infallible> {
    Ok(input * 2)
}

#[test]
fn async_fn_generates_a_working_pipe() {
    let out = block_on(Pipe::call(&double, 21)).expect("infallible double");
    assert_eq!(out, 42);
}

// ---- UnpinPipe: plain fn, wrapped in core::future::ready, pollable in place ----

#[piped]
fn halve(input: u64) -> Result<u64, &'static str> {
    if input.is_multiple_of(2) {
        Ok(input / 2)
    } else {
        Err("odd input")
    }
}

#[test]
fn sync_fn_generates_a_working_unpin_pipe() {
    let mut call = UnpinPipe::call(&halve, 10);
    // the whole point of the tier: poll it where it sits, no Box, no unsafe.
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    match Pin::new(&mut call).poll(&mut cx) {
        Poll::Ready(out) => assert_eq!(out, Ok(5)),
        Poll::Pending => panic!("core::future::ready is always immediately ready"),
    }
}

#[test]
fn sync_fn_pipe_surfaces_its_error() {
    let out = block_on(UnpinPipe::call(&halve, 7));
    assert_eq!(out, Err("odd input"));
}

#[test]
fn ready_future_from_a_generated_unpin_pipe_is_unpin() {
    let call = UnpinPipe::call(&halve, 4);
    assert_unpin(&call);
}

// ---- SendPipe: #[piped(send)] on an async fn ----

#[piped(send)]
async fn increment(input: u64) -> Result<u64, &'static str> {
    input.checked_add(1).ok_or("overflow")
}

#[test]
fn send_async_fn_generates_a_working_send_pipe() {
    let out = block_on(SendPipe::call(&increment, 41)).expect("no overflow");
    assert_eq!(out, 42);
}

#[test]
fn send_pipe_future_is_actually_send() {
    let future = SendPipe::call(&increment, 1);
    assert_send(&future);
}

// ---- UnpinSendPipe: #[piped(send)] on a plain fn ----

#[piped(send)]
fn triple(input: u64) -> Result<u64, &'static str> {
    input.checked_mul(3).ok_or("overflow")
}

#[test]
fn send_sync_fn_generates_a_working_unpin_send_pipe() {
    let out = block_on(UnpinSendPipe::call(&triple, 14)).expect("no overflow");
    assert_eq!(out, 42);
}

#[test]
fn unpin_send_pipe_future_is_send_and_unpin() {
    let future = UnpinSendPipe::call(&triple, 1);
    assert_send(&future);
    assert_unpin(&future);
}

// ---- name = Ident override ----

#[piped(name = CustomName)]
fn identity_fn(input: u64) -> Result<u64, Infallible> {
    Ok(input)
}

#[test]
fn name_arg_overrides_the_generated_struct_name() {
    let out = block_on(UnpinPipe::call(&CustomName, 9)).expect("infallible identity");
    assert_eq!(out, 9);
}

/// An explicit name cannot collide with the fn, so the fn keeps its own and
/// stays directly callable. This is the escape hatch the tutorial promises for
/// wanting both the plain function and a pipe; calling it here is the proof.
#[test]
fn an_explicitly_named_pipe_leaves_the_fn_callable() {
    let direct = identity_fn(4).expect("infallible identity");
    assert_eq!(direct, 4);
}

// ---- zero-arg fn: the source form, In = () ----

#[piped]
fn always_seven() -> Result<u64, Infallible> {
    Ok(7)
}

#[test]
fn zero_arg_fn_generates_a_source_pipe() {
    let out = block_on(UnpinPipe::call(&always_seven, ())).expect("infallible source");
    assert_eq!(out, 7);
}

// ---- #[piped(unpin)] on a sync fn: redundant assertion, not an error ----

#[piped(unpin)]
fn negate(input: bool) -> Result<bool, Infallible> {
    Ok(!input)
}

#[test]
fn explicit_unpin_on_sync_fn_still_compiles_and_runs() {
    let out = block_on(UnpinPipe::call(&negate, true)).expect("infallible negate");
    assert!(!out);
}

// ---- proof this is useful: a macro-generated UnpinPipe source drops into
// the real FanIn merge with no adapter, same as a hand-written one would.
// `counter_source` is a stateless zero-sized handle onto a shared counter —
// the "read a shared ring register" shape FanIn's own doc calls out —  so
// every array slot legitimately shares the one static.

static COUNTER_SOURCE_CALLS: AtomicU8 = AtomicU8::new(0);

#[piped]
fn counter_source(_: ()) -> Result<u8, Exhausted> {
    let value = COUNTER_SOURCE_CALLS.fetch_add(1, Ordering::Relaxed);
    if value < 6 { Ok(value) } else { Err(Exhausted) }
}

impl DropSafe for counter_source {}

#[test]
fn generated_unpin_pipe_composes_into_the_real_fan_in() {
    let fan = FanIn::new(
        [counter_source, counter_source, counter_source],
        Select::RoundRobin,
    );
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);

    let mut collected = Vec::new();
    loop {
        let mut call = Pipe::call(&fan, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => collected.push(value),
            Poll::Ready(Err(Exhausted)) => break,
            Poll::Pending => panic!("counter_source never pends"),
        }
    }

    collected.sort_unstable();
    assert_eq!(
        collected,
        vec![0, 1, 2, 3, 4, 5],
        "the macro-generated UnpinPipe merged through FanIn like any hand-written source"
    );
}

// ---- proof this is useful, part two: a Handler-shaped `#[piped(send)]` fn
// also gets `impl From<Struct> for MountTarget`, so it mounts directly — no
// `MountTarget::Handle(into_handle(..))` wrapper. Dispatched through the
// real `App`/router, not a mock of one, same bar as the FanIn proof above.

#[piped(send)]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::test]
async fn handler_shaped_send_pipe_mounts_directly_and_dispatches() {
    let app = App::new().expect("app");
    app.mount("/", hello).expect("mount");

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
    assert_eq!(&body[..], b"hello, proxima\n");
}

// ---- affordance A: the generated struct derives Clone unconditionally, so
// a leaf pipe can drop straight into a combinator requiring `Inner: Clone`
// (RateLimit, Retry, Delay, Isolate, Diff, Transform, Validate) with no
// hand-rolled struct+impl. ----

#[piped(send)]
async fn always_ok(_input: u64) -> Result<u64, Infallible> {
    Ok(1)
}

#[test]
fn generated_pipe_struct_is_clone() {
    let first = always_ok;
    let second = first.clone();
    let out = block_on(SendPipe::call(&second, 0)).expect("infallible");
    assert_eq!(out, 1);
}

// ---- affordance B: the stateful impl-block form ----

#[derive(Clone)]
struct Multiplier {
    factor: u64,
}

// async-fn shape: `In -> Result<Out, Err>`, relocated verbatim into
// `async move { .. }`, landing on SendPipe via `send`.
#[piped(send)]
impl Multiplier {
    async fn call(&self, input: u64) -> Result<u64, Infallible> {
        Ok(input * self.factor)
    }
}

#[proxima::test]
async fn impl_block_async_form_generates_a_working_send_pipe() {
    let pipe = Multiplier { factor: 3 };
    let out = SendPipe::call(&pipe, 14).await.expect("infallible");
    assert_eq!(out, 42);
}

#[test]
fn impl_block_async_form_future_is_send() {
    let pipe = Multiplier { factor: 3 };
    let future = SendPipe::call(&pipe, 1);
    assert_send(&future);
}

// sync-fn shape: `call` already returns `impl Future<..> + Unpin` — a
// hand-rolled `Ready`-backed future, relocated unwrapped. Proves the
// `UnpinPipe` half of the impl-block form, mirroring `examples/gate`'s
// `BackendQueue`.
struct FixedQueue {
    remaining: Cell<u8>,
}

#[piped]
impl FixedQueue {
    fn call(&self, (): ()) -> impl Future<Output = Result<u8, Exhausted>> + Unpin {
        let value = self.remaining.get();
        if value == 0 {
            ready(Err(Exhausted))
        } else {
            self.remaining.set(value - 1);
            ready(Ok(value))
        }
    }
}

#[test]
fn impl_block_sync_form_generates_a_working_unpin_pipe() {
    let queue = FixedQueue {
        remaining: Cell::new(2),
    };
    let mut call = UnpinPipe::call(&queue, ());
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    match Pin::new(&mut call).poll(&mut cx) {
        Poll::Ready(out) => assert_eq!(out, Ok(2)),
        Poll::Pending => panic!("Ready is always immediately ready"),
    }
    let second = block_on(UnpinPipe::call(&queue, ()));
    assert_eq!(second, Ok(1));
    let exhausted = block_on(UnpinPipe::call(&queue, ()));
    assert_eq!(exhausted, Err(Exhausted));
}

#[test]
fn impl_block_sync_form_future_is_unpin() {
    let queue = FixedQueue {
        remaining: Cell::new(1),
    };
    let call = UnpinPipe::call(&queue, ());
    assert_unpin(&call);
}

// a helper method alongside `call` survives, relocated into a leftover
// inherent impl — same struct, still callable directly.
struct WithHelper {
    base: u64,
}

#[piped]
impl WithHelper {
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> + Unpin {
        ready(Ok(self.scaled(input)))
    }

    fn scaled(&self, input: u64) -> u64 {
        input + self.base
    }
}

#[test]
fn impl_block_form_preserves_direct_access_via_the_original_type() {
    let pipe = WithHelper { base: 10 };
    assert_eq!(pipe.scaled(5), 15);
    let out = block_on(UnpinPipe::call(&pipe, 5)).expect("infallible");
    assert_eq!(out, 15);
}
