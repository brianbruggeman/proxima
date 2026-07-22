//! The sugar layer over the pipe algebra, run for real. Nothing here adds a
//! new noun to the algebra (`proxima-primitives/src/pipe/primitives.rs`) —
//! every demo below is one of: `PipeExt` (`.and_then`/`.filter`), a leaf
//! macro (`pipe!`/`filter!`/`fanout!`/`fanin!`), `#[proxima::piped]`'s
//! impl-all tier closure, or `App::mount`'s four accepted shapes. Companion
//! to `docs/tutorials/01-ergonomics.md`, which cites this file by line.
//!
//! Run: `cargo run --example pipe_ergonomics`

// demonstration code: a panic on unexpected failure is the point, not a
// production error path (same convention as `examples/gate/main.rs`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::convert::Infallible;
use core::future::Future;

use bytes::Bytes;
use proxima::pipe::{
    Exhausted, Pipe, PipeExt, Request, Response, SendPipe, UnpinPipe, UnpinSendPipe,
};
use proxima::{App, ProximaError, fanin, fanout, filter, pipe};

// ---- 1. pipe! lifts a closure into a Pipe value at the call site ----
async fn leaf_macro_demo() {
    let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let out = Pipe::call(&doubled, 21).await.expect("infallible");
    assert_eq!(out, 42);
    println!("pipe! : 21 -> {out}");
}

// ---- 2. PipeExt::and_then chains two pipe!-lifted closures ----
async fn and_then_demo() {
    let increment = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input + 1) });
    let double = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let chain = increment.and_then(double);
    let out = Pipe::call(&chain, 5).await.expect("infallible");
    assert_eq!(out, 12);
    println!("and_then: 5 -> increment -> double -> {out}");
}

// ---- 3. filter! + PipeExt::filter gates an inner pipe ----
#[derive(Debug, PartialEq, Eq)]
struct Odd;

async fn filter_demo() {
    let reject_odd = filter!(|input: u64| -> Result<u64, Odd> {
        if input.is_multiple_of(2) { Ok(input) } else { Err(Odd) }
    });
    let double = pipe!(|input: u64| -> Result<u64, Odd> { Ok(input * 2) });
    let gated = double.filter(reject_odd);

    let admitted = Pipe::call(&gated, 4).await.expect("4 is even");
    assert_eq!(admitted, 8);
    let rejected = Pipe::call(&gated, 3).await.expect_err("3 is odd");
    assert_eq!(rejected, Odd);
    println!("filter! : 4 -> {admitted}, 3 -> rejected({rejected:?})");
}

// ---- 4. fanout! builds a FanOut over N closure arms in one call ----
async fn fanout_demo() {
    let fan = fanout!(
        |input: u32| -> Result<(), Infallible> {
            println!("  fanout! arm a saw {input}");
            Ok(())
        },
        |input: u32| -> Result<(), Infallible> {
            println!("  fanout! arm b saw {input}");
            Ok(())
        },
    );
    Pipe::call(&fan, 7).await.expect("both arms accept");
}

// ---- 5. fanin! builds a FanIn over N source arms, merged round-robin ----
fn fanin_demo() {
    let merged = fanin!(
        |(): ()| -> Result<u8, Exhausted> { Ok(1) },
        |(): ()| -> Result<u8, Exhausted> { Ok(2) },
    );
    let waker = core::task::Waker::noop();
    let mut context = core::task::Context::from_waker(waker);
    let mut drained = [0u8; 2];
    let mut count = 0;
    for _ in 0..8 {
        if count == 2 {
            break;
        }
        let mut call = UnpinPipe::call(&merged, ());
        if let core::task::Poll::Ready(Ok(value)) =
            core::pin::Pin::new(&mut call).poll(&mut context)
        {
            drained[count] = value;
            count += 1;
        }
    }
    drained.sort_unstable();
    assert_eq!(drained, [1, 2]);
    println!("fanin! : merged both sources -> {drained:?}");
}

// ---- 6. #[proxima::piped(send)] on a sync fn implements ALL four tiers ----
#[proxima::piped(send)]
fn triple(input: u64) -> Result<u64, Infallible> {
    Ok(input * 3)
}

fn assert_pipe<Value: Pipe>() {}
fn assert_send_pipe<Value: SendPipe>() {}
fn assert_unpin_pipe<Value: UnpinPipe>() {}
fn assert_unpin_send_pipe<Value: UnpinSendPipe>() {}

async fn piped_impl_all_demo() {
    // if `#[proxima::piped(send)]` picked only one tier, only one of these
    // four lines would compile — all four compiling is the impl-all proof.
    assert_pipe::<triple>();
    assert_send_pipe::<triple>();
    assert_unpin_pipe::<triple>();
    assert_unpin_send_pipe::<triple>();

    let out = Pipe::call(&triple, 14).await.expect("infallible");
    assert_eq!(out, 42);
    println!("#[proxima::piped(send)] on a sync fn: reaches Pipe+SendPipe+UnpinPipe+UnpinSendPipe");
}

// ---- 7. App::mount accepts a handler pipe and a bare async fn ----
#[proxima::piped(send)]
async fn echo(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    let (_, body) = request.body_bytes().await?;
    Ok(Response::ok(body))
}

async fn echo_fn(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    let (_, body) = request.body_bytes().await?;
    Ok(Response::ok(body))
}

fn mount_shapes_demo() {
    let app = App::new().expect("app");
    app.mount("/via-pipe", echo).expect("mount a handler-shaped pipe");
    app.mount("/via-fn", echo_fn).expect("mount a bare async fn");
    println!("mount: ViaPipe and ViaFn both accepted by the same App::mount");
}

#[proxima::main]
async fn main() {
    leaf_macro_demo().await;
    and_then_demo().await;
    filter_demo().await;
    fanout_demo().await;
    fanin_demo();
    piped_impl_all_demo().await;
    mount_shapes_demo();
    println!("all pipe-ergonomics claims verified");
}
