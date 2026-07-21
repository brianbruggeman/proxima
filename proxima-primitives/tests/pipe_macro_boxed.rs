//! Proof that `#[proxima::piped(unpin, boxed)]` — the third exit from the
//! `unpin`-on-`async fn` refusal — actually works and composes into the
//! real merge.
//!
//! `#![cfg(feature = "alloc")]` gates this whole file the same way
//! `src/pipe/mod.rs` gates every other alloc-only module (`body`,
//! `bounded`, ...), and the macro emits the identical gate on the generated
//! struct/impl (see `boxed_shape_gates_the_original_fn_too_not_only_the_impl`
//! in `proxima-macros/src/pipe_attr.rs`, which proves that deterministically
//! from the macro's own token output).
//!
//! This file's own gate is NOT independently provable via `cargo test -p
//! proxima-primitives --no-default-features` here: this crate's
//! `[dev-dependencies]` already include `proxima` (for `test-support`,
//! predating this change), and `proxima` depends back on
//! `proxima-primitives` with `std`+`alloc` always on — a dev-dependency
//! cycle Cargo is allowed to have, but one that forces feature unification
//! across the whole graph the moment ANY test target is built, regardless
//! of what `--no-default-features` requests on the `-p` selection. The
//! actual T0-floor proof is `cargo build -p proxima-primitives
//! --no-default-features --lib`: verified (`cargo build -pv`) to pass
//! neither `--cfg feature="alloc"` nor an `--extern proxima_macros` edge at
//! all — `boxed` is categorically absent from that build, not merely
//! cfg'd out, because this file (the only place `boxed` is invoked in this
//! crate) is dev-only and never part of the lib target.
//!
//! With `alloc` (or `std`, which implies it) on, the boxed source below
//! composes into the real [`FanIn`] merge exactly like the hand-written
//! `UnpinPipe` in `src/pipe/fan_in.rs`'s own tests, or the ready()-wrapped
//! one proven in `proxima-macros/tests/expand_pipe.rs`.
#![cfg(feature = "alloc")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

use proxima_core::markers::DropSafe;
use proxima_macros::piped;
use proxima_primitives::pipe::{Exhausted, FanIn, Pipe, Select, UnpinPipe};

// a stateless handle onto a shared counter — the same shape a real `async fn
// recv()` over a channel or socket would have: it awaits (nothing to await
// here, but the compiler doesn't know that — it is a genuine `async fn`),
// and the only way to reach `UnpinPipe` from it, short of hand-writing a
// poll struct, is to pay the box.
static RECV_CALLS: AtomicU8 = AtomicU8::new(0);

#[piped(unpin, boxed)]
async fn recv(_: ()) -> Result<u8, Exhausted> {
    let value = RECV_CALLS.fetch_add(1, Ordering::Relaxed);
    if value < 4 { Ok(value) } else { Err(Exhausted) }
}

impl DropSafe for recv {}

#[test]
fn boxed_async_source_generates_a_working_unpin_pipe() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut call = UnpinPipe::call(&recv, ());
    match Pin::new(&mut call).poll(&mut cx) {
        Poll::Ready(Ok(value)) => assert_eq!(value, 0),
        other => panic!("Box::pin(recv(())) must resolve immediately, got {other:?}"),
    }
}

#[test]
fn boxed_async_source_composes_into_the_real_fan_in() {
    // RECV_CALLS is process-wide; the previous test already advanced it, so
    // only assert on shape (monotonic, terminates), not on absolute values —
    // this test does not own the static exclusively.
    let fan = FanIn::new([recv, recv, recv], Select::RoundRobin);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);

    let mut collected = Vec::new();
    loop {
        let mut call = Pipe::call(&fan, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => collected.push(value),
            Poll::Ready(Err(Exhausted)) => break,
            Poll::Pending => panic!("recv never pends in this test"),
        }
    }

    assert!(
        !collected.is_empty(),
        "the boxed macro-generated UnpinPipe must merge through FanIn like any hand-written source"
    );
    assert!(
        collected.windows(2).all(|pair| pair[0] < pair[1]),
        "RECV_CALLS only increases, so drained values must be strictly increasing: {collected:?}"
    );
}
