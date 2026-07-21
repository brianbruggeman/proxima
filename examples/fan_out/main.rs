#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Fan-out: one request delivered to N downstream arms — the dual of fan-in
//! (`FanIn` merges N sources into one; `FanOut` broadcasts one input to N
//! sinks). `proxima_primitives::pipe::FanOut<S, Policy>` composes over ordinary
//! `SendPipe`s: a "sink" needs no bespoke trait, it is just another arm of
//! the pipe algebra — the same `Pipe` shape `transform` taught, used here in
//! its sink role (`In -> ()`).
//!
//! One `Message` goes in; a primary arm and a mirror arm each get their own
//! clone of it and record it under their own label, proving the broadcast
//! reaches every arm independently.
//!
//! Run: `cargo run --example fan_out`

use core::convert::Infallible;
use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex};

use proxima_macros::piped;
use proxima_primitives::pipe::FanOut;
use proxima_primitives::pipe::SendPipe;

fn main() {
    block_on_ready(async {
        let primary_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mirror_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let primary = CapturingSink {
            label: "primary",
            log: Arc::clone(&primary_log),
        };
        let mirror = CapturingSink {
            label: "mirror",
            log: Arc::clone(&mirror_log),
        };

        let fan = FanOut::all_or_nothing(vec![primary, mirror]);
        println!("fanning one request to {} arms", fan.sink_count());

        fan.call(Message("checkout order 42".into()))
            .await
            .expect("fan-out delivers to every arm");

        let primary_received = primary_log.lock().expect("primary lock").clone();
        let mirror_received = mirror_log.lock().expect("mirror lock").clone();

        println!("primary arm received: {primary_received:?}");
        println!("mirror arm received:  {mirror_received:?}");

        assert_eq!(
            primary_received,
            vec!["primary: checkout order 42".to_string()],
            "primary arm received the request"
        );
        assert_eq!(
            mirror_received,
            vec!["mirror: checkout order 42".to_string()],
            "mirror arm received the SAME request, independently"
        );

        println!("both arms received the one request, independently processed: fan-out proven");
    });
}

// ── shared driver ───────────────────────────────────────────────────────────

// every future here resolves on its first poll (a mutex lock, no real I/O),
// so a one-shot poll is a legitimate block_on — no executor dependency
// needed to prove the pattern.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("fan_out example futures resolve on first poll"),
    }
}

// ── domain ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
struct Message(String);

// a sink that records every message it receives under its own label, so the
// example can assert which arms the fan-out actually reached and prove each
// arm's copy is independent of the others.
#[derive(Clone)]
struct CapturingSink {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[piped(send)]
impl CapturingSink {
    async fn call(&self, message: Message) -> Result<(), Infallible> {
        let log = Arc::clone(&self.log);
        let label = self.label;
        log.lock()
            .expect("capture lock")
            .push(format!("{label}: {}", message.0));
        Ok(())
    }
}
