#![allow(clippy::expect_used)]

//! C3 walkthrough — the DNS-TCP no-alloc rung driven end to end through the
//! generic `codec_pipe` composition (`FrameCodecPipe<DnsTcpCodec>` `AndThen`
//! `OnFrame<SummarizeQuery>`), over REAL encoded wire bytes
//! ([`proxima_protocols::dns::encode::encode_query`]), with a dependency-free
//! noop-waker executor — no I/O trait, no tokio, no async runtime at all.
//!
//! Run with the no-alloc tier active (`proxima_alloc` unset — `alloc` is
//! NOT in the feature list below):
//!
//! ```text
//! cargo run --example dns_tcp_no_alloc_walkthrough -p proxima-protocols \
//!     --no-default-features --features dns-codec-trait
//! ```
//!
//! Two outcomes are driven through the SAME pipe: a well-formed short-name
//! query (`Query`), and a hand-built message whose name exceeds
//! `DNS_NAME_DOTTED_CAP` (`Violation(NameTooLong)`) — proving the reject
//! path is reachable through the full composed pipe, not just the bare
//! `own_frame` unit tested in `tests/dns_no_alloc_alloc_counter.rs`.
//!
//! What this walkthrough shows: `own_frame`'s own name-render step performs
//! no heap allocation on this tier. It does NOT claim the surrounding
//! `Bytes` window, or every other stage of a real DNS listener, is
//! allocation-free — `Bytes` is already-allocated storage handed in from
//! outside this pipe, same scope note as the sibling 0-alloc test.

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

use bytes::Bytes;
use proxima_primitives::pipe::{AndThen, Pipe};
use proxima_protocols::codec_pipe::{FrameCodecPipe, OnFrame};
use proxima_protocols::dns::encode;
use proxima_protocols::dns::frame_codec::{DnsTcpCodec, DnsTcpFrameError, DnsTcpOwnedFrame};

const MAX_MESSAGE_BYTES: usize = 512;

/// Dependency-free executor for the always-ready futures this pipe
/// produces — mirrors `codec_pipe`'s own test-only `block_on` helper
/// (see its module doc) rather than pulling in a runtime for a walkthrough
/// that never actually suspends.
fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
    }
}

/// The `App` half of `OnFrame<App>` — prints what `own_frame` handed back,
/// distinguishing a usable query from a rejected one. `Err = DnsTcpFrameError`
/// only to satisfy `AndThen`'s `Second::Err: From<First::Err>` composition
/// seam (identity `From`); this pipe never actually produces an `Err`.
struct SummarizeQuery;

impl Pipe for SummarizeQuery {
    type In = DnsTcpOwnedFrame;
    type Out = ();
    type Err = DnsTcpFrameError;

    fn call(&self, input: DnsTcpOwnedFrame) -> impl Future<Output = Result<(), DnsTcpFrameError>> {
        async move {
            match input {
                DnsTcpOwnedFrame::Query(query) => println!(
                    "Query   {{ id: {}, name: {:?}, qtype: {}, qclass: {}, rd: {} }}",
                    query.id,
                    query.name.as_str(),
                    query.qtype,
                    query.qclass,
                    query.recursion_desired,
                ),
                DnsTcpOwnedFrame::Violation(violation) => {
                    println!("Violation({violation:?})");
                }
            }
            Ok(())
        }
    }
}

fn framed(message: &[u8]) -> Bytes {
    let mut framed = Vec::with_capacity(2 + message.len());
    framed.extend_from_slice(&u16::try_from(message.len()).expect("fits u16").to_be_bytes());
    framed.extend_from_slice(message);
    Bytes::from(framed)
}

/// A real, single-question query, encoded through the actual wire encoder.
fn well_formed_query(id: u16) -> Bytes {
    let mut message = Vec::new();
    encode::encode_query(
        id,
        true,
        encode::EncodeQuestion {
            name: "example.com.",
            qtype: 1,
            qclass: 1,
        },
        &mut message,
    )
    .expect("fixture name encodes cleanly");
    framed(&message)
}

/// A hand-built single-question message whose name's dotted rendering (320
/// chars: 5 labels of 63 bytes each) exceeds `DNS_NAME_DOTTED_CAP` (255) —
/// `encode::encode_name` itself refuses to build a name this long, so this
/// is assembled by hand to model a non-conforming or adversarial peer.
fn oversized_name_query(id: u16) -> Bytes {
    let mut message = Vec::new();
    message.extend_from_slice(&id.to_be_bytes());
    message.extend_from_slice(&0x0100u16.to_be_bytes()); // QR=0, RD=1
    message.extend_from_slice(&1u16.to_be_bytes()); // qdcount
    message.extend_from_slice(&0u16.to_be_bytes()); // ancount
    message.extend_from_slice(&0u16.to_be_bytes()); // nscount
    message.extend_from_slice(&0u16.to_be_bytes()); // arcount
    for _ in 0..5 {
        message.push(63u8);
        message.extend(std::iter::repeat_n(b'a', 63));
    }
    message.push(0); // root terminator
    message.extend_from_slice(&1u16.to_be_bytes()); // qtype A
    message.extend_from_slice(&1u16.to_be_bytes()); // qclass IN
    framed(&message)
}

fn main() {
    let pipe = AndThen::new(
        FrameCodecPipe::new(DnsTcpCodec::new(MAX_MESSAGE_BYTES)),
        OnFrame::new(SummarizeQuery),
    );

    println!("-- well-formed query --");
    let outcome = block_on(pipe.call(well_formed_query(1234))).expect("infallible");
    assert!(outcome.is_some(), "a complete frame produces Some");

    println!("-- oversized-name query (no-alloc reject path) --");
    let outcome = block_on(pipe.call(oversized_name_query(4321))).expect("infallible");
    assert!(outcome.is_some(), "a complete frame produces Some");
}
