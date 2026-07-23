#![allow(clippy::unwrap_used, clippy::expect_used)]
// Whole-crate gate: this file's every test assumes the no-alloc tier
// (`DnsTcpQuery::name` = `heapless::String`, `DnsTcpViolation::NameTooLong`
// exists). `dns-codec-trait` alone (`--no-default-features`) resolves
// there; plain `--features dns,dns-codec-trait` (default-features on,
// C3's own `default = ["alloc"]`) resolves to the alloc tier instead —
// this whole binary compiles to zero tests in that combination rather
// than reference no-alloc-only API that legitimately does not exist there.
#![cfg(not(proxima_alloc))]

//! C3 — the no-alloc rung's mechanically-reprovable proof: on the no-alloc
//! tier (`--no-default-features --features dns-codec-trait`, `proxima_alloc`
//! UNSET), `DnsTcpCodec::own_frame`'s question-name render allocates ZERO
//! times, because [`proxima_protocols::dns::frame_codec::DnsTcpQuery::name`]
//! is a `heapless::String` inline buffer, not a heap `String`.
//!
//! Reuses `stats_alloc::{Region, StatsAlloc}` — the SAME counting substrate
//! `tests/memcached_frame_ownership_alloc.rs` uses (RISC, P1) — over a REAL
//! encoded wire query built via [`proxima_protocols::dns::encode::encode_query`]
//! (P9: real bytes, not a hand-rolled struct literal), framed exactly like
//! `dns::frame_codec`'s own test fixtures.
//!
//! What this proves: `own_frame`'s own allocation count on this one code
//! path is 0. What it does NOT prove: that a whole request/response
//! round-trip, or any other pipe stage, is allocation-free — `Bytes`
//! itself (the `OwnFrame::Source` window `own_frame` re-owns FROM) is
//! already-allocated storage built outside the measured window, same as
//! the memcached sibling test.

use bytes::Bytes;
use proxima_codec::FrameCodec;
use proxima_protocols::codec_pipe::OwnFrame;
use proxima_protocols::dns::encode;
use proxima_protocols::dns::frame_codec::{DnsTcpCodec, DnsTcpOwnedFrame, DnsTcpViolation};
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static ALLOC: StatsAlloc<std::alloc::System> = StatsAlloc::system();

const MAX_MESSAGE_BYTES: usize = 65_535;

/// The no-alloc tier's `heapless::String<DNS_NAME_DOTTED_CAP>` capacity
/// (RFC 1035 §2.3.4) — kept in sync by hand since the const itself is
/// crate-private; a drift here would fail
/// [`an_over_capacity_name_is_rejected_not_truncated`] loudly (either the
/// oversized fixture stops being oversized, or a legitimate name starts
/// getting rejected) rather than silently.
const DNS_NAME_DOTTED_CAP: usize = 255;

fn framed(message: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(2 + message.len());
    framed.extend_from_slice(&u16::try_from(message.len()).unwrap().to_be_bytes());
    framed.extend_from_slice(message);
    framed
}

/// One real, single-question query, encoded through the actual wire
/// encoder — the same fixture shape `dns::frame_codec`'s own tests use.
fn single_question_query(id: u16, name: &str) -> Vec<u8> {
    let mut message = Vec::new();
    encode::encode_query(
        id,
        true,
        encode::EncodeQuestion {
            name,
            qtype: 1,
            qclass: 1,
        },
        &mut message,
    )
    .expect("fixture name encodes cleanly");
    framed(&message)
}

/// A single-question message whose name's dotted rendering exceeds
/// `DNS_NAME_DOTTED_CAP` — hand-built (not via `encode::encode_name`,
/// which itself enforces the RFC 1035 §2.3.4 wire limit and would refuse
/// to build this) to exercise the no-alloc tier's own reject path against
/// wire bytes the encoder could never have produced but a non-conforming
/// or adversarial peer could send. 5 labels of 63 bytes each: wire
/// `5 * 64 + 1 = 321` bytes, dotted `5 * 64 = 320` chars — over the
/// 255-byte cap the encoder itself targets, and (since `Name::labels`
/// does not itself enforce a total-length cap while parsing) a message
/// [`super::parse_message`] parses successfully all the same.
fn oversized_name_query(id: u16) -> Vec<u8> {
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

#[test]
fn own_frame_allocates_zero_times_rendering_a_bounded_name() {
    let codec = DnsTcpCodec::new(MAX_MESSAGE_BYTES);
    let wire = single_question_query(1234, "example.com.");
    let source = Bytes::from(wire.clone());
    let (frame, _consumed) = codec.parse_frame(&source).expect("frames cleanly");

    let region = Region::new(&ALLOC);
    let before = region.change();
    let owned = DnsTcpCodec::own_frame(&source, &frame);
    let after = region.change();

    assert_eq!(
        after.allocations - before.allocations,
        0,
        "own_frame must not allocate on the no-alloc tier (heapless::String \
         backs DnsTcpQuery::name here) — got a non-zero allocation delta"
    );

    match owned {
        DnsTcpOwnedFrame::Query(query) => {
            assert_eq!(query.name.as_str(), "example.com.");
            assert_eq!(query.id, 1234);
        }
        DnsTcpOwnedFrame::Violation(violation) => {
            panic!("expected a well-formed query, got {violation:?}");
        }
    }
}

#[test]
fn an_over_capacity_name_is_rejected_not_truncated() {
    let codec = DnsTcpCodec::new(MAX_MESSAGE_BYTES);
    let wire = oversized_name_query(4321);
    let source = Bytes::from(wire.clone());
    let (frame, _consumed) = codec.parse_frame(&source).expect("frames cleanly");

    // sanity: the fixture really is over the cap, or this test proves nothing.
    let dotted_len = 5 * 64;
    assert!(
        dotted_len > DNS_NAME_DOTTED_CAP,
        "fixture must exceed DNS_NAME_DOTTED_CAP to exercise the reject path"
    );

    let owned = DnsTcpCodec::own_frame(&source, &frame);
    assert_eq!(
        owned,
        DnsTcpOwnedFrame::Violation(DnsTcpViolation::NameTooLong)
    );
}
