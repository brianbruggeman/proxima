//! [`MemcachedCodec`] — the TCP-direction `proxima_codec::FrameCodec` +
//! `codec_pipe::OwnFrame`/`Incomplete` impl memcached needs to plug into
//! `proxima_listen::any::FramedAny`, the generic stateless `AnyProtocol`
//! driver. Reuses [`super::parse_command`] (decode) and
//! [`super::reply::encode_reply`] / [`super::codec_trait::encode_command`]
//! (encode) UNCHANGED — no wire logic is rewritten here, only wrapped in
//! the trait shapes `FramedAny` composes against.
//!
//! memcached's wire is genuinely asymmetric: a REQUEST is a [`Command`],
//! a REPLY is a [`super::Reply`] — two unrelated shapes, unlike RESP's
//! single recursive `Frame` (`crate::redis::codec_trait`'s docs) or a
//! symmetric echo protocol. [`proxima_codec::FrameCodec::Frame`] is
//! nonetheless ONE associated type shared by `parse_frame` (decode) and
//! `encode_frame` (encode) — [`MemcachedFrame`] resolves that by being a
//! sum over both directions. The one real cost: [`MemcachedCodec::own_frame`]
//! becomes a partial function over that sum (a `Reply` frame it can never
//! actually receive, since `parse_frame` never produces one) — see its
//! own doc.
//!
//! A memcached command that fails to parse (unknown verb, a malformed
//! numeric field, ...) or that would grow the connection's buffer past
//! [`MemcachedCodec::max_message_bytes`] before completing is NOT
//! signalled as a hard [`proxima_codec::FrameCodec::Error`] — the ONLY
//! error this codec ever raises is [`NeedMoreBytes`] ("keep reading").
//! Both harder cases are folded into a SUCCESSFULLY parsed
//! [`MemcachedFrame::Violation`] that consumes the WHOLE buffered window,
//! so the generic driver still writes the matching reply and the
//! App-level `keep_serving() == false` (see `proxima-memcached`'s
//! `framed_app` module) closes the connection cleanly afterward — no
//! attempt to resynchronize past a value body of unknown length, which
//! is the exact safety reason
//! `proxima_protocols::memcached::connection::Advanced::ProtocolError`'s
//! own docs give for closing outright rather than skipping ahead to the
//! next line.

use alloc::vec::Vec;
use core::fmt;

use bytes::Bytes;
use proxima_codec::FrameCodec;

use crate::codec_pipe::{Incomplete, OwnFrame};
use crate::memcached::codec_trait::encode_command;
use crate::memcached::pipe_contract::MemcachedRequest;
use crate::memcached::reply::{Reply, encode_reply};
use crate::memcached::{Command, ParseError, parse_command};

/// A hard framing problem this codec resolves WITHOUT ever surfacing a
/// [`FrameCodec::Error`] — see the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    /// An unknown verb, or a malformed/non-numeric field — renders as
    /// the bare `ERROR\r\n` `proxima_protocols::memcached::reply::Reply::Error`
    /// already carries; no detail text distinguishes which parse rule
    /// tripped (matching the FSM driver this replaces).
    Protocol,
    /// A still-incomplete command already exceeds `limit` buffered bytes.
    MessageTooLarge { limit: usize },
}

/// [`FrameCodec::Frame`] for memcached: the SUM of both wire directions
/// (see module doc). [`MemcachedCodec::parse_frame`] only ever produces
/// `Request`/`Violation`; `Reply` only ever appears on the encode side,
/// borrowed from a handler's owned outcome
/// (`proxima-memcached`'s `framed_app::MemcachedOutcome::as_frame`).
#[derive(Debug, Clone)]
pub enum MemcachedFrame<'a> {
    Request(Command<'a>),
    Violation(Violation),
    Reply(&'a Reply),
}

/// The one error [`MemcachedCodec::parse_frame`] ever raises: "the
/// buffer does not hold a complete command yet." Every harder failure
/// (unknown verb, malformed field, oversized value) is folded into
/// [`MemcachedFrame::Violation`] instead — see the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedMoreBytes;

impl fmt::Display for NeedMoreBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("need more bytes: command not yet complete")
    }
}

impl core::error::Error for NeedMoreBytes {}

impl Incomplete for NeedMoreBytes {
    fn is_incomplete(&self) -> bool {
        true
    }
}

/// memcached (text protocol) [`FrameCodec`]. Carries
/// [`Self::max_message_bytes`] (mirrors
/// `proxima_protocols::memcached::connection::Limits`) — the DoS cap
/// [`Self::parse_frame`] enforces directly, since a `FrameCodec` is
/// stateless per call and `FramedAny`'s driver hands it the WHOLE
/// currently-buffered window on every attempt (re-parsing from byte
/// zero; see `proxima_listen::any::FramedAny`'s own doc).
#[derive(Debug, Clone, Copy)]
pub struct MemcachedCodec {
    pub max_message_bytes: usize,
}

impl MemcachedCodec {
    #[must_use]
    pub const fn new(max_message_bytes: usize) -> Self {
        Self { max_message_bytes }
    }
}

impl FrameCodec for MemcachedCodec {
    type Frame<'a> = MemcachedFrame<'a>;
    type Error = NeedMoreBytes;

    fn parse_frame<'a>(
        &self,
        buf: &'a [u8],
    ) -> Result<(MemcachedFrame<'a>, usize), NeedMoreBytes> {
        match parse_command(buf) {
            Ok((command, consumed)) => Ok((MemcachedFrame::Request(command), consumed)),
            Err(ParseError::Short) => Err(NeedMoreBytes),
            Err(ParseError::PartialValue(_declared)) => {
                if buf.len() > self.max_message_bytes {
                    Ok((
                        MemcachedFrame::Violation(Violation::MessageTooLarge {
                            limit: self.max_message_bytes,
                        }),
                        buf.len(),
                    ))
                } else {
                    Err(NeedMoreBytes)
                }
            }
            Err(_hard_error) => Ok((MemcachedFrame::Violation(Violation::Protocol), buf.len())),
        }
    }

    fn encode_frame(
        &self,
        frame: &MemcachedFrame<'_>,
        dest: &mut Vec<u8>,
    ) -> Result<(), NeedMoreBytes> {
        match frame {
            MemcachedFrame::Request(command) => encode_command(command, dest),
            MemcachedFrame::Reply(reply) => encode_reply(reply, dest),
            MemcachedFrame::Violation(_) => {
                // never constructed on the encode side — a handler's
                // outcome only ever borrows a `Reply` variant (see
                // `MemcachedOutcome::as_frame` in `proxima-memcached`).
                unreachable!(
                    "a Violation frame is never encoded; the App layer renders it as a Reply first"
                )
            }
        }
        Ok(())
    }
}

/// [`OwnFrame::Owned`] for [`MemcachedCodec`] — the owned mirror of
/// [`MemcachedFrame::Request`]/[`MemcachedFrame::Violation`] (never
/// `Reply`; that variant only ever appears on the encode side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemcachedOwnedFrame {
    Request(MemcachedRequest),
    Violation(Violation),
}

impl OwnFrame for MemcachedCodec {
    type Owned = MemcachedOwnedFrame;

    fn own_frame(_source: &Bytes, frame: &MemcachedFrame<'_>) -> MemcachedOwnedFrame {
        match frame {
            MemcachedFrame::Request(command) => {
                MemcachedOwnedFrame::Request(MemcachedRequest::from_command(command))
            }
            MemcachedFrame::Violation(kind) => MemcachedOwnedFrame::Violation(*kind),
            MemcachedFrame::Reply(_) => {
                // `own_frame`'s own contract (see `codec_pipe::OwnFrame`'s
                // doc) is "given the Bytes window it was PARSED from" —
                // `parse_frame` never produces a `Reply` frame, so this
                // arm is unreachable by construction, not by convention.
                unreachable!("own_frame is only ever called on parse_frame's own output")
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memcached::StoreMode;

    fn codec() -> MemcachedCodec {
        MemcachedCodec::new(1024)
    }

    #[test]
    fn parse_frame_returns_a_complete_request() {
        let (frame, consumed) = codec().parse_frame(b"get k\r\n").expect("parses");
        assert_eq!(consumed, b"get k\r\n".len());
        match frame {
            MemcachedFrame::Request(Command::Get { keys, gets }) => {
                assert_eq!(keys, b"k");
                assert!(!gets);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_frame_short_line_needs_more_bytes() {
        let outcome = codec().parse_frame(b"get incomp");
        assert_eq!(outcome.unwrap_err(), NeedMoreBytes);
    }

    #[test]
    fn parse_frame_partial_value_under_the_cap_needs_more_bytes() {
        let outcome = codec().parse_frame(b"set k 0 0 500\r\nabc");
        assert_eq!(outcome.unwrap_err(), NeedMoreBytes);
    }

    #[test]
    fn parse_frame_partial_value_over_the_cap_is_a_message_too_large_violation() {
        let codec = MemcachedCodec::new(8);
        let buf = b"set k 0 0 500\r\nabc";
        let (frame, consumed) = codec.parse_frame(buf).expect("folds into a violation frame");
        assert_eq!(consumed, buf.len());
        assert!(matches!(
            frame,
            MemcachedFrame::Violation(Violation::MessageTooLarge { limit: 8 })
        ));
    }

    #[test]
    fn parse_frame_unknown_verb_is_a_protocol_violation_consuming_the_whole_buffer() {
        let buf = b"bogus\r\n";
        let (frame, consumed) = codec().parse_frame(buf).expect("folds into a violation frame");
        assert_eq!(consumed, buf.len());
        assert!(matches!(frame, MemcachedFrame::Violation(Violation::Protocol)));
    }

    #[test]
    fn encode_frame_renders_a_request() {
        let mut dest = Vec::new();
        let command = Command::Delete {
            key: b"k",
            noreply: false,
        };
        codec()
            .encode_frame(&MemcachedFrame::Request(command), &mut dest)
            .expect("encode");
        assert_eq!(dest, b"delete k\r\n");
    }

    #[test]
    fn encode_frame_renders_a_reply() {
        let mut dest = Vec::new();
        let reply = Reply::Stored;
        codec()
            .encode_frame(&MemcachedFrame::Reply(&reply), &mut dest)
            .expect("encode");
        assert_eq!(dest, b"STORED\r\n");
    }

    #[test]
    fn own_frame_reowns_a_request_into_a_memcached_request() {
        let (frame, _) = codec().parse_frame(b"set k 5 60 5\r\nhello\r\n").expect("parses");
        let owned = MemcachedCodec::own_frame(&Bytes::from_static(b"unused"), &frame);
        assert_eq!(
            owned,
            MemcachedOwnedFrame::Request(MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: b"k".to_vec(),
                flags: 5,
                exptime: 60,
                value: b"hello".to_vec(),
                noreply: false,
            })
        );
    }

    #[test]
    fn own_frame_reowns_a_violation_verbatim() {
        let frame = MemcachedFrame::Violation(Violation::MessageTooLarge { limit: 16 });
        let owned = MemcachedCodec::own_frame(&Bytes::new(), &frame);
        assert_eq!(
            owned,
            MemcachedOwnedFrame::Violation(Violation::MessageTooLarge { limit: 16 })
        );
    }
}
