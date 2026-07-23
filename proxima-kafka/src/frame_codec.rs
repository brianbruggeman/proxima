//! [`KafkaCodec`] — the TCP-direction `proxima_codec::FrameCodec` +
//! `proxima_protocols::codec_pipe::OwnFrame`/`Incomplete` impl Kafka needs
//! to plug into `proxima_listen::any::FramedAny`, the generic stateless
//! `AnyProtocol` driver. Reuses [`proxima_protocols::kafka::parse_frame`]/
//! [`proxima_protocols::kafka::parse_request_header`] (envelope + header
//! decode) and [`crate::wire::decode_request`]/[`crate::wire::ResponseBody::encode`]
//! (body decode/encode) UNCHANGED — no wire logic is rewritten here, only
//! wrapped in the trait shapes `FramedAny` composes against.
//!
//! Kafka's wire is genuinely asymmetric: a REQUEST decodes into a
//! [`crate::wire::RequestBody`], a REPLY encodes from a
//! [`crate::wire::ResponseBody`] — two unrelated shapes, exactly the
//! memcached precedent this module follows
//! (`proxima_protocols::memcached::frame_codec`'s own doc).
//! [`proxima_codec::FrameCodec::Frame`] is nonetheless ONE associated
//! type shared by `parse_frame` (decode) and `encode_frame` (encode) —
//! [`KafkaFrame`] resolves that by being a sum over both directions. The
//! one real cost: [`KafkaCodec::encode_frame`] becomes a partial function
//! over that sum (a `Request`/`Violation` frame it can never actually be
//! asked to encode, since `parse_frame` never produces one on the
//! caller-facing encode path) — see its own doc.
//!
//! A request that fails to parse (malformed envelope/header, an
//! unrecognized or unsupported `api_key`/`api_version`, a malformed body)
//! is NOT signalled as a hard [`proxima_codec::FrameCodec::Error`] — the
//! ONLY error this codec ever raises is "not enough bytes yet"
//! ([`KafkaCodecError::Incomplete`]). Every harder case is folded into a
//! successfully parsed [`KafkaFrame::Violation`] that consumes the WHOLE
//! parsed (or, for a still-incomplete oversized frame, the whole
//! currently-buffered) window, so the generic driver still writes
//! whatever reply this facade can honestly render and the App-level
//! `keep_serving() == false` (see `crate::framed_app`) closes the
//! connection afterward — the same "close outright, don't try to
//! resynchronize past an untrusted length" safety reasoning the deleted
//! `connection::Advanced::ProtocolError`/`Advanced::MessageTooLarge` arms
//! already used.

use bytes::Bytes;
use proxima_codec::FrameCodec;
use proxima_protocols::codec_pipe::{Incomplete, OwnFrame};
use proxima_protocols::kafka::{
    KafkaFrameCodec, ParseError as EnvelopeParseError, RequestHeader, parse_frame, parse_request_header,
};

use crate::wire::{self, ApiKey, RequestBody, ResponseBody, WireError};

/// A hard framing/decode problem this codec resolves WITHOUT ever
/// surfacing a [`FrameCodec::Error`] — see the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// The envelope or request header could not be parsed at all — no
    /// trustworthy `correlation_id` to answer against, so this closes
    /// with no reply (mirrors the deleted `Advanced::ProtocolError`).
    Protocol,
    /// A still-incomplete frame already declares a size past `limit` —
    /// closes with no reply (there is no complete header yet to answer
    /// against either).
    MessageTooLarge { limit: usize },
    /// The header parsed fine but the body for this `api_key`/
    /// `api_version` could not be decoded (short/malformed field, or an
    /// unrecognized `api_key`) — closes with no reply even though
    /// `correlation_id` is known, mirroring the deleted
    /// `connection::dispatch`'s catch-all `FrameOutcome::Close`.
    MalformedBody,
    /// The envelope, header, and `api_key` all resolved fine, but the
    /// client declared an `api_version` this facade does not support —
    /// a well-formed, data-free reply under the same `correlation_id`;
    /// the connection stays open.
    UnsupportedVersion { correlation_id: i32, body: ResponseBody },
}

/// [`FrameCodec::Frame`] for Kafka: the SUM of both wire directions (see
/// module doc). [`KafkaCodec::parse_frame`] only ever produces
/// `Request`/`Violation`; `Reply` only ever appears on the encode side,
/// borrowed from a handler's owned outcome
/// (`crate::framed_app::KafkaOutcome::as_frame`).
#[derive(Debug, Clone)]
pub enum KafkaFrame<'a> {
    Request { header: RequestHeader<'a>, body: &'a [u8] },
    Violation(Violation),
    Reply { correlation_id: i32, body: &'a ResponseBody },
}

/// The one error [`KafkaCodec::parse_frame`] ever raises ("the buffer
/// does not hold a complete frame yet"), plus the one
/// [`KafkaCodec::encode_frame`] can raise (a response too large for
/// Kafka's signed 32-bit length prefix — never hit by this facade's own
/// bodies in practice, but a real `FrameCodec::Error` rather than a
/// silently-swallowed empty reply, unlike the deleted
/// `connection::encode_response`'s own log-and-return-empty fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KafkaCodecError {
    #[error("incomplete: frame not yet complete")]
    Incomplete,
    #[error("response of {len} bytes exceeds the i32 length prefix")]
    ResponseTooLarge { len: usize },
}

impl Incomplete for KafkaCodecError {
    fn is_incomplete(&self) -> bool {
        matches!(self, KafkaCodecError::Incomplete)
    }
}

/// Kafka broker-facade [`FrameCodec`]. Carries [`Self::max_message_bytes`]
/// (mirrors `crate::config::KafkaServerConfig::max_message_bytes`) — the
/// DoS cap [`Self::parse_frame`] enforces directly, since a `FrameCodec`
/// is stateless per call and `FramedAny`'s driver hands it the WHOLE
/// currently-buffered window on every attempt (re-parsing from byte zero;
/// see `proxima_listen::any::FramedAny`'s own doc).
#[derive(Debug, Clone, Copy)]
pub struct KafkaCodec {
    pub max_message_bytes: usize,
}

impl KafkaCodec {
    #[must_use]
    pub const fn new(max_message_bytes: usize) -> Self {
        Self { max_message_bytes }
    }
}

impl FrameCodec for KafkaCodec {
    type Frame<'a> = KafkaFrame<'a>;
    type Error = KafkaCodecError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(KafkaFrame<'a>, usize), KafkaCodecError> {
        match parse_frame(buf) {
            Ok((payload, consumed)) => match parse_request_header(payload) {
                Ok((header, body_offset)) => Ok((
                    KafkaFrame::Request {
                        header,
                        body: &payload[body_offset..],
                    },
                    consumed,
                )),
                Err(_malformed_header) => Ok((KafkaFrame::Violation(Violation::Protocol), consumed)),
            },
            Err(EnvelopeParseError::PartialFrame(size)) => {
                if 4 + size as usize > self.max_message_bytes {
                    Ok((
                        KafkaFrame::Violation(Violation::MessageTooLarge {
                            limit: self.max_message_bytes,
                        }),
                        buf.len(),
                    ))
                } else {
                    Err(KafkaCodecError::Incomplete)
                }
            }
            Err(EnvelopeParseError::Short) => Err(KafkaCodecError::Incomplete),
            Err(_invalid_size) => Ok((KafkaFrame::Violation(Violation::Protocol), buf.len())),
        }
    }

    fn encode_frame(&self, frame: &KafkaFrame<'_>, dest: &mut Vec<u8>) -> Result<(), KafkaCodecError> {
        let KafkaFrame::Reply { correlation_id, body } = frame else {
            // never constructed on the encode side — a handler's outcome
            // only ever borrows the `Reply` variant (see
            // `crate::framed_app::KafkaOutcome::as_frame`).
            unreachable!(
                "a Request/Violation frame is never encoded; the App layer renders it as a Reply first"
            )
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(&correlation_id.to_be_bytes());
        payload.extend_from_slice(&body.encode());
        KafkaFrameCodec
            .encode_frame(&payload.as_slice(), dest)
            .map_err(|_frame_too_large| KafkaCodecError::ResponseTooLarge { len: payload.len() })
    }
}

/// [`OwnFrame::Owned`] for [`KafkaCodec`] — the owned mirror of
/// [`KafkaFrame::Request`]/[`KafkaFrame::Violation`] (never `Reply`; that
/// variant only ever appears on the encode side). Deferring
/// [`wire::decode_request`] to here (rather than inside `parse_frame`)
/// means the body is decoded exactly once, matching the deleted
/// `connection::dispatch`'s own single decode call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KafkaOwnedFrame {
    Request { correlation_id: i32, api_key: i16, body: RequestBody },
    Violation(Violation),
}

impl OwnFrame for KafkaCodec {
    type Owned = KafkaOwnedFrame;

    fn own_frame(_source: &Bytes, frame: &KafkaFrame<'_>) -> KafkaOwnedFrame {
        match frame {
            KafkaFrame::Request { header, body } => {
                match wire::decode_request(header.api_key, header.api_version, body) {
                    Ok(decoded) => KafkaOwnedFrame::Request {
                        correlation_id: header.correlation_id,
                        api_key: header.api_key,
                        body: decoded,
                    },
                    Err(WireError::UnsupportedVersion { api_key, .. }) => {
                        KafkaOwnedFrame::Violation(Violation::UnsupportedVersion {
                            correlation_id: header.correlation_id,
                            body: empty_response_for(api_key),
                        })
                    }
                    Err(_malformed_body) => KafkaOwnedFrame::Violation(Violation::MalformedBody),
                }
            }
            KafkaFrame::Violation(violation) => KafkaOwnedFrame::Violation(violation.clone()),
            KafkaFrame::Reply { .. } => {
                // `own_frame`'s own contract (see `codec_pipe::OwnFrame`'s
                // doc) is "given the Bytes window it was PARSED from" —
                // `parse_frame` never produces a `Reply` frame, so this
                // arm is unreachable by construction, not by convention.
                unreachable!("own_frame is only ever called on parse_frame's own output")
            }
        }
    }
}

/// A data-free but well-formed response for `api_key` — used both when a
/// client's declared `api_version` is unsupported ([`OwnFrame::own_frame`])
/// and when the handler pipe itself errors (`crate::framed_app::dispatch`),
/// matching the deleted `connection::empty_response_for` exactly. Every
/// caller already has `api_key` confirmed present in
/// [`wire::SUPPORTED_API_VERSIONS`] (Produce/Fetch/Metadata/ApiVersions) —
/// `ApiKey::Other` is unreachable by construction, not by convention.
pub(crate) fn empty_response_for(api_key: i16) -> ResponseBody {
    match ApiKey::from_i16(api_key) {
        ApiKey::Produce => ResponseBody::Produce(wire::ProduceResponse::default()),
        ApiKey::Fetch => ResponseBody::Fetch(wire::FetchResponse::default()),
        ApiKey::Metadata => ResponseBody::Metadata(wire::MetadataResponse::default()),
        ApiKey::ApiVersions => ResponseBody::ApiVersions(wire::ApiVersionsResponse::supported()),
        ApiKey::Other(other) => unreachable!(
            "empty_response_for called with api_key {other}, which is not present in \
             SUPPORTED_API_VERSIONS — every caller already confirmed this before calling"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::wire::ApiVersionsResponse;

    fn codec() -> KafkaCodec {
        KafkaCodec::new(1024)
    }

    fn encode_request(api_key: i16, api_version: i16, correlation_id: i32, body: &[u8]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&api_key.to_be_bytes());
        payload.extend_from_slice(&api_version.to_be_bytes());
        payload.extend_from_slice(&correlation_id.to_be_bytes());
        payload.extend_from_slice(&(-1_i16).to_be_bytes()); // null client_id
        payload.extend_from_slice(body);

        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&payload.as_slice(), &mut wire)
            .expect("encode");
        wire
    }

    fn api_versions_request(correlation_id: i32) -> Vec<u8> {
        encode_request(ApiKey::ApiVersions.to_i16(), 0, correlation_id, b"")
    }

    #[test]
    fn parse_frame_needs_more_bytes_on_an_empty_buffer() {
        let outcome = codec().parse_frame(b"");
        assert_eq!(outcome.unwrap_err(), KafkaCodecError::Incomplete);
    }

    #[test]
    fn parse_frame_returns_a_complete_request_header() {
        let wire = api_versions_request(7);
        let (frame, consumed) = codec().parse_frame(&wire).expect("parses");
        assert_eq!(consumed, wire.len());
        match frame {
            KafkaFrame::Request { header, body } => {
                assert_eq!(header.api_key, ApiKey::ApiVersions.to_i16());
                assert_eq!(header.correlation_id, 7);
                assert!(body.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_frame_trips_message_too_large_on_a_declared_oversized_frame() {
        let codec = KafkaCodec::new(10);
        // declares a 1000-byte payload but supplies none of it yet.
        let buf = 1000_i32.to_be_bytes();
        let (frame, consumed) = codec.parse_frame(&buf).expect("folds into a violation frame");
        assert_eq!(consumed, buf.len());
        assert!(matches!(
            frame,
            KafkaFrame::Violation(Violation::MessageTooLarge { limit: 10 })
        ));
    }

    #[test]
    fn parse_frame_reports_protocol_violation_on_a_malformed_header() {
        // a frame whose payload is shorter than a v0 header requires.
        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&[0_u8, 1].as_slice(), &mut wire)
            .expect("encode");
        let (frame, consumed) = codec().parse_frame(&wire).expect("folds into a violation frame");
        assert_eq!(consumed, wire.len());
        assert!(matches!(frame, KafkaFrame::Violation(Violation::Protocol)));
    }

    #[test]
    fn own_frame_reowns_a_supported_request_into_a_decoded_body() {
        let wire = api_versions_request(3);
        let (frame, _) = codec().parse_frame(&wire).expect("parses");
        let owned = KafkaCodec::own_frame(&Bytes::new(), &frame);
        assert_eq!(
            owned,
            KafkaOwnedFrame::Request {
                correlation_id: 3,
                api_key: ApiKey::ApiVersions.to_i16(),
                body: RequestBody::ApiVersions,
            }
        );
    }

    #[test]
    fn own_frame_folds_an_unsupported_version_into_a_violation() {
        let wire = encode_request(ApiKey::Produce.to_i16(), 9, 5, b"");
        let (frame, _) = codec().parse_frame(&wire).expect("parses the envelope+header");
        let owned = KafkaCodec::own_frame(&Bytes::new(), &frame);
        assert_eq!(
            owned,
            KafkaOwnedFrame::Violation(Violation::UnsupportedVersion {
                correlation_id: 5,
                body: ResponseBody::Produce(wire::ProduceResponse::default()),
            })
        );
    }

    #[test]
    fn own_frame_folds_a_malformed_body_into_a_violation() {
        // Produce v0 with a truncated body (declares fields it doesn't supply).
        let wire = encode_request(ApiKey::Produce.to_i16(), 0, 1, &[0, 1]);
        let (frame, _) = codec().parse_frame(&wire).expect("parses the envelope+header");
        let owned = KafkaCodec::own_frame(&Bytes::new(), &frame);
        assert_eq!(owned, KafkaOwnedFrame::Violation(Violation::MalformedBody));
    }

    #[test]
    fn own_frame_reowns_a_violation_verbatim() {
        let frame = KafkaFrame::Violation(Violation::MessageTooLarge { limit: 16 });
        let owned = KafkaCodec::own_frame(&Bytes::new(), &frame);
        assert_eq!(owned, KafkaOwnedFrame::Violation(Violation::MessageTooLarge { limit: 16 }));
    }

    #[test]
    fn encode_frame_renders_a_reply_with_its_correlation_id() {
        let mut dest = Vec::new();
        let body = ResponseBody::ApiVersions(ApiVersionsResponse::supported());
        codec()
            .encode_frame(
                &KafkaFrame::Reply {
                    correlation_id: 42,
                    body: &body,
                },
                &mut dest,
            )
            .expect("encode");
        let correlation_id = i32::from_be_bytes([dest[4], dest[5], dest[6], dest[7]]);
        assert_eq!(correlation_id, 42);
    }

    #[test]
    #[should_panic(expected = "never encoded")]
    fn encode_frame_panics_on_a_non_reply_frame() {
        let _ = codec().encode_frame(&KafkaFrame::Violation(Violation::Protocol), &mut Vec::new());
    }
}
