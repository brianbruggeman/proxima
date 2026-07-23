//! Sans-IO AMQP 0-9-1 connection state machine — bytes in
//! ([`Connection::feed_bytes`]), one decoded event out per
//! [`Connection::advance`] call, no socket (workspace principle 11).
//! Mirrors `proxima_protocols::redis::connection::Connection`'s
//! `feed_bytes`/`advance`/[`Advanced`] shape, built on top of
//! [`proxima_protocols::amqp::parse_frame`] (the envelope) +
//! [`crate::method::decode`] (the method arguments) this crate adds.
//!
//! Two things a bare per-frame parser doesn't give a broker for free, both
//! handled here:
//!
//! - **The protocol header.** Before any framed byte, a client sends the
//!   literal 8 bytes `"AMQP\0\0\x09\x01"` — not itself an AMQP frame.
//!   [`Connection::advance`] consumes it once and returns
//!   [`Advanced::ProtocolHeader`]; a mismatch is
//!   [`Advanced::ProtocolError`] (the driver replies with the server's own
//!   protocol header, per AMQP 0-9-1 §4.2.2, then closes).
//! - **Content reassembly.** `basic.publish` (client -> server) and
//!   `basic.deliver` (server -> client) are each a `Method` frame followed
//!   by exactly one `Header` frame and zero or more `Body` frames on the
//!   SAME channel — the only two content-bearing methods this crate
//!   implements. [`Connection`] is direction-agnostic: whichever side
//!   receives one tracks its reassembly state per channel (multiple
//!   channels multiplex independently, and content frames from other
//!   channels may legally interleave between them) and only emits
//!   [`Advanced::Publish`]/[`Advanced::Deliver`] once the declared
//!   `body_size` is fully received, capped by
//!   [`Limits::message_max_bytes`] — the DoS guard against declaring a huge
//!   `body_size` and trickling bytes forever.

// `Advanced` is used as BOTH the success value `advance()` returns to every
// caller AND the internal error carried through `Result<Option<Advanced>,
// Advanced>` while a single frame is being handled (a content-reassembly
// violation reported mid-dispatch) — one shared shape, not a hot per-byte
// path (per-frame only, and `OwnedFrame` already copies each frame's bytes
// once regardless), so boxing it here would add an allocation without a
// real win.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;

use proxima_protocols::amqp::{Frame, ParseError, parse_frame};

use crate::method::{Method, MethodError, decode, id};

/// The literal bytes a client sends before any framed AMQP traffic:
/// `"AMQP"` + protocol-id `0` + major `0` + minor `9` + revision `1`.
pub const PROTOCOL_HEADER: [u8; 8] = *b"AMQP\0\0\x09\x01";

/// DoS caps this connection enforces regardless of what a client's
/// `connection.tune-ok` claims — the server-advertised ceiling always
/// wins (a client cannot negotiate a larger cap than the server offered).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// hard cap on one frame's total wire size (7-byte header + payload +
    /// 1-byte end marker)
    pub frame_max_bytes: usize,
    /// hard cap on one reassembled `basic.publish` message body
    pub message_max_bytes: usize,
}

/// What the driver must do next — mirrors
/// `proxima_protocols::redis::connection::Advanced`'s role for this
/// protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum Advanced {
    /// No complete frame buffered yet — read more bytes, `feed_bytes`, and
    /// call `advance` again.
    NeedMore,
    /// The client's protocol header was consumed and matched.
    ProtocolHeader,
    /// One fully decoded method with no content to follow.
    Frame {
        channel: u16,
        method: Method,
    },
    /// A `basic.publish` method + its content-header + every content-body
    /// frame, fully reassembled. Sent client -> server.
    Publish {
        channel: u16,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        mandatory: bool,
        immediate: bool,
        properties: Vec<u8>,
        body: Vec<u8>,
    },
    /// A `basic.deliver` method + its content-header + every content-body
    /// frame, fully reassembled. Sent server -> client — the symmetric
    /// content-bearing counterpart to [`Advanced::Publish`], reassembled
    /// through the SAME per-channel state machine (content reassembly is
    /// direction-agnostic: whichever side receives a content-bearing
    /// method reassembles it the same way).
    Deliver {
        channel: u16,
        consumer_tag: Vec<u8>,
        delivery_tag: u64,
        redelivered: bool,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        properties: Vec<u8>,
        body: Vec<u8>,
    },
    Heartbeat,
    /// A framing or method-decode violation. The driver renders
    /// `connection.close` (or, pre-handshake, the raw protocol-header
    /// mismatch reply) and ends the connection — class 1 in AMQP's own
    /// error taxonomy (§4.2.2), never silently recovered from.
    ProtocolError {
        reason: String,
    },
    /// One frame's wire size exceeds [`Limits::frame_max_bytes`].
    FrameTooLarge {
        limit: usize,
    },
    /// A reassembled message body exceeds [`Limits::message_max_bytes`].
    MessageTooLarge {
        limit: usize,
    },
}

/// The content-bearing method that opened this channel's reassembly —
/// whichever side's own [`Method`] carried no body itself but declared one
/// is coming ([`Method::BasicPublish`] client -> server,
/// [`Method::BasicDeliver`] server -> client).
#[derive(Debug, Clone)]
enum PendingContent {
    Publish {
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        mandatory: bool,
        immediate: bool,
    },
    Deliver {
        consumer_tag: Vec<u8>,
        delivery_tag: u64,
        redelivered: bool,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
    },
}

#[derive(Debug, Clone)]
enum ChannelState {
    Idle,
    AwaitingHeader {
        pending: PendingContent,
    },
    AwaitingBody {
        pending: PendingContent,
        body_size: u64,
        properties: Vec<u8>,
        received: Vec<u8>,
    },
}

pub struct Connection {
    inbox: Vec<u8>,
    limits: Limits,
    header_seen: bool,
    channels: BTreeMap<u16, ChannelState>,
}

impl Connection {
    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            inbox: Vec::with_capacity(8192),
            limits,
            header_seen: false,
            channels: BTreeMap::new(),
        }
    }

    /// For client-side use: a broker never sends the literal protocol-header
    /// bytes back (only the client sends them, once, up front, as raw bytes
    /// outside the framed-frame grammar entirely) — a client-mode
    /// `Connection` starts straight past the header gate, in framed-frame
    /// mode, since it will never see one arrive.
    #[must_use]
    pub fn for_client(limits: Limits) -> Self {
        let mut connection = Self::with_limits(limits);
        connection.header_seen = true;
        connection
    }

    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.inbox.extend_from_slice(bytes);
    }

    /// Drops per-channel content-reassembly state — call on `channel.close`
    /// (and connection close) so a stale mid-publish state never leaks
    /// past a channel's lifetime.
    pub fn close_channel(&mut self, channel: u16) {
        self.channels.remove(&channel);
    }

    /// Advances the state machine by at most one event. Loops internally
    /// over frames that only update reassembly bookkeeping (a `Header`
    /// frame that isn't yet the last one, e.g.) without yielding anything
    /// to the driver.
    pub fn advance(&mut self) -> Advanced {
        if !self.header_seen {
            return self.advance_header();
        }
        loop {
            // reject an oversized frame from its declared length in the
            // 7-byte envelope header alone — `parse_frame` only succeeds
            // once the FULL declared payload has already arrived, so
            // checking only its `Ok` result would force buffering an
            // attacker's entire oversized payload before ever rejecting
            // it (unbounded memory growth while `NeedMore` loops).
            if let Some(declared_total) = declared_frame_len(&self.inbox)
                && declared_total > self.limits.frame_max_bytes
            {
                return Advanced::FrameTooLarge {
                    limit: self.limits.frame_max_bytes,
                };
            }
            match parse_frame(&self.inbox) {
                Err(ParseError::Short | ParseError::PartialFrame(_)) => return Advanced::NeedMore,
                Err(other) => {
                    return Advanced::ProtocolError {
                        reason: other.to_string(),
                    };
                }
                Ok((frame, consumed)) => {
                    // `frame` borrows `self.inbox` (its byte slices are
                    // zero-copy views into the buffer parse_frame just
                    // read) — copying it to an owned `OwnedFrame` HERE ends
                    // that borrow immediately, which is what lets the next
                    // two lines both touch `self` mutably (drain the
                    // buffer, then dispatch against `self.channels`).
                    // `Method::decode` inside `handle_owned_frame` would
                    // have copied these same bytes into owned `Vec<u8>`
                    // fields anyway; this only moves WHEN that copy
                    // happens, not whether.
                    let owned = OwnedFrame::from(&frame);
                    self.inbox.drain(..consumed);
                    let outcome = self.handle_owned_frame(owned);
                    match outcome {
                        Ok(Some(advanced)) => return advanced,
                        Ok(None) => continue,
                        Err(advanced) => return advanced,
                    }
                }
            }
        }
    }

    fn advance_header(&mut self) -> Advanced {
        if self.inbox.len() < PROTOCOL_HEADER.len() {
            return Advanced::NeedMore;
        }
        if self.inbox[..PROTOCOL_HEADER.len()] != PROTOCOL_HEADER {
            return Advanced::ProtocolError {
                reason: "client sent an unsupported AMQP protocol header".into(),
            };
        }
        self.inbox.drain(..PROTOCOL_HEADER.len());
        self.header_seen = true;
        Advanced::ProtocolHeader
    }

    fn handle_owned_frame(&mut self, frame: OwnedFrame) -> Result<Option<Advanced>, Advanced> {
        match frame {
            OwnedFrame::Heartbeat => Ok(Some(Advanced::Heartbeat)),
            OwnedFrame::Method {
                channel,
                class_id,
                method_id,
                args,
            } => self.handle_method(channel, class_id, method_id, &args),
            OwnedFrame::Header {
                channel,
                class_id,
                body_size,
                properties,
            } => self.handle_header(channel, class_id, body_size, &properties),
            OwnedFrame::Body { channel, payload } => self.handle_body(channel, &payload),
        }
    }

    fn handle_method(
        &mut self,
        channel: u16,
        class_id: u16,
        method_id: u16,
        args: &[u8],
    ) -> Result<Option<Advanced>, Advanced> {
        let state = self.channels.entry(channel).or_insert(ChannelState::Idle);
        if !matches!(state, ChannelState::Idle) {
            return Err(Advanced::ProtocolError {
                reason: format!("method frame on channel {channel} while content is in flight"),
            });
        }
        let method = decode(class_id, method_id, args).map_err(method_error_to_advanced)?;
        let pending = match &method {
            Method::BasicPublish {
                exchange,
                routing_key,
                mandatory,
                immediate,
            } => Some(PendingContent::Publish {
                exchange: exchange.clone(),
                routing_key: routing_key.clone(),
                mandatory: *mandatory,
                immediate: *immediate,
            }),
            Method::BasicDeliver {
                consumer_tag,
                delivery_tag,
                redelivered,
                exchange,
                routing_key,
            } => Some(PendingContent::Deliver {
                consumer_tag: consumer_tag.clone(),
                delivery_tag: *delivery_tag,
                redelivered: *redelivered,
                exchange: exchange.clone(),
                routing_key: routing_key.clone(),
            }),
            _ => None,
        };
        if let Some(pending) = pending {
            *state = ChannelState::AwaitingHeader { pending };
            return Ok(None);
        }
        Ok(Some(Advanced::Frame { channel, method }))
    }

    fn handle_header(
        &mut self,
        channel: u16,
        class_id: u16,
        body_size: u64,
        properties: &[u8],
    ) -> Result<Option<Advanced>, Advanced> {
        if class_id != id::BASIC {
            return Err(Advanced::ProtocolError {
                reason: format!(
                    "content header on channel {channel} names unsupported class {class_id}"
                ),
            });
        }
        if body_size > self.limits.message_max_bytes as u64 {
            return Err(Advanced::MessageTooLarge {
                limit: self.limits.message_max_bytes,
            });
        }
        let state = self.channels.entry(channel).or_insert(ChannelState::Idle);
        let ChannelState::AwaitingHeader { pending } =
            core::mem::replace(state, ChannelState::Idle)
        else {
            return Err(Advanced::ProtocolError {
                reason: format!("unexpected content header on channel {channel}"),
            });
        };
        if body_size == 0 {
            return Ok(Some(content_event(
                channel,
                pending,
                properties.to_vec(),
                Vec::new(),
            )));
        }
        *self.channels.entry(channel).or_insert(ChannelState::Idle) = ChannelState::AwaitingBody {
            pending,
            body_size,
            properties: properties.to_vec(),
            received: Vec::new(),
        };
        Ok(None)
    }

    fn handle_body(&mut self, channel: u16, payload: &[u8]) -> Result<Option<Advanced>, Advanced> {
        let state = self.channels.entry(channel).or_insert(ChannelState::Idle);
        let ChannelState::AwaitingBody {
            pending,
            body_size,
            properties,
            mut received,
        } = core::mem::replace(state, ChannelState::Idle)
        else {
            return Err(Advanced::ProtocolError {
                reason: format!("unexpected content body on channel {channel}"),
            });
        };
        received.extend_from_slice(payload);
        if received.len() as u64 > body_size {
            return Err(Advanced::MessageTooLarge {
                limit: self.limits.message_max_bytes,
            });
        }
        if received.len() as u64 == body_size {
            return Ok(Some(content_event(channel, pending, properties, received)));
        }
        *self.channels.entry(channel).or_insert(ChannelState::Idle) = ChannelState::AwaitingBody {
            pending,
            body_size,
            properties,
            received,
        };
        Ok(None)
    }
}

/// An owned copy of one [`Frame`] — see the comment at its one call site in
/// [`Connection::advance`] for why the copy happens eagerly instead of
/// threading the borrowed `Frame<'_>` further in.
enum OwnedFrame {
    Heartbeat,
    Method {
        channel: u16,
        class_id: u16,
        method_id: u16,
        args: Vec<u8>,
    },
    Header {
        channel: u16,
        class_id: u16,
        body_size: u64,
        properties: Vec<u8>,
    },
    Body {
        channel: u16,
        payload: Vec<u8>,
    },
}

impl From<&Frame<'_>> for OwnedFrame {
    fn from(frame: &Frame<'_>) -> Self {
        match *frame {
            Frame::Heartbeat { .. } => Self::Heartbeat,
            Frame::Method {
                channel,
                class_id,
                method_id,
                args,
            } => Self::Method {
                channel,
                class_id,
                method_id,
                args: args.to_vec(),
            },
            Frame::Header {
                channel,
                class_id,
                body_size,
                properties,
                ..
            } => Self::Header {
                channel,
                class_id,
                body_size,
                properties: properties.to_vec(),
            },
            Frame::Body { channel, payload } => Self::Body {
                channel,
                payload: payload.to_vec(),
            },
        }
    }
}

/// The frame's total wire size (7-byte header + payload + 1-byte end
/// marker) from its declared length field alone — `None` until at least
/// the 7-byte header has arrived.
fn declared_frame_len(buf: &[u8]) -> Option<usize> {
    if buf.len() < 7 {
        return None;
    }
    let declared_payload = u32::from_be_bytes([buf[3], buf[4], buf[5], buf[6]]) as usize;
    Some(7 + declared_payload + 1)
}

fn content_event(
    channel: u16,
    pending: PendingContent,
    properties: Vec<u8>,
    body: Vec<u8>,
) -> Advanced {
    match pending {
        PendingContent::Publish {
            exchange,
            routing_key,
            mandatory,
            immediate,
        } => Advanced::Publish {
            channel,
            exchange,
            routing_key,
            mandatory,
            immediate,
            properties,
            body,
        },
        PendingContent::Deliver {
            consumer_tag,
            delivery_tag,
            redelivered,
            exchange,
            routing_key,
        } => Advanced::Deliver {
            channel,
            consumer_tag,
            delivery_tag,
            redelivered,
            exchange,
            routing_key,
            properties,
            body,
        },
    }
}

fn method_error_to_advanced(error: MethodError) -> Advanced {
    Advanced::ProtocolError {
        reason: error.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::frame::{encode_body_frames, encode_header_frame, encode_method_frame};
    use crate::wire::FieldTable;

    fn limits() -> Limits {
        Limits {
            frame_max_bytes: 128 * 1024,
            message_max_bytes: 1024,
        }
    }

    #[test]
    fn needs_more_before_the_protocol_header_arrives() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(b"AMQP\0\0");
        assert_eq!(connection.advance(), Advanced::NeedMore);
    }

    #[test]
    fn protocol_header_is_consumed_exactly_once() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);
        assert_eq!(connection.advance(), Advanced::NeedMore);
    }

    #[test]
    fn mismatched_protocol_header_is_a_protocol_error() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(b"GET / HTTP/1.1\r\n");
        match connection.advance() {
            Advanced::ProtocolError { .. } => {}
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn decodes_a_plain_method_frame_with_no_content() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            0,
            &Method::ConnectionOpen {
                virtual_host: b"/".to_vec(),
            },
        );
        connection.feed_bytes(&wire);

        match connection.advance() {
            Advanced::Frame { channel, method } => {
                assert_eq!(channel, 0);
                assert_eq!(
                    method,
                    Method::ConnectionOpen {
                        virtual_host: b"/".to_vec()
                    }
                );
            }
            other => panic!("expected Frame, got {other:?}"),
        }
    }

    #[test]
    fn reassembles_a_basic_publish_across_method_header_and_body_frames() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            1,
            &Method::BasicPublish {
                exchange: b"orders".to_vec(),
                routing_key: b"orders.eu".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_header_frame(&mut wire, 1, id::BASIC, 11, b"");
        encode_body_frames(&mut wire, 1, b"hello world", 4);
        connection.feed_bytes(&wire);

        match connection.advance() {
            Advanced::Publish {
                channel,
                exchange,
                routing_key,
                body,
                ..
            } => {
                assert_eq!(channel, 1);
                assert_eq!(exchange, b"orders");
                assert_eq!(routing_key, b"orders.eu");
                assert_eq!(body, b"hello world");
            }
            other => panic!("expected Publish, got {other:?}"),
        }
        assert_eq!(connection.advance(), Advanced::NeedMore);
    }

    #[test]
    fn zero_length_body_publishes_immediately_after_the_header_frame() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            1,
            &Method::BasicPublish {
                exchange: Vec::new(),
                routing_key: b"orders".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_header_frame(&mut wire, 1, id::BASIC, 0, b"");
        connection.feed_bytes(&wire);

        match connection.advance() {
            Advanced::Publish { body, .. } => assert!(body.is_empty()),
            other => panic!("expected Publish, got {other:?}"),
        }
    }

    #[test]
    fn oversized_declared_body_size_is_message_too_large() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            1,
            &Method::BasicPublish {
                exchange: Vec::new(),
                routing_key: b"orders".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_header_frame(&mut wire, 1, id::BASIC, 10_000, b"");
        connection.feed_bytes(&wire);

        assert_eq!(
            connection.advance(),
            Advanced::MessageTooLarge { limit: 1024 }
        );
    }

    #[test]
    fn interleaved_channels_reassemble_independently() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            1,
            &Method::BasicPublish {
                exchange: Vec::new(),
                routing_key: b"q1".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_method_frame(
            &mut wire,
            2,
            &Method::BasicPublish {
                exchange: Vec::new(),
                routing_key: b"q2".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_header_frame(&mut wire, 2, id::BASIC, 2, b"");
        encode_header_frame(&mut wire, 1, id::BASIC, 2, b"");
        encode_body_frames(&mut wire, 2, b"b2", 128);
        encode_body_frames(&mut wire, 1, b"b1", 128);
        connection.feed_bytes(&wire);

        let mut routing_keys = Vec::new();
        loop {
            match connection.advance() {
                Advanced::Publish { routing_key, .. } => routing_keys.push(routing_key),
                Advanced::NeedMore => break,
                other => panic!("unexpected: {other:?}"),
            }
        }
        assert_eq!(routing_keys, vec![b"q2".to_vec(), b"q1".to_vec()]);
    }

    #[test]
    fn a_method_frame_while_content_is_in_flight_is_a_protocol_error() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            1,
            &Method::BasicPublish {
                exchange: Vec::new(),
                routing_key: b"q1".to_vec(),
                mandatory: false,
                immediate: false,
            },
        );
        encode_method_frame(&mut wire, 1, &Method::ChannelOpen);
        connection.feed_bytes(&wire);

        match connection.advance() {
            Advanced::ProtocolError { .. } => {}
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn heartbeat_frame_yields_a_heartbeat_event() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        crate::frame::encode_heartbeat_frame(&mut wire);
        connection.feed_bytes(&wire);
        assert_eq!(connection.advance(), Advanced::Heartbeat);
    }

    #[test]
    fn unsupported_method_surfaces_as_a_protocol_error_not_a_panic() {
        let mut connection = Connection::with_limits(limits());
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        // tx.select (90, 10) — deliberately out of the implemented subset.
        crate::frame::encode_frame(&mut wire, proxima_protocols::amqp::FrameType::Method, 0, &{
            let mut payload = Vec::new();
            payload.extend_from_slice(&90_u16.to_be_bytes());
            payload.extend_from_slice(&10_u16.to_be_bytes());
            payload
        });
        connection.feed_bytes(&wire);

        match connection.advance() {
            Advanced::ProtocolError { .. } => {}
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn frame_exceeding_the_configured_max_is_rejected() {
        let mut connection = Connection::with_limits(Limits {
            frame_max_bytes: 16,
            message_max_bytes: 1024,
        });
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        let mut wire = Vec::new();
        let mut arguments = FieldTable::new();
        arguments.insert(
            "padding".into(),
            crate::wire::FieldValue::LongString(vec![0_u8; 64]),
        );
        encode_method_frame(
            &mut wire,
            0,
            &Method::QueueDeclare {
                queue: b"q".to_vec(),
                passive: false,
                durable: false,
                exclusive: false,
                auto_delete: false,
                no_wait: false,
                arguments,
            },
        );
        connection.feed_bytes(&wire);

        assert_eq!(connection.advance(), Advanced::FrameTooLarge { limit: 16 });
    }

    // proves the DoS cap trips from the 7-byte envelope header alone — a
    // streaming attacker who declares a huge length and then trickles (or
    // never sends) the payload must never force unbounded buffering while
    // `advance` keeps answering `NeedMore`.
    #[test]
    fn frame_too_large_is_detected_from_the_header_alone_without_the_full_payload() {
        let mut connection = Connection::with_limits(Limits {
            frame_max_bytes: 16,
            message_max_bytes: 1024,
        });
        connection.feed_bytes(&PROTOCOL_HEADER);
        assert_eq!(connection.advance(), Advanced::ProtocolHeader);

        // type(1) + channel(2) + declared length(4) = 1_000_000, no payload
        // bytes follow at all.
        let mut header_only = vec![1_u8, 0, 0];
        header_only.extend_from_slice(&1_000_000_u32.to_be_bytes());
        connection.feed_bytes(&header_only);

        assert_eq!(connection.advance(), Advanced::FrameTooLarge { limit: 16 });
    }
}
