//! Sans-IO AMQP 0-9-1 client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`.
//! The client-side mirror of `proxima_redis::client::session::ClientSession`:
//! it owns the startup handshake (protocol header ->
//! `connection.start`/`start-ok` -> `tune`/`tune-ok` -> `open`/`open-ok` ->
//! `channel.open`/`open-ok`) and, once `Ready`, `basic.publish` /
//! `basic.consume`, but never touches a socket (workspace principle 11). It
//! reuses [`crate::fsm::Connection`] for inbound decode + content
//! reassembly — the SAME state machine the listener side uses, since
//! content reassembly is direction-agnostic (see that module's docs).
//!
//! Unlike redis's strictly request/reply wire, AMQP interleaves control
//! replies (`basic.consume-ok`) and asynchronous pushes (`basic.deliver`)
//! on the same connection even outside a dedicated push-only mode, so
//! `advance()` folds both into one [`Step`] enum rather than redis's split
//! `Step`/`PushStep` — a caller drains whichever variant it gets.

use crate::frame::{encode_body_frames, encode_header_frame, encode_method_frame};
use crate::fsm::{Advanced, Connection as Fsm, Limits, PROTOCOL_HEADER};
use crate::method::{Method, id};
use crate::wire::{FieldTable, FieldValue};

use crate::client::config::AmqpClientConfig;

/// Fixed client-side DoS ceiling against a misbehaving broker — this MVP
/// client does not yet negotiate its own cap down from what the broker's
/// `connection.tune` offers (it always accepts the broker's own values),
/// so this is the outer safety bound regardless.
const CLIENT_FRAME_MAX_BYTES: usize = 64 * 1024 * 1024;
const CLIENT_MESSAGE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// The one channel this client opens and uses for every publish/consume —
/// a client wanting multiple concurrent channels is out of this MVP's
/// scope (see the crate-level gap notes).
const CLIENT_CHANNEL: u16 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The broker closed the connection or channel with a reason
    /// (`connection.close`/`channel.close`).
    #[error("server: {0}")]
    Server(String),
    #[error("server connection closed mid-reply")]
    Closed,
    #[error("protocol: {0}")]
    Protocol(String),
}

/// What the driver must do next to advance the session. The driver owns
/// I/O; the session owns the protocol.
#[derive(Debug)]
pub enum Step {
    /// Bytes are queued — write `take_outbound()` to the transport, then
    /// call `advance()` again.
    Send,
    /// No progress without more inbound bytes — read, `feed()`, then
    /// `advance()` again.
    Recv,
    /// Handshake complete and no pending event; the session is idle and
    /// ready for `queue_publish`/`queue_consume`.
    Ready,
    /// The broker's reply to a `queue_consume` call.
    ConsumeOk { consumer_tag: Vec<u8> },
    /// One pushed `basic.deliver`, fully reassembled.
    Delivery {
        consumer_tag: Vec<u8>,
        delivery_tag: u64,
        redelivered: bool,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        properties: Vec<u8>,
        body: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    AwaitingStart,
    AwaitingTune,
    AwaitingOpenOk,
    AwaitingChannelOpenOk,
    Ready,
}

pub struct ClientSession {
    fsm: Fsm,
    outbound: Vec<u8>,
    phase: Phase,
    virtual_host: Vec<u8>,
    username: Vec<u8>,
    password: Vec<u8>,
    negotiated_frame_max: usize,
}

impl ClientSession {
    /// Builds a session and queues the client's protocol header — the
    /// literal 8 bytes sent before any framed traffic (not itself a frame,
    /// so it rides `outbound` directly rather than through
    /// `crate::frame::encode_frame`).
    #[must_use]
    pub fn new(config: &AmqpClientConfig) -> Self {
        let mut outbound = Vec::with_capacity(8192);
        outbound.extend_from_slice(&PROTOCOL_HEADER);
        Self {
            fsm: Fsm::for_client(Limits {
                frame_max_bytes: CLIENT_FRAME_MAX_BYTES,
                message_max_bytes: CLIENT_MESSAGE_MAX_BYTES,
            }),
            outbound,
            phase: Phase::AwaitingStart,
            virtual_host: config.virtual_host.clone().into_bytes(),
            username: config.username.clone().into_bytes(),
            password: config.password.clone().into_bytes(),
            negotiated_frame_max: CLIENT_FRAME_MAX_BYTES,
        }
    }

    pub fn take_outbound(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.outbound)
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        self.fsm.feed_bytes(bytes);
    }

    /// `basic.publish` on the client's one channel, followed by the
    /// content-header and content-body frames. Fire-and-forget — AMQP 0-9-1
    /// has no synchronous per-publish reply outside publisher-confirms (not
    /// implemented; see the crate-level gap notes).
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if called before the session reaches
    /// [`Step::Ready`].
    pub fn queue_publish(
        &mut self,
        exchange: &[u8],
        routing_key: &[u8],
        mandatory: bool,
        immediate: bool,
        properties: &[u8],
        body: &[u8],
    ) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("publish before ready".into()));
        }
        encode_method_frame(
            &mut self.outbound,
            CLIENT_CHANNEL,
            &Method::BasicPublish {
                exchange: exchange.to_vec(),
                routing_key: routing_key.to_vec(),
                mandatory,
                immediate,
            },
        );
        encode_header_frame(
            &mut self.outbound,
            CLIENT_CHANNEL,
            id::BASIC,
            body.len() as u64,
            properties,
        );
        encode_body_frames(
            &mut self.outbound,
            CLIENT_CHANNEL,
            body,
            self.negotiated_frame_max.saturating_sub(8),
        );
        Ok(())
    }

    /// `basic.consume` on the client's one channel. The reply (`Step::ConsumeOk`)
    /// arrives on a later `advance()`; every subsequent `basic.deliver`
    /// yields `Step::Delivery`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if called before the session reaches
    /// [`Step::Ready`].
    pub fn queue_consume(
        &mut self,
        queue: &[u8],
        consumer_tag: &[u8],
        no_ack: bool,
    ) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("consume before ready".into()));
        }
        encode_method_frame(
            &mut self.outbound,
            CLIENT_CHANNEL,
            &Method::BasicConsume {
                queue: queue.to_vec(),
                consumer_tag: consumer_tag.to_vec(),
                no_local: false,
                no_ack,
                exclusive: false,
                no_wait: false,
                arguments: FieldTable::new(),
            },
        );
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then processes
    /// inbound frames — driving the handshake while not yet `Ready`, or
    /// surfacing the next `ConsumeOk`/`Delivery` event once it is.
    ///
    /// # Errors
    /// [`ClientError`] on a broker-reported close or a malformed frame.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        loop {
            match self.fsm.advance() {
                Advanced::NeedMore => return Ok(Step::Recv),
                Advanced::ProtocolHeader => {
                    // never produced for a client-mode `Fsm` (constructed via
                    // `Fsm::for_client`, which starts past the header gate).
                    continue;
                }
                Advanced::Heartbeat => continue,
                Advanced::Frame { method, .. } => match self.handle_method(method)? {
                    Some(step) => return Ok(step),
                    None => continue,
                },
                Advanced::Deliver {
                    consumer_tag,
                    delivery_tag,
                    redelivered,
                    exchange,
                    routing_key,
                    properties,
                    body,
                    ..
                } => {
                    return Ok(Step::Delivery {
                        consumer_tag,
                        delivery_tag,
                        redelivered,
                        exchange,
                        routing_key,
                        properties,
                        body,
                    });
                }
                Advanced::Publish { .. } => {
                    return Err(ClientError::Protocol(
                        "broker sent basic.publish (server-only method)".into(),
                    ));
                }
                Advanced::ProtocolError { reason } => return Err(ClientError::Protocol(reason)),
                Advanced::FrameTooLarge { limit } => {
                    return Err(ClientError::Protocol(format!(
                        "frame exceeds the {limit}-byte cap"
                    )));
                }
                Advanced::MessageTooLarge { limit } => {
                    return Err(ClientError::Protocol(format!(
                        "message body exceeds the {limit}-byte cap"
                    )));
                }
            }
        }
    }

    fn handle_method(&mut self, method: Method) -> Result<Option<Step>, ClientError> {
        match (self.phase, method) {
            (Phase::AwaitingStart, Method::ConnectionStart { .. }) => {
                self.outbound_start_ok();
                self.phase = Phase::AwaitingTune;
                Ok(Some(Step::Send))
            }
            (
                Phase::AwaitingTune,
                Method::ConnectionTune {
                    channel_max,
                    frame_max,
                    heartbeat,
                },
            ) => {
                self.negotiated_frame_max = frame_max as usize;
                encode_method_frame(
                    &mut self.outbound,
                    0,
                    &Method::ConnectionTuneOk {
                        channel_max,
                        frame_max,
                        heartbeat,
                    },
                );
                encode_method_frame(
                    &mut self.outbound,
                    0,
                    &Method::ConnectionOpen {
                        virtual_host: self.virtual_host.clone(),
                    },
                );
                self.phase = Phase::AwaitingOpenOk;
                Ok(Some(Step::Send))
            }
            (Phase::AwaitingOpenOk, Method::ConnectionOpenOk) => {
                encode_method_frame(&mut self.outbound, CLIENT_CHANNEL, &Method::ChannelOpen);
                self.phase = Phase::AwaitingChannelOpenOk;
                Ok(Some(Step::Send))
            }
            (Phase::AwaitingChannelOpenOk, Method::ChannelOpenOk) => {
                self.phase = Phase::Ready;
                Ok(Some(Step::Ready))
            }
            (Phase::Ready, Method::BasicConsumeOk { consumer_tag }) => {
                Ok(Some(Step::ConsumeOk { consumer_tag }))
            }
            (
                _,
                Method::ConnectionClose {
                    reply_code,
                    reply_text,
                    ..
                },
            ) => Err(ClientError::Server(format!(
                "connection.close {reply_code}: {}",
                String::from_utf8_lossy(&reply_text)
            ))),
            (
                _,
                Method::ChannelClose {
                    reply_code,
                    reply_text,
                    ..
                },
            ) => Err(ClientError::Server(format!(
                "channel.close {reply_code}: {}",
                String::from_utf8_lossy(&reply_text)
            ))),
            // heartbeats, basic.qos-ok (unsent by this client so unseen in
            // practice), or a method arriving out of the expected handshake
            // order — drained without surfacing a `Step`, mirroring redis's
            // "keep parsing buffered frames" loop.
            (_, _) => Ok(None),
        }
    }

    fn outbound_start_ok(&mut self) {
        let mut response = Vec::with_capacity(2 + self.username.len() + self.password.len());
        response.push(0);
        response.extend_from_slice(&self.username);
        response.push(0);
        response.extend_from_slice(&self.password);

        let mut client_properties = FieldTable::new();
        client_properties.insert(
            "product".into(),
            FieldValue::LongString(b"proxima-amqp".to_vec()),
        );
        client_properties.insert(
            "version".into(),
            FieldValue::LongString(env!("CARGO_PKG_VERSION").as_bytes().to_vec()),
        );

        encode_method_frame(
            &mut self.outbound,
            0,
            &Method::ConnectionStartOk {
                client_properties,
                mechanism: b"PLAIN".to_vec(),
                response,
                locale: b"en_US".to_vec(),
            },
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::wire::FieldTable as WireFieldTable;

    fn config() -> AmqpClientConfig {
        AmqpClientConfig::builder().virtual_host("/").build()
    }

    fn server_start_frame() -> Vec<u8> {
        let mut wire = Vec::new();
        encode_method_frame(
            &mut wire,
            0,
            &Method::ConnectionStart {
                version_major: 0,
                version_minor: 9,
                server_properties: WireFieldTable::new(),
                mechanisms: b"PLAIN".to_vec(),
                locales: b"en_US".to_vec(),
            },
        );
        wire
    }

    fn drive_to_ready(session: &mut ClientSession) {
        assert!(matches!(session.advance().unwrap(), Step::Send));
        assert_eq!(session.take_outbound(), PROTOCOL_HEADER);

        session.feed(&server_start_frame());
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let _start_ok = session.take_outbound();

        let mut tune = Vec::new();
        encode_method_frame(
            &mut tune,
            0,
            &Method::ConnectionTune {
                channel_max: 2047,
                frame_max: 131_072,
                heartbeat: 60,
            },
        );
        session.feed(&tune);
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let _tune_ok_and_open = session.take_outbound();

        let mut open_ok = Vec::new();
        encode_method_frame(&mut open_ok, 0, &Method::ConnectionOpenOk);
        session.feed(&open_ok);
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let _channel_open = session.take_outbound();

        let mut channel_open_ok = Vec::new();
        encode_method_frame(&mut channel_open_ok, CLIENT_CHANNEL, &Method::ChannelOpenOk);
        session.feed(&channel_open_ok);
        assert!(matches!(session.advance().unwrap(), Step::Ready));
    }

    #[test]
    fn handshake_reaches_ready_through_every_negotiation_step() {
        let mut session = ClientSession::new(&config());
        drive_to_ready(&mut session);
    }

    #[test]
    fn publish_before_ready_is_rejected() {
        let mut session = ClientSession::new(&config());
        assert!(
            session
                .queue_publish(b"", b"q", false, false, b"", b"body")
                .is_err()
        );
    }

    #[test]
    fn publish_encodes_method_header_and_body_frames() {
        let mut session = ClientSession::new(&config());
        drive_to_ready(&mut session);

        session
            .queue_publish(b"", b"orders", false, false, b"", b"hello")
            .expect("queue_publish");
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let wire = session.take_outbound();

        let (frame, consumed) = proxima_protocols::amqp::parse_frame(&wire).expect("method frame");
        assert!(matches!(
            frame,
            proxima_protocols::amqp::Frame::Method { .. }
        ));
        let rest = &wire[consumed..];
        let (frame, consumed) = proxima_protocols::amqp::parse_frame(rest).expect("header frame");
        assert!(matches!(
            frame,
            proxima_protocols::amqp::Frame::Header { .. }
        ));
        let rest = &rest[consumed..];
        let (frame, _) = proxima_protocols::amqp::parse_frame(rest).expect("body frame");
        match frame {
            proxima_protocols::amqp::Frame::Body { payload, .. } => assert_eq!(payload, b"hello"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn consume_ok_and_delivery_round_trip() {
        let mut session = ClientSession::new(&config());
        drive_to_ready(&mut session);

        session
            .queue_consume(b"orders", b"", false)
            .expect("queue_consume");
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let _consume = session.take_outbound();

        let mut consume_ok = Vec::new();
        encode_method_frame(
            &mut consume_ok,
            CLIENT_CHANNEL,
            &Method::BasicConsumeOk {
                consumer_tag: b"ctag-1".to_vec(),
            },
        );
        session.feed(&consume_ok);
        match session.advance().unwrap() {
            Step::ConsumeOk { consumer_tag } => assert_eq!(consumer_tag, b"ctag-1"),
            other => panic!("expected ConsumeOk, got {other:?}"),
        }

        let mut deliver_wire = Vec::new();
        encode_method_frame(
            &mut deliver_wire,
            CLIENT_CHANNEL,
            &Method::BasicDeliver {
                consumer_tag: b"ctag-1".to_vec(),
                delivery_tag: 1,
                redelivered: false,
                exchange: b"".to_vec(),
                routing_key: b"orders".to_vec(),
            },
        );
        encode_header_frame(&mut deliver_wire, CLIENT_CHANNEL, id::BASIC, 5, b"");
        encode_body_frames(&mut deliver_wire, CLIENT_CHANNEL, b"hello", 128);
        session.feed(&deliver_wire);

        match session.advance().unwrap() {
            Step::Delivery {
                consumer_tag,
                body,
                routing_key,
                ..
            } => {
                assert_eq!(consumer_tag, b"ctag-1");
                assert_eq!(body, b"hello");
                assert_eq!(routing_key, b"orders");
            }
            other => panic!("expected Delivery, got {other:?}"),
        }
    }

    #[test]
    fn connection_close_during_handshake_surfaces_as_a_server_error() {
        let mut session = ClientSession::new(&config());
        assert!(matches!(session.advance().unwrap(), Step::Send));
        let _ = session.take_outbound();

        let mut close = Vec::new();
        encode_method_frame(
            &mut close,
            0,
            &Method::ConnectionClose {
                reply_code: 530,
                reply_text: b"NOT_ALLOWED".to_vec(),
                class_id: 0,
                method_id: 0,
            },
        );
        session.feed(&close);
        match session.advance() {
            Err(ClientError::Server(message)) => assert!(message.contains("NOT_ALLOWED")),
            other => panic!("expected a server error, got {other:?}"),
        }
    }
}
