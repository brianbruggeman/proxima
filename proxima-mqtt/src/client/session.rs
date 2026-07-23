//! Sans-IO MQTT client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`.
//! The client-side mirror of `proxima_redis::client::session::ClientSession`:
//! it owns the `CONNECT`/`CONNACK` handshake and the request/reply
//! exchange, but never touches a socket (workspace principle 11). The
//! `PipeFactory` client wraps it — that is what makes the client agnostic
//! to the transport shape.
//!
//! The FSM is a two-state enum ([`Phase`]): `Handshake` sends the queued
//! `CONNECT` and awaits `CONNACK`; `Ready` accepts one request at a time
//! (`PUBLISH`/`UNSUBSCRIBE`/`PINGREQ`) and yields its single reply.
//! `SUBSCRIBE` leaves the request/reply rhythm — mirrors redis's own
//! SUBSCRIBE/PSUBSCRIBE scope: after queuing it the driver reads pushed
//! frames with [`ClientSession::poll_push`], whose first frame IS the
//! `SUBACK` (the same "the subscribe ack is just the stream's first item"
//! shape `RedisClientUpstream::stream` uses).
//!
//! Scope boundary: this client drives QoS 0 (`PUBLISH` completes once
//! sent, no wire wait) and QoS 1 (`PUBLISH` waits for `PUBACK`). QoS 2's
//! `PUBREC`/`PUBREL`/`PUBCOMP` three-way handshake is not implemented —
//! [`ClientSession::submit_publish`] rejects `qos == 2` — the same
//! incremental-scope call `proxima_protocols::mqtt`'s module doc makes for
//! MQTT v5.

use std::collections::VecDeque;

use proxima_protocols::mqtt::encode::{
    encode_connect, encode_pingreq, encode_publish, encode_subscribe, encode_unsubscribe,
};
use proxima_protocols::mqtt::{MqttReply, PacketType, ParseError, Packet, parse_packet};

use crate::client::config::MqttClientConfig;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A non-zero `CONNACK` return code during the handshake (bad
    /// credentials, unacceptable protocol version, ...).
    #[error("server: {0}")]
    Server(String),
    #[error("server connection closed mid-reply")]
    Closed,
    #[error("protocol: {0}")]
    Protocol(&'static str),
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
    /// Idle: no request is pending. `PUBLISH`/`UNSUBSCRIBE`/`PING` may be
    /// submitted.
    Ready,
    /// The in-flight request's reply.
    Complete(MqttReply),
}

/// One step of the `SUBSCRIBE` push loop, driven after
/// [`ClientSession::queue_subscribe`].
#[derive(Debug)]
pub enum PushStep {
    /// Need more inbound bytes for a complete frame.
    Recv,
    /// One frame: the `SUBACK` (always first) or a subsequently pushed
    /// `PUBLISH`.
    Frame(MqttReply),
}

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    Handshake,
    Ready,
}

/// The single in-flight request/reply this session tracks — MQTT has no
/// pipelining, so only one may be outstanding at a time (mirrors
/// `RedisClientUpstream`'s pool-of-one cached connection: callers serialize
/// through a mutex).
#[derive(Debug, Clone, Copy)]
enum Pending {
    PubAck(u16),
    UnsubAck(u16),
    PingResp,
}

pub struct ClientSession {
    inbox: Vec<u8>,
    outbound: Vec<u8>,
    phase: Phase,
    pending: Option<Pending>,
    /// Set the moment a QoS 0 `PUBLISH` is queued — the next `advance()`
    /// call completes it immediately with no wire wait, since QoS 0 has no
    /// acknowledgement packet.
    immediate_complete: VecDeque<MqttReply>,
    next_packet_id: u16,
}

impl ClientSession {
    /// Builds a session and queues the `CONNECT` handshake packet derived
    /// from `config`.
    #[must_use]
    pub fn new(config: &MqttClientConfig) -> Self {
        let mut outbound = Vec::with_capacity(64);
        let username = (!config.username.is_empty()).then_some(config.username.as_bytes());
        let password = (!config.password.is_empty()).then_some(config.password.as_bytes());
        encode_connect(
            config.client_id.as_bytes(),
            config.clean_session,
            config.keep_alive,
            username,
            password,
            &mut outbound,
        );
        Self {
            inbox: Vec::with_capacity(8192),
            outbound,
            phase: Phase::Handshake,
            pending: None,
            immediate_complete: VecDeque::new(),
            next_packet_id: 1,
        }
    }

    /// Drains the bytes the driver must send.
    pub fn take_outbound(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.outbound)
    }

    /// Appends bytes the driver read from the transport.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.inbox.extend_from_slice(bytes);
    }

    fn alloc_packet_id(&mut self) -> u16 {
        let id = self.next_packet_id;
        self.next_packet_id = if self.next_packet_id == u16::MAX {
            1
        } else {
            self.next_packet_id + 1
        };
        id
    }

    /// Queues a `PUBLISH`. QoS 0 completes on the next `advance()` with no
    /// wire wait; QoS 1 waits for `PUBACK`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready, a reply is already pending,
    /// or `qos == 2` (unsupported — see the module doc).
    pub fn submit_publish(
        &mut self,
        topic: &[u8],
        payload: &[u8],
        qos: u8,
        retain: bool,
    ) -> Result<(), ClientError> {
        self.guard_ready_and_idle()?;
        match qos {
            0 => {
                encode_publish(topic, None, payload, 0, false, retain, &mut self.outbound);
                self.immediate_complete.push_back(MqttReply::Published);
            }
            1 => {
                let id = self.alloc_packet_id();
                encode_publish(topic, Some(id), payload, 1, false, retain, &mut self.outbound);
                self.pending = Some(Pending::PubAck(id));
            }
            _ => return Err(ClientError::Protocol("qos 2 publish is not supported by this client")),
        }
        Ok(())
    }

    /// Queues a `SUBSCRIBE`. Does not set a pending reply — the driver
    /// switches to [`Self::poll_push`] afterward; its first frame is the
    /// `SUBACK`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready.
    pub fn queue_subscribe(&mut self, filters: &[(&[u8], u8)]) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("subscribe before ready"));
        }
        let id = self.alloc_packet_id();
        encode_subscribe(id, filters, &mut self.outbound);
        Ok(())
    }

    /// Queues an `UNSUBSCRIBE`, awaiting `UNSUBACK`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready or a reply is already pending.
    pub fn submit_unsubscribe(&mut self, filters: &[&[u8]]) -> Result<(), ClientError> {
        self.guard_ready_and_idle()?;
        let id = self.alloc_packet_id();
        encode_unsubscribe(id, filters, &mut self.outbound);
        self.pending = Some(Pending::UnsubAck(id));
        Ok(())
    }

    /// Queues a `PINGREQ`, awaiting `PINGRESP`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready or a reply is already pending.
    pub fn submit_ping(&mut self) -> Result<(), ClientError> {
        self.guard_ready_and_idle()?;
        encode_pingreq(&mut self.outbound);
        self.pending = Some(Pending::PingResp);
        Ok(())
    }

    fn guard_ready_and_idle(&self) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("request before ready"));
        }
        if self.pending.is_some() {
            return Err(ClientError::Protocol("request while a reply is pending"));
        }
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then parses inbound
    /// frames until it needs more bytes or reaches a checkpoint.
    ///
    /// # Errors
    /// [`ClientError`] on a server error during the handshake, a malformed
    /// frame, or a reply that does not match the pending request.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        if let Some(reply) = self.immediate_complete.pop_front() {
            return Ok(Step::Complete(reply));
        }
        match self.phase {
            Phase::Handshake => self.advance_handshake(),
            Phase::Ready => self.advance_ready(),
        }
    }

    /// Reads one pushed frame (the `SUBACK`, then subsequent `PUBLISH`
    /// deliveries) without sending.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] on a malformed frame or an unexpected
    /// packet type.
    pub fn poll_push(&mut self) -> Result<PushStep, ClientError> {
        match self.next_inbound()? {
            None => Ok(PushStep::Recv),
            Some(Inbound::SubAck { packet_id, granted }) => {
                Ok(PushStep::Frame(MqttReply::SubAck { packet_id, granted }))
            }
            Some(Inbound::Publish { topic, payload, qos, retain }) => {
                Ok(PushStep::Frame(MqttReply::Publish { topic, payload, qos, retain }))
            }
            Some(_) => Err(ClientError::Protocol("unexpected packet while awaiting pub/sub pushes")),
        }
    }

    fn advance_handshake(&mut self) -> Result<Step, ClientError> {
        match self.next_inbound()? {
            None => Ok(Step::Recv),
            Some(Inbound::ConnAck { return_code }) => {
                if return_code != 0 {
                    return Err(ClientError::Server(connack_error(return_code)));
                }
                self.phase = Phase::Ready;
                Ok(Step::Ready)
            }
            Some(_) => Err(ClientError::Protocol("expected CONNACK during the handshake")),
        }
    }

    fn advance_ready(&mut self) -> Result<Step, ClientError> {
        let Some(pending) = self.pending else {
            return Ok(Step::Ready);
        };
        match self.next_inbound()? {
            None => Ok(Step::Recv),
            Some(inbound) => {
                let reply = complete_pending(pending, inbound)?;
                self.pending = None;
                Ok(Step::Complete(reply))
            }
        }
    }

    /// Parses one logical packet from the inbox, owning it and draining
    /// the consumed bytes.
    fn next_inbound(&mut self) -> Result<Option<Inbound>, ClientError> {
        let (packet, consumed) = match parse_packet(&self.inbox) {
            Err(ParseError::Short | ParseError::PartialPacket(_)) => return Ok(None),
            Err(ParseError::InvalidPacketType(_)) => {
                return Err(ClientError::Protocol("invalid packet type"));
            }
            Err(ParseError::RemainingLengthOverflow) => {
                return Err(ClientError::Protocol("remaining-length varint exceeds 4 bytes"));
            }
            Err(ParseError::Malformed(reason)) => return Err(ClientError::Protocol(reason)),
            Ok(result) => result,
        };
        let inbound = match packet {
            Packet::ConnAck { return_code, .. } => Inbound::ConnAck { return_code },
            Packet::Ack { packet_type: PacketType::PubAck, packet_id } => {
                Inbound::PubAck { packet_id }
            }
            Packet::Ack { packet_type: PacketType::UnsubAck, packet_id } => {
                Inbound::UnsubAck { packet_id }
            }
            Packet::SubAck { packet_id, return_codes } => {
                Inbound::SubAck { packet_id, granted: return_codes.to_vec() }
            }
            Packet::PingResp => Inbound::PingResp,
            Packet::Publish { flags, topic, payload, .. } => Inbound::Publish {
                topic: topic.to_vec(),
                payload: payload.to_vec(),
                qos: flags.qos,
                retain: flags.retain,
            },
            _ => return Err(ClientError::Protocol("unexpected packet from the broker")),
        };
        self.inbox.drain(..consumed);
        Ok(Some(inbound))
    }
}

/// Owned, correlation-relevant view of an inbound packet — the client's
/// analogue of `RespValue::from_frame`'s borrowed-to-owned lift, scoped to
/// exactly the packet shapes a broker legitimately sends this client.
enum Inbound {
    ConnAck { return_code: u8 },
    PubAck { packet_id: u16 },
    SubAck { packet_id: u16, granted: Vec<u8> },
    UnsubAck { packet_id: u16 },
    PingResp,
    Publish { topic: Vec<u8>, payload: Vec<u8>, qos: u8, retain: bool },
}

fn complete_pending(pending: Pending, inbound: Inbound) -> Result<MqttReply, ClientError> {
    match (pending, inbound) {
        (Pending::PubAck(expected), Inbound::PubAck { packet_id }) if packet_id == expected => {
            Ok(MqttReply::PubAck { packet_id })
        }
        (Pending::UnsubAck(expected), Inbound::UnsubAck { packet_id }) if packet_id == expected => {
            Ok(MqttReply::UnsubAck { packet_id })
        }
        (Pending::PingResp, Inbound::PingResp) => Ok(MqttReply::Pong),
        _ => Err(ClientError::Protocol("reply did not match the pending request")),
    }
}

fn connack_error(return_code: u8) -> String {
    let reason = match return_code {
        1 => "unacceptable protocol version",
        2 => "identifier rejected",
        3 => "server unavailable",
        4 => "bad username or password",
        5 => "not authorized",
        _ => "unknown CONNACK return code",
    };
    format!("CONNACK {return_code}: {reason}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::mqtt::encode::{encode_connack, encode_pingresp, encode_publish as encode_publish_frame, encode_suback};

    fn drive_handshake(session: &mut ClientSession, server_replies: &[&[u8]]) {
        let mut reply_index = 0;
        loop {
            match session.advance().expect("advance") {
                Step::Send => {
                    let _sent = session.take_outbound();
                    if reply_index < server_replies.len() {
                        session.feed(server_replies[reply_index]);
                        reply_index += 1;
                    }
                }
                Step::Recv => {
                    if reply_index < server_replies.len() {
                        session.feed(server_replies[reply_index]);
                        reply_index += 1;
                    } else {
                        panic!("session wants more bytes but the script is exhausted");
                    }
                }
                Step::Ready => return,
                Step::Complete(_) => panic!("unexpected reply during handshake"),
            }
        }
    }

    fn connack(return_code: u8) -> Vec<u8> {
        let mut out = Vec::new();
        encode_connack(false, return_code, &mut out);
        out
    }

    #[test]
    fn handshake_sends_connect_then_becomes_ready_on_connack() {
        let config = MqttClientConfig::builder().client_id("c1").build();
        let mut session = ClientSession::new(&config);

        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send (CONNECT), got {other:?}"),
        }
        let sent = session.take_outbound();
        assert_eq!(sent[0], 0x10, "CONNECT fixed header");

        session.feed(&connack(0));
        match session.advance().expect("advance") {
            Step::Ready => {}
            other => panic!("expected Ready after CONNACK, got {other:?}"),
        }
    }

    #[test]
    fn handshake_surfaces_a_non_zero_connack_return_code_as_a_server_error() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        let _ = session.advance().expect("advance");
        let _ = session.take_outbound();
        session.feed(&connack(5));
        match session.advance() {
            Err(ClientError::Server(message)) => assert!(message.contains('5')),
            other => panic!("expected server error, got {other:?}"),
        }
    }

    #[test]
    fn qos0_publish_completes_immediately_with_no_wire_wait() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[connack(0).as_slice()]);

        session.submit_publish(b"a/b", b"hi", 0, false).expect("submit");
        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send (PUBLISH), got {other:?}"),
        }
        let _ = session.take_outbound();
        match session.advance().expect("advance") {
            Step::Complete(MqttReply::Published) => {}
            other => panic!("expected immediate Complete(Published), got {other:?}"),
        }
    }

    #[test]
    fn qos1_publish_waits_for_a_matching_puback() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[connack(0).as_slice()]);

        session.submit_publish(b"a/b", b"hi", 1, false).expect("submit");
        let sent = session.take_outbound();
        let (packet, _) = proxima_protocols::mqtt::parse_packet(&sent).expect("valid PUBLISH");
        let proxima_protocols::mqtt::Packet::Publish { packet_id: Some(id), .. } = packet else {
            panic!("expected a QoS 1 PUBLISH with a packet id");
        };

        let mut ack = Vec::new();
        proxima_protocols::mqtt::encode::encode_ack(proxima_protocols::mqtt::PacketType::PubAck, id, &mut ack);
        session.feed(&ack);
        match session.advance().expect("advance") {
            Step::Complete(MqttReply::PubAck { packet_id }) => assert_eq!(packet_id, id),
            other => panic!("expected Complete(PubAck), got {other:?}"),
        }
    }

    #[test]
    fn qos2_publish_is_rejected() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[connack(0).as_slice()]);
        assert!(matches!(
            session.submit_publish(b"a/b", b"hi", 2, false),
            Err(ClientError::Protocol(_))
        ));
    }

    #[test]
    fn subscribe_push_loop_yields_suback_then_pushed_publishes() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[connack(0).as_slice()]);

        session.queue_subscribe(&[(b"news/#", 0)]).expect("queue");
        let _ = session.take_outbound();

        let mut suback = Vec::new();
        encode_suback(1, &[0], &mut suback);
        session.feed(&suback);
        match session.poll_push().expect("poll") {
            PushStep::Frame(MqttReply::SubAck { granted, .. }) => assert_eq!(granted, vec![0]),
            other => panic!("expected the SUBACK first, got {other:?}"),
        }

        let mut published = Vec::new();
        encode_publish_frame(b"news/tech", None, b"hi", 0, false, false, &mut published);
        session.feed(&published);
        match session.poll_push().expect("poll") {
            PushStep::Frame(MqttReply::Publish { topic, payload, .. }) => {
                assert_eq!(topic, b"news/tech");
                assert_eq!(payload, b"hi");
            }
            other => panic!("expected a pushed PUBLISH, got {other:?}"),
        }
        assert!(matches!(session.poll_push().expect("poll"), PushStep::Recv));
    }

    #[test]
    fn ping_waits_for_pingresp() {
        let config = MqttClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[connack(0).as_slice()]);

        session.submit_ping().expect("submit");
        let _ = session.take_outbound();
        let mut pong = Vec::new();
        encode_pingresp(&mut pong);
        session.feed(&pong);
        assert!(matches!(session.advance().expect("advance"), Step::Complete(MqttReply::Pong)));
    }
}
