//! Sans-IO Kafka client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`.
//! The client-side mirror of `proxima_redis::client::session::ClientSession`:
//! it opens with an `ApiVersions` handshake (the real thing every modern
//! Kafka client sends first) validating the broker answers with
//! `error_code == NONE`, then accepts one Produce/Fetch request at a time
//! and yields its single reply — never touching a socket (workspace
//! principle 11).
//!
//! One wrinkle redis's session never has: a Kafka response header carries
//! only `correlation_id`, never `api_key` — the client alone remembers
//! what it asked for. [`ClientSession`] tracks exactly one in-flight
//! request as `pending: Option<(correlation_id, api_key)>` and uses that
//! recorded `api_key` to pick [`crate::wire::decode_response`]'s decoder.

use proxima_codec::FrameCodec;
use proxima_protocols::kafka::{KafkaFrameCodec, ParseError as FrameParseError, parse_frame};

use crate::client::config::KafkaClientConfig;
use crate::wire::{self, ApiKey, RequestBody, ResponseBody, WireError, error_code};

/// v0 request/response layouts only — see `crate::wire`'s module doc.
const API_VERSION: i16 = 0;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The broker's `ApiVersions` handshake reply carried a non-`NONE`
    /// error code — this facade cannot negotiate a lower version, so the
    /// handshake simply fails rather than silently downgrading.
    #[error("server: {0}")]
    Server(String),
    #[error("server connection closed mid-reply")]
    Closed,
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("wire: {0}")]
    Wire(#[from] WireError),
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
    /// Handshake complete; the session is idle and ready for `submit`.
    Ready,
    /// The in-flight request's reply.
    Complete(ResponseBody),
}

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    Handshake,
    Ready,
}

pub struct ClientSession {
    inbox: Vec<u8>,
    outbound: Vec<u8>,
    client_id: String,
    next_correlation_id: i32,
    /// The one in-flight request this session ever has at a time — Kafka's
    /// response header has no `api_key` of its own, so decoding the reply
    /// needs this recorded alongside the `correlation_id` it must match.
    pending: Option<(i32, ApiKey)>,
    phase: Phase,
}

impl ClientSession {
    /// Builds a session for `config`. The `ApiVersions` handshake is
    /// queued lazily on the first [`Self::advance`] call, mirroring
    /// redis's own lazily-queued `HELLO`.
    #[must_use]
    pub fn new(config: &KafkaClientConfig) -> Self {
        Self {
            inbox: Vec::with_capacity(8192),
            outbound: Vec::new(),
            client_id: config.client_id.clone(),
            next_correlation_id: 0,
            pending: None,
            phase: Phase::Handshake,
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

    /// Queues a Produce or Fetch request. Only valid once `Ready` and with
    /// no other reply outstanding.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if the session is not yet ready or a
    /// reply is already pending.
    pub fn submit(&mut self, request: RequestBody) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol(
                "submit before the handshake completed".into(),
            ));
        }
        if self.pending.is_some() {
            return Err(ClientError::Protocol(
                "submit while a reply is pending".into(),
            ));
        }
        let api_key = api_key_of(&request);
        self.queue(api_key, &request);
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then parses inbound
    /// frames until it needs more bytes or reaches a checkpoint.
    ///
    /// # Errors
    /// [`ClientError`] on a server error during the handshake, a malformed
    /// frame, or a decode failure.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        match self.phase {
            Phase::Handshake => self.advance_handshake(),
            Phase::Ready => self.advance_ready(),
        }
    }

    fn advance_handshake(&mut self) -> Result<Step, ClientError> {
        if self.pending.is_none() {
            self.queue(ApiKey::ApiVersions, &RequestBody::ApiVersions);
            return Ok(Step::Send);
        }
        match self.next_reply()? {
            None => Ok(Step::Recv),
            Some((correlation_id, body)) => {
                let (expected_id, api_key) = self.take_pending(correlation_id)?;
                let decoded = wire::decode_response(api_key.to_i16(), &body)?;
                let ResponseBody::ApiVersions(response) = decoded else {
                    return Err(ClientError::Protocol(format!(
                        "expected an ApiVersions reply for correlation_id {expected_id}, decoded a different shape"
                    )));
                };
                if response.error_code != error_code::NONE {
                    return Err(ClientError::Server(format!(
                        "ApiVersions handshake failed: error_code={}",
                        response.error_code
                    )));
                }
                self.phase = Phase::Ready;
                Ok(Step::Ready)
            }
        }
    }

    fn advance_ready(&mut self) -> Result<Step, ClientError> {
        if self.pending.is_none() {
            return Ok(Step::Ready);
        }
        match self.next_reply()? {
            None => Ok(Step::Recv),
            Some((correlation_id, body)) => {
                let (_, api_key) = self.take_pending(correlation_id)?;
                let decoded = wire::decode_response(api_key.to_i16(), &body)?;
                Ok(Step::Complete(decoded))
            }
        }
    }

    /// Confirms `correlation_id` matches the one outstanding request and
    /// takes it, or a protocol error naming the mismatch.
    fn take_pending(&mut self, correlation_id: i32) -> Result<(i32, ApiKey), ClientError> {
        let (expected_id, api_key) = self
            .pending
            .take()
            .ok_or_else(|| ClientError::Protocol("reply with no pending request".into()))?;
        if correlation_id != expected_id {
            return Err(ClientError::Protocol(format!(
                "correlation_id mismatch: expected {expected_id}, got {correlation_id}"
            )));
        }
        Ok((expected_id, api_key))
    }

    fn queue(&mut self, api_key: ApiKey, request: &RequestBody) {
        let correlation_id = self.next_correlation_id;
        self.next_correlation_id += 1;

        let mut payload = Vec::new();
        wire::write_i16(&mut payload, api_key.to_i16());
        wire::write_i16(&mut payload, API_VERSION);
        wire::write_i32(&mut payload, correlation_id);
        wire::write_string(&mut payload, &self.client_id);
        payload.extend_from_slice(&request.encode());

        let mut wire_bytes = Vec::new();
        if let Err(error) = KafkaFrameCodec.encode_frame(&payload.as_slice(), &mut wire_bytes) {
            // unreachable in practice (no request this client builds
            // approaches the i32 length-prefix ceiling); still handled,
            // not unwrapped, per house rule.
            tracing::error!(error = %error, "kafka client request exceeds the wire's i32 length prefix");
        }
        self.outbound.extend_from_slice(&wire_bytes);
        self.pending = Some((correlation_id, api_key));
    }

    /// Parses one logical reply from the inbox, owning it and draining the
    /// consumed bytes.
    fn next_reply(&mut self) -> Result<Option<(i32, Vec<u8>)>, ClientError> {
        match parse_frame(&self.inbox) {
            Err(FrameParseError::Short | FrameParseError::PartialFrame(_)) => Ok(None),
            Err(error) => Err(ClientError::Protocol(format!("bad frame: {error}"))),
            Ok((payload, consumed)) => {
                if payload.len() < 4 {
                    return Err(ClientError::Protocol(
                        "response shorter than a correlation_id".into(),
                    ));
                }
                let correlation_id =
                    i32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
                let body = payload[4..].to_vec();
                self.inbox.drain(..consumed);
                Ok(Some((correlation_id, body)))
            }
        }
    }
}

fn api_key_of(request: &RequestBody) -> ApiKey {
    match request {
        RequestBody::Produce(_) => ApiKey::Produce,
        RequestBody::Fetch(_) => ApiKey::Fetch,
        RequestBody::Metadata(_) => ApiKey::Metadata,
        RequestBody::ApiVersions => ApiKey::ApiVersions,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn api_versions_reply(correlation_id: i32, error_code: i16) -> Vec<u8> {
        let response = crate::wire::ApiVersionsResponse {
            error_code,
            api_versions: crate::wire::SUPPORTED_API_VERSIONS
                .iter()
                .map(
                    |&(api_key, min_version, max_version)| crate::wire::ApiVersionRange {
                        api_key,
                        min_version,
                        max_version,
                    },
                )
                .collect(),
        };
        let body = ResponseBody::ApiVersions(response).encode();
        let mut payload = Vec::new();
        payload.extend_from_slice(&correlation_id.to_be_bytes());
        payload.extend_from_slice(&body);
        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&payload.as_slice(), &mut wire)
            .expect("encode");
        wire
    }

    fn drive_handshake(session: &mut ClientSession, error_code: i16) {
        loop {
            match session.advance().expect("advance") {
                Step::Send => {
                    let _sent = session.take_outbound();
                    session.feed(&api_versions_reply(0, error_code));
                }
                Step::Recv => panic!("handshake reply already fed ahead of the Send step"),
                Step::Ready => return,
                Step::Complete(_) => panic!("unexpected reply during handshake"),
            }
        }
    }

    #[test]
    fn handshake_sends_apiversions_then_becomes_ready() {
        let config = KafkaClientConfig::default();
        let mut session = ClientSession::new(&config);

        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send (ApiVersions), got {other:?}"),
        }
        let sent = session.take_outbound();
        assert!(!sent.is_empty());

        session.feed(&api_versions_reply(0, error_code::NONE));
        match session.advance().expect("advance") {
            Step::Ready => {}
            other => panic!("expected Ready after ApiVersions reply, got {other:?}"),
        }
    }

    #[test]
    fn handshake_surfaces_a_non_none_error_code_as_a_server_error() {
        let config = KafkaClientConfig::default();
        let mut session = ClientSession::new(&config);
        let _ = session.advance().expect("advance");
        let _ = session.take_outbound();
        session.feed(&api_versions_reply(0, error_code::UNSUPPORTED_VERSION));
        match session.advance() {
            Err(ClientError::Server(message)) => assert!(message.contains("35")),
            other => panic!("expected a server error, got {other:?}"),
        }
    }

    #[test]
    fn produce_round_trips_a_reply_after_the_handshake() {
        let config = KafkaClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, error_code::NONE);

        session
            .submit(RequestBody::Produce(crate::wire::ProduceRequest {
                acks: 1,
                timeout_ms: 1000,
                topics: vec![crate::wire::ProduceTopicData {
                    topic: "orders".to_string(),
                    partitions: vec![crate::wire::ProducePartitionData {
                        partition: 0,
                        record_set: Bytes::from_static(b"hello"),
                    }],
                }],
            }))
            .expect("submit");
        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send, got {other:?}"),
        }
        let _sent = session.take_outbound();

        let response = crate::wire::ProduceResponse {
            topics: vec![crate::wire::ProduceTopicResult {
                topic: "orders".to_string(),
                partitions: vec![crate::wire::ProducePartitionResult {
                    partition: 0,
                    error_code: error_code::NONE,
                    base_offset: 0,
                }],
            }],
        };
        let body = ResponseBody::Produce(response.clone()).encode();
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_i32.to_be_bytes()); // correlation_id 1: handshake used 0
        payload.extend_from_slice(&body);
        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&payload.as_slice(), &mut wire)
            .expect("encode");
        session.feed(&wire);

        match session.advance().expect("advance") {
            Step::Complete(ResponseBody::Produce(decoded)) => assert_eq!(decoded, response),
            other => panic!("expected Complete(Produce), got {other:?}"),
        }
        // back to idle-ready
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn mismatched_correlation_id_is_a_protocol_error() {
        let config = KafkaClientConfig::default();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, error_code::NONE);

        session
            .submit(RequestBody::Fetch(crate::wire::FetchRequest {
                replica_id: -1,
                max_wait_ms: 0,
                min_bytes: 0,
                topics: Vec::new(),
            }))
            .expect("submit");
        let _sent = session.take_outbound();

        let mut payload = Vec::new();
        payload.extend_from_slice(&999_i32.to_be_bytes()); // wrong correlation_id
        payload.extend_from_slice(
            &ResponseBody::Fetch(crate::wire::FetchResponse::default()).encode(),
        );
        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&payload.as_slice(), &mut wire)
            .expect("encode");
        session.feed(&wire);

        match session.advance() {
            Err(ClientError::Protocol(reason)) => {
                assert!(reason.contains("correlation_id mismatch"))
            }
            other => panic!("expected a correlation_id mismatch error, got {other:?}"),
        }
    }
}
