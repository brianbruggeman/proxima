//! Sans-IO Kafka connection state machine, plus the per-connection I/O
//! driver built on top of it.
//!
//! [`Connection`] mirrors `proxima_protocols::redis::connection::Connection`'s
//! `feed_bytes`/`advance`/`consume` shape (itself mirroring
//! `h1_connection::Connection`) — one growing read buffer, a cursor so
//! pipelined requests don't memcpy, and a typed [`Advanced`] outcome the
//! driver matches on. It exists in THIS crate rather than
//! `proxima_protocols::kafka` because that module only lifts the wire
//! ENVELOPE ([`proxima_protocols::kafka::parse_frame`] +
//! [`proxima_protocols::kafka::parse_request_header`]) — no persistent,
//! buffer-owning connection type. [`Connection::advance`] is exactly that
//! envelope-parsing pair wired to an owned growing buffer, plus the same
//! DoS guard redis's own `Connection` applies: a still-incomplete frame
//! whose DECLARED size already exceeds [`Limits::max_message_bytes`] trips
//! [`Advanced::MessageTooLarge`] before the connection is allowed to buffer
//! that many bytes.
//!
//! [`serve_connection`] is the I/O driver: reads bytes, feeds the FSM,
//! decodes each request's body ([`crate::wire::decode_request`]), and
//! dispatches to a [`crate::pipes::KafkaPipeHandle`] — EXCEPT
//! `ApiVersions`, answered protocol-level (never reaching the handler,
//! never admission-gated), the same way redis's PING never reaches its
//! business handler. Unlike redis, there is no separate push/pub-sub
//! channel to race in the outer loop: `Fetch`'s long-poll wait
//! ([`crate::broker::KafkaBroker::produce`]'s wake fan-out) happens INSIDE
//! the handler call itself, so one request in flight is exactly one
//! `.await`, sequential, no `select!` needed here.

use bytes::Bytes;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use proxima_codec::FrameCodec;

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, RequestAdmit};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_protocols::kafka::{
    KafkaFrameCodec, ParseError as FrameParseError, RequestHeader, parse_frame,
    parse_request_header,
};

use crate::config::KafkaServerConfig;
use crate::error::KafkaServeError;
use crate::pipes::KafkaPipeHandle;
use crate::wire::{
    self, ApiVersionsResponse, FetchResponse, MetadataResponse, ProduceResponse, ResponseBody,
    WireError,
};

/// A connection stays under this many buffered-but-unparsed bytes before a
/// still-incomplete frame is treated as an oversized message.
const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Once the consumed prefix exceeds this many bytes, [`Connection::consume`]
/// compacts the buffer instead of just moving the cursor — mirrors
/// `proxima_protocols::redis::connection::Connection`'s identical
/// threshold-triggered compaction.
const COMPACT_THRESHOLD_BYTES: usize = 8 * 1024;

/// Byte caps a [`Connection`] enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_message_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

/// Typed outcome of [`Connection::advance`]. `Command`'s `header`/`body`
/// borrow from the connection's internal buffer — the driver must extract
/// whatever it needs before calling [`Connection::consume`] or `advance`
/// again.
pub enum Advanced<'a> {
    /// The buffer holds a prefix of a frame; read more bytes and retry.
    NeedMore,
    /// One full length-prefixed frame parsed and its header decoded.
    /// `consumed` is the byte length to pass to [`Connection::consume`].
    Command {
        header: RequestHeader<'a>,
        body: &'a [u8],
        consumed: usize,
    },
    /// The buffered bytes violate Kafka framing or header layout. The
    /// driver closes the connection — there is no trustworthy
    /// `correlation_id` to reply against.
    ProtocolError { reason: String },
    /// A still-incomplete frame already declares a size past
    /// [`Limits::max_message_bytes`] — the DoS guard tripped.
    MessageTooLarge,
}

/// Sans-IO Kafka connection state machine.
pub struct Connection {
    buffer: Vec<u8>,
    cursor: usize,
    limits: Limits,
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            limits,
        }
    }

    /// Append bytes read off the wire.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Drive the state machine one step: try to parse one length-prefixed
    /// frame, then its request header, from the unconsumed buffer region.
    pub fn advance(&mut self) -> Advanced<'_> {
        let unparsed = &self.buffer[self.cursor..];
        match parse_frame(unparsed) {
            Ok((payload, consumed)) => match parse_request_header(payload) {
                Ok((header, body_offset)) => Advanced::Command {
                    header,
                    body: &payload[body_offset..],
                    consumed,
                },
                Err(error) => Advanced::ProtocolError {
                    reason: format!("bad request header: {error}"),
                },
            },
            Err(FrameParseError::PartialFrame(size)) => {
                if 4 + size as usize > self.limits.max_message_bytes {
                    Advanced::MessageTooLarge
                } else {
                    Advanced::NeedMore
                }
            }
            Err(FrameParseError::Short) => Advanced::NeedMore,
            Err(error) => Advanced::ProtocolError {
                reason: format!("bad frame: {error}"),
            },
        }
    }

    /// Advance past a parsed frame's bytes. Compacts the buffer once the
    /// consumed prefix grows past [`COMPACT_THRESHOLD_BYTES`], or clears it
    /// outright once every buffered byte is consumed.
    pub fn consume(&mut self, amount: usize) {
        self.cursor += amount;
        if self.cursor >= self.buffer.len() {
            self.buffer.clear();
            self.cursor = 0;
        } else if self.cursor > COMPACT_THRESHOLD_BYTES {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }
}

// --------------------------------------------------------------- driver

async fn read_some<S: AsyncRead + Unpin>(
    stream: &mut S,
    scratch: &mut [u8],
) -> std::io::Result<usize> {
    stream.read(scratch).await
}

async fn flush_out<S: AsyncWrite + Unpin>(
    stream: &mut S,
    out: &mut Vec<u8>,
) -> std::io::Result<()> {
    if !out.is_empty() {
        stream.write_all(out).await?;
        out.clear();
    }
    stream.flush().await
}

/// Encodes one response body under `correlation_id` into a full
/// length-prefixed wire frame, reusing
/// [`proxima_protocols::kafka::KafkaFrameCodec`] for the envelope — the
/// mechanical inverse of [`parse_frame`] this crate composes on top of
/// rather than reimplementing.
fn encode_response(correlation_id: i32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.extend_from_slice(&correlation_id.to_be_bytes());
    payload.extend_from_slice(body);

    let mut wire = Vec::with_capacity(4 + payload.len());
    if let Err(error) = KafkaFrameCodec.encode_frame(&payload.as_slice(), &mut wire) {
        tracing::error!(error = %error, "kafka response exceeds the wire's i32 length prefix");
        return Vec::new();
    }
    wire
}

/// A data-free but well-formed response for `api_key` — used for both the
/// admission-shed path and a decode failure whose api_key IS one this
/// facade recognizes (just with no content this facade can honestly
/// supply).
fn empty_response_for(api_key: i16) -> Option<ResponseBody> {
    match wire::ApiKey::from_i16(api_key) {
        wire::ApiKey::Produce => Some(ResponseBody::Produce(ProduceResponse::default())),
        wire::ApiKey::Fetch => Some(ResponseBody::Fetch(FetchResponse::default())),
        wire::ApiKey::Metadata => Some(ResponseBody::Metadata(MetadataResponse::default())),
        wire::ApiKey::ApiVersions => {
            Some(ResponseBody::ApiVersions(ApiVersionsResponse::supported()))
        }
        wire::ApiKey::Other(_) => None,
    }
}

fn build_request(api_key: i16, payload: wire::RequestBody) -> crate::pipes::KafkaPipeRequest {
    let method = match wire::ApiKey::from_i16(api_key) {
        wire::ApiKey::Produce => Method::from_bytes(b"PRODUCE"),
        wire::ApiKey::Fetch => Method::from_bytes(b"FETCH"),
        wire::ApiKey::Metadata => Method::from_bytes(b"METADATA"),
        wire::ApiKey::ApiVersions => Method::from_bytes(b"APIVERSIONS"),
        wire::ApiKey::Other(_) => Method::from_bytes(b"UNKNOWN"),
    };
    Request {
        method,
        path: Bytes::new(),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload,
        stream: None,
        context: RequestContext::default(),
    }
}

/// Outcome of dispatching one parsed frame.
enum FrameOutcome {
    /// Write these wire bytes; connection stays open.
    Reply(Vec<u8>),
    /// Malformed input with no trustworthy `correlation_id` — write
    /// nothing, close cleanly.
    Close,
    /// The handler pipe itself failed: write the courtesy empty reply,
    /// then close with the error.
    HandlerError(Vec<u8>, ProximaError),
}

async fn dispatch(
    header: &RequestHeader<'_>,
    body: &[u8],
    handler: &KafkaPipeHandle,
    admission: &ConnAdmission,
) -> FrameOutcome {
    let correlation_id = header.correlation_id;
    match wire::decode_request(header.api_key, header.api_version, body) {
        Ok(wire::RequestBody::ApiVersions) => {
            // protocol-level, mirrors redis's PING: never reaches the
            // handler, never admission-gated.
            FrameOutcome::Reply(encode_response(
                correlation_id,
                &ResponseBody::ApiVersions(ApiVersionsResponse::supported()).encode(),
            ))
        }
        Ok(decoded) => {
            if let RequestAdmit::Shed { reason } = admission.request_admit() {
                tracing::warn!(
                    api_key = header.api_key,
                    ?reason,
                    "kafka request shed under admission policy"
                );
                let body = empty_response_for(header.api_key)
                    .map(|body| body.encode())
                    .unwrap_or_default();
                return FrameOutcome::Reply(encode_response(correlation_id, &body));
            }
            let request = build_request(header.api_key, decoded);
            let dispatched = handler.call(request).await;
            admission.request_release();
            match dispatched {
                Ok(response) => {
                    FrameOutcome::Reply(encode_response(correlation_id, &response.payload.encode()))
                }
                Err(error) => {
                    tracing::error!(error = %error, api_key = header.api_key, "kafka handler error");
                    let body = empty_response_for(header.api_key)
                        .map(|body| body.encode())
                        .unwrap_or_default();
                    FrameOutcome::HandlerError(encode_response(correlation_id, &body), error)
                }
            }
        }
        Err(WireError::UnsupportedVersion { api_key, version }) => {
            tracing::warn!(api_key, version, "kafka unsupported api_version");
            let body = empty_response_for(api_key)
                .map(|body| body.encode())
                .unwrap_or_default();
            FrameOutcome::Reply(encode_response(correlation_id, &body))
        }
        Err(error) => {
            // an api_key this facade has never heard of, or a malformed
            // body for one it does — no schema to answer with honestly.
            tracing::error!(error = %error, api_key = header.api_key, "kafka request rejected");
            FrameOutcome::Close
        }
    }
}

/// Serves one accepted connection to completion. Sequential await-per-
/// request (pipelining requires N replies in request order; never spawn
/// per-request).
pub async fn serve_connection<S>(
    mut stream: S,
    handler: KafkaPipeHandle,
    config: &KafkaServerConfig,
    admission: ConnAdmission,
) -> Result<(), KafkaServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut connection = Connection::with_limits(Limits {
        max_message_bytes: config.max_message_bytes,
    });
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let mut scratch = vec![0_u8; config.read_buffer_bytes];

    loop {
        loop {
            match connection.advance() {
                Advanced::NeedMore => break,
                Advanced::Command {
                    header,
                    body,
                    consumed,
                } => {
                    let api_key = header.api_key;
                    let api_version = header.api_version;
                    let correlation_id = header.correlation_id;
                    let owned_body = body.to_vec();
                    connection.consume(consumed);
                    let owned_header = RequestHeader {
                        api_key,
                        api_version,
                        correlation_id,
                        client_id: None,
                    };
                    let outcome = dispatch(&owned_header, &owned_body, &handler, &admission).await;
                    match outcome {
                        FrameOutcome::Reply(bytes) => out.extend_from_slice(&bytes),
                        FrameOutcome::Close => {
                            flush_out(&mut stream, &mut out).await?;
                            return Ok(());
                        }
                        FrameOutcome::HandlerError(bytes, error) => {
                            out.extend_from_slice(&bytes);
                            flush_out(&mut stream, &mut out).await?;
                            return Err(KafkaServeError::Pipe(error));
                        }
                    }
                    if out.len() >= config.write_high_water_bytes {
                        flush_out(&mut stream, &mut out).await?;
                    }
                }
                Advanced::ProtocolError { reason } => {
                    tracing::error!(reason, "kafka protocol violation");
                    flush_out(&mut stream, &mut out).await?;
                    return Ok(());
                }
                Advanced::MessageTooLarge => {
                    tracing::error!(limit = config.max_message_bytes, "kafka message too large");
                    return Err(KafkaServeError::MessageTooLarge {
                        limit: config.max_message_bytes,
                    });
                }
            }
        }
        flush_out(&mut stream, &mut out).await?;

        let read = read_some(&mut stream, &mut scratch).await?;
        if read == 0 {
            return Ok(());
        }
        connection.feed_bytes(&scratch[..read]);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use proxima_primitives::pipe::request::Response;

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
        encode_request(wire::ApiKey::ApiVersions.to_i16(), 0, correlation_id, b"")
    }

    #[test]
    fn advance_needs_more_on_an_empty_buffer() {
        let mut connection = Connection::new();
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn advance_parses_one_full_frame_and_its_header() {
        let mut connection = Connection::new();
        connection.feed_bytes(&api_versions_request(7));
        match connection.advance() {
            Advanced::Command {
                header,
                body,
                consumed,
            } => {
                assert_eq!(header.api_key, wire::ApiKey::ApiVersions.to_i16());
                assert_eq!(header.correlation_id, 7);
                assert!(body.is_empty());
                connection.consume(consumed);
            }
            Advanced::NeedMore => panic!("expected Command, got NeedMore"),
            Advanced::ProtocolError { reason } => {
                panic!("expected Command, got ProtocolError({reason})")
            }
            Advanced::MessageTooLarge => panic!("expected Command, got MessageTooLarge"),
        }
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn advance_trips_message_too_large_on_a_declared_oversized_frame() {
        let mut connection = Connection::with_limits(Limits {
            max_message_bytes: 10,
        });
        // declare a 1000-byte payload but supply none of it yet.
        connection.feed_bytes(&1000_i32.to_be_bytes());
        assert!(matches!(connection.advance(), Advanced::MessageTooLarge));
    }

    #[test]
    fn advance_reports_protocol_error_on_a_malformed_header() {
        let mut connection = Connection::new();
        // a frame whose payload is shorter than a v0 header requires.
        let mut wire = Vec::new();
        KafkaFrameCodec
            .encode_frame(&[0_u8, 1].as_slice(), &mut wire)
            .expect("encode");
        connection.feed_bytes(&wire);
        assert!(matches!(
            connection.advance(),
            Advanced::ProtocolError { .. }
        ));
    }

    struct ScriptedSocket {
        read_data: std::io::Cursor<Vec<u8>>,
        write_data: Arc<std::sync::Mutex<Vec<u8>>>,
    }

    impl AsyncRead for ScriptedSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(self.read_data.read(buf))
        }
    }

    impl AsyncWrite for ScriptedSocket {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.write_data
                .lock()
                .expect("write_data lock")
                .extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    struct EchoProduceHandler;

    impl SendPipe for EchoProduceHandler {
        type In = crate::pipes::KafkaPipeRequest;
        type Out = crate::pipes::KafkaPipeReply;
        type Err = ProximaError;

        async fn call(
            &self,
            request: crate::pipes::KafkaPipeRequest,
        ) -> Result<Self::Out, ProximaError> {
            match request.payload {
                wire::RequestBody::Produce(_) => Ok(Response::typed(
                    200,
                    ResponseBody::Produce(ProduceResponse::default()),
                )),
                _ => Err(ProximaError::Upstream("unexpected api".into())),
            }
        }
    }

    async fn drive(wire: &[u8]) -> Vec<u8> {
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire.to_vec()),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            crate::pipes::into_kafka_handle(EchoProduceHandler),
            &KafkaServerConfig::default(),
            ConnAdmission::unbounded(),
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        write_data.lock().expect("write_data lock").clone()
    }

    #[proxima::test(runtime = "tokio")]
    async fn api_versions_is_answered_protocol_level_without_reaching_the_handler() {
        let response = drive(&api_versions_request(11)).await;
        // 4-byte length prefix + 4-byte correlation_id header we can check directly.
        assert!(response.len() > 8);
        let correlation_id =
            i32::from_be_bytes([response[4], response[5], response[6], response[7]]);
        assert_eq!(correlation_id, 11);
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsupported_version_gets_a_well_formed_empty_reply_not_a_dropped_connection() {
        let request = encode_request(wire::ApiKey::Produce.to_i16(), 9, 3, b"");
        let response = drive(&request).await;
        assert!(
            !response.is_empty(),
            "connection must reply, not silently close"
        );
        let correlation_id =
            i32::from_be_bytes([response[4], response[5], response[6], response[7]]);
        assert_eq!(correlation_id, 3);
    }
}
