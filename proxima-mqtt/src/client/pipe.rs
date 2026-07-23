//! Async MQTT client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam `proxima_redis::client::pipe::RedisClientUpstream` uses,
//! so the client is agnostic to the wire (prime, tokio, TLS-wrapped). It
//! drives the sans-IO [`ClientSession`] over a futures-io connection.
//!
//! `CONNECT` is never a caller-visible `Request` — like redis's
//! `HELLO`/`AUTH`/`SELECT` handshake, it runs automatically the first time
//! [`MqttClientUpstream::connect`] dials, driven internally by
//! [`ClientSession::new`]. The caller-visible verbs are `PUBLISH` /
//! `SUBSCRIBE` / `UNSUBSCRIBE` / `PING` / `DISCONNECT`.
//!
//! `Request.payload` carries NUL-delimited byte segments — the same
//! flat-bytes convention `RedisClientUpstream::argv_of` uses for RESP
//! args, adapted to MQTT's per-verb field shape (`PUBLISH`'s segments are
//! `topic \0 qos \0 retain \0 payload`, split with `splitn(4, ..)` so the
//! payload segment — the last one — stays binary-safe even if it embeds a
//! NUL byte; `SUBSCRIBE`'s are alternating `filter \0 qos` pairs;
//! `UNSUBSCRIBE`'s are a flat filter list). `Response.payload` carries the
//! reply's raw MQTT-encoded wire bytes (mirrors
//! `RedisClientUpstream`'s `Bytes::from(value.encode())`). `SUBSCRIBE`
//! rides `Response.stream` instead: one re-encoded wire frame per pushed
//! item, first the `SUBACK` then every subsequent `PUBLISH`.

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::lock::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::body::ResponseStream;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

use proxima_protocols::mqtt::encode::{
    encode_ack, encode_connack, encode_disconnect, encode_pingresp, encode_publish as encode_publish_frame,
    encode_suback,
};
use proxima_protocols::mqtt::{MqttReply, PacketType, is_streaming, verb};

use crate::client::config::MqttClientConfig;
use crate::client::session::{ClientError, ClientSession, PushStep, Step};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// MQTT client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached, already-`CONNECT`ed
/// connection (pool of one) reused across request/reply calls. A
/// `SUBSCRIBE` consumes the connection for the lifetime of the returned
/// stream, so the cache is dropped and the next call reconnects.
pub struct MqttClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    config: MqttClientConfig,
    cached: Arc<Mutex<Option<Cached<U::Conn>>>>,
}

struct Cached<C> {
    conn: C,
    session: ClientSession,
}

impl<U: StreamUpstream> MqttClientUpstream<U> {
    /// Builds a client over `upstream` with `config`. The transport is
    /// injected (runtime object); the config is the declarative half —
    /// the same split `RedisClientUpstream::new` uses.
    pub fn new(upstream: U, config: MqttClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            config,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    async fn exchange(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let command = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let (_request, body) = request.body_bytes().await?;

        if is_streaming(&command) {
            return self.subscribe(&body).await;
        }
        match command.as_str() {
            cmd if cmd.eq_ignore_ascii_case(verb::PUBLISH) => self.publish(&body).await,
            cmd if cmd.eq_ignore_ascii_case(verb::UNSUBSCRIBE) => self.unsubscribe(&body).await,
            cmd if cmd.eq_ignore_ascii_case(verb::PING) => self.ping().await,
            cmd if cmd.eq_ignore_ascii_case(verb::DISCONNECT) => self.disconnect().await,
            other => Err(ProximaError::Upstream(format!("mqtt: unsupported verb '{other}'"))),
        }
    }

    async fn publish(&self, body: &[u8]) -> Result<Response<Bytes>, ProximaError> {
        let (topic, qos, retain, payload) = split_publish_body(body);
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("mqtt cache empty".into()))?;

        cached
            .session
            .submit_publish(&topic, &payload, qos, retain)
            .map_err(client_error_to_proxima)?;
        match drive_to_complete(&mut cached.session, &mut cached.conn).await {
            Ok(reply) => Ok(Response::ok(Bytes::from(encode_reply(&reply)))),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn unsubscribe(&self, body: &[u8]) -> Result<Response<Bytes>, ProximaError> {
        let filters = split_filters(body);
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("mqtt cache empty".into()))?;

        let refs: Vec<&[u8]> = filters.iter().map(Vec::as_slice).collect();
        cached
            .session
            .submit_unsubscribe(&refs)
            .map_err(client_error_to_proxima)?;
        match drive_to_complete(&mut cached.session, &mut cached.conn).await {
            Ok(reply) => Ok(Response::ok(Bytes::from(encode_reply(&reply)))),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn ping(&self) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("mqtt cache empty".into()))?;

        cached.session.submit_ping().map_err(client_error_to_proxima)?;
        match drive_to_complete(&mut cached.session, &mut cached.conn).await {
            Ok(reply) => Ok(Response::ok(Bytes::from(encode_reply(&reply)))),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    /// Sends `DISCONNECT` (best effort — MQTT defines no acknowledgement)
    /// and drops the cached connection.
    async fn disconnect(&self) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        if let Some(mut cached) = guard.take() {
            let mut bytes = Vec::new();
            encode_disconnect(&mut bytes);
            let _ = cached.conn.write_all(&bytes).await;
            let _ = cached.conn.flush().await;
        }
        Ok(Response::ok(Bytes::new()))
    }

    /// A `SUBSCRIBE`: send it, then hand the (session, conn) to a stream
    /// that yields each frame — first the `SUBACK`, then every pushed
    /// `PUBLISH` — as re-encoded wire bytes. The pool-of-one cache is
    /// taken for the stream's lifetime.
    async fn subscribe(&self, body: &[u8]) -> Result<Response<Bytes>, ProximaError> {
        let filters = split_filters_with_qos(body);
        let mut guard = self.cached.lock().await;
        let mut cached = match guard.take() {
            Some(cached) => cached,
            None => self.connect().await?,
        };
        drop(guard);

        let refs: Vec<(&[u8], u8)> = filters.iter().map(|(filter, qos)| (filter.as_slice(), *qos)).collect();
        cached
            .session
            .queue_subscribe(&refs)
            .map_err(client_error_to_proxima)?;
        flush(&mut cached.session, &mut cached.conn)
            .await
            .map_err(client_error_to_proxima)?;

        let stream = futures::stream::unfold(StreamState::Active(cached), push_step);
        Ok(Response::streamed(ResponseStream::new(stream)))
    }

    async fn connect(&self) -> Result<Cached<U::Conn>, ProximaError> {
        let mut conn = self
            .upstream
            .connect()
            .await
            .map_err(|err| ProximaError::Upstream(format!("mqtt connect: {err}")))?;
        let mut session = ClientSession::new(&self.config);
        drive_until_ready(&mut session, &mut conn)
            .await
            .map_err(client_error_to_proxima)?;
        Ok(Cached { conn, session })
    }
}

impl<U: StreamUpstream> SendPipe for MqttClientUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { self.exchange(request).await }
    }
}

/// `topic \0 qos \0 retain \0 payload`. `splitn(4, ..)` keeps the payload
/// segment binary-safe even if it embeds a NUL byte — MQTT topics never
/// legally contain one ([MQTT-1.5.3-2]), so the first three splits are
/// exact.
fn split_publish_body(body: &[u8]) -> (Vec<u8>, u8, bool, Vec<u8>) {
    let mut segments = body.splitn(4, |byte| *byte == 0);
    let topic = segments.next().unwrap_or(&[]).to_vec();
    let qos = segments
        .next()
        .and_then(|segment| core::str::from_utf8(segment).ok())
        .and_then(|text| text.parse::<u8>().ok())
        .unwrap_or(0);
    let retain = segments.next() == Some(b"1");
    let payload = segments.next().unwrap_or(&[]).to_vec();
    (topic, qos, retain, payload)
}

/// Alternating `filter \0 qos` pairs.
fn split_filters_with_qos(body: &[u8]) -> Vec<(Vec<u8>, u8)> {
    let segments: Vec<&[u8]> = body.split(|byte| *byte == 0).collect();
    segments
        .chunks(2)
        .filter_map(|pair| {
            let filter = pair.first()?.to_vec();
            let qos = pair
                .get(1)
                .and_then(|segment| core::str::from_utf8(segment).ok())
                .and_then(|text| text.parse::<u8>().ok())
                .unwrap_or(0);
            Some((filter, qos))
        })
        .collect()
}

/// A flat, NUL-delimited filter list.
fn split_filters(body: &[u8]) -> Vec<Vec<u8>> {
    body.split(|byte| *byte == 0)
        .filter(|segment| !segment.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
}

/// Re-encodes a typed [`MqttReply`] back to its MQTT wire packet — the
/// same "protocol-out is bytes the caller may re-decode" shape
/// `RedisClientUpstream` gets for free from `RespValue::encode`.
fn encode_reply(reply: &MqttReply) -> Vec<u8> {
    let mut out = Vec::new();
    match reply {
        MqttReply::Published | MqttReply::Disconnected => {}
        MqttReply::ConnAck { session_present, return_code } => {
            encode_connack(*session_present, *return_code, &mut out);
        }
        MqttReply::PubAck { packet_id } => encode_ack(PacketType::PubAck, *packet_id, &mut out),
        MqttReply::SubAck { packet_id, granted } => encode_suback(*packet_id, granted, &mut out),
        MqttReply::UnsubAck { packet_id } => encode_ack(PacketType::UnsubAck, *packet_id, &mut out),
        MqttReply::Pong => encode_pingresp(&mut out),
        MqttReply::Publish { topic, payload, qos, retain } => {
            encode_publish_frame(topic, None, payload, *qos, false, *retain, &mut out);
        }
    }
    out
}

async fn drive_to_complete<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<MqttReply, ClientError> {
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Complete(reply) => return Ok(reply),
            Step::Ready => return Err(ClientError::Protocol("ready without a reply")),
        }
    }
}

async fn drive_until_ready<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Ready => return Ok(()),
            Step::Complete(_) => return Err(ClientError::Protocol("reply before ready")),
        }
    }
}

async fn flush<C: StreamConnection>(session: &mut ClientSession, conn: &mut C) -> Result<(), ClientError> {
    let bytes = session.take_outbound();
    conn.write_all(&bytes).await?;
    conn.flush().await?;
    Ok(())
}

async fn recv<C: StreamConnection>(session: &mut ClientSession, conn: &mut C) -> Result<(), ClientError> {
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    let read = conn.read(&mut chunk).await?;
    if read == 0 {
        return Err(ClientError::Closed);
    }
    session.feed(&chunk[..read]);
    Ok(())
}

/// The `SUBSCRIBE` stream's state: either an active (session, conn) pair,
/// or `Done` once the connection closed or errored.
enum StreamState<C> {
    Active(Cached<C>),
    Done,
}

/// One `unfold` step: read pushed frames, yielding each as re-encoded MQTT
/// wire bytes.
async fn push_step<C: StreamConnection>(
    state: StreamState<C>,
) -> Option<(Result<Bytes, ProximaError>, StreamState<C>)> {
    let mut cached = match state {
        StreamState::Active(cached) => cached,
        StreamState::Done => return None,
    };
    loop {
        match cached.session.poll_push() {
            Ok(PushStep::Frame(reply)) => {
                return Some((Ok(Bytes::from(encode_reply(&reply))), StreamState::Active(cached)));
            }
            Ok(PushStep::Recv) => {
                let mut chunk = [0_u8; READ_CHUNK_BYTES];
                match cached.conn.read(&mut chunk).await {
                    Ok(0) => return None,
                    Ok(read) => cached.session.feed(&chunk[..read]),
                    Err(error) => return Some((Err(ProximaError::Io(error)), StreamState::Done)),
                }
            }
            Err(error) => return Some((Err(client_error_to_proxima(error)), StreamState::Done)),
        }
    }
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("mqtt client: {error}"))
}
