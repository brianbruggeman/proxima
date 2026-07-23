//! Async Kafka client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam `proxima_redis::client::pipe::RedisClientUpstream` uses,
//! so the client is agnostic to the wire (prime, tokio, TLS-wrapped). It
//! drives the sans-IO [`ClientSession`] over a futures-io connection.
//!
//! Pipe-boundary contract (this facade's own — Kafka has no
//! `proxima_protocols::kafka::pipe_contract` to lift the way redis's RESP
//! argv convention exists): `Request.method` names the operation
//! (`b"PRODUCE"` / `b"FETCH"` / `b"METADATA"`) and `Request.payload` /
//! `Response.payload` carry the SAME binary body encoding
//! [`crate::wire::RequestBody::encode`] / [`crate::wire::ResponseBody::encode`]
//! already produce for the real wire — no separate JSON/argv scheme to
//! keep in sync with the wire codec. [`KafkaClientUpstream::produce`] /
//! [`KafkaClientUpstream::fetch`] are the ergonomic typed entry points that
//! build this convention for a caller so nobody outside this module needs
//! to know it exists.

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::lock::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

use crate::client::config::KafkaClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};
use crate::wire::{self, ApiKey, RequestBody, ResponseBody};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// Kafka client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached, handshaken connection
/// (pool of one) reused across request/reply calls — the `ApiVersions`
/// handshake runs once per connection, so keep-alive matters.
pub struct KafkaClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    config: KafkaClientConfig,
    cached: Arc<Mutex<Option<Cached<U::Conn>>>>,
}

struct Cached<C> {
    conn: C,
    session: ClientSession,
}

impl<U: StreamUpstream> KafkaClientUpstream<U> {
    /// Builds a client over `upstream` with `config`. The transport is
    /// injected (runtime object); the config is the declarative half — the
    /// same split `RedisClientUpstream::new` uses.
    pub fn new(upstream: U, config: KafkaClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            config,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    /// Produce one batch to `topic`/`partition`. Ergonomic typed entry
    /// point over [`SendPipe::call`]'s generic `Request<Bytes>` contract.
    ///
    /// # Errors
    /// [`ProximaError`] on transport failure, a malformed reply, or a
    /// non-Produce reply shape (a handler bug on the broker side).
    pub async fn produce(
        &self,
        request: wire::ProduceRequest,
    ) -> Result<wire::ProduceResponse, ProximaError> {
        match self.exchange_typed(RequestBody::Produce(request)).await? {
            ResponseBody::Produce(response) => Ok(response),
            other => Err(unexpected_shape("Produce", &other)),
        }
    }

    /// Fetch from `topic`/`partition` starting at an offset. Ergonomic
    /// typed entry point over [`SendPipe::call`]'s generic `Request<Bytes>`
    /// contract.
    ///
    /// # Errors
    /// [`ProximaError`] on transport failure, a malformed reply, or a
    /// non-Fetch reply shape.
    pub async fn fetch(
        &self,
        request: wire::FetchRequest,
    ) -> Result<wire::FetchResponse, ProximaError> {
        match self.exchange_typed(RequestBody::Fetch(request)).await? {
            ResponseBody::Fetch(response) => Ok(response),
            other => Err(unexpected_shape("Fetch", &other)),
        }
    }

    async fn exchange_typed(&self, request: RequestBody) -> Result<ResponseBody, ProximaError> {
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("kafka cache empty".into()))?;

        match run_request(&mut cached.session, &mut cached.conn, request).await {
            Ok(response) => Ok(response),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn connect(&self) -> Result<Cached<U::Conn>, ProximaError> {
        let mut conn = self
            .upstream
            .connect()
            .await
            .map_err(|error| ProximaError::Upstream(format!("kafka connect: {error}")))?;
        let mut session = ClientSession::new(&self.config);
        drive_until_ready(&mut session, &mut conn)
            .await
            .map_err(client_error_to_proxima)?;
        Ok(Cached { conn, session })
    }
}

impl<U: StreamUpstream> SendPipe for KafkaClientUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let api_key = api_key_of_method(&request.method)?;
            let (_request, body_bytes) = request.body_bytes().await?;
            let decoded =
                wire::decode_request(api_key.to_i16(), 0, &body_bytes).map_err(|error| {
                    ProximaError::Upstream(format!("kafka request decode: {error}"))
                })?;
            let response = self.exchange_typed(decoded).await?;
            Ok(Response::ok(Bytes::from(response.encode())))
        }
    }
}

fn api_key_of_method(method: &Method) -> Result<ApiKey, ProximaError> {
    match method.as_bytes() {
        b"PRODUCE" => Ok(ApiKey::Produce),
        b"FETCH" => Ok(ApiKey::Fetch),
        b"METADATA" => Ok(ApiKey::Metadata),
        other => Err(ProximaError::Config(format!(
            "kafka client: unrecognized method {:?} (expected PRODUCE/FETCH/METADATA)",
            String::from_utf8_lossy(other)
        ))),
    }
}

fn unexpected_shape(expected: &str, got: &ResponseBody) -> ProximaError {
    ProximaError::Upstream(format!(
        "kafka client: expected a {expected} reply, got {got:?}"
    ))
}

/// Build the generic `Request<Bytes>` this pipe's own convention expects,
/// for a caller that wants to go through [`SendPipe::call`] directly
/// (e.g. `proxima::Client`'s registered-protocol dispatch) instead of the
/// typed [`KafkaClientUpstream::produce`]/[`KafkaClientUpstream::fetch`].
#[must_use]
pub fn request_of(api_key: ApiKey, body: &RequestBody) -> Request<Bytes> {
    let method = match api_key {
        ApiKey::Produce => Method::from_bytes(b"PRODUCE"),
        ApiKey::Fetch => Method::from_bytes(b"FETCH"),
        ApiKey::Metadata => Method::from_bytes(b"METADATA"),
        ApiKey::ApiVersions | ApiKey::Other(_) => Method::from_bytes(b"UNKNOWN"),
    };
    Request {
        method,
        path: Bytes::new(),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: Bytes::from(body.encode()),
        stream: None,
        context: RequestContext::default(),
    }
}

async fn run_request<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
    request: RequestBody,
) -> Result<ResponseBody, ClientError> {
    session.submit(request)?;
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Complete(response) => return Ok(response),
            Step::Ready => return Err(ClientError::Protocol("ready without a reply".into())),
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
            Step::Complete(_) => return Err(ClientError::Protocol("reply before ready".into())),
        }
    }
}

async fn flush<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let bytes = session.take_outbound();
    conn.write_all(&bytes).await?;
    conn.flush().await?;
    Ok(())
}

async fn recv<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    let read = conn.read(&mut chunk).await?;
    if read == 0 {
        return Err(ClientError::Closed);
    }
    session.feed(&chunk[..read]);
    Ok(())
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("kafka client: {error}"))
}
