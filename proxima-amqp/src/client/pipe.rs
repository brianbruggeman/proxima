//! Async AMQP 0-9-1 client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam `proxima_redis::client::pipe::RedisClientUpstream` uses,
//! so the client is agnostic to the wire (prime, tokio, TLS-wrapped). It
//! drives the sans-IO [`ClientSession`] over a futures-io connection.
//!
//! The request contract mirrors redis's own generic `Request<Bytes>` ->
//! `Response<Bytes>` Pipe shape rather than exposing [`ClientSession`]'s
//! richer typed surface directly — `request.method` selects the verb
//! (`PUBLISH`/`CONSUME`), `request.payload` carries NUL-delimited args (the
//! SAME wire convention redis's own `argv_of` uses for its client Pipe):
//! `PUBLISH` is `exchange \0 routing_key \0 body` (a `splitn(3, ...)` so an
//! embedded NUL in `body` itself is never mis-split); `CONSUME` is a bare
//! queue name. `CONSUME`'s reply streams each delivery's body via
//! `Response.stream` (one chunk per `basic.deliver`) — properties/exchange/
//! routing-key metadata do not ride this generic Bytes contract; a caller
//! that needs them drives [`ClientSession`] (or the blocking
//! [`crate::client::AmqpClient`]) directly.

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

use crate::client::config::AmqpClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// AMQP 0-9-1 client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached, already-handshaken
/// connection (pool of one) reused across `PUBLISH` calls. `CONSUME`
/// consumes the connection for the lifetime of the returned stream (a
/// `basic.consume`'d channel is not simultaneously usable for further
/// `PUBLISH`/`CONSUME` calls in this MVP — mirrors redis's own pub/sub
/// cache-take behavior).
pub struct AmqpClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    config: AmqpClientConfig,
    cached: Arc<Mutex<Option<Cached<U::Conn>>>>,
}

struct Cached<C> {
    conn: C,
    session: ClientSession,
}

impl<U: StreamUpstream> AmqpClientUpstream<U> {
    /// `new` never touches the network — the protocol-header handshake
    /// only runs lazily on the first `.call()`:
    ///
    /// ```
    /// use proxima_amqp::{AmqpClientConfig, AmqpClientUpstream};
    /// use proxima_net::prime::PrimeTcpUpstream;
    ///
    /// let addr = "127.0.0.1:5672".parse().expect("valid socket address");
    /// let transport = PrimeTcpUpstream::new(addr);
    /// let client = AmqpClientUpstream::new(transport, AmqpClientConfig::default());
    /// # let _ = client;
    /// ```
    pub fn new(upstream: U, config: AmqpClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            config,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    async fn exchange(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let command = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let (_request, body) = request.body_bytes().await?;
        match command.as_str() {
            "PUBLISH" => self.publish(&body).await,
            "CONSUME" => self.consume(&body).await,
            other => Err(ProximaError::Config(format!(
                "amqp client: unsupported command '{other}'"
            ))),
        }
    }

    async fn publish(&self, body: &[u8]) -> Result<Response<Bytes>, ProximaError> {
        let mut parts = body.splitn(3, |byte| *byte == 0);
        let exchange = parts.next().unwrap_or_default();
        let routing_key = parts.next().unwrap_or_default();
        let payload = parts.next().unwrap_or_default();

        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("amqp cache empty".into()))?;

        match drive_publish(
            &mut cached.session,
            &mut cached.conn,
            exchange,
            routing_key,
            payload,
        )
        .await
        {
            Ok(()) => Ok(Response::ok(Bytes::new())),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn consume(&self, queue: &[u8]) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        let mut cached = match guard.take() {
            Some(cached) => cached,
            None => self.connect().await?,
        };
        drop(guard);

        if let Err(error) = drive_consume_start(&mut cached.session, &mut cached.conn, queue).await
        {
            return Err(client_error_to_proxima(error));
        }

        let stream = futures::stream::unfold(StreamState::Active(cached), delivery_step);
        Ok(Response::streamed(ResponseStream::new(stream)))
    }

    async fn connect(&self) -> Result<Cached<U::Conn>, ProximaError> {
        let mut conn = self
            .upstream
            .connect()
            .await
            .map_err(|error| ProximaError::Upstream(format!("amqp connect: {error}")))?;
        let mut session = ClientSession::new(&self.config);
        drive_until_ready(&mut session, &mut conn)
            .await
            .map_err(client_error_to_proxima)?;
        Ok(Cached { conn, session })
    }
}

impl<U: StreamUpstream> SendPipe for AmqpClientUpstream<U> {
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

async fn drive_until_ready<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Ready => return Ok(()),
            Step::ConsumeOk { .. } | Step::Delivery { .. } => {
                return Err(ClientError::Protocol(
                    "unexpected event before ready".into(),
                ));
            }
        }
    }
}

async fn drive_publish<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
    exchange: &[u8],
    routing_key: &[u8],
    body: &[u8],
) -> Result<(), ClientError> {
    session.queue_publish(exchange, routing_key, false, false, b"", body)?;
    flush(session, conn).await
}

async fn drive_consume_start<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
    queue: &[u8],
) -> Result<(), ClientError> {
    session.queue_consume(queue, b"", false)?;
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::ConsumeOk { .. } => return Ok(()),
            Step::Ready => {}
            Step::Delivery { .. } => {
                return Err(ClientError::Protocol(
                    "delivery arrived before consume-ok".into(),
                ));
            }
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

enum StreamState<C> {
    Active(Cached<C>),
    Done,
}

async fn delivery_step<C: StreamConnection>(
    state: StreamState<C>,
) -> Option<(Result<Bytes, ProximaError>, StreamState<C>)> {
    let mut cached = match state {
        StreamState::Active(cached) => cached,
        StreamState::Done => return None,
    };
    loop {
        match cached.session.advance() {
            Ok(Step::Delivery { body, .. }) => {
                return Some((Ok(Bytes::from(body)), StreamState::Active(cached)));
            }
            Ok(Step::Recv) => {
                let mut chunk = [0_u8; READ_CHUNK_BYTES];
                match cached.conn.read(&mut chunk).await {
                    Ok(0) => return None,
                    Ok(read) => cached.session.feed(&chunk[..read]),
                    Err(error) => return Some((Err(ProximaError::Io(error)), StreamState::Done)),
                }
            }
            Ok(Step::Send) => {
                let bytes = cached.session.take_outbound();
                if let Err(error) = cached.conn.write_all(&bytes).await {
                    return Some((Err(ProximaError::Io(error)), StreamState::Done));
                }
            }
            Ok(Step::Ready | Step::ConsumeOk { .. }) => {}
            Err(error) => return Some((Err(client_error_to_proxima(error)), StreamState::Done)),
        }
    }
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("amqp client: {error}"))
}
