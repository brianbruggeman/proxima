//! Async Redis/Valkey client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam pgwire's `PgwireClientUpstream` uses, so the client is
//! agnostic to the wire (prime, tokio, TLS-wrapped). It drives the sans-IO
//! [`ClientSession`] over a futures-io connection and maps the RESP-over-Pipe
//! contract (verb in `Request.method`, NUL-delimited args in `Request.payload`)
//! to RESP-encoded bytes in `Response.payload`. Pub/sub and MONITOR ride
//! `Response.stream` instead: one pushed frame per chunk, re-encoded as RESP bytes.

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::lock::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::body::ResponseStream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

use proxima_protocols::redis::RespValue;
use proxima_protocols::redis::pipe_contract::is_streaming;

use crate::client::config::RedisClientConfig;
use crate::client::session::{ClientError, ClientSession, PushStep, Step};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// Redis/Valkey client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached authenticated connection (pool of
/// one) reused across request/reply calls — the `HELLO`/`AUTH`/`SELECT`
/// handshake runs once per connection, so keep-alive matters. A streaming
/// command (pub/sub, MONITOR) consumes the connection for the lifetime of the
/// stream, so the cache is dropped and the next call reconnects.
pub struct RedisClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    config: RedisClientConfig,
    cached: Arc<Mutex<Option<Cached<U::Conn>>>>,
}

struct Cached<C> {
    conn: C,
    session: ClientSession,
}

impl<U: StreamUpstream> RedisClientUpstream<U> {
    /// Builds a client over `upstream` with `config`. The transport is injected
    /// (runtime object); the config is the declarative half — the same split as
    /// `PgwireClientUpstream::new`.
    pub fn new(upstream: U, config: RedisClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            config,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    async fn exchange(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let command = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let (_request, body) = request.body_bytes().await?;
        let argv = argv_of(&command, &body);

        if is_streaming(&command) {
            return self.stream(argv).await;
        }
        self.request_reply(argv).await
    }

    /// One request/reply over the cached connection. A server error reply (a
    /// `-ERR` / `!blob-error`) is a normal typed [`RespValue`] on the carry —
    /// the connection stays usable. Only a transport/protocol failure drops the
    /// cached connection and surfaces `Err`.
    async fn request_reply(&self, argv: Vec<Vec<u8>>) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("redis cache empty".into()))?;

        match run_command(&mut cached.session, &mut cached.conn, &argv).await {
            Ok(value) => Ok(Response::ok(Bytes::from(value.encode()))),
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    /// A subscribe/MONITOR command: send it, then hand the (session, conn) to a
    /// stream that yields each pushed frame as RESP bytes. The pool-of-one cache
    /// is taken for the stream's lifetime.
    async fn stream(&self, argv: Vec<Vec<u8>>) -> Result<Response<Bytes>, ProximaError> {
        let mut guard = self.cached.lock().await;
        let mut cached = match guard.take() {
            Some(cached) => cached,
            None => self.connect().await?,
        };
        drop(guard);

        let refs = borrow(&argv);
        cached
            .session
            .queue_command(&refs)
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
            .map_err(|err| ProximaError::Upstream(format!("redis connect: {err}")))?;
        let mut session = ClientSession::new(&self.config);
        drive_until_ready(&mut session, &mut conn)
            .await
            .map_err(client_error_to_proxima)?;
        Ok(Cached { conn, session })
    }
}

impl<U: StreamUpstream> SendPipe for RedisClientUpstream<U> {
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


/// `[verb] ++ args`, where args are the NUL-delimited segments in `body`.
/// A body with no NUL bytes is a single arg; empty body adds no args.
fn argv_of(command: &str, body: &[u8]) -> Vec<Vec<u8>> {
    let mut argv = vec![command.as_bytes().to_vec()];
    if !body.is_empty() {
        argv.extend(body.split(|byte| *byte == 0).map(|seg| seg.to_vec()));
    }
    argv
}

fn borrow(argv: &[Vec<u8>]) -> Vec<&[u8]> {
    argv.iter().map(Vec::as_slice).collect()
}

async fn run_command<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
    argv: &[Vec<u8>],
) -> Result<RespValue, ClientError> {
    let refs = borrow(argv);
    session.submit(&refs)?;
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Complete(value) => return Ok(value),
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

/// The pub/sub stream's state: either an active (session, conn) pair, or `Done`
/// once the connection closed or errored.
enum StreamState<C> {
    Active(Cached<C>),
    Done,
}

/// One `unfold` step: read pushed frames, yielding each as re-encoded RESP
/// bytes. A clean server close ends the stream; an I/O / protocol failure yields
/// one terminal `Err` then ends.
async fn push_step<C: StreamConnection>(
    state: StreamState<C>,
) -> Option<(Result<Bytes, ProximaError>, StreamState<C>)> {
    let mut cached = match state {
        StreamState::Active(cached) => cached,
        StreamState::Done => return None,
    };
    loop {
        match cached.session.poll_push() {
            Ok(PushStep::Frame(value)) => {
                return Some((Ok(Bytes::from(value.encode())), StreamState::Active(cached)));
            }
            Ok(PushStep::Recv) => {
                let mut chunk = [0_u8; READ_CHUNK_BYTES];
                match cached.conn.read(&mut chunk).await {
                    Ok(0) => return None,
                    Ok(read) => cached.session.feed(&chunk[..read]),
                    Err(error) => {
                        return Some((Err(ProximaError::Io(error)), StreamState::Done));
                    }
                }
            }
            Err(error) => return Some((Err(client_error_to_proxima(error)), StreamState::Done)),
        }
    }
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("redis client: {error}"))
}
