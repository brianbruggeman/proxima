//! The per-connection I/O driver: reads bytes, feeds the sans-IO
//! `proxima_protocols::memcached::Connection` FSM, dispatches each parsed
//! command, and writes the reply back onto the wire.
//!
//! Mirrors `proxima_redis::connection`'s `main_loop`/`read_some`/
//! `flush_out` shape, minus the pub/sub `select!` arm redis needs — memcached
//! never pushes anything unsolicited, so the only thing to race against a
//! socket read is shutdown. Composes `proxima_protocols::memcached`
//! (`Connection`, `parse_reply`'s sibling `encode_reply`) over any
//! `futures::io` stream — no runtime, no socket type, no TLS knowledge.
//!
//! Pipelining is answered by reading every already-buffered command to
//! completion before the next socket read (the inner loop below); replies
//! are written in request order because each command is awaited to
//! completion — one at a time — before the next is dispatched. Never spawn
//! per-command: pipelining requires N replies in request order, which a
//! spawned-and-raced dispatch cannot guarantee.
//!
//! A `noreply`-flagged command ([`MemcachedRequest::is_noreply`]) never
//! writes anything to the wire regardless of the handler's outcome — that
//! is the real protocol's contract, not an optimization this driver adds.

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::{Request, RequestContext};

use proxima_protocols::memcached::{
    Advanced, Connection as WireConnection, MemcachedRequest, Reply, StoreMode, encode_reply,
};

use crate::config::MemcachedServerConfig;
use crate::error::MemcachedServeError;
use crate::pipes::{MemcachedPipeHandle, MemcachedPipeRequest};

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

fn method_for(request: &MemcachedRequest) -> Method {
    let verb: &[u8] = match request {
        MemcachedRequest::Get { gets: false, .. } => b"GET",
        MemcachedRequest::Get { gets: true, .. } => b"GETS",
        MemcachedRequest::Store {
            mode: StoreMode::Set,
            ..
        } => b"SET",
        MemcachedRequest::Store {
            mode: StoreMode::Add,
            ..
        } => b"ADD",
        MemcachedRequest::Store {
            mode: StoreMode::Replace,
            ..
        } => b"REPLACE",
        MemcachedRequest::Store {
            mode: StoreMode::Append,
            ..
        } => b"APPEND",
        MemcachedRequest::Store {
            mode: StoreMode::Prepend,
            ..
        } => b"PREPEND",
        MemcachedRequest::Cas { .. } => b"CAS",
        MemcachedRequest::Delete { .. } => b"DELETE",
        MemcachedRequest::Counter {
            increment: true, ..
        } => b"INCR",
        MemcachedRequest::Counter {
            increment: false, ..
        } => b"DECR",
        MemcachedRequest::Touch { .. } => b"TOUCH",
        MemcachedRequest::FlushAll { .. } => b"FLUSH_ALL",
        MemcachedRequest::Stats { .. } => b"STATS",
        MemcachedRequest::Version => b"VERSION",
        MemcachedRequest::Quit => b"QUIT",
    };
    Method::from_bytes(verb)
}

fn build_request(payload: MemcachedRequest) -> MemcachedPipeRequest {
    Request {
        method: method_for(&payload),
        path: Bytes::new(),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload,
        stream: None,
        context: RequestContext::default(),
    }
}

/// Outcome of dispatching one parsed command — what the driver writes (or
/// doesn't) after `Connection::consume` runs.
enum FrameOutcome {
    Reply(Reply),
    /// A `noreply`-flagged command: the wire contract forbids ANY reply,
    /// success or failure alike.
    Silent,
    Close,
    InternalError(ProximaError),
}

async fn dispatch_request(
    request: MemcachedRequest,
    handler: &MemcachedPipeHandle,
    admission: &proxima_listen::admission::ConnAdmission,
) -> FrameOutcome {
    if matches!(request, MemcachedRequest::Quit) {
        return FrameOutcome::Close;
    }
    let noreply = request.is_noreply();

    if let proxima_listen::admission::RequestAdmit::Shed { reason } = admission.request_admit() {
        return if noreply {
            FrameOutcome::Silent
        } else {
            FrameOutcome::Reply(Reply::ServerError(
                format!("server is shedding requests ({reason:?}); retry shortly").into_bytes(),
            ))
        };
    }

    let dispatched = SendPipe::call(handler.as_ref(), build_request(request)).await;
    admission.request_release();
    match dispatched {
        Ok(_response) if noreply => FrameOutcome::Silent,
        Ok(response) => FrameOutcome::Reply(response.payload),
        Err(_error) if noreply => FrameOutcome::Silent,
        Err(error) => FrameOutcome::InternalError(error),
    }
}

/// Serves one accepted connection to completion. Sequential await-per-command
/// (mandatory — pipelining requires N replies in request order; never spawn
/// per-command). The `select!` at the bottom races: (a) more socket bytes,
/// (b) shutdown.
pub async fn serve_connection<S>(
    mut stream: S,
    handler: MemcachedPipeHandle,
    config: &MemcachedServerConfig,
    shutdown: oneshot::Receiver<()>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), MemcachedServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut connection = WireConnection::with_limits(proxima_protocols::memcached::Limits {
        max_message_bytes: config.max_message_bytes,
    });
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let mut scratch = vec![0_u8; config.read_buffer_bytes];

    main_loop(
        &mut stream,
        &mut connection,
        &mut out,
        &mut scratch,
        &handler,
        config,
        shutdown,
        &admission,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn main_loop<S>(
    stream: &mut S,
    connection: &mut WireConnection,
    out: &mut Vec<u8>,
    scratch: &mut [u8],
    handler: &MemcachedPipeHandle,
    config: &MemcachedServerConfig,
    mut shutdown: oneshot::Receiver<()>,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<(), MemcachedServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    loop {
        loop {
            match connection.advance() {
                Advanced::NeedMore => break,
                Advanced::Command { command, consumed } => {
                    // extract owned request here — `command` borrows
                    // `connection` (via `Advanced`'s lifetime), and dispatch
                    // below needs `&mut connection` (indirectly, through
                    // `consume`); ending the borrow before that call is what
                    // releases it (NLL).
                    let request = MemcachedRequest::from_command(&command);
                    connection.consume(consumed);
                    let outcome = dispatch_request(request, handler, admission).await;
                    match outcome {
                        FrameOutcome::Reply(reply) => encode_reply(&reply, out),
                        FrameOutcome::Silent => {}
                        FrameOutcome::Close => {
                            flush_out(stream, out).await?;
                            return Ok(());
                        }
                        FrameOutcome::InternalError(error) => {
                            tracing::error!(error = %error, "memcached handler error");
                            encode_reply(&Reply::ServerError(b"internal error".to_vec()), out);
                            flush_out(stream, out).await?;
                            return Err(MemcachedServeError::Pipe(error));
                        }
                    }
                    if out.len() >= config.write_high_water_bytes {
                        flush_out(stream, out).await?;
                    }
                }
                Advanced::ProtocolError { error } => {
                    tracing::error!(reason = %error, "memcached protocol violation");
                    encode_reply(&Reply::Error, out);
                    flush_out(stream, out).await?;
                    return Ok(());
                }
                Advanced::MessageTooLarge => {
                    tracing::error!(limit = config.max_message_bytes, "memcached message too large");
                    encode_reply(
                        &Reply::ServerError(
                            format!(
                                "message exceeds {} byte limit",
                                config.max_message_bytes
                            )
                            .into_bytes(),
                        ),
                        out,
                    );
                    flush_out(stream, out).await?;
                    return Err(MemcachedServeError::MessageTooLarge {
                        limit: config.max_message_bytes,
                    });
                }
            }
        }
        flush_out(stream, out).await?;

        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            read = read_some(stream, scratch).fuse() => {
                match read? {
                    0 => return Ok(()),
                    count => connection.feed_bytes(&scratch[..count]),
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;
    use std::io::Read;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct EchoHandler;

    impl SendPipe for EchoHandler {
        type In = MemcachedPipeRequest;
        type Out = crate::pipes::MemcachedPipeReply;
        type Err = ProximaError;

        fn call(
            &self,
            request: MemcachedPipeRequest,
        ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
            async move {
                let reply = match request.payload {
                    MemcachedRequest::Get { keys, .. } if keys == vec![b"k".to_vec()] => {
                        Reply::Values(vec![proxima_protocols::memcached::StoredValue {
                            key: b"k".to_vec(),
                            flags: 0,
                            data: b"stub-value".to_vec(),
                            cas_unique: None,
                        }])
                    }
                    MemcachedRequest::Get { .. } => Reply::Values(Vec::new()),
                    MemcachedRequest::Store { .. } => Reply::Stored,
                    MemcachedRequest::Delete { .. } => Reply::Deleted,
                    _ => Reply::Error,
                };
                Ok(Response::typed(200, reply))
            }
        }
    }

    fn handler() -> MemcachedPipeHandle {
        crate::pipes::into_memcached_handle(EchoHandler)
    }

    /// A read-once / write-to-shared-vec fake, mirroring
    /// `proxima_redis::connection`'s `ScriptedSocket` test double — a
    /// one-shot scripted client conversation with no live push traffic
    /// (QUIT returns before `main_loop` ever reaches its `select!`).
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

    async fn drive(wire: &[u8], config: &MemcachedServerConfig) -> Vec<u8> {
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire.to_vec()),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler(),
            config,
            shutdown_rx,
            proxima_listen::admission::ConnAdmission::unbounded(),
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        write_data.lock().expect("write_data lock").clone()
    }

    #[proxima::test(runtime = "tokio")]
    async fn get_hit_reaches_the_handler() {
        let mut wire = Vec::new();
        wire.extend_from_slice(b"get k\r\n");
        wire.extend_from_slice(b"quit\r\n");
        let config = MemcachedServerConfig::default();
        let response = drive(&wire, &config).await;
        assert_eq!(response, b"VALUE k 0 10\r\nstub-value\r\nEND\r\n");
    }

    #[proxima::test(runtime = "tokio")]
    async fn set_reaches_the_handler_and_replies_stored() {
        let mut wire = Vec::new();
        wire.extend_from_slice(b"set k 0 0 5\r\nhello\r\n");
        wire.extend_from_slice(b"quit\r\n");
        let config = MemcachedServerConfig::default();
        let response = drive(&wire, &config).await;
        assert_eq!(response, b"STORED\r\n");
    }

    #[proxima::test(runtime = "tokio")]
    async fn noreply_set_never_writes_a_reply() {
        let mut wire = Vec::new();
        wire.extend_from_slice(b"set k 0 0 5 noreply\r\nhello\r\n");
        wire.extend_from_slice(b"quit\r\n");
        let config = MemcachedServerConfig::default();
        let response = drive(&wire, &config).await;
        assert_eq!(response, b"", "noreply must suppress the STORED reply");
    }

    #[proxima::test(runtime = "tokio")]
    async fn unknown_command_closes_the_connection_with_an_error() {
        let mut wire = Vec::new();
        wire.extend_from_slice(b"bogus\r\n");
        let config = MemcachedServerConfig::default();
        let response = drive(&wire, &config).await;
        assert_eq!(response, b"ERROR\r\n");
    }

    // The listener's admission policy (quiesce/drain/capacity), not the
    // business handler, decides whether a command reaches the engine.
    #[proxima::test(runtime = "tokio")]
    async fn business_command_is_shed_with_a_server_error_reply_while_admission_is_quiescing() {
        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        admission.begin_quiesce();

        let mut wire = Vec::new();
        wire.extend_from_slice(b"delete k\r\n");
        wire.extend_from_slice(b"quit\r\n");

        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler(),
            &MemcachedServerConfig::default(),
            shutdown_rx,
            admission,
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        let response = write_data.lock().expect("write_data lock").clone();
        let response_text = String::from_utf8_lossy(&response);
        assert!(
            response_text.starts_with("SERVER_ERROR server is shedding requests"),
            "expected a shed error reply, got: {response_text:?}"
        );
    }
}
