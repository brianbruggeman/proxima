//! The per-connection I/O driver: reads bytes, feeds the sans-IO
//! `proxima_protocols::redis::Connection` FSM, dispatches each parsed
//! command, and writes the RESP reply back onto the wire.
//!
//! Mirrors `proxima_pgwire::connection`'s `main_loop`/`read_some`/
//! `flush_out` shape, plus the datagram-listener multi-source merge
//! pattern (here racing the socket read against this connection's pub/sub
//! push channel instead of pgwire's LISTEN/NOTIFY `notify_rx`) — via
//! [`proxima_listen::wait_for_wire_event`], the shared outer-wait driver
//! (see `crate::wait_sources`). Composes `proxima_protocols::redis`
//! (parse/encode/`Connection`) over any `futures::io` stream — no runtime,
//! no socket type, no TLS knowledge.
//!
//! Pipelining is answered by reading every already-buffered command to
//! completion before the next socket read (the inner loop below); replies
//! are written in request order because each command is awaited to
//! completion — one at a time — before the next is dispatched. Never spawn
//! per-command: pipelining requires N replies in request order, which a
//! spawned-and-raced dispatch cannot guarantee.

use std::collections::BTreeMap;

use bytes::Bytes;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use proxima_core::ProximaError;
use proxima_primitives::pipe::{FanIn, Pipe, PipeExt, Select, SendPipe, SubscriptionId};

use proxima_protocols::redis::{Advanced, ConnMode, Connection as RespConnection, Frame, Limits, RedisRequest, RespValue};

use crate::broker::{PushSink, RedisBroker};
use crate::config::RedisServerConfig;
use crate::error::RedisServeError;
use crate::pipes::RedisPipeHandle;
use crate::wait_sources::RedisConnSource;

async fn flush_out<S: AsyncWrite + Unpin>(stream: &mut S, out: &mut Vec<u8>) -> std::io::Result<()> {
    if !out.is_empty() {
        stream.write_all(out).await?;
        out.clear();
    }
    stream.flush().await
}

fn write_reply(out: &mut Vec<u8>, value: &RespValue) {
    out.extend_from_slice(&value.encode());
}

/// The subscriptions THIS connection holds, keyed by the exact channel /
/// pattern bytes so a later UNSUBSCRIBE/PUNSUBSCRIBE/SUNSUBSCRIBE (or
/// connection close) can remove precisely the right [`SubscriptionId`] from
/// the shared [`RedisBroker`]. `shard_channels` is a namespace distinct from
/// `channels` — real Redis (7.0+) never crosses `SSUBSCRIBE` with
/// `SUBSCRIBE`.
#[derive(Default)]
struct SubscriberState {
    channels: BTreeMap<Vec<u8>, SubscriptionId>,
    patterns: BTreeMap<Vec<u8>, SubscriptionId>,
    shard_channels: BTreeMap<Vec<u8>, SubscriptionId>,
}

impl SubscriberState {
    fn unsubscribe_all(&self, broker: &RedisBroker) {
        for (channel, id) in &self.channels {
            broker.unsubscribe_channel(channel, *id);
        }
        for (pattern, id) in &self.patterns {
            broker.unsubscribe_pattern(pattern, *id);
        }
        for (channel, id) in &self.shard_channels {
            broker.unsubscribe_shard_channel(channel, *id);
        }
    }
}

/// Gate on the business-handler dispatch while the connection is in
/// [`ConnMode::Subscriber`] — `PipeExt::filter(predicate)` composed onto
/// the handler pipe, per the pipe-algebra map (workspace principle 1): a
/// canonical `Connection::admits(verb)` check, not a hand-rolled routing
/// layer. Rejection rides `ProximaError::Forbidden` — the SAME "deliberate
/// refusal, not a failure" variant
/// `proxima_primitives::pipe::KeyedLiveFilter` already uses for a rejected
/// filter predicate — so it renders as a normal `RespValue::Error` reply
/// (the connection stays alive) rather than the internal-error/close path.
struct SubscriberGate {
    admitted: bool,
    verb: Vec<u8>,
}

impl Pipe for SubscriberGate {
    type In = RedisRequest;
    type Out = RedisRequest;
    type Err = ProximaError;

    fn call(
        &self,
        request: RedisRequest,
    ) -> impl core::future::Future<Output = Result<RedisRequest, ProximaError>> {
        let outcome = self.admission_result(request);
        async move { outcome }
    }
}

impl SendPipe for SubscriberGate {
    type In = RedisRequest;
    type Out = RedisRequest;
    type Err = ProximaError;

    fn call(
        &self,
        request: RedisRequest,
    ) -> impl core::future::Future<Output = Result<RedisRequest, ProximaError>> + Send {
        let outcome = self.admission_result(request);
        async move { outcome }
    }
}

impl SubscriberGate {
    fn admission_result(&self, request: RedisRequest) -> Result<RedisRequest, ProximaError> {
        if self.admitted {
            Ok(request)
        } else {
            Err(ProximaError::Forbidden(format!(
                "Can't execute '{}': only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / PING / QUIT / \
                 RESET are allowed in this context",
                String::from_utf8_lossy(&self.verb).to_lowercase()
            )))
        }
    }
}

fn extract_args(frame: &Frame<'_>) -> Result<Vec<Vec<u8>>, &'static str> {
    let Frame::Array(elements) = frame else {
        return Err("expected a multi-bulk command array");
    };
    let mut args = Vec::with_capacity(elements.len());
    for element in elements {
        let Frame::BlobString(bytes) = element else {
            return Err("expected every command element to be a bulk string");
        };
        args.push((*bytes).to_vec());
    }
    Ok(args)
}

fn ping_reply(args: &[Vec<u8>]) -> RespValue {
    match args {
        [] => RespValue::SimpleString("PONG".to_string()),
        [message] => RespValue::BulkString(message.clone()),
        _ => RespValue::Error("ERR wrong number of arguments for 'ping' command".to_string()),
    }
}

fn subscribe_ack(kind: &'static str, channel: Option<&[u8]>, count: usize) -> RespValue {
    let channel_value = match channel {
        Some(bytes) => RespValue::BulkString(bytes.to_vec()),
        None => RespValue::Null,
    };
    RespValue::Array(vec![
        RespValue::BulkString(kind.as_bytes().to_vec()),
        channel_value,
        RespValue::Integer(i64::try_from(count).unwrap_or(i64::MAX)),
    ])
}

/// Outcome of dispatching one parsed frame — what the driver writes (and
/// whether it must close) after `Connection::consume` runs.
enum FrameOutcome {
    Reply(RespValue),
    Frames(Vec<RespValue>),
    Close,
    /// Class 2 stays alive: a normal `RespValue::Error` reply. Class 1/3
    /// (framing violations, handler failures) are NOT modeled here — they
    /// come back through `Advanced::ProtocolError`/`MessageTooLarge` (from
    /// `Connection::advance` itself) or `InternalError` below.
    InternalError(ProximaError),
}

/// Dispatches one command's already-extracted argument list. Takes owned
/// `args` (not a borrowed `Frame<'_>`) deliberately: `Frame<'_>` borrows
/// from the `Connection`'s internal buffer via `Advanced::Command`, and
/// this function also needs `&mut Connection` (to subscribe/unsubscribe,
/// check `admits`, etc.) — holding both at once is exactly the aliasing
/// `h1_connection`'s typestate handles prevent at compile time; here the
/// caller extracts `args` (ending the frame borrow) before calling in.
/// Each arm is a 1:1 mechanical promotion of the former raw byte-`match`:
/// same broker calls, same `ConnMode` transitions, same `admits()` gate on
/// the `Command` arm in Subscriber mode — the only change is dispatching on
/// [`RedisRequest::from_args`]'s FSM-aware carry instead of a bare
/// `verb.as_slice()` match.
#[allow(clippy::too_many_arguments)]
async fn dispatch_args(
    args: Vec<Vec<u8>>,
    connection: &mut RespConnection,
    handler: &RedisPipeHandle,
    broker: &RedisBroker,
    push_sink: &PushSink,
    state: &mut SubscriberState,
    admission: &proxima_listen::admission::ConnAdmission,
) -> FrameOutcome {
    if args.is_empty() {
        return FrameOutcome::Reply(RespValue::Error("ERR unknown command ''".to_string()));
    }
    match RedisRequest::from_args(args) {
        RedisRequest::Subscribe { channels } => {
            let mut frames = Vec::with_capacity(channels.len());
            for channel in channels {
                if connection.subscribe(channel.clone()) {
                    let id = broker.subscribe_channel(&channel, push_sink.clone());
                    state.channels.insert(channel.clone(), id);
                }
                frames.push(subscribe_ack(
                    "subscribe",
                    Some(&channel),
                    connection.subscription_count(),
                ));
            }
            FrameOutcome::Frames(frames)
        }
        RedisRequest::Unsubscribe { channels } => {
            let targets: Vec<Vec<u8>> = if channels.is_empty() {
                state.channels.keys().cloned().collect()
            } else {
                channels
            };
            if targets.is_empty() {
                FrameOutcome::Frames(vec![subscribe_ack(
                    "unsubscribe",
                    None,
                    connection.subscription_count(),
                )])
            } else {
                let mut frames = Vec::with_capacity(targets.len());
                for channel in &targets {
                    connection.unsubscribe(channel);
                    if let Some(id) = state.channels.remove(channel) {
                        broker.unsubscribe_channel(channel, id);
                    }
                    frames.push(subscribe_ack(
                        "unsubscribe",
                        Some(channel),
                        connection.subscription_count(),
                    ));
                }
                FrameOutcome::Frames(frames)
            }
        }
        RedisRequest::Psubscribe { patterns } => {
            let mut frames = Vec::with_capacity(patterns.len());
            for pattern in patterns {
                if connection.psubscribe(pattern.clone()) {
                    let id = broker.subscribe_pattern(&pattern, push_sink.clone());
                    state.patterns.insert(pattern.clone(), id);
                }
                frames.push(subscribe_ack(
                    "psubscribe",
                    Some(&pattern),
                    connection.subscription_count(),
                ));
            }
            FrameOutcome::Frames(frames)
        }
        RedisRequest::Punsubscribe { patterns } => {
            let targets: Vec<Vec<u8>> = if patterns.is_empty() {
                state.patterns.keys().cloned().collect()
            } else {
                patterns
            };
            if targets.is_empty() {
                FrameOutcome::Frames(vec![subscribe_ack(
                    "punsubscribe",
                    None,
                    connection.subscription_count(),
                )])
            } else {
                let mut frames = Vec::with_capacity(targets.len());
                for pattern in &targets {
                    connection.punsubscribe(pattern);
                    if let Some(id) = state.patterns.remove(pattern) {
                        broker.unsubscribe_pattern(pattern, id);
                    }
                    frames.push(subscribe_ack(
                        "punsubscribe",
                        Some(pattern),
                        connection.subscription_count(),
                    ));
                }
                FrameOutcome::Frames(frames)
            }
        }
        RedisRequest::Ssubscribe { channels } => {
            let mut frames = Vec::with_capacity(channels.len());
            for channel in channels {
                if connection.subscribe_shard(channel.clone()) {
                    let id = broker.subscribe_shard_channel(&channel, push_sink.clone());
                    state.shard_channels.insert(channel.clone(), id);
                }
                frames.push(subscribe_ack(
                    "ssubscribe",
                    Some(&channel),
                    connection.shard_subscription_count(),
                ));
            }
            FrameOutcome::Frames(frames)
        }
        RedisRequest::Sunsubscribe { channels } => {
            let targets: Vec<Vec<u8>> = if channels.is_empty() {
                state.shard_channels.keys().cloned().collect()
            } else {
                channels
            };
            if targets.is_empty() {
                FrameOutcome::Frames(vec![subscribe_ack(
                    "sunsubscribe",
                    None,
                    connection.shard_subscription_count(),
                )])
            } else {
                let mut frames = Vec::with_capacity(targets.len());
                for channel in &targets {
                    connection.unsubscribe_shard(channel);
                    if let Some(id) = state.shard_channels.remove(channel) {
                        broker.unsubscribe_shard_channel(channel, id);
                    }
                    frames.push(subscribe_ack(
                        "sunsubscribe",
                        Some(channel),
                        connection.shard_subscription_count(),
                    ));
                }
                FrameOutcome::Frames(frames)
            }
        }
        RedisRequest::Command { verb, args } => match verb.as_slice() {
            b"PING" => FrameOutcome::Reply(ping_reply(&args)),
            b"QUIT" => FrameOutcome::Close,
            b"PUBLISH" if args.len() == 2 => match broker.publish(&args[0], &args[1]).await {
                Ok(count) => {
                    FrameOutcome::Reply(RespValue::Integer(i64::try_from(count).unwrap_or(i64::MAX)))
                }
                Err(error) => FrameOutcome::InternalError(error),
            },
            b"SPUBLISH" if args.len() == 2 => match broker.publish_shard(&args[0], &args[1]).await {
                Ok(count) => {
                    FrameOutcome::Reply(RespValue::Integer(i64::try_from(count).unwrap_or(i64::MAX)))
                }
                Err(error) => FrameOutcome::InternalError(error),
            },
            _ => {
                // Request-level admission: a business command dispatched to the
                // handler is redis's natural "one request" unit (mirrors h1 per
                // request, h2 per stream, pgwire per message). PING/SUBSCRIBE/
                // PUBLISH etc. above are protocol-level framing, not business
                // dispatch, and stay ungated — a health-check PING should not
                // get shed during quiesce/drain.
                if let proxima_listen::admission::RequestAdmit::Shed { reason } =
                    admission.request_admit()
                {
                    return FrameOutcome::Reply(RespValue::Error(format!(
                        "ERR server is shedding requests ({reason:?}); retry shortly"
                    )));
                }
                let admitted = connection.admits(&verb);
                let mode = connection.mode();
                let request = RedisRequest::Command {
                    verb: verb.clone(),
                    args,
                };
                let dispatched = if mode == ConnMode::Subscriber {
                    let gate = SubscriberGate { admitted, verb };
                    SendPipe::call(&handler.clone().filter(gate), request).await
                } else {
                    SendPipe::call(handler.as_ref(), request).await
                };
                admission.request_release();
                match dispatched {
                    Ok(response) => FrameOutcome::Reply(response),
                    Err(ProximaError::Forbidden(reason)) => {
                        FrameOutcome::Reply(RespValue::Error(format!("ERR {reason}")))
                    }
                    Err(error) => FrameOutcome::InternalError(error),
                }
            }
        },
    }
}

/// Serves one accepted connection to completion. Sequential await-per-frame
/// (mandatory — pipelining requires N replies in request order; never spawn
/// per-command). The outer wait races: (a) more socket bytes, (b) this
/// connection's pub/sub push channel (frames `RedisBroker` pushed via
/// `KeyedFanOut`), (c) shutdown — via
/// [`proxima_listen::wait_for_wire_event`], the shared outer-wait driver
/// (see `crate::wait_sources`'s doc for why the inner decode+dispatch loop
/// stays sequential and outside that race).
pub async fn serve_connection<S>(
    stream: S,
    handler: RedisPipeHandle,
    broker: std::sync::Arc<RedisBroker>,
    config: &RedisServerConfig,
    shutdown: oneshot::Receiver<()>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), RedisServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut connection = RespConnection::with_limits(Limits {
        max_message_bytes: config.max_message_bytes,
    });
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let (push_tx, push_rx) = mpsc::unbounded::<Bytes>();
    let push_sink = PushSink::new(push_tx);
    let mut state = SubscriberState::default();

    let (read_half, mut write_half) = stream.split();
    // declaration order == `Select::Fifo` scan order == the old
    // `select_biased!`'s arm order (shutdown, then push, then read) — the
    // SAME tie-break priority, just expressed as array position instead of
    // macro-arm position.
    let sources = FanIn::new(
        [
            RedisConnSource::shutdown(shutdown),
            RedisConnSource::push(push_rx),
            RedisConnSource::read(read_half, config.read_buffer_bytes),
        ],
        Select::Fifo,
    );

    let outcome = main_loop(
        &mut write_half,
        &mut connection,
        &mut out,
        &handler,
        &broker,
        &push_sink,
        &mut state,
        config,
        &sources,
        &admission,
    )
    .await;
    state.unsubscribe_all(&broker);
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn main_loop<S>(
    write_half: &mut futures::io::WriteHalf<S>,
    connection: &mut RespConnection,
    out: &mut Vec<u8>,
    handler: &RedisPipeHandle,
    broker: &RedisBroker,
    push_sink: &PushSink,
    state: &mut SubscriberState,
    config: &RedisServerConfig,
    sources: &FanIn<RedisConnSource<futures::io::ReadHalf<S>>, Select, 3>,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<(), RedisServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    loop {
        loop {
            match connection.advance() {
                Advanced::NeedMore => break,
                Advanced::Command { frame, consumed } => {
                    // extract owned args here — `frame` borrows `connection`
                    // (via `Advanced`'s lifetime), and dispatch below also
                    // needs `&mut connection`; ending the frame's last use
                    // before that call is what releases the borrow (NLL),
                    // exactly the aliasing `h1_connection`'s typestate
                    // handles avoid by construction instead.
                    let extracted = extract_args(&frame);
                    connection.consume(consumed);
                    let outcome = match extracted {
                        Ok(args) => {
                            dispatch_args(
                                args, connection, handler, broker, push_sink, state, admission,
                            )
                            .await
                        }
                        Err(reason) => {
                            tracing::error!(reason, "redis protocol violation");
                            write_reply(
                                out,
                                &RespValue::Error(format!("ERR Protocol error: {reason}")),
                            );
                            flush_out(write_half, out).await?;
                            return Ok(());
                        }
                    };
                    match outcome {
                        FrameOutcome::Reply(value) => write_reply(out, &value),
                        FrameOutcome::Frames(values) => {
                            for value in &values {
                                write_reply(out, value);
                            }
                        }
                        FrameOutcome::Close => {
                            write_reply(out, &RespValue::SimpleString("OK".to_string()));
                            flush_out(write_half, out).await?;
                            return Ok(());
                        }
                        FrameOutcome::InternalError(error) => {
                            tracing::error!(error = %error, "redis handler error");
                            write_reply(out, &RespValue::Error("ERR internal error".to_string()));
                            flush_out(write_half, out).await?;
                            return Err(RedisServeError::Pipe(error));
                        }
                    }
                    if out.len() >= config.write_high_water_bytes {
                        flush_out(write_half, out).await?;
                    }
                }
                Advanced::ProtocolError { reason, .. } => {
                    tracing::error!(reason, "redis malformed frame");
                    write_reply(out, &RespValue::Error(format!("ERR Protocol error: {reason}")));
                    flush_out(write_half, out).await?;
                    return Ok(());
                }
                Advanced::MessageTooLarge => {
                    tracing::error!(limit = config.max_message_bytes, "redis message too large");
                    write_reply(
                        out,
                        &RespValue::Error(format!(
                            "ERR Protocol error: message exceeds {} byte limit",
                            config.max_message_bytes
                        )),
                    );
                    flush_out(write_half, out).await?;
                    return Err(RedisServeError::MessageTooLarge {
                        limit: config.max_message_bytes,
                    });
                }
            }
        }
        flush_out(write_half, out).await?;

        match proxima_listen::wait_for_wire_event(sources, write_half).await? {
            Some(bytes) => connection.feed_bytes(&bytes),
            None => return Ok(()),
        }
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

    struct EchoHandler;

    impl SendPipe for EchoHandler {
        type In = RedisRequest;
        type Out = RespValue;
        type Err = ProximaError;

        fn call(
            &self,
            request: RedisRequest,
        ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
            async move {
                let reply = match request {
                    RedisRequest::Command { verb, .. } if verb == b"GET" => {
                        RespValue::BulkString(b"stub-value".to_vec())
                    }
                    _ => RespValue::Error("ERR unknown command".to_string()),
                };
                Ok(reply)
            }
        }
    }

    fn handler() -> RedisPipeHandle {
        crate::pipes::into_redis_handle(EchoHandler)
    }

    /// A read-once / write-to-shared-vec fake, mirroring
    /// `proxima_pgwire::pipe`'s `ScriptedSocket` test double — sufficient
    /// for a one-shot scripted client conversation with no live push
    /// traffic in flight (QUIT returns before `main_loop` ever reaches its
    /// `select!`). `write_data` is a shared handle (not owned uniquely) so
    /// the test can read it back after `serve_connection` consumes the
    /// socket by value.
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

    async fn drive(wire: &[u8], broker: Arc<RedisBroker>, config: &RedisServerConfig) -> Vec<u8> {
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire.to_vec()),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler(),
            broker,
            config,
            shutdown_rx,
            proxima_listen::admission::ConnAdmission::unbounded(),
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        write_data.lock().expect("write_data lock").clone()
    }

    #[proxima::test(runtime = "tokio")]
    async fn ping_replies_pong() {
        let mut wire = Vec::new();
        proxima_protocols::redis::encode_command(&[b"PING"], &mut wire);
        proxima_protocols::redis::encode_command(&[b"QUIT"], &mut wire);
        let config = RedisServerConfig::default();
        let response = drive(&wire, Arc::new(RedisBroker::new()), &config).await;
        assert_eq!(response, b"+PONG\r\n+OK\r\n");
    }

    #[proxima::test(runtime = "tokio")]
    async fn get_reaches_the_handler_in_command_mode() {
        let mut wire = Vec::new();
        proxima_protocols::redis::encode_command(&[b"GET", b"k"], &mut wire);
        proxima_protocols::redis::encode_command(&[b"QUIT"], &mut wire);
        let config = RedisServerConfig::default();
        let response = drive(&wire, Arc::new(RedisBroker::new()), &config).await;
        assert_eq!(response, b"$10\r\nstub-value\r\n+OK\r\n");
    }

    // The listener's admission policy (quiesce/drain/capacity), not the
    // business handler, decides whether a command reaches the engine.
    // Proves the Shed path renders a real `-ERR` reply (connection stays
    // alive — the next command still gets a normal reply) instead of
    // silently dropping the command or the connection.
    #[proxima::test(runtime = "tokio")]
    async fn business_command_is_shed_with_an_err_reply_while_admission_is_quiescing() {
        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        admission.begin_quiesce();

        let mut wire = Vec::new();
        proxima_protocols::redis::encode_command(&[b"GET", b"k"], &mut wire);
        proxima_protocols::redis::encode_command(&[b"QUIT"], &mut wire);

        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler(),
            Arc::new(RedisBroker::new()),
            &RedisServerConfig::default(),
            shutdown_rx,
            admission,
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        let response = write_data.lock().expect("write_data lock").clone();
        let response_text = String::from_utf8_lossy(&response);
        assert!(
            response_text.starts_with("-ERR server is shedding requests"),
            "expected a shed error reply, got: {response_text:?}"
        );
        assert!(
            response_text.ends_with("+OK\r\n"),
            "QUIT must still be answered normally; connection stays alive through the shed: {response_text:?}"
        );
    }

    // PING/SUBSCRIBE/PUBLISH etc. are protocol-level framing, not business
    // dispatch — they must never be shed even while admission is quiescing
    // (a health-check PING should always succeed).
    #[proxima::test(runtime = "tokio")]
    async fn ping_is_never_shed_even_while_admission_is_quiescing() {
        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        admission.begin_quiesce();

        let mut wire = Vec::new();
        proxima_protocols::redis::encode_command(&[b"PING"], &mut wire);
        proxima_protocols::redis::encode_command(&[b"QUIT"], &mut wire);

        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire),
            write_data: Arc::clone(&write_data),
        };
        serve_connection(
            socket,
            handler(),
            Arc::new(RedisBroker::new()),
            &RedisServerConfig::default(),
            shutdown_rx,
            admission,
        )
        .await
        .expect("serve_connection");
        let response = write_data.lock().expect("write_data lock").clone();
        assert_eq!(response, b"+PONG\r\n+OK\r\n");
    }
}
