//! The per-connection I/O driver: reads bytes, feeds the sans-IO
//! [`crate::fsm::Connection`], drives the server-initiated
//! connection/channel handshake, dispatches every broker-level method
//! (exchange/queue declare, `basic.consume`/`cancel`/`qos`) directly
//! against [`AmqpBroker`], sends every `basic.publish` through the
//! business [`AmqpPipeHandle`] before routing it, and writes AMQP frames
//! back onto the wire.
//!
//! Mirrors `proxima_redis::connection`'s `main_loop`/`read_some`/
//! `flush_out` shape, plus the same multi-source `select!` pattern (racing
//! the socket read against this connection's consumer push channel).
//!
//! Sequential await-per-event (mandatory, same reasoning as redis):
//! `basic.publish` replies (implicitly, via routing) must observe request
//! order, and interleaving is answered by reading every already-buffered
//! frame to completion before the next socket read.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::mpsc::{self, UnboundedReceiver};
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use futures::stream::StreamExt;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, RequestContext};

use crate::broker::{AmqpBroker, ConsumerSink, ExchangeKind};
use crate::config::AmqpServerConfig;
use crate::error::AmqpServeError;
use crate::frame::encode_method_frame;
use crate::fsm::{Advanced, Connection as Fsm, Limits};
use crate::method::Method;
use crate::pipes::{AmqpMessage, AmqpPipeHandle, AmqpPipeRequest};
use crate::wire::{FieldTable, FieldValue};

mod reply_code {
    pub const NOT_FOUND: u16 = 404;
    pub const PRECONDITION_FAILED: u16 = 406;
    pub const FRAME_ERROR: u16 = 501;
    pub const SYNTAX_ERROR: u16 = 502;
    pub const COMMAND_INVALID: u16 = 503;
    pub const RESOURCE_ERROR: u16 = 506;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    AwaitingStartOk,
    AwaitingTuneOk,
    AwaitingOpen,
    Ready,
}

/// Per-connection consumer bookkeeping: `basic.consume` registers here so a
/// later `basic.cancel`, `channel.close`, or connection close can remove
/// exactly the right [`proxima_primitives::pipe::SubscriptionId`] from the
/// shared [`AmqpBroker`] — mirrors redis's `SubscriberState`.
#[derive(Default)]
struct ConsumerState {
    subscriptions: BTreeMap<(u16, Vec<u8>), (Vec<u8>, proxima_primitives::pipe::SubscriptionId)>,
    next_consumer_tag: u64,
    next_queue_name: u64,
}

impl ConsumerState {
    fn generate_consumer_tag(&mut self) -> Vec<u8> {
        self.next_consumer_tag += 1;
        format!("amq.ctag-{}", self.next_consumer_tag).into_bytes()
    }

    fn generate_queue_name(&mut self) -> Vec<u8> {
        self.next_queue_name += 1;
        format!("amq.gen-{}", self.next_queue_name).into_bytes()
    }

    fn cancel_all_on_channel(&mut self, channel: u16, broker: &AmqpBroker) {
        let keys: Vec<(u16, Vec<u8>)> = self
            .subscriptions
            .keys()
            .filter(|(channel_of, _)| *channel_of == channel)
            .cloned()
            .collect();
        for key in keys {
            if let Some((queue, id)) = self.subscriptions.remove(&key) {
                broker.unsubscribe_queue(&queue, id);
            }
        }
    }

    fn cancel_all(&mut self, broker: &AmqpBroker) {
        for (queue, id) in self.subscriptions.values() {
            broker.unsubscribe_queue(queue, *id);
        }
        self.subscriptions.clear();
    }
}

fn server_properties() -> FieldTable {
    let mut table = FieldTable::new();
    table.insert(
        "product".into(),
        FieldValue::LongString(b"proxima-amqp".to_vec()),
    );
    table.insert(
        "version".into(),
        FieldValue::LongString(env!("CARGO_PKG_VERSION").as_bytes().to_vec()),
    );
    table
}

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

fn write_connection_close(out: &mut Vec<u8>, reply_code: u16, reply_text: &str) {
    encode_method_frame(
        out,
        0,
        &Method::ConnectionClose {
            reply_code,
            reply_text: reply_text.as_bytes().to_vec(),
            class_id: 0,
            method_id: 0,
        },
    );
}

fn write_channel_close(out: &mut Vec<u8>, channel: u16, reply_code: u16, reply_text: &str) {
    encode_method_frame(
        out,
        channel,
        &Method::ChannelClose {
            reply_code,
            reply_text: reply_text.as_bytes().to_vec(),
            class_id: 0,
            method_id: 0,
        },
    );
}

/// What the caller does after one dispatched event: keep reading, or the
/// connection ended cleanly. A wire-level protocol violation always renders
/// its own `connection.close`/`channel.close` reply and maps to `Close` —
/// there is no separate internal-error outcome here (unlike redis's
/// `FrameOutcome::InternalError`) because every `dispatch_method` failure
/// mode has a real AMQP reply-code to send; only the FSM-level failures
/// handled directly in `main_loop` (`FrameTooLarge`/`MessageTooLarge`)
/// return a hard [`AmqpServeError`].
enum Outcome {
    Continue,
    Close,
}

/// Serves one accepted connection to completion.
#[allow(clippy::too_many_arguments)]
pub async fn serve_connection<S>(
    mut stream: S,
    handler: AmqpPipeHandle,
    broker: std::sync::Arc<AmqpBroker>,
    config: &AmqpServerConfig,
    shutdown: oneshot::Receiver<()>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), AmqpServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut fsm = Fsm::with_limits(Limits {
        frame_max_bytes: config.frame_max_bytes,
        message_max_bytes: config.message_max_bytes,
    });
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let mut scratch = vec![0_u8; config.read_buffer_bytes];
    let (push_tx, push_rx) = mpsc::unbounded::<Bytes>();
    let mut consumers = ConsumerState::default();
    let mut open_channels: BTreeSet<u16> = BTreeSet::new();
    let mut phase = Phase::AwaitingStartOk;

    let outcome = main_loop(
        &mut stream,
        &mut fsm,
        &mut out,
        &mut scratch,
        &handler,
        &broker,
        &push_tx,
        &mut consumers,
        &mut open_channels,
        &mut phase,
        config,
        push_rx,
        shutdown,
        &admission,
    )
    .await;
    consumers.cancel_all(&broker);
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn main_loop<S>(
    stream: &mut S,
    fsm: &mut Fsm,
    out: &mut Vec<u8>,
    scratch: &mut [u8],
    handler: &AmqpPipeHandle,
    broker: &AmqpBroker,
    push_tx: &mpsc::UnboundedSender<Bytes>,
    consumers: &mut ConsumerState,
    open_channels: &mut BTreeSet<u16>,
    phase: &mut Phase,
    config: &AmqpServerConfig,
    mut push_rx: UnboundedReceiver<Bytes>,
    mut shutdown: oneshot::Receiver<()>,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<(), AmqpServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    loop {
        loop {
            match fsm.advance() {
                Advanced::NeedMore => break,
                Advanced::ProtocolHeader => {
                    encode_method_frame(
                        out,
                        0,
                        &Method::ConnectionStart {
                            version_major: 0,
                            version_minor: 9,
                            server_properties: server_properties(),
                            mechanisms: b"PLAIN".to_vec(),
                            locales: b"en_US".to_vec(),
                        },
                    );
                }
                Advanced::Heartbeat => {}
                Advanced::Frame { channel, method } => {
                    match dispatch_method(
                        channel,
                        method,
                        out,
                        phase,
                        fsm,
                        broker,
                        push_tx,
                        consumers,
                        open_channels,
                        config,
                    ) {
                        Outcome::Continue => {}
                        Outcome::Close => {
                            flush_out(stream, out).await?;
                            return Ok(());
                        }
                    }
                }
                Advanced::Publish {
                    exchange,
                    routing_key,
                    mandatory,
                    immediate,
                    properties,
                    body,
                    ..
                } => {
                    dispatch_publish(
                        exchange,
                        routing_key,
                        mandatory,
                        immediate,
                        properties,
                        body,
                        broker,
                        handler,
                        admission,
                    )
                    .await;
                }
                Advanced::Deliver { .. } => {
                    // `basic.deliver` is server -> client only; the FSM's
                    // content reassembly is direction-agnostic (the client
                    // driver reuses the SAME event to receive pushed
                    // messages — see `crate::client::session`), but a
                    // well-behaved client must never send it to us.
                    tracing::error!("amqp protocol violation: client sent basic.deliver");
                    write_connection_close(
                        out,
                        reply_code::SYNTAX_ERROR,
                        "unexpected basic.deliver from client",
                    );
                    flush_out(stream, out).await?;
                    return Ok(());
                }
                Advanced::ProtocolError { reason } => {
                    tracing::error!(reason, "amqp protocol violation");
                    write_connection_close(out, reply_code::SYNTAX_ERROR, &reason);
                    flush_out(stream, out).await?;
                    return Ok(());
                }
                Advanced::FrameTooLarge { limit } => {
                    tracing::error!(limit, "amqp frame exceeds frame-max");
                    write_connection_close(out, reply_code::FRAME_ERROR, "frame exceeds frame-max");
                    flush_out(stream, out).await?;
                    return Err(AmqpServeError::FrameTooLarge { limit });
                }
                Advanced::MessageTooLarge { limit } => {
                    tracing::error!(limit, "amqp message body exceeds limit");
                    write_connection_close(
                        out,
                        reply_code::RESOURCE_ERROR,
                        "message body too large",
                    );
                    flush_out(stream, out).await?;
                    return Err(AmqpServeError::MessageTooLarge { limit });
                }
            }
            if out.len() >= config.write_high_water_bytes {
                flush_out(stream, out).await?;
            }
        }
        flush_out(stream, out).await?;

        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            pushed = push_rx.next().fuse() => {
                match pushed {
                    Some(bytes) => {
                        out.extend_from_slice(&bytes);
                        while let Ok(more) = push_rx.try_recv() {
                            out.extend_from_slice(&more);
                        }
                        flush_out(stream, out).await?;
                    }
                    None => {
                        // the sender half lives on `push_tx`, held by this
                        // same task — `None` cannot happen while the loop
                        // runs; parking (not panicking) is the house style
                        // for a violated-by-construction invariant.
                    }
                }
            }
            read = read_some(stream, scratch).fuse() => {
                match read? {
                    0 => return Ok(()),
                    count => fsm.feed_bytes(&scratch[..count]),
                }
            }
        }
    }
}

/// Dispatches one non-content method frame. Purely synchronous: every
/// branch here is protocol/broker bookkeeping (connection/channel
/// lifecycle, exchange/queue declare, consumer registration) — sync
/// [`AmqpBroker`] calls, never a business-handler round trip. The one
/// method that DOES need the async business handler, `basic.publish`,
/// never reaches here — the FSM diverts it straight to
/// [`crate::fsm::Advanced::Publish`], dispatched by [`dispatch_publish`].
#[allow(clippy::too_many_arguments)]
fn dispatch_method(
    channel: u16,
    method: Method,
    out: &mut Vec<u8>,
    phase: &mut Phase,
    fsm: &mut Fsm,
    broker: &AmqpBroker,
    push_tx: &mpsc::UnboundedSender<Bytes>,
    consumers: &mut ConsumerState,
    open_channels: &mut BTreeSet<u16>,
    config: &AmqpServerConfig,
) -> Outcome {
    if *phase != Phase::Ready {
        return dispatch_handshake(method, out, phase, config);
    }
    match method {
        Method::ChannelOpen => {
            if open_channels.len() >= config.channel_max as usize {
                write_connection_close(out, reply_code::RESOURCE_ERROR, "channel-max exceeded");
                return Outcome::Close;
            }
            open_channels.insert(channel);
            encode_method_frame(out, channel, &Method::ChannelOpenOk);
            Outcome::Continue
        }
        Method::ChannelClose { .. } => {
            consumers.cancel_all_on_channel(channel, broker);
            open_channels.remove(&channel);
            // drops any stale content-reassembly state for this channel
            // number — without this, reopening the SAME channel number
            // mid-`basic.publish` would resume the old (now orphaned)
            // reassembly instead of starting clean.
            fsm.close_channel(channel);
            encode_method_frame(out, channel, &Method::ChannelCloseOk);
            Outcome::Continue
        }
        Method::ChannelCloseOk => Outcome::Continue,
        Method::ExchangeDeclare {
            exchange,
            kind,
            no_wait,
            ..
        } => {
            let Some(exchange_kind) = ExchangeKind::parse(&kind) else {
                write_channel_close(
                    out,
                    channel,
                    reply_code::COMMAND_INVALID,
                    "unsupported exchange type (only direct/fanout/topic)",
                );
                return Outcome::Continue;
            };
            match broker.declare_exchange(exchange, exchange_kind) {
                Ok(()) => {
                    if !no_wait {
                        encode_method_frame(out, channel, &Method::ExchangeDeclareOk);
                    }
                    Outcome::Continue
                }
                Err(_existing_kind) => {
                    write_channel_close(
                        out,
                        channel,
                        reply_code::PRECONDITION_FAILED,
                        "exchange already declared with a different type",
                    );
                    Outcome::Continue
                }
            }
        }
        Method::QueueDeclare { queue, no_wait, .. } => {
            let queue = if queue.is_empty() {
                consumers.generate_queue_name()
            } else {
                queue
            };
            if !no_wait {
                let consumer_count = broker.queue_consumer_count(&queue);
                encode_method_frame(
                    out,
                    channel,
                    &Method::QueueDeclareOk {
                        queue,
                        message_count: 0,
                        consumer_count: consumer_count as u32,
                    },
                );
            }
            Outcome::Continue
        }
        Method::QueueBind {
            queue,
            exchange,
            routing_key,
            no_wait,
            ..
        } => {
            if !broker.bind_queue(&exchange, queue, routing_key) {
                write_channel_close(out, channel, reply_code::NOT_FOUND, "no such exchange");
                return Outcome::Continue;
            }
            if !no_wait {
                encode_method_frame(out, channel, &Method::QueueBindOk);
            }
            Outcome::Continue
        }
        Method::BasicQos { .. } => {
            // no prefetch/QoS enforcement — see the crate-level gap notes.
            encode_method_frame(out, channel, &Method::BasicQosOk);
            Outcome::Continue
        }
        Method::BasicConsume {
            queue,
            consumer_tag,
            no_wait,
            ..
        } => {
            let consumer_tag = if consumer_tag.is_empty() {
                consumers.generate_consumer_tag()
            } else {
                consumer_tag
            };
            let sink = ConsumerSink::new(
                channel,
                consumer_tag.clone(),
                push_tx.clone(),
                config.frame_max_bytes,
            );
            let id = broker.subscribe_queue(&queue, sink);
            consumers
                .subscriptions
                .insert((channel, consumer_tag.clone()), (queue, id));
            if !no_wait {
                encode_method_frame(out, channel, &Method::BasicConsumeOk { consumer_tag });
            }
            Outcome::Continue
        }
        Method::BasicCancel {
            consumer_tag,
            no_wait,
        } => {
            if let Some((queue, id)) = consumers
                .subscriptions
                .remove(&(channel, consumer_tag.clone()))
            {
                broker.unsubscribe_queue(&queue, id);
            }
            if !no_wait {
                encode_method_frame(out, channel, &Method::BasicCancelOk { consumer_tag });
            }
            Outcome::Continue
        }
        Method::BasicAck { .. } | Method::BasicNack { .. } => {
            // no publisher/consumer confirm tracking — see the crate-level
            // gap notes; every delivery behaves as if `no_ack` were set.
            Outcome::Continue
        }
        Method::ConnectionClose { .. } => {
            encode_method_frame(out, 0, &Method::ConnectionCloseOk);
            Outcome::Close
        }
        Method::ConnectionCloseOk => Outcome::Close,
        Method::ExchangeDeclareOk
        | Method::QueueDeclareOk { .. }
        | Method::QueueBindOk
        | Method::BasicQosOk
        | Method::BasicConsumeOk { .. }
        | Method::BasicCancelOk { .. }
        | Method::ConnectionStart { .. }
        | Method::ConnectionStartOk { .. }
        | Method::ConnectionTune { .. }
        | Method::ConnectionTuneOk { .. }
        | Method::ConnectionOpen { .. }
        | Method::ConnectionOpenOk
        | Method::ChannelOpenOk
        | Method::BasicPublish { .. }
        | Method::BasicDeliver { .. } => {
            // server-direction replies replayed back at us, or a content-
            // bearing method (BasicPublish/BasicDeliver) the FSM already
            // diverts upstream before it ever reaches this match — ignored,
            // not a close.
            Outcome::Continue
        }
    }
}

fn dispatch_handshake(
    method: Method,
    out: &mut Vec<u8>,
    phase: &mut Phase,
    config: &AmqpServerConfig,
) -> Outcome {
    match (*phase, method) {
        (Phase::AwaitingStartOk, Method::ConnectionStartOk { .. }) => {
            encode_method_frame(
                out,
                0,
                &Method::ConnectionTune {
                    channel_max: config.channel_max,
                    frame_max: config.frame_max_bytes as u32,
                    heartbeat: config.heartbeat_seconds,
                },
            );
            *phase = Phase::AwaitingTuneOk;
            Outcome::Continue
        }
        (Phase::AwaitingTuneOk, Method::ConnectionTuneOk { .. }) => {
            *phase = Phase::AwaitingOpen;
            Outcome::Continue
        }
        (Phase::AwaitingOpen, Method::ConnectionOpen { .. }) => {
            encode_method_frame(out, 0, &Method::ConnectionOpenOk);
            *phase = Phase::Ready;
            Outcome::Continue
        }
        (_, Method::ConnectionClose { .. }) => {
            encode_method_frame(out, 0, &Method::ConnectionCloseOk);
            Outcome::Close
        }
        (expected, _) => {
            write_connection_close(
                out,
                reply_code::COMMAND_INVALID,
                &format!("expected the next handshake method for {expected:?}"),
            );
            Outcome::Close
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_publish(
    exchange: Vec<u8>,
    routing_key: Vec<u8>,
    mandatory: bool,
    immediate: bool,
    properties: Vec<u8>,
    body: Vec<u8>,
    broker: &AmqpBroker,
    handler: &AmqpPipeHandle,
    admission: &proxima_listen::admission::ConnAdmission,
) {
    if let proxima_listen::admission::RequestAdmit::Shed { reason } = admission.request_admit() {
        tracing::warn!(?reason, "amqp publish shed while admission is quiescing");
        return;
    }
    let request: AmqpPipeRequest = Request {
        method: proxima_primitives::pipe::method::Method::from_bytes(b"PUBLISH"),
        path: Bytes::new(),
        query: proxima_primitives::pipe::header_list::HeaderList::new(),
        metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
        payload: AmqpMessage {
            exchange: exchange.clone(),
            routing_key: routing_key.clone(),
            properties: properties.clone(),
            body: body.clone(),
            mandatory,
            immediate,
        },
        stream: None,
        context: RequestContext::default(),
    };
    let dispatched = SendPipe::call(handler.as_ref(), request).await;
    admission.request_release();
    match dispatched {
        Ok(_reply) => {
            if let Err(error) = broker
                .publish(&exchange, &routing_key, properties, body)
                .await
            {
                tracing::error!(error = %error, "amqp broker publish failed");
            }
        }
        Err(ProximaError::Forbidden(reason)) => {
            tracing::debug!(reason, "amqp publish rejected by handler");
        }
        Err(error) => {
            tracing::error!(error = %error, "amqp publish handler failed");
        }
    }
}
