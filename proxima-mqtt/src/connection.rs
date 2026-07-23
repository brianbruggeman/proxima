//! The per-connection I/O driver: reads bytes, feeds the sans-IO
//! `proxima_protocols::mqtt::Connection` FSM, dispatches each parsed
//! packet, and writes the reply packet back onto the wire.
//!
//! Mirrors `proxima_redis::connection`'s `main_loop`/`read_some`/
//! `flush_out` shape, plus the datagram-listener multi-source `select!`
//! pattern (here racing the socket read against this connection's pub/sub
//! push channel instead of redis's PUBLISH-driven one). Composes
//! `proxima_protocols::mqtt` (`Connection`/`parse_packet`/`encode`) over
//! any `futures::io` stream — no runtime, no socket type, no TLS
//! knowledge.
//!
//! Every packet gets exactly one reply packet, in order (or none, for
//! `DISCONNECT`/ignored stray acks) — MQTT v3.1.1 has no pipelined
//! multi-reply shape the way RESP does, so there is no analogue of
//! redis's `Frames(Vec<RespValue>)` outcome.

use std::collections::BTreeMap;

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::mpsc::{self, UnboundedReceiver};
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use futures::stream::StreamExt;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_primitives::pipe::HeaderList;

use proxima_protocols::mqtt::encode::{
    encode_ack, encode_connack, encode_pingresp, encode_suback, iter_subscribe_filters,
    iter_unsubscribe_filters,
};
use proxima_protocols::mqtt::{
    Advanced, Connection as MqttConnection, Limits, MqttRequest, Packet, PacketType, read_string,
};

use crate::broker::{MqttBroker, PushSink};
use crate::config::MqttServerConfig;
use crate::error::MqttServeError;
use crate::pipes::{MqttPipeHandle, MqttPipeRequest};
use crate::topic_filter::is_valid_filter;

async fn read_some<S: AsyncRead + Unpin>(
    stream: &mut S,
    scratch: &mut [u8],
) -> std::io::Result<usize> {
    stream.read(scratch).await
}

async fn flush_out<S: AsyncWrite + Unpin>(stream: &mut S, out: &mut Vec<u8>) -> std::io::Result<()> {
    if !out.is_empty() {
        stream.write_all(out).await?;
        out.clear();
    }
    stream.flush().await
}

/// This connection's own subscriptions, keyed by the exact filter bytes so
/// a later `UNSUBSCRIBE` (or connection close) can remove precisely the
/// right `SubscriptionId` from the shared [`MqttBroker`].
#[derive(Default)]
struct SubscriberState {
    filters: BTreeMap<Vec<u8>, proxima_primitives::pipe::SubscriptionId>,
}

impl SubscriberState {
    fn unsubscribe_all(&self, broker: &MqttBroker) {
        for (filter, id) in &self.filters {
            broker.unsubscribe(filter, *id);
        }
    }
}

/// Outcome of dispatching one parsed packet — what the driver writes (and
/// whether it must close) after `Connection::advance` runs.
enum FrameOutcome {
    /// Write these encoded bytes; the connection stays open.
    Reply(Vec<u8>),
    /// The packet was handled with nothing to send back (an ignored stray
    /// ack, or a `PUBLISH` fan-out with no wire reply of its own).
    NoReply,
    /// Write these encoded bytes (may be empty — `DISCONNECT` has none),
    /// then close.
    Close(Vec<u8>),
    /// A handler-pipe failure. `MessageTooLarge`/malformed-frame failures
    /// come back through `Advanced` itself, not this variant.
    InternalError(ProximaError),
}

fn build_connect_request(
    client_id: &[u8],
    clean_session: bool,
    keep_alive: u16,
    username: Option<&[u8]>,
    password: Option<&[u8]>,
) -> MqttPipeRequest {
    Request {
        method: proxima_primitives::pipe::method::Method::from_bytes(b"CONNECT"),
        path: Bytes::new(),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: MqttRequest::Connect {
            client_id: client_id.to_vec(),
            clean_session,
            keep_alive,
            username: username.map(<[u8]>::to_vec),
            password: password.map(<[u8]>::to_vec),
        },
        stream: None,
        context: RequestContext::default(),
    }
}

/// The `[User Name]`/`[Password]` fields walked off a `CONNECT` payload's
/// remainder by [`parse_connect_credentials`].
struct ConnectCredentials {
    username: Option<Vec<u8>>,
    password: Option<Vec<u8>>,
}

/// Walks a `CONNECT` payload's remainder (everything after the client ID)
/// per the connect-flags bits: `[Will Topic, Will Message]` (skipped — out
/// of scope, see the crate's module doc), then `[User Name]`, then
/// `[Password]`.
fn parse_connect_credentials(
    connect_flags: u8,
    rest: &[u8],
) -> Result<ConnectCredentials, &'static str> {
    let mut cursor = rest;
    if connect_flags & 0x04 != 0 {
        let (_will_topic, after_topic) = read_string(cursor).map_err(|_| "malformed will topic")?;
        let (_will_message, after_message) =
            read_string(after_topic).map_err(|_| "malformed will message")?;
        cursor = after_message;
    }
    let username = if connect_flags & 0x80 != 0 {
        let (value, after) = read_string(cursor).map_err(|_| "malformed username")?;
        cursor = after;
        Some(value.to_vec())
    } else {
        None
    };
    let password = if connect_flags & 0x40 != 0 {
        let (value, _after) = read_string(cursor).map_err(|_| "malformed password")?;
        Some(value.to_vec())
    } else {
        None
    };
    Ok(ConnectCredentials { username, password })
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_connect(
    protocol_name: &[u8],
    protocol_level: u8,
    connect_flags: u8,
    keep_alive: u16,
    client_id: &[u8],
    rest: &[u8],
    connected: &mut bool,
    handler: &MqttPipeHandle,
) -> FrameOutcome {
    if *connected {
        // [MQTT-3.1.0-2]: a second CONNECT on an already-open session is a
        // protocol violation — no CONNACK, just close.
        return FrameOutcome::Close(Vec::new());
    }
    if protocol_name != b"MQTT" || protocol_level != 4 {
        let mut out = Vec::new();
        encode_connack(false, 1, &mut out); // 1 = unacceptable protocol version
        return FrameOutcome::Close(out);
    }
    let clean_session = connect_flags & 0x02 != 0;
    let ConnectCredentials { username, password } = match parse_connect_credentials(connect_flags, rest) {
        Ok(credentials) => credentials,
        Err(reason) => {
            return FrameOutcome::InternalError(ProximaError::Upstream(format!(
                "mqtt: {reason}"
            )));
        }
    };
    let request = build_connect_request(
        client_id,
        clean_session,
        keep_alive,
        username.as_deref(),
        password.as_deref(),
    );
    match SendPipe::call(handler, request).await {
        Ok(_) => {
            *connected = true;
            let mut out = Vec::new();
            encode_connack(false, 0, &mut out);
            FrameOutcome::Reply(out)
        }
        Err(ProximaError::Forbidden(_)) => {
            let mut out = Vec::new();
            encode_connack(false, 5, &mut out); // 5 = not authorized
            FrameOutcome::Close(out)
        }
        Err(error) => FrameOutcome::InternalError(error),
    }
}

async fn dispatch_publish(
    topic: &[u8],
    packet_id: Option<u16>,
    payload: &[u8],
    qos: u8,
    broker: &MqttBroker,
) -> FrameOutcome {
    if let Err(error) = broker.publish(topic, payload).await {
        return FrameOutcome::InternalError(error);
    }
    match (qos, packet_id) {
        (0, _) => FrameOutcome::NoReply,
        (1, Some(id)) => {
            let mut out = Vec::new();
            encode_ack(PacketType::PubAck, id, &mut out);
            FrameOutcome::Reply(out)
        }
        (2, Some(id)) => {
            let mut out = Vec::new();
            encode_ack(PacketType::PubRec, id, &mut out);
            FrameOutcome::Reply(out)
        }
        // qos > 0 with no packet_id is a framing violation `parse_publish`
        // already prevents; unreachable in practice.
        _ => FrameOutcome::NoReply,
    }
}

fn dispatch_subscribe(
    packet_id: u16,
    payload: &[u8],
    broker: &MqttBroker,
    push_sink: &PushSink,
    state: &mut SubscriberState,
) -> FrameOutcome {
    let mut granted = Vec::new();
    for (filter, _requested_qos) in iter_subscribe_filters(payload) {
        if !is_valid_filter(filter) {
            granted.push(0x80);
            continue;
        }
        if !state.filters.contains_key(filter) {
            let id = broker.subscribe(filter, push_sink.clone());
            state.filters.insert(filter.to_vec(), id);
        }
        granted.push(0); // delivery is always QoS 0 — see broker module docs
    }
    let mut out = Vec::new();
    encode_suback(packet_id, &granted, &mut out);
    FrameOutcome::Reply(out)
}

fn dispatch_unsubscribe(
    packet_id: u16,
    payload: &[u8],
    broker: &MqttBroker,
    state: &mut SubscriberState,
) -> FrameOutcome {
    for filter in iter_unsubscribe_filters(payload) {
        if let Some(id) = state.filters.remove(filter) {
            broker.unsubscribe(filter, id);
        }
    }
    let mut out = Vec::new();
    encode_ack(PacketType::UnsubAck, packet_id, &mut out);
    FrameOutcome::Reply(out)
}

/// Serves one accepted connection to completion. Sequential await-per-packet
/// — MQTT has no pipelining concept, but the same "never spawn per-packet"
/// discipline redis's driver documents applies for reply ordering. The
/// `select!` at the bottom races: (a) more socket bytes, (b) this
/// connection's pub/sub push channel (frames `MqttBroker` pushed via
/// `KeyedFanOut`), (c) shutdown.
pub async fn serve_connection<S>(
    mut stream: S,
    handler: MqttPipeHandle,
    broker: std::sync::Arc<MqttBroker>,
    config: &MqttServerConfig,
    shutdown: oneshot::Receiver<()>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), MqttServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut connection = MqttConnection::with_limits(Limits {
        max_message_bytes: config.max_message_bytes,
    });
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let mut scratch = vec![0_u8; config.read_buffer_bytes];
    let (push_tx, push_rx) = mpsc::unbounded::<Bytes>();
    let push_sink = PushSink::new(push_tx);
    let mut state = SubscriberState::default();
    let mut connected = false;

    let outcome = main_loop(
        &mut stream,
        &mut connection,
        &mut out,
        &mut scratch,
        &handler,
        &broker,
        &push_sink,
        &mut state,
        &mut connected,
        config,
        push_rx,
        shutdown,
        &admission,
    )
    .await;
    state.unsubscribe_all(&broker);
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn main_loop<S>(
    stream: &mut S,
    connection: &mut MqttConnection,
    out: &mut Vec<u8>,
    scratch: &mut [u8],
    handler: &MqttPipeHandle,
    broker: &MqttBroker,
    push_sink: &PushSink,
    state: &mut SubscriberState,
    connected: &mut bool,
    config: &MqttServerConfig,
    mut push_rx: UnboundedReceiver<Bytes>,
    mut shutdown: oneshot::Receiver<()>,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<(), MqttServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    loop {
        loop {
            match connection.advance() {
                Advanced::NeedMore => break,
                Advanced::Command { packet, consumed } => {
                    // [MQTT-3.1.0-1]: the first packet on a connection must
                    // be CONNECT.
                    if !*connected && !matches!(packet, Packet::Connect { .. }) {
                        return Ok(());
                    }
                    let outcome = dispatch_packet(
                        packet, connected, handler, broker, push_sink, state, admission,
                    )
                    .await;
                    connection.consume(consumed);
                    match outcome {
                        FrameOutcome::Reply(bytes) => out.extend_from_slice(&bytes),
                        FrameOutcome::NoReply => {}
                        FrameOutcome::Close(bytes) => {
                            out.extend_from_slice(&bytes);
                            flush_out(stream, out).await?;
                            return Ok(());
                        }
                        FrameOutcome::InternalError(error) => {
                            tracing::error!(error = %error, "mqtt handler error");
                            return Err(MqttServeError::Pipe(error));
                        }
                    }
                    if out.len() >= config.write_high_water_bytes {
                        flush_out(stream, out).await?;
                    }
                }
                Advanced::ProtocolError { reason, .. } => {
                    // MQTT v3.1.1 has no error-report packet — a broker
                    // closes on a framing violation rather than replying.
                    tracing::error!(reason, "mqtt protocol violation");
                    return Ok(());
                }
                Advanced::MessageTooLarge => {
                    tracing::error!(limit = config.max_message_bytes, "mqtt message too large");
                    return Err(MqttServeError::MessageTooLarge {
                        limit: config.max_message_bytes,
                    });
                }
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
                        // sender half lives on `push_sink`, held by this same
                        // task — `None` cannot happen while the loop runs;
                        // parking (not panicking) is the house style for a
                        // violated-by-construction invariant.
                    }
                }
            }
            read = read_some(stream, scratch).fuse() => {
                match read? {
                    0 => return Ok(()),
                    count => connection.feed_bytes(&scratch[..count]),
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_packet(
    packet: Packet<'_>,
    connected: &mut bool,
    handler: &MqttPipeHandle,
    broker: &MqttBroker,
    push_sink: &PushSink,
    state: &mut SubscriberState,
    admission: &proxima_listen::admission::ConnAdmission,
) -> FrameOutcome {
    match packet {
        Packet::Connect {
            protocol_name,
            protocol_level,
            connect_flags,
            keep_alive,
            client_id,
            rest,
        } => {
            if let proxima_listen::admission::RequestAdmit::Shed { .. } = admission.request_admit()
            {
                return FrameOutcome::Close(Vec::new());
            }
            let outcome = dispatch_connect(
                protocol_name,
                protocol_level,
                connect_flags,
                keep_alive,
                client_id,
                rest,
                connected,
                handler,
            )
            .await;
            admission.request_release();
            outcome
        }
        Packet::Publish { flags, topic, packet_id, payload } => {
            dispatch_publish(topic, packet_id, payload, flags.qos, broker).await
        }
        Packet::Subscribe { packet_id, topic_filters } => {
            dispatch_subscribe(packet_id, topic_filters, broker, push_sink, state)
        }
        Packet::Unsubscribe { packet_id, topic_filters } => {
            dispatch_unsubscribe(packet_id, topic_filters, broker, state)
        }
        Packet::Ack { packet_type: PacketType::PubRel, packet_id } => {
            let mut out = Vec::new();
            encode_ack(PacketType::PubComp, packet_id, &mut out);
            FrameOutcome::Reply(out)
        }
        // a client legitimately sends us only PUBREL among the Ack family
        // (our delivery is always QoS 0 — see the broker module docs — so
        // PUBACK/PUBCOMP/UNSUBACK never arrive here); ignore the rest
        // rather than closing over a harmless stray ack.
        Packet::Ack { .. } => FrameOutcome::NoReply,
        Packet::PingReq => {
            let mut out = Vec::new();
            encode_pingresp(&mut out);
            FrameOutcome::Reply(out)
        }
        Packet::Disconnect => FrameOutcome::Close(Vec::new()),
        // server-to-client-only packets arriving from a client are a
        // protocol violation; close rather than guess at intent.
        Packet::ConnAck { .. } | Packet::SubAck { .. } | Packet::PingResp => {
            FrameOutcome::Close(Vec::new())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;
    use proxima_protocols::mqtt::MqttReply;
    use proxima_protocols::mqtt::encode::{encode_connect, encode_disconnect, encode_publish, encode_subscribe};
    use std::io::Read;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct AcceptAllHandler;

    impl SendPipe for AcceptAllHandler {
        type In = MqttPipeRequest;
        type Out = crate::pipes::MqttPipeReply;
        type Err = ProximaError;

        fn call(
            &self,
            _request: MqttPipeRequest,
        ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
            async move { Ok(Response::typed(200, MqttReply::ConnAck { session_present: false, return_code: 0 })) }
        }
    }

    struct RejectHandler;

    impl SendPipe for RejectHandler {
        type In = MqttPipeRequest;
        type Out = crate::pipes::MqttPipeReply;
        type Err = ProximaError;

        fn call(
            &self,
            _request: MqttPipeRequest,
        ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
            async move { Err(ProximaError::Forbidden("nope".into())) }
        }
    }

    fn handler() -> MqttPipeHandle {
        crate::pipes::into_mqtt_handle(AcceptAllHandler)
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

    async fn drive(
        wire: &[u8],
        handler: MqttPipeHandle,
        broker: Arc<MqttBroker>,
        config: &MqttServerConfig,
    ) -> Vec<u8> {
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire.to_vec()),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler,
            broker,
            config,
            shutdown_rx,
            proxima_listen::admission::ConnAdmission::unbounded(),
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        write_data.lock().expect("write_data lock").clone()
    }

    fn connect_wire(client_id: &str) -> Vec<u8> {
        let mut wire = Vec::new();
        encode_connect(client_id.as_bytes(), true, 60, None, None, &mut wire);
        wire
    }

    #[proxima::test(runtime = "tokio")]
    async fn connect_then_disconnect_replies_connack_then_closes() {
        let mut wire = connect_wire("c1");
        encode_disconnect(&mut wire);
        let response = drive(&wire, handler(), Arc::new(MqttBroker::new()), &MqttServerConfig::default()).await;
        assert_eq!(response, vec![0x20, 0x02, 0x00, 0x00]);
    }

    #[proxima::test(runtime = "tokio")]
    async fn pingreq_replies_pingresp() {
        let mut wire = connect_wire("c1");
        proxima_protocols::mqtt::encode::encode_pingreq(&mut wire);
        encode_disconnect(&mut wire);
        let response = drive(&wire, handler(), Arc::new(MqttBroker::new()), &MqttServerConfig::default()).await;
        assert_eq!(response, vec![0x20, 0x02, 0x00, 0x00, 0xD0, 0x00]);
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_before_connect_closes_without_a_reply() {
        let mut wire = Vec::new();
        encode_publish(b"a/b", None, b"hi", 0, false, false, &mut wire);
        let response = drive(&wire, handler(), Arc::new(MqttBroker::new()), &MqttServerConfig::default()).await;
        assert!(response.is_empty());
    }

    #[proxima::test(runtime = "tokio")]
    async fn connect_is_rejected_with_connack_5_when_the_handler_forbids_it() {
        let wire = connect_wire("c1");
        let response = drive(
            &wire,
            crate::pipes::into_mqtt_handle(RejectHandler),
            Arc::new(MqttBroker::new()),
            &MqttServerConfig::default(),
        )
        .await;
        assert_eq!(response, vec![0x20, 0x02, 0x00, 0x05]);
    }

    #[proxima::test(runtime = "tokio")]
    async fn subscribe_grants_qos0_and_registers_on_the_broker() {
        let broker = Arc::new(MqttBroker::new());
        let mut wire = connect_wire("subscriber");
        encode_subscribe(1, &[(b"news/#", 1)], &mut wire);
        let response = drive(&wire, handler(), Arc::clone(&broker), &MqttServerConfig::default()).await;
        // CONNACK(accepted) + SUBACK(granted=[0], QoS 0 regardless of the
        // requested QoS — see the broker module docs)
        assert_eq!(response, vec![0x20, 0x02, 0x00, 0x00, 0x90, 0x03, 0x00, 0x01, 0x00]);
        assert_eq!(
            broker.subscription_count(b"news/#"),
            0,
            "the connection closed on EOF and unsubscribed everything"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_qos1_replies_puback() {
        let mut wire = connect_wire("publisher");
        encode_publish(b"a/b", Some(9), b"hi", 1, false, false, &mut wire);
        encode_disconnect(&mut wire);
        let response = drive(&wire, handler(), Arc::new(MqttBroker::new()), &MqttServerConfig::default()).await;
        assert_eq!(response, vec![0x20, 0x02, 0x00, 0x00, 0x40, 0x02, 0x00, 0x09]);
    }

    // The listener's admission policy (quiesce/drain/capacity), not the
    // business handler, decides whether a CONNECT reaches the auth hook.
    #[proxima::test(runtime = "tokio")]
    async fn connect_is_closed_without_a_connack_while_admission_is_quiescing() {
        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        admission.begin_quiesce();

        let wire = connect_wire("c1");
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let write_data = Arc::new(std::sync::Mutex::new(Vec::new()));
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(wire),
            write_data: Arc::clone(&write_data),
        };
        let outcome = serve_connection(
            socket,
            handler(),
            Arc::new(MqttBroker::new()),
            &MqttServerConfig::default(),
            shutdown_rx,
            admission,
        )
        .await;
        assert!(outcome.is_ok(), "serve_connection: {outcome:?}");
        assert!(write_data.lock().expect("write_data lock").is_empty());
    }
}
