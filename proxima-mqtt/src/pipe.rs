//! `MqttConnectionPipe` — the connection layer as a real `Pipe`.
//!
//! A freshly accepted socket arrives as a `CONNECT` request (the pipe
//! `Request`, not to be confused with the MQTT `CONNECT` packet the wire
//! driver parses); the pipe answers with `Response.upgrade` (the
//! `proxima_primitives::pipe::upgrade` raw-socket-hijack seam). The
//! upgrade handler runs [`crate::connection::serve_connection`] over the
//! hijacked stream, calling the business-handler pipe once — on the MQTT
//! `CONNECT` packet, for auth. Mirrors `proxima_redis::pipe::RedisConnectionPipe`.
//!
//! MQTT-over-TLS is whole-connection TLS from byte 0 — there is no in-band
//! STARTTLS-style negotiation — so this pipe carries no TLS state of its
//! own; TLS composes as the generic `Listener::builder().tls(config)`
//! decorator over whatever `MqttAnyProtocol` resolves, exactly like the
//! h2/h3 listeners and `RedisConnectionPipe`.

use std::sync::Arc;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncWrite};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};

use crate::broker::MqttBroker;
use crate::config::MqttServerConfig;
use crate::connection::serve_connection;
use crate::error::MqttServeError;
use crate::pipes::MqttPipeHandle;

/// The connection layer as a `Pipe`. Its `call` (for a `CONNECT` pipe
/// request) returns a `Response.upgrade` that drives `serve_connection`
/// over the hijacked stream, calling `handler` once the wire-level MQTT
/// `CONNECT` packet arrives. One `MqttConnectionPipe` owns one
/// [`MqttBroker`], shared by every connection it upgrades — that shared
/// instance IS the PUBLISH/SUBSCRIBE fabric.
#[derive(Clone)]
pub struct MqttConnectionPipe {
    handler: MqttPipeHandle,
    broker: Arc<MqttBroker>,
    config: Arc<MqttServerConfig>,
    label: String,
    /// The listener-wide request-admission handle `MqttAnyProtocol`
    /// installs via [`Self::with_admission`]. `None` (the default a bare
    /// `MqttConnectionPipe::new` gets, e.g. in this file's own unit tests)
    /// resolves to an unbounded, never-quiesced/-drained handle at `call`
    /// time — behavior-preserving for every caller that predates
    /// request-level admission.
    admission: Option<proxima_listen::admission::ConnAdmission>,
}

impl MqttConnectionPipe {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        handler: MqttPipeHandle,
        config: Arc<MqttServerConfig>,
    ) -> Self {
        Self {
            handler,
            broker: Arc::new(MqttBroker::new()),
            config,
            label: label.into(),
            admission: None,
        }
    }

    /// Installs the listener-wide [`proxima_listen::admission::ConnAdmission`]
    /// handle `MqttAnyProtocol::drive` clones into a fresh
    /// `MqttConnectionPipe` per accepted connection.
    #[must_use]
    pub fn with_admission(mut self, admission: proxima_listen::admission::ConnAdmission) -> Self {
        self.admission = Some(admission);
        self
    }

    /// Overrides the broker `new` constructs fresh. `MqttAnyProtocol`
    /// builds a FRESH `MqttConnectionPipe` per accepted connection (to
    /// carry that connection's own `ConnAdmission` clone) but must NOT
    /// give each one its own broker — PUBLISH/SUBSCRIBE only works across
    /// connections when they all share the SAME broker instance, built
    /// once at `MqttAnyProtocol::new` and installed here on every
    /// per-connection pipe.
    #[must_use]
    pub fn with_broker(mut self, broker: Arc<MqttBroker>) -> Self {
        self.broker = broker;
        self
    }

    /// This pipe's label, set at construction.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The shared pub/sub broker every connection this pipe upgrades
    /// registers with.
    #[must_use]
    pub fn broker(&self) -> &Arc<MqttBroker> {
        &self.broker
    }
}

async fn drive_session<S>(
    stream: S,
    handler: MqttPipeHandle,
    broker: Arc<MqttBroker>,
    config: Arc<MqttServerConfig>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), MqttServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let (_shutdown_tx, shutdown_rx) = oneshot::channel();
    serve_connection(stream, handler, broker, &config, shutdown_rx, admission).await
}

impl SendPipe for MqttConnectionPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let handler = self.handler.clone();
        let broker = Arc::clone(&self.broker);
        let config = Arc::clone(&self.config);
        let admission = self
            .admission
            .clone()
            .unwrap_or_else(proxima_listen::admission::ConnAdmission::unbounded);
        let handler_fn = UpgradeHandler::new(move |hijacked: HijackedSocket| async move {
            let HijackedSocket { stream, leftover } = hijacked;
            if !leftover.is_empty() {
                // a raw mqtt socket has no prior protocol head, so the
                // upgrade seam should never hand us pre-buffered bytes;
                // dropping them silently would corrupt the CONNECT packet.
                return Err(ProximaError::Upstream(
                    "mqtt upgrade received pre-buffered bytes before the first packet".into(),
                ));
            }
            drive_session(stream, handler, broker, config, admission)
                .await
                .map_err(|error| ProximaError::Upstream(format!("mqtt session: {error}")))
        });
        Ok(Response::new(200).with_upgrade(handler_fn))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::io::Read;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use bytes::Bytes;
    use proxima_core::ProximaError;
    use proxima_primitives::pipe::method::Method;
    use proxima_primitives::pipe::request::{Request, RequestContext, Response};
    use proxima_primitives::pipe::upgrade::HijackedSocket;
    use proxima_protocols::mqtt::MqttReply;

    use crate::config::MqttServerConfig;
    use crate::pipes::{MqttPipeReply, MqttPipeRequest, into_mqtt_handle};

    use super::*;

    struct AcceptAllPipe;

    impl SendPipe for AcceptAllPipe {
        type In = MqttPipeRequest;
        type Out = MqttPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
            Ok(Response::typed(
                200,
                MqttReply::ConnAck { session_present: false, return_code: 0 },
            ))
        }
    }

    struct ScriptedSocket {
        read_data: std::io::Cursor<Vec<u8>>,
        write_data: Vec<u8>,
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
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.write_data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    fn connect_request() -> Request<Bytes> {
        Request {
            method: Method::Connect,
            path: Bytes::new(),
            query: proxima_primitives::pipe::header_list::HeaderList::new(),
            metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
            payload: Bytes::new(),
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn connect_request_answers_with_an_upgrade() {
        let pipe = MqttConnectionPipe::new(
            "mqtt",
            into_mqtt_handle(AcceptAllPipe),
            std::sync::Arc::new(MqttServerConfig::default()),
        );

        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");

        assert_eq!(response.status, 200);
        assert!(
            response.upgrade.is_some(),
            "connect response must carry an upgrade"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn upgrade_handler_drives_a_session_to_clean_completion() {
        let pipe = MqttConnectionPipe::new(
            "mqtt",
            into_mqtt_handle(AcceptAllPipe),
            std::sync::Arc::new(MqttServerConfig::default()),
        );
        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");
        let handler = response.upgrade.expect("upgrade must be present");

        let mut scripted = Vec::new();
        proxima_protocols::mqtt::encode::encode_connect(b"c1", true, 30, None, None, &mut scripted);
        proxima_protocols::mqtt::encode::encode_disconnect(&mut scripted);
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(scripted),
            write_data: Vec::new(),
        };
        let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());

        handler
            .invoke(hijacked)
            .await
            .expect("session must complete cleanly through the upgrade");
    }
}
