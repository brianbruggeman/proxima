//! `AmqpConnectionPipe` — the connection layer as a real `Pipe`.
//!
//! A freshly accepted socket arrives as a `CONNECT` request; the pipe
//! answers with `Response.upgrade` (the `proxima_primitives::pipe::upgrade`
//! raw-socket-hijack seam). The upgrade handler runs
//! [`crate::connection::serve_connection`] over the hijacked stream,
//! calling the business `basic.publish` handler pipe once per reassembled
//! message. Mirrors `proxima_redis::pipe::RedisConnectionPipe`.
//!
//! AMQP-over-TLS (AMQPS) is whole-connection TLS from byte 0 — same as
//! redis's RESP-over-TLS — so this pipe carries no TLS state of its own;
//! TLS composes as the generic `Listener::builder().tls(config)` decorator.

use std::sync::Arc;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncWrite};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};

use crate::broker::AmqpBroker;
use crate::config::AmqpServerConfig;
use crate::connection::serve_connection;
use crate::error::AmqpServeError;
use crate::pipes::AmqpPipeHandle;

/// The connection layer as a `Pipe`. Its `call` (for a `CONNECT` request)
/// returns a `Response.upgrade` that drives `serve_connection` over the
/// hijacked stream, calling `handler` per reassembled `basic.publish`. One
/// `AmqpConnectionPipe` owns one [`AmqpBroker`], shared by every connection
/// it upgrades — that shared instance IS the exchange -> queue routing
/// fabric.
#[derive(Clone)]
pub struct AmqpConnectionPipe {
    handler: AmqpPipeHandle,
    broker: Arc<AmqpBroker>,
    config: Arc<AmqpServerConfig>,
    label: String,
    /// The listener-wide request-admission handle `AmqpAnyProtocol`
    /// installs via [`Self::with_admission`]. `None` (the default a bare
    /// `AmqpConnectionPipe::new` gets, e.g. in this file's own unit tests)
    /// resolves to an unbounded, never-quiesced/-drained handle at `call`
    /// time — behavior-preserving for every caller that predates
    /// request-level admission.
    admission: Option<proxima_listen::admission::ConnAdmission>,
}

impl AmqpConnectionPipe {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        handler: AmqpPipeHandle,
        config: Arc<AmqpServerConfig>,
    ) -> Self {
        Self {
            handler,
            broker: Arc::new(AmqpBroker::new()),
            config,
            label: label.into(),
            admission: None,
        }
    }

    /// Installs the listener-wide [`proxima_listen::admission::ConnAdmission`]
    /// handle `AmqpAnyProtocol::drive` clones into a fresh
    /// `AmqpConnectionPipe` per accepted connection.
    #[must_use]
    pub fn with_admission(mut self, admission: proxima_listen::admission::ConnAdmission) -> Self {
        self.admission = Some(admission);
        self
    }

    /// Overrides the broker `new` constructs fresh. `AmqpAnyProtocol`
    /// builds a FRESH `AmqpConnectionPipe` per accepted connection (to
    /// carry that connection's own `ConnAdmission` clone) but must NOT
    /// give each one its own broker — exchange/queue routing only works
    /// across connections when they all share the SAME broker instance,
    /// built once at `AmqpAnyProtocol::new` and installed here on every
    /// per-connection pipe.
    #[must_use]
    pub fn with_broker(mut self, broker: Arc<AmqpBroker>) -> Self {
        self.broker = broker;
        self
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn broker(&self) -> &Arc<AmqpBroker> {
        &self.broker
    }
}

async fn drive_session<S>(
    stream: S,
    handler: AmqpPipeHandle,
    broker: Arc<AmqpBroker>,
    config: Arc<AmqpServerConfig>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), AmqpServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let (_shutdown_tx, shutdown_rx) = oneshot::channel();
    serve_connection(stream, handler, broker, &config, shutdown_rx, admission).await
}

impl SendPipe for AmqpConnectionPipe {
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
                // a raw AMQP socket has no prior protocol head, so the
                // upgrade seam should never hand us pre-buffered bytes;
                // dropping them silently would corrupt the protocol header.
                return Err(ProximaError::Upstream(
                    "amqp upgrade received pre-buffered bytes before the protocol header".into(),
                ));
            }
            drive_session(stream, handler, broker, config, admission)
                .await
                .map_err(|error| ProximaError::Upstream(format!("amqp session: {error}")))
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

    use crate::config::AmqpServerConfig;
    use crate::pipes::{AmqpPipeReply, AmqpPipeRequest, into_amqp_handle};

    use super::*;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = AmqpPipeRequest;
        type Out = AmqpPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: AmqpPipeRequest) -> Result<AmqpPipeReply, ProximaError> {
            Ok(Response::typed(200, ()))
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
        let pipe = AmqpConnectionPipe::new(
            "amqp",
            into_amqp_handle(EchoPipe),
            std::sync::Arc::new(AmqpServerConfig::default()),
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
    async fn upgrade_handler_drives_the_protocol_header_and_a_clean_close() {
        let pipe = AmqpConnectionPipe::new(
            "amqp",
            into_amqp_handle(EchoPipe),
            std::sync::Arc::new(AmqpServerConfig::default()),
        );
        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");
        let handler = response.upgrade.expect("upgrade must be present");

        // a client that sends only the protocol header then closes: the
        // server writes connection.start and keeps waiting — the socket
        // read then yields 0 bytes (EOF), ending the session cleanly.
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(crate::fsm::PROTOCOL_HEADER.to_vec()),
            write_data: Vec::new(),
        };
        let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());

        handler
            .invoke(hijacked)
            .await
            .expect("session must complete cleanly through the upgrade");
    }
}
