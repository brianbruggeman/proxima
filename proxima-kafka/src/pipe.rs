//! `KafkaConnectionPipe` — the connection layer as a real `Pipe`.
//!
//! A freshly accepted socket arrives as a `CONNECT` request; the pipe
//! answers with `Response.upgrade` (the `proxima_primitives::pipe::upgrade`
//! raw-socket-hijack seam). The upgrade handler runs
//! [`crate::connection::serve_connection`] over the hijacked stream,
//! calling the business-handler pipe once per non-`ApiVersions` request.
//! Mirrors `proxima_redis::pipe::RedisConnectionPipe`.
//!
//! Unlike redis, there is no separate broker `Arc` field here: Kafka's
//! Produce/Fetch/Metadata dispatch runs entirely through `handler` (see
//! `crate::broker`'s module doc for why the broker collapses into the
//! handler pipe itself, rather than sitting alongside it the way redis's
//! pub/sub bookkeeping does).

use std::sync::Arc;

use bytes::Bytes;
use futures::io::{AsyncRead, AsyncWrite};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};

use crate::config::KafkaServerConfig;
use crate::connection::serve_connection;
use crate::error::KafkaServeError;
use crate::pipes::KafkaPipeHandle;

/// The connection layer as a `Pipe`. Its `call` (for a `CONNECT` request)
/// returns a `Response.upgrade` that drives `serve_connection` over the
/// hijacked stream, calling `handler` per Produce/Fetch/Metadata request.
#[derive(Clone)]
pub struct KafkaConnectionPipe {
    handler: KafkaPipeHandle,
    config: Arc<KafkaServerConfig>,
    label: String,
    /// The listener-wide request-admission handle `KafkaAnyProtocol`
    /// installs via [`Self::with_admission`]. `None` (the default a bare
    /// `KafkaConnectionPipe::new` gets, e.g. in this file's own unit
    /// tests) resolves to an unbounded, never-quiesced/-drained handle at
    /// `call` time.
    admission: Option<proxima_listen::admission::ConnAdmission>,
}

impl KafkaConnectionPipe {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        handler: KafkaPipeHandle,
        config: Arc<KafkaServerConfig>,
    ) -> Self {
        Self {
            handler,
            config,
            label: label.into(),
            admission: None,
        }
    }

    /// Installs the listener-wide [`proxima_listen::admission::ConnAdmission`]
    /// handle `KafkaAnyProtocol::drive` clones into a fresh
    /// `KafkaConnectionPipe` per accepted connection.
    #[must_use]
    pub fn with_admission(mut self, admission: proxima_listen::admission::ConnAdmission) -> Self {
        self.admission = Some(admission);
        self
    }

    /// This pipe's label, set at construction.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

async fn drive_session<S>(
    stream: S,
    handler: KafkaPipeHandle,
    config: Arc<KafkaServerConfig>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), KafkaServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    serve_connection(stream, handler, &config, admission).await
}

impl SendPipe for KafkaConnectionPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let handler = self.handler.clone();
        let config = Arc::clone(&self.config);
        let admission = self
            .admission
            .clone()
            .unwrap_or_else(proxima_listen::admission::ConnAdmission::unbounded);
        let handler_fn = UpgradeHandler::new(move |hijacked: HijackedSocket| async move {
            let HijackedSocket { stream, leftover } = hijacked;
            if !leftover.is_empty() {
                // a raw kafka socket has no prior protocol head, so the
                // upgrade seam should never hand us pre-buffered bytes;
                // dropping them silently would corrupt the first request.
                return Err(ProximaError::Upstream(
                    "kafka upgrade received pre-buffered bytes before the first request".into(),
                ));
            }
            drive_session(stream, handler, config, admission)
                .await
                .map_err(|error| ProximaError::Upstream(format!("kafka session: {error}")))
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

    use crate::config::KafkaServerConfig;
    use crate::pipes::{KafkaPipeReply, KafkaPipeRequest, into_kafka_handle};
    use crate::wire::{ApiVersionsResponse, ResponseBody};

    use super::*;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = KafkaPipeRequest;
        type Out = KafkaPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: KafkaPipeRequest) -> Result<KafkaPipeReply, ProximaError> {
            Ok(Response::typed(
                200,
                ResponseBody::ApiVersions(ApiVersionsResponse::supported()),
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
        let pipe = KafkaConnectionPipe::new(
            "kafka",
            into_kafka_handle(EchoPipe),
            std::sync::Arc::new(KafkaServerConfig::default()),
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
    async fn upgrade_handler_drives_an_apiversions_request_to_clean_completion() {
        let pipe = KafkaConnectionPipe::new(
            "kafka",
            into_kafka_handle(EchoPipe),
            std::sync::Arc::new(KafkaServerConfig::default()),
        );
        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");
        let handler = response.upgrade.expect("upgrade must be present");

        let mut scripted = Vec::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&18_i16.to_be_bytes()); // ApiVersions
        payload.extend_from_slice(&0_i16.to_be_bytes());
        payload.extend_from_slice(&1_i32.to_be_bytes());
        payload.extend_from_slice(&(-1_i16).to_be_bytes());
        scripted.extend_from_slice(&(payload.len() as i32).to_be_bytes());
        scripted.extend_from_slice(&payload);
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(scripted),
            write_data: Vec::new(),
        };
        let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());

        // the scripted socket returns EOF once the one scripted request is
        // consumed and replied to; `serve_connection`'s next read sees 0
        // bytes and returns `Ok(())` — a clean close, matching real
        // socket-close semantics (mirrors redis's own `QUIT`-terminated
        // upgrade test).
        handler
            .invoke(hijacked)
            .await
            .expect("session must complete cleanly through the upgrade");
    }
}
