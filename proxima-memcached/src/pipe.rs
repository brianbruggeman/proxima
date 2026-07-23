//! `MemcachedConnectionPipe` — the connection layer as a real `Pipe`.
//!
//! A freshly accepted socket arrives as a `CONNECT` request; the pipe
//! answers with `Response.upgrade` (the `proxima_primitives::pipe::upgrade`
//! raw-socket-hijack seam). The upgrade handler runs
//! [`crate::connection::serve_connection`] over the hijacked stream,
//! calling the business-handler pipe once per non-`quit` command. Mirrors
//! `proxima_redis::pipe::RedisConnectionPipe`, minus the broker field redis
//! carries for PUBLISH/SUBSCRIBE — memcached has no pub/sub fabric to
//! share across connections.
//!
//! Memcached-over-TLS is whole-connection TLS from byte 0 — there is no
//! in-band STARTTLS-style negotiation — so this pipe carries no TLS state
//! of its own; TLS composes as the generic `Listener::builder().tls(config)`
//! decorator over whatever `MemcachedAnyProtocol` resolves.

use std::sync::Arc;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncWrite};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};

use crate::config::MemcachedServerConfig;
use crate::connection::serve_connection;
use crate::error::MemcachedServeError;
use crate::pipes::MemcachedPipeHandle;

/// The connection layer as a `Pipe`. Its `call` (for a `CONNECT` request)
/// returns a `Response.upgrade` that drives `serve_connection` over the
/// hijacked stream, calling `handler` per non-`quit` command.
#[derive(Clone)]
pub struct MemcachedConnectionPipe {
    handler: MemcachedPipeHandle,
    config: Arc<MemcachedServerConfig>,
    label: String,
    /// The listener-wide request-admission handle `MemcachedAnyProtocol`
    /// installs via [`Self::with_admission`]. `None` (the default a bare
    /// `MemcachedConnectionPipe::new` gets, e.g. in this file's own unit
    /// tests) resolves to an unbounded, never-quiesced/-drained handle at
    /// `call` time — behavior-preserving for every caller that predates
    /// request-level admission.
    admission: Option<proxima_listen::admission::ConnAdmission>,
}

impl MemcachedConnectionPipe {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        handler: MemcachedPipeHandle,
        config: Arc<MemcachedServerConfig>,
    ) -> Self {
        Self {
            handler,
            config,
            label: label.into(),
            admission: None,
        }
    }

    /// Installs the listener-wide [`proxima_listen::admission::ConnAdmission`]
    /// handle `MemcachedAnyProtocol::drive` clones into a fresh
    /// `MemcachedConnectionPipe` per accepted connection.
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
    handler: MemcachedPipeHandle,
    config: Arc<MemcachedServerConfig>,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), MemcachedServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let (_shutdown_tx, shutdown_rx) = oneshot::channel();
    serve_connection(stream, handler, &config, shutdown_rx, admission).await
}

impl SendPipe for MemcachedConnectionPipe {
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
                // a raw memcached socket has no prior protocol head, so the
                // upgrade seam should never hand us pre-buffered bytes;
                // dropping them silently would corrupt the first command.
                return Err(ProximaError::Upstream(
                    "memcached upgrade received pre-buffered bytes before the first command"
                        .into(),
                ));
            }
            drive_session(stream, handler, config, admission)
                .await
                .map_err(|error| ProximaError::Upstream(format!("memcached session: {error}")))
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

    use crate::config::MemcachedServerConfig;
    use crate::pipes::{MemcachedPipeReply, MemcachedPipeRequest, into_memcached_handle};

    use super::*;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = MemcachedPipeRequest;
        type Out = MemcachedPipeReply;
        type Err = ProximaError;

        async fn call(
            &self,
            _request: MemcachedPipeRequest,
        ) -> Result<MemcachedPipeReply, ProximaError> {
            Ok(Response::typed(200, proxima_protocols::memcached::Reply::Ok))
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
        let pipe = MemcachedConnectionPipe::new(
            "memcached",
            into_memcached_handle(EchoPipe),
            std::sync::Arc::new(MemcachedServerConfig::default()),
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
    async fn upgrade_handler_drives_a_command_to_clean_completion() {
        let pipe = MemcachedConnectionPipe::new(
            "memcached",
            into_memcached_handle(EchoPipe),
            std::sync::Arc::new(MemcachedServerConfig::default()),
        );
        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");
        let handler = response.upgrade.expect("upgrade must be present");

        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(b"quit\r\n".to_vec()),
            write_data: Vec::new(),
        };
        let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());

        handler
            .invoke(hijacked)
            .await
            .expect("session must complete cleanly through the upgrade");
    }
}
