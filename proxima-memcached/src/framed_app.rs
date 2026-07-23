//! The memcached business-handler pipe, wired as the `App` half of
//! `proxima_listen::any::FramedAny<MemcachedCodec, MemcachedFramedApp, _, _>`
//! — the generic stateless `AnyProtocol` driver replacing this crate's
//! former bespoke `serve_connection`/`main_loop` (see git history:
//! `connection.rs`, `pipe.rs`). `build_request`/`method_for` are moved
//! here verbatim from that driver — no wire-mapping logic is rewritten,
//! only relocated next to their one remaining caller.
//!
//! [`MemcachedOutcome`] is the sentinel `FramedAny` asked for: a plain
//! reply and admission-shed reply both keep serving; `quit` and a
//! [`Violation`] close the connection afterward (`CloseSilent`/
//! `CloseWithReply`), mirroring exactly what the deleted `main_loop`'s
//! `FrameOutcome::Close`/`Advanced::ProtocolError`/`Advanced::MessageTooLarge`
//! arms used to do, now expressed through `AsFrame::as_frame`
//! (`Option`) + `AsFrame::keep_serving` (`bool`) instead of an early
//! `return` inside a bespoke loop.

use bytes::Bytes;

use proxima_core::ProximaError;
use proxima_listen::any::AsFrame;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::{Request, RequestContext};

use proxima_protocols::memcached::frame_codec::{MemcachedCodec, MemcachedFrame, MemcachedOwnedFrame, Violation};
use proxima_protocols::memcached::{MemcachedRequest, Reply, StoreMode};

use crate::pipes::{MemcachedPipeHandle, MemcachedPipeRequest};

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

/// A parsed frame's outcome — what [`proxima_listen::any::FramedAny`]'s
/// generic `drive` loop should do with it. `Reply`/`Silent` keep
/// serving; `CloseWithReply`/`CloseSilent` write (or don't) a final
/// frame and then stop, matching `quit`'s "no reply" and a protocol
/// violation's "one last reply, then close" shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemcachedOutcome {
    Reply(Reply),
    Silent,
    CloseWithReply(Reply),
    CloseSilent,
}

impl AsFrame<MemcachedCodec> for MemcachedOutcome {
    fn as_frame(&self) -> Option<MemcachedFrame<'_>> {
        match self {
            MemcachedOutcome::Reply(reply) | MemcachedOutcome::CloseWithReply(reply) => {
                Some(MemcachedFrame::Reply(reply))
            }
            MemcachedOutcome::Silent | MemcachedOutcome::CloseSilent => None,
        }
    }

    fn keep_serving(&self) -> bool {
        !matches!(
            self,
            MemcachedOutcome::CloseWithReply(_) | MemcachedOutcome::CloseSilent
        )
    }
}

/// Local wrapper around [`ProximaError`] — `AndThen`'s
/// `Second::Err: From<First::Err>` composition seam needs a `From<NeedMoreBytes>`
/// impl, and orphan rules forbid implementing a foreign trait on a
/// foreign type from this crate; a local newtype error satisfies it
/// trivially. In practice this conversion is never exercised:
/// `NeedMoreBytes::is_incomplete()` is unconditionally `true`, so
/// `proxima_protocols::codec_pipe::FrameCodecPipe` always collapses it to
/// `Ok(None)` before this App is ever called.
#[derive(Debug, thiserror::Error)]
pub enum MemcachedAppError {
    #[error("handler pipe: {0}")]
    Handler(#[from] ProximaError),
    #[error("codec reported it needed more bytes after FrameCodecPipe should have collapsed that to Ok(None)")]
    UnexpectedIncomplete,
}

impl From<proxima_protocols::memcached::frame_codec::NeedMoreBytes> for MemcachedAppError {
    fn from(_: proxima_protocols::memcached::frame_codec::NeedMoreBytes) -> Self {
        MemcachedAppError::UnexpectedIncomplete
    }
}

/// The memcached business-handler pipe as `FramedAny`'s `App`: dispatches
/// a parsed [`MemcachedRequest`] to the wrapped [`MemcachedPipeHandle`],
/// and resolves `quit`/protocol violations directly (no handler call).
/// Admission-shedding is NOT this type's concern — `FramedAny` wraps
/// every `App` in its own generic `AdmittedApp`, so a shed connection
/// never reaches [`Self::call`] at all.
#[derive(Clone)]
pub struct MemcachedFramedApp {
    handler: MemcachedPipeHandle,
}

impl MemcachedFramedApp {
    #[must_use]
    pub fn new(handler: MemcachedPipeHandle) -> Self {
        Self { handler }
    }
}

impl SendPipe for MemcachedFramedApp {
    type In = MemcachedOwnedFrame;
    type Out = MemcachedOutcome;
    type Err = MemcachedAppError;

    async fn call(&self, input: MemcachedOwnedFrame) -> Result<MemcachedOutcome, MemcachedAppError> {
        match input {
            MemcachedOwnedFrame::Violation(Violation::Protocol) => {
                tracing::error!("memcached protocol violation");
                Ok(MemcachedOutcome::CloseWithReply(Reply::Error))
            }
            MemcachedOwnedFrame::Violation(Violation::MessageTooLarge { limit }) => {
                tracing::error!(limit, "memcached message too large");
                Ok(MemcachedOutcome::CloseWithReply(Reply::ServerError(
                    format!("message exceeds {limit} byte limit").into_bytes(),
                )))
            }
            MemcachedOwnedFrame::Request(MemcachedRequest::Quit) => Ok(MemcachedOutcome::CloseSilent),
            MemcachedOwnedFrame::Request(request) => Ok(dispatch(&self.handler, request).await),
        }
    }
}

/// Calls the wrapped business handler for every command except `quit`
/// (resolved by [`MemcachedFramedApp::call`] before this is reached) —
/// mirrors the deleted `connection::dispatch_request`'s handler-call arm
/// exactly, minus the admission check `FramedAny`'s `AdmittedApp` now
/// performs generically.
async fn dispatch(handler: &MemcachedPipeHandle, request: MemcachedRequest) -> MemcachedOutcome {
    let noreply = request.is_noreply();
    let dispatched = SendPipe::call(handler.as_ref(), build_request(request)).await;
    match dispatched {
        Ok(_response) if noreply => MemcachedOutcome::Silent,
        Ok(response) => MemcachedOutcome::Reply(response.payload),
        Err(_error) if noreply => MemcachedOutcome::Silent,
        Err(error) => {
            tracing::error!(error = %error, "memcached handler error");
            MemcachedOutcome::Reply(Reply::ServerError(b"internal error".to_vec()))
        }
    }
}

/// Renders the listener-wide admission-shed reply — installed as
/// `FramedAny`'s `Shed` closure. Matches the deleted
/// `connection::dispatch_request`'s shed-reply text exactly, now with the
/// same per-request judgment that driver made: `FramedAny`'s
/// `Shed: Fn(ShedReason, &App::In) -> App::Out` hands this closure the
/// frame that got shed, so it can special-case exactly what the deleted
/// driver did — `quit` closes rather than being answered with an error
/// (it never reaches the wire either way), and a `noreply`-flagged
/// command that gets shed stays silent, matching the same command's own
/// [`dispatch`] outcome had admission let it through.
#[must_use]
pub fn shed_reply(
    reason: proxima_listen::admission::ShedReason,
    input: &MemcachedOwnedFrame,
) -> MemcachedOutcome {
    match input {
        MemcachedOwnedFrame::Request(MemcachedRequest::Quit) => MemcachedOutcome::CloseSilent,
        MemcachedOwnedFrame::Request(request) if request.is_noreply() => MemcachedOutcome::Silent,
        _ => MemcachedOutcome::Reply(Reply::ServerError(
            format!("server is shedding requests ({reason:?}); retry shortly").into_bytes(),
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;

    struct EchoHandler;

    impl SendPipe for EchoHandler {
        type In = MemcachedPipeRequest;
        type Out = crate::pipes::MemcachedPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: MemcachedPipeRequest) -> Result<Self::Out, ProximaError> {
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

    fn app() -> MemcachedFramedApp {
        MemcachedFramedApp::new(crate::pipes::into_memcached_handle(EchoHandler))
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_get_hit_dispatches_to_the_handler_and_replies() {
        let outcome = app()
            .call(MemcachedOwnedFrame::Request(MemcachedRequest::Get {
                keys: vec![b"k".to_vec()],
                gets: false,
            }))
            .await
            .expect("call");
        assert_eq!(
            outcome,
            MemcachedOutcome::Reply(Reply::Values(vec![proxima_protocols::memcached::StoredValue {
                key: b"k".to_vec(),
                flags: 0,
                data: b"stub-value".to_vec(),
                cas_unique: None,
            }]))
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn noreply_store_is_silent_even_on_a_successful_dispatch() {
        let outcome = app()
            .call(MemcachedOwnedFrame::Request(MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: b"k".to_vec(),
                flags: 0,
                exptime: 0,
                value: b"v".to_vec(),
                noreply: true,
            }))
            .await
            .expect("call");
        assert_eq!(outcome, MemcachedOutcome::Silent);
    }

    #[proxima::test(runtime = "tokio")]
    async fn quit_closes_silently() {
        let outcome = app()
            .call(MemcachedOwnedFrame::Request(MemcachedRequest::Quit))
            .await
            .expect("call");
        assert_eq!(outcome, MemcachedOutcome::CloseSilent);
        assert!(!outcome.keep_serving());
        assert!(outcome.as_frame().is_none());
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_protocol_violation_closes_with_a_bare_error_reply() {
        let outcome = app()
            .call(MemcachedOwnedFrame::Violation(Violation::Protocol))
            .await
            .expect("call");
        assert_eq!(outcome, MemcachedOutcome::CloseWithReply(Reply::Error));
        assert!(!outcome.keep_serving());
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_message_too_large_violation_closes_with_the_limit_in_the_reply() {
        let outcome = app()
            .call(MemcachedOwnedFrame::Violation(Violation::MessageTooLarge {
                limit: 1024,
            }))
            .await
            .expect("call");
        assert_eq!(
            outcome,
            MemcachedOutcome::CloseWithReply(Reply::ServerError(
                b"message exceeds 1024 byte limit".to_vec()
            ))
        );
    }

    #[test]
    fn shed_reply_renders_a_server_error_that_keeps_serving() {
        let outcome = shed_reply(proxima_listen::admission::ShedReason::Draining);
        assert!(outcome.keep_serving());
        match outcome {
            MemcachedOutcome::Reply(Reply::ServerError(message)) => {
                assert!(String::from_utf8_lossy(&message).starts_with("server is shedding requests"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
