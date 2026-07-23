//! The Kafka business-handler pipe, wired as the `App` half of
//! `proxima_listen::any::FramedAny<KafkaCodec, KafkaFramedApp, _, _>` —
//! the generic stateless `AnyProtocol` driver replacing this crate's
//! former bespoke `serve_connection`/`KafkaConnectionPipe` CONNECT/upgrade
//! indirection (see git history: `connection.rs`, `pipe.rs`).
//!
//! [`KafkaOutcome`] is the sentinel `FramedAny` asked for: `Reply` keeps
//! serving, `CloseWithReply` writes one final courtesy reply then stops
//! (a handler-pipe failure), `Close` stops with no reply at all (a
//! framing/header/body [`crate::frame_codec::Violation`] this facade has
//! no trustworthy `correlation_id` — or chooses — to answer against).
//! Mirrors exactly what the deleted `main_loop`'s
//! `FrameOutcome::Close`/`Advanced::ProtocolError`/
//! `Advanced::MessageTooLarge` arms used to do, now expressed through
//! `AsFrame::as_frame` (`Option`) + `AsFrame::keep_serving` (`bool`)
//! instead of an early `return` inside a bespoke loop.
//!
//! `ApiVersions` is answered protocol-level, same as the deleted driver:
//! it never reaches [`KafkaPipeHandle`]. Because `FramedAny`'s generic
//! `AdmittedApp` wrapper checks admission on EVERY frame uniformly —
//! `ApiVersions` and every [`crate::frame_codec::Violation`] included —
//! [`shed_reply`] (installed as `FramedAny`'s `Shed`) reproduces the
//! SAME outcome [`KafkaFramedApp::call`] would have rendered for those
//! two cases, ignoring the shed reason entirely (via the shared
//! [`resolve_violation`]/`apiversions_reply` helpers): the deleted
//! driver's own admission check ran only AFTER `ApiVersions` and every
//! parse-time violation had already short-circuited to their answer, so
//! neither was ever actually sheddable. Only a genuine dispatchable
//! request (Produce/Fetch/Metadata) gets the "busy" empty-reply
//! treatment when shed.

use proxima_listen::admission::ShedReason;
use proxima_listen::any::AsFrame;
use proxima_primitives::pipe::SendPipe;

use crate::frame_codec::{KafkaCodec, KafkaCodecError, KafkaFrame, KafkaOwnedFrame, Violation, empty_response_for};
use crate::pipes::KafkaPipeHandle;
use crate::wire::{ApiVersionsResponse, RequestBody, ResponseBody};

fn apiversions_reply(correlation_id: i32) -> KafkaOutcome {
    KafkaOutcome::Reply {
        correlation_id,
        body: ResponseBody::ApiVersions(ApiVersionsResponse::supported()),
    }
}

/// Resolves a pre-parsed [`Violation`] into its final [`KafkaOutcome`] —
/// shared by [`KafkaFramedApp::call`] and [`shed_reply`] so a violation
/// answers identically whether or not admission happens to be shedding
/// at the moment it arrives: the outcome was already decided at parse
/// time (`crate::frame_codec::KafkaCodec::own_frame`), before any
/// handler or admission concern.
fn resolve_violation(violation: &Violation) -> KafkaOutcome {
    match violation {
        Violation::Protocol => {
            tracing::error!("kafka protocol violation");
            KafkaOutcome::Close
        }
        Violation::MessageTooLarge { limit } => {
            tracing::error!(limit, "kafka message too large");
            KafkaOutcome::Close
        }
        Violation::MalformedBody => {
            tracing::error!("kafka request body could not be decoded");
            KafkaOutcome::Close
        }
        Violation::UnsupportedVersion { correlation_id, body } => {
            tracing::warn!(correlation_id, "kafka unsupported api_version");
            KafkaOutcome::Reply {
                correlation_id: *correlation_id,
                body: body.clone(),
            }
        }
    }
}

/// A parsed frame's outcome — what [`proxima_listen::any::FramedAny`]'s
/// generic `drive` loop should do with it. `Reply` keeps serving;
/// `CloseWithReply`/`Close` write (or don't) a final frame and then stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KafkaOutcome {
    Reply { correlation_id: i32, body: ResponseBody },
    CloseWithReply { correlation_id: i32, body: ResponseBody },
    Close,
}

impl AsFrame<KafkaCodec> for KafkaOutcome {
    fn as_frame(&self) -> Option<KafkaFrame<'_>> {
        match self {
            KafkaOutcome::Reply { correlation_id, body }
            | KafkaOutcome::CloseWithReply { correlation_id, body } => Some(KafkaFrame::Reply {
                correlation_id: *correlation_id,
                body,
            }),
            KafkaOutcome::Close => None,
        }
    }

    fn keep_serving(&self) -> bool {
        !matches!(self, KafkaOutcome::CloseWithReply { .. } | KafkaOutcome::Close)
    }
}

/// Local wrapper around [`KafkaCodecError`] — `AndThen`'s
/// `Second::Err: From<First::Err>` composition seam needs a
/// `From<KafkaCodecError>` impl, and orphan rules forbid implementing a
/// foreign trait on a foreign type from this crate; a local newtype error
/// satisfies it trivially. In practice this conversion is never
/// exercised: `KafkaCodecError::Incomplete` is the only variant
/// `KafkaCodec::parse_frame` ever returns, and `is_incomplete()` is
/// unconditionally `true` for it, so
/// `proxima_protocols::codec_pipe::FrameCodecPipe` always collapses it to
/// `Ok(None)` before this App is ever called; `ResponseTooLarge` is an
/// encode-side-only error `FramedAny::drive` handles directly, never
/// routed through this conversion either.
#[derive(Debug, thiserror::Error)]
pub enum KafkaAppError {
    #[error("codec: {0}")]
    Codec(#[from] KafkaCodecError),
}

/// The Kafka business-handler pipe as `FramedAny`'s `App`: dispatches a
/// parsed [`RequestBody`] to the wrapped [`KafkaPipeHandle`], and
/// resolves `ApiVersions`/violations directly (no handler call).
/// Admission-shedding is NOT this type's concern — `FramedAny` wraps
/// every `App` in its own generic `AdmittedApp`, so a shed connection
/// never reaches [`Self::call`] at all.
#[derive(Clone)]
pub struct KafkaFramedApp {
    handler: KafkaPipeHandle,
}

impl KafkaFramedApp {
    #[must_use]
    pub fn new(handler: KafkaPipeHandle) -> Self {
        Self { handler }
    }
}

impl SendPipe for KafkaFramedApp {
    type In = KafkaOwnedFrame;
    type Out = KafkaOutcome;
    type Err = KafkaAppError;

    async fn call(&self, input: KafkaOwnedFrame) -> Result<KafkaOutcome, KafkaAppError> {
        match input {
            KafkaOwnedFrame::Violation(violation) => Ok(resolve_violation(&violation)),
            KafkaOwnedFrame::Request {
                correlation_id,
                body: RequestBody::ApiVersions,
                ..
            } => Ok(apiversions_reply(correlation_id)),
            KafkaOwnedFrame::Request {
                correlation_id,
                api_key,
                body,
            } => Ok(dispatch(&self.handler, correlation_id, api_key, body).await),
        }
    }
}

/// Calls the wrapped business handler for every request except
/// `ApiVersions` (resolved by [`KafkaFramedApp::call`] before this is
/// reached) — mirrors the deleted `connection::dispatch`'s handler-call
/// arm exactly, minus the admission check `FramedAny`'s `AdmittedApp` now
/// performs generically.
async fn dispatch(handler: &KafkaPipeHandle, correlation_id: i32, api_key: i16, body: RequestBody) -> KafkaOutcome {
    match SendPipe::call(handler.as_ref(), body).await {
        Ok(response) => KafkaOutcome::Reply {
            correlation_id,
            body: response,
        },
        Err(error) => {
            tracing::error!(error = %error, api_key, "kafka handler error");
            KafkaOutcome::CloseWithReply {
                correlation_id,
                body: empty_response_for(api_key),
            }
        }
    }
}

/// Renders the listener-wide admission-shed reply — installed as
/// `FramedAny`'s `Shed` closure. See the module doc: `ApiVersions` and
/// every parse-time [`Violation`] are exempt from shedding (they answer
/// exactly as they would outside admission pressure); only a genuine
/// dispatchable request gets the "busy" empty-reply treatment.
#[must_use]
pub fn shed_reply(reason: ShedReason, input: &KafkaOwnedFrame) -> KafkaOutcome {
    match input {
        KafkaOwnedFrame::Violation(violation) => resolve_violation(violation),
        KafkaOwnedFrame::Request {
            correlation_id,
            body: RequestBody::ApiVersions,
            ..
        } => apiversions_reply(*correlation_id),
        KafkaOwnedFrame::Request {
            correlation_id,
            api_key,
            ..
        } => {
            tracing::warn!(api_key, ?reason, "kafka request shed under admission policy");
            KafkaOutcome::Reply {
                correlation_id: *correlation_id,
                body: empty_response_for(*api_key),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_core::ProximaError;

    use crate::wire;
    use crate::wire::ApiKey;

    struct EchoHandler;

    impl SendPipe for EchoHandler {
        type In = RequestBody;
        type Out = ResponseBody;
        type Err = ProximaError;

        async fn call(&self, request: RequestBody) -> Result<Self::Out, ProximaError> {
            match request {
                RequestBody::Produce(_) => Ok(ResponseBody::Produce(wire::ProduceResponse::default())),
                _ => Err(ProximaError::Upstream("unexpected api".into())),
            }
        }
    }

    fn app() -> KafkaFramedApp {
        KafkaFramedApp::new(crate::pipes::into_kafka_handle(EchoHandler))
    }

    #[proxima::test(runtime = "tokio")]
    async fn api_versions_is_answered_without_reaching_the_handler() {
        let outcome = app()
            .call(KafkaOwnedFrame::Request {
                correlation_id: 11,
                api_key: ApiKey::ApiVersions.to_i16(),
                body: RequestBody::ApiVersions,
            })
            .await
            .expect("call");
        assert_eq!(
            outcome,
            KafkaOutcome::Reply {
                correlation_id: 11,
                body: ResponseBody::ApiVersions(ApiVersionsResponse::supported()),
            }
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_produce_request_dispatches_to_the_handler_and_replies() {
        let outcome = app()
            .call(KafkaOwnedFrame::Request {
                correlation_id: 4,
                api_key: ApiKey::Produce.to_i16(),
                body: RequestBody::Produce(wire::ProduceRequest {
                    acks: 1,
                    timeout_ms: 100,
                    topics: Vec::new(),
                }),
            })
            .await
            .expect("call");
        assert_eq!(
            outcome,
            KafkaOutcome::Reply {
                correlation_id: 4,
                body: ResponseBody::Produce(wire::ProduceResponse::default()),
            }
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_handler_error_closes_with_a_courtesy_empty_reply() {
        let outcome = app()
            .call(KafkaOwnedFrame::Request {
                correlation_id: 6,
                api_key: ApiKey::Fetch.to_i16(),
                body: RequestBody::Fetch(wire::FetchRequest {
                    replica_id: -1,
                    max_wait_ms: 0,
                    min_bytes: 0,
                    topics: Vec::new(),
                }),
            })
            .await
            .expect("call");
        assert_eq!(
            outcome,
            KafkaOutcome::CloseWithReply {
                correlation_id: 6,
                body: ResponseBody::Fetch(wire::FetchResponse::default()),
            }
        );
        assert!(!outcome.keep_serving());
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_protocol_violation_closes_with_no_reply() {
        let outcome = app()
            .call(KafkaOwnedFrame::Violation(Violation::Protocol))
            .await
            .expect("call");
        assert_eq!(outcome, KafkaOutcome::Close);
        assert!(!outcome.keep_serving());
        assert!(outcome.as_frame().is_none());
    }

    #[proxima::test(runtime = "tokio")]
    async fn a_message_too_large_violation_closes_with_no_reply() {
        let outcome = app()
            .call(KafkaOwnedFrame::Violation(Violation::MessageTooLarge { limit: 1024 }))
            .await
            .expect("call");
        assert_eq!(outcome, KafkaOutcome::Close);
        assert!(!outcome.keep_serving());
    }

    #[proxima::test(runtime = "tokio")]
    async fn an_unsupported_version_violation_replies_and_keeps_serving() {
        let outcome = app()
            .call(KafkaOwnedFrame::Violation(Violation::UnsupportedVersion {
                correlation_id: 9,
                body: ResponseBody::Produce(wire::ProduceResponse::default()),
            }))
            .await
            .expect("call");
        assert_eq!(
            outcome,
            KafkaOutcome::Reply {
                correlation_id: 9,
                body: ResponseBody::Produce(wire::ProduceResponse::default()),
            }
        );
        assert!(outcome.keep_serving());
    }

    #[test]
    fn shed_reply_renders_an_empty_reply_that_keeps_serving_for_an_ordinary_request() {
        let input = KafkaOwnedFrame::Request {
            correlation_id: 2,
            api_key: ApiKey::Metadata.to_i16(),
            body: RequestBody::Metadata(wire::MetadataRequest { topics: None }),
        };
        let outcome = shed_reply(ShedReason::Draining, &input);
        assert!(outcome.keep_serving());
        assert_eq!(
            outcome,
            KafkaOutcome::Reply {
                correlation_id: 2,
                body: ResponseBody::Metadata(wire::MetadataResponse::default()),
            }
        );
    }

    #[test]
    fn shed_reply_still_answers_api_versions_normally() {
        let input = KafkaOwnedFrame::Request {
            correlation_id: 8,
            api_key: ApiKey::ApiVersions.to_i16(),
            body: RequestBody::ApiVersions,
        };
        let outcome = shed_reply(ShedReason::Draining, &input);
        assert_eq!(
            outcome,
            KafkaOutcome::Reply {
                correlation_id: 8,
                body: ResponseBody::ApiVersions(ApiVersionsResponse::supported()),
            }
        );
    }

    #[test]
    fn shed_reply_resolves_a_violation_the_same_as_a_non_shed_call() {
        let input = KafkaOwnedFrame::Violation(Violation::Protocol);
        let outcome = shed_reply(ShedReason::Draining, &input);
        assert_eq!(outcome, KafkaOutcome::Close);
    }
}
