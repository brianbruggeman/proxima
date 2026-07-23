//! The DNS-over-TCP business-handler pipe, wired as the `App` half of
//! `proxima_listen::any::FramedAny<DnsTcpCodec, DnsFramedApp, _, _>` — the
//! generic stateless `AnyProtocol` driver replacing this crate's former
//! bespoke `serve_tcp_connection`/`handle_one_message` (see git history:
//! `any_protocol.rs`). `to_dns_query`/`build_request` are moved here
//! verbatim from that driver — no wire-mapping logic is rewritten, only
//! relocated next to their one remaining caller.
//!
//! [`DnsTcpOutcome`] is the sentinel `FramedAny` asked for: `Reply` writes
//! the pre-encoded response bytes, `Silent` writes nothing. Both keep
//! serving — DNS-over-TCP has no `quit`-shaped command, and the deleted
//! driver's own "malformed input: warn and skip, connection stays open"
//! contract ([`crate::any_protocol`]'s old `handle_one_message`) had
//! nothing that ever closed a connection from inside the per-message
//! loop; the two cases that DID close (an over-declared length, a reply
//! too large for the wire) are now [`proxima_protocols::dns::DnsTcpFrameError`]
//! hard errors, resolved one layer down in
//! [`proxima_listen::any::FramedAny`]'s own drive loop before this `App`
//! is ever called.

use proxima_listen::admission::ShedReason;
use proxima_listen::any::AsFrame;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::{Request, RequestContext};

use proxima_protocols::dns::{DnsTcpCodec, DnsTcpFrameError, DnsTcpOwnedFrame, DnsTcpQuery, DnsTcpViolation};
use proxima_telemetry::warn;

use crate::pipes::{DnsAnswer, DnsPipeHandle, DnsQuery};
use crate::wire::answer_to_wire;

const METHOD_LABEL: &[u8] = b"DNS-TCP";

fn to_dns_query(query: DnsTcpQuery) -> DnsQuery {
    DnsQuery {
        id: query.id,
        recursion_desired: query.recursion_desired,
        name: query.name,
        qtype: query.qtype,
        qclass: query.qclass,
    }
}

fn build_request(query: DnsQuery) -> crate::pipes::DnsPipeRequest {
    Request {
        method: Method::from_wire(bytes::Bytes::from_static(METHOD_LABEL)),
        path: bytes::Bytes::from_static(b"/"),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: query,
        stream: None,
        context: RequestContext::default(),
    }
}

/// A framed message's outcome — what [`proxima_listen::any::FramedAny`]'s
/// generic `drive` loop should do with it. Both variants keep serving
/// (see the module doc); `Silent` is the "nothing to send back" case
/// (malformed input, a non-single-question message, a handler failure,
/// or an encode failure) — logged at the point it's decided, then the
/// connection carries on to the next frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnsTcpOutcome {
    Reply(Vec<u8>),
    Silent,
}

impl AsFrame<DnsTcpCodec> for DnsTcpOutcome {
    fn as_frame(&self) -> Option<&[u8]> {
        match self {
            DnsTcpOutcome::Reply(bytes) => Some(bytes.as_slice()),
            DnsTcpOutcome::Silent => None,
        }
    }
}

/// Local wrapper around [`DnsTcpFrameError`] — `AndThen`'s
/// `Second::Err: From<First::Err>` composition seam needs this `From` impl.
/// Unlike memcached's equivalent (never exercised, since its codec's only
/// error is unconditionally incomplete), this conversion IS reachable: a
/// [`DnsTcpFrameError::MessageTooLarge`] is a real hard error the codec
/// raises deliberately (see `proxima_protocols::dns::frame_codec`'s module
/// doc) — surfacing it here is what makes [`proxima_listen::any::FramedAny`]
/// close the connection for it, matching the deleted driver's own
/// close-immediately behavior for an over-declared length.
#[derive(Debug, thiserror::Error)]
pub enum DnsFramedAppError {
    #[error("dns-tcp framing: {0}")]
    Framing(#[from] DnsTcpFrameError),
}

/// The DNS-over-TCP business-handler pipe as `FramedAny`'s `App`:
/// dispatches a parsed [`DnsTcpQuery`] to the wrapped [`DnsPipeHandle`],
/// and resolves a [`DnsTcpViolation`] directly (no handler call).
/// Admission-shedding is NOT this type's concern — `FramedAny` wraps
/// every `App` in its own generic `AdmittedApp`, so a shed connection
/// never reaches [`Self::call`] at all.
#[derive(Clone)]
pub struct DnsFramedApp {
    handler: DnsPipeHandle,
    label: String,
}

impl DnsFramedApp {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: DnsPipeHandle) -> Self {
        Self {
            handler,
            label: label.into(),
        }
    }
}

impl SendPipe for DnsFramedApp {
    type In = DnsTcpOwnedFrame;
    type Out = DnsTcpOutcome;
    type Err = DnsFramedAppError;

    async fn call(&self, input: DnsTcpOwnedFrame) -> Result<DnsTcpOutcome, DnsFramedAppError> {
        let query = match input {
            DnsTcpOwnedFrame::Violation(DnsTcpViolation::Malformed) => {
                warn!(label = %self.label, "dns-tcp message failed to parse; skipping");
                return Ok(DnsTcpOutcome::Silent);
            }
            DnsTcpOwnedFrame::Violation(DnsTcpViolation::NotSingleQuestion) => {
                warn!(label = %self.label, "dns-tcp message is not exactly one question; skipping");
                return Ok(DnsTcpOutcome::Silent);
            }
            DnsTcpOwnedFrame::Query(query) => to_dns_query(query),
        };

        let outcome = SendPipe::call(&self.handler, build_request(query.clone())).await;
        let answer = match outcome {
            Ok(reply) => reply.payload,
            Err(error) => {
                warn!(label = %self.label, ?error, "dns-tcp handler pipe failed; skipping");
                return Ok(DnsTcpOutcome::Silent);
            }
        };

        Ok(render_reply(&self.label, &query, &answer))
    }
}

/// Encode `answer` for `query`, folding an encode failure into
/// [`DnsTcpOutcome::Silent`] — mirrors the deleted `handle_one_message`'s
/// own "warn and skip" contract for [`crate::wire::answer_to_wire`]
/// failures.
fn render_reply(label: &str, query: &DnsQuery, answer: &DnsAnswer) -> DnsTcpOutcome {
    let mut out = Vec::new();
    match answer_to_wire(query, answer, &mut out) {
        Ok(()) => DnsTcpOutcome::Reply(out),
        Err(error) => {
            warn!(label = %label, ?error, "dns-tcp answer failed to encode; skipping");
            DnsTcpOutcome::Silent
        }
    }
}

/// Renders the listener-wide admission-shed reply — installed as
/// `FramedAny`'s `Shed` closure. Matches the deleted
/// `handle_one_message`'s shed-reply behavior exactly: SERVFAIL (RFC 1035
/// §4.1.1 RCODE 2), since DNS's own wire-specific rejection is a
/// server-failure answer, not a dropped connection. A shed
/// [`DnsTcpViolation`] stays silent — there is no valid question to
/// answer either way.
#[must_use]
pub fn shed_reply(reason: ShedReason, input: &DnsTcpOwnedFrame) -> DnsTcpOutcome {
    match input {
        DnsTcpOwnedFrame::Violation(_) => DnsTcpOutcome::Silent,
        DnsTcpOwnedFrame::Query(query) => {
            warn!(?reason, "dns-tcp request shed; replying servfail");
            let query = to_dns_query(query.clone());
            let answer = DnsAnswer {
                rcode: 2,
                authoritative: false,
                recursion_available: true,
                records: Vec::new(),
            };
            let mut out = Vec::new();
            match answer_to_wire(&query, &answer, &mut out) {
                Ok(()) => DnsTcpOutcome::Reply(out),
                Err(_error) => DnsTcpOutcome::Silent,
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_core::ProximaError;
    use proxima_primitives::pipe::request::Response;

    struct EchoHandler;

    impl SendPipe for EchoHandler {
        type In = crate::pipes::DnsPipeRequest;
        type Out = crate::pipes::DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: Self::In) -> Result<Self::Out, ProximaError> {
            let record = crate::pipes::DnsAnswerRecord {
                name: request.payload.name.clone(),
                rtype: 1,
                rclass: 1,
                ttl: 60,
                rdata: proxima_protocols::dns::encode::ipv4_rdata(core::net::Ipv4Addr::new(93, 184, 216, 34))
                    .to_vec(),
            };
            Ok(Response::typed(200, DnsAnswer::ok(vec![record])))
        }
    }

    fn app() -> DnsFramedApp {
        DnsFramedApp::new("dns-tcp-test", crate::pipes::into_dns_handle(EchoHandler))
    }

    fn query(id: u16) -> DnsTcpQuery {
        DnsTcpQuery {
            id,
            recursion_desired: true,
            name: "example.com.".to_string(),
            qtype: 1,
            qclass: 1,
        }
    }

    #[proxima::test]
    async fn a_query_dispatches_to_the_handler_and_replies() {
        let outcome = app()
            .call(DnsTcpOwnedFrame::Query(query(1234)))
            .await
            .expect("call");
        match outcome {
            DnsTcpOutcome::Reply(bytes) => {
                let message = proxima_protocols::dns::parse_message(&bytes).unwrap();
                assert_eq!(message.header.id, 1234);
                assert!(message.header.flags.is_response());
                assert_eq!(message.header.ancount, 1);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[proxima::test]
    async fn a_malformed_violation_is_silent_and_keeps_serving() {
        let outcome = app()
            .call(DnsTcpOwnedFrame::Violation(DnsTcpViolation::Malformed))
            .await
            .expect("call");
        assert_eq!(outcome, DnsTcpOutcome::Silent);
        assert!(outcome.keep_serving());
        assert!(outcome.as_frame().is_none());
    }

    #[proxima::test]
    async fn a_not_single_question_violation_is_silent_and_keeps_serving() {
        let outcome = app()
            .call(DnsTcpOwnedFrame::Violation(DnsTcpViolation::NotSingleQuestion))
            .await
            .expect("call");
        assert_eq!(outcome, DnsTcpOutcome::Silent);
        assert!(outcome.keep_serving());
    }

    #[test]
    fn shed_reply_renders_a_servfail_that_keeps_serving_for_a_query() {
        let input = DnsTcpOwnedFrame::Query(query(42));
        let outcome = shed_reply(ShedReason::Draining, &input);
        assert!(outcome.keep_serving());
        match outcome {
            DnsTcpOutcome::Reply(bytes) => {
                let message = proxima_protocols::dns::parse_message(&bytes).unwrap();
                assert_eq!(message.header.id, 42);
                assert_eq!(message.header.flags.rcode(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn shed_reply_stays_silent_for_a_shed_violation() {
        let input = DnsTcpOwnedFrame::Violation(DnsTcpViolation::Malformed);
        let outcome = shed_reply(ShedReason::Draining, &input);
        assert_eq!(outcome, DnsTcpOutcome::Silent);
        assert!(outcome.keep_serving());
        assert!(outcome.as_frame().is_none());
    }
}
