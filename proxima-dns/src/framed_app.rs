//! The DNS-over-TCP business-handler pipe, wired as the `App` half of
//! `proxima_listen::any::FramedAny<DnsTcpCodec, DnsFramedApp, _, _>` — the
//! generic stateless `AnyProtocol` driver. Everything here is per-message
//! business logic; framing, admission, and the read/write loop all live one
//! layer down in `FramedAny`.
//!
//! [`DnsTcpOutcome`] is the sentinel `FramedAny` asked for: `Reply` writes
//! the pre-encoded response bytes, `Silent` writes nothing. Both keep
//! serving — DNS-over-TCP has no `quit`-shaped command, and malformed input
//! is warn-and-skip with the connection open. The two cases that DO close it
//! (an over-declared length, a reply too large for the wire) are
//! [`DnsTcpFrameError`] hard errors [`proxima_listen::any::FramedAny`]'s own
//! drive loop resolves before this `App` is ever called — which is why this
//! `App` names the codec's error type directly rather than wrapping it:
//! `FramedAny` asks only for `App::Err: From<C::Error>`, and a type is
//! trivially `From` itself.

use proxima_listen::admission::ShedReason;
use proxima_listen::any::AsFrame;
use proxima_primitives::pipe::SendPipe;

use proxima_protocols::dns::{
    DnsTcpCodec, DnsTcpFrameError, DnsTcpOwnedFrame, DnsTcpQuery, DnsTcpViolation,
};
use proxima_telemetry::warn;

use crate::pipes::{DnsAnswer, DnsPipeHandle, DnsQuery, TCP_TRANSPORT, build_request};
use crate::wire::answer_to_wire;

fn to_dns_query(query: DnsTcpQuery) -> DnsQuery {
    DnsQuery {
        id: query.id,
        recursion_desired: query.recursion_desired,
        name: query.name,
        qtype: query.qtype,
        qclass: query.qclass,
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
    type Err = DnsTcpFrameError;

    async fn call(&self, input: DnsTcpOwnedFrame) -> Result<DnsTcpOutcome, DnsTcpFrameError> {
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

        let request = build_request(TCP_TRANSPORT, query.clone());
        let outcome = SendPipe::call(&self.handler, request).await;
        let answer = match outcome {
            Ok(reply) => {
                if reply.status == 204 {
                    return Ok(DnsTcpOutcome::Silent);
                }
                reply.payload
            }
            Err(error) => {
                warn!(label = %self.label, ?error, "dns-tcp handler pipe failed; skipping");
                return Ok(DnsTcpOutcome::Silent);
            }
        };

        Ok(render_reply(&self.label, &query, &answer))
    }
}

/// Encode `answer` for `query`, folding an encode failure into
/// [`DnsTcpOutcome::Silent`]: a reply this server cannot render is not worth
/// closing an otherwise healthy connection over.
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
/// `FramedAny`'s `Shed` closure. SERVFAIL (RFC 1035 §4.1.1 RCODE 2), because
/// DNS's own wire-specific way to say "not right now" is a server-failure
/// answer, not a dropped connection: a resolver that gets an answer retries
/// or fails over, a resolver that gets silence waits out its whole timeout.
/// A shed [`DnsTcpViolation`] stays silent — there is no valid question to
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
                rdata: proxima_protocols::dns::encode::ipv4_rdata(core::net::Ipv4Addr::new(
                    93, 184, 216, 34,
                ))
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
            .call(DnsTcpOwnedFrame::Violation(
                DnsTcpViolation::NotSingleQuestion,
            ))
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
