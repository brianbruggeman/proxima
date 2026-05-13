//! W3C trace-context + baggage propagation, sans-IO.
//!
//! Proxima is an intermediary: it lifts `traceparent` / `baggage` off an
//! inbound request and stamps them onto the outbound request to the origin,
//! so a single trace spans the hop. This module is the pure header codec —
//! parse/validate on extract, `insert_if_absent` on inject. The seams that
//! own real headers (listeners on ingress, upstream pipes on egress) call
//! [`extract`] and [`inject`]; everything here is allocation-light and
//! testable without a socket.
//!
//! Moved here (from `proxima-pipe`) alongside [`crate::id`] so `proxima-pipe`
//! carries no trace-specific type: `proxima-pipe::RequestContext` stores only
//! the byte form (`Option<Arc<[u8]>>`) and calls
//! [`establish_trace_context`] (via the listener that already depends on
//! this crate) to populate it — `proxima-pipe` never needs to depend on
//! `proxima-telemetry`.

#![cfg(feature = "alloc")]

use bytes::Bytes;
use proxima_primitives::pipe::HeaderList;

pub const TRACEPARENT: &str = "traceparent";
pub const TRACESTATE: &str = "tracestate";
pub const BAGGAGE: &str = "baggage";

// W3C Baggage caps the serialized header at 8192 bytes.
const BAGGAGE_MAX_LEN: usize = 8192;

/// Trace-context + baggage lifted off the wire, in verbatim W3C header byte
/// form. `None` when the inbound request carried no such header or it failed
/// validation.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Propagation {
    pub traceparent: Option<Bytes>,
    pub baggage: Option<Bytes>,
}

impl Propagation {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.traceparent.is_none() && self.baggage.is_none()
    }
}

/// A `traceparent` worth forwarding is one that parses as a valid W3C value.
/// Forwarding a malformed traceparent breaks the trace at the next hop.
#[must_use]
pub fn is_valid_traceparent(value: &[u8]) -> bool {
    crate::id::parse_traceparent(value).is_some()
}

/// A `baggage` value safe to forward: non-empty, within the 8192-byte cap,
/// free of header-injection bytes (CR/LF/NUL), and shaped like at least one
/// `key=value` member. A forwarder must not relay junk that could smuggle a
/// second header onto the wire.
#[must_use]
pub fn is_valid_baggage(value: &[u8]) -> bool {
    if value.is_empty() || value.len() > BAGGAGE_MAX_LEN {
        return false;
    }
    if value.iter().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
        return false;
    }
    value.contains(&b'=')
}

/// Lift `traceparent` + `baggage` off inbound headers, keeping each only when
/// it validates.
#[must_use]
pub fn extract(headers: &HeaderList) -> Propagation {
    let traceparent = headers
        .get(TRACEPARENT)
        .filter(|value| is_valid_traceparent(value.as_ref()))
        .cloned();
    let baggage = headers
        .get(BAGGAGE)
        .filter(|value| is_valid_baggage(value.as_ref()))
        .cloned();
    Propagation {
        traceparent,
        baggage,
    }
}

/// Stamp `traceparent` + `baggage` onto outbound headers without clobbering a
/// value a caller already set (`insert_if_absent` semantics).
pub fn inject(propagation: &Propagation, headers: &mut HeaderList) {
    if let Some(traceparent) = &propagation.traceparent {
        headers.insert_if_absent(TRACEPARENT, traceparent.clone());
    }
    if let Some(baggage) = &propagation.baggage {
        headers.insert_if_absent(BAGGAGE, baggage.clone());
    }
}

/// Establish the byte-form trace id + baggage payload for a request's
/// `RequestContext` from inbound headers — the logic
/// `RequestContext::extract_propagation` used to run in `proxima-pipe`
/// before trace ownership moved here. Adopts the inbound `traceparent` when
/// it validates (preserving the inbound span id, restamped to the canonical
/// byte form) or originates a fresh trace + span when none is present;
/// baggage passes through verbatim when valid.
///
/// Under `no_std` (no id generation available) this falls back to verbatim
/// forward of a valid inbound `traceparent` only — already span-id-preserving.
///
/// The caller (a listener, already depending on `proxima-telemetry`) hands
/// the result straight to
/// [`RequestContext::adopt_trace_context`](proxima_primitives::pipe::request::RequestContext::adopt_trace_context).
#[must_use]
pub fn establish_trace_context(headers: &HeaderList) -> (Option<Bytes>, Option<Bytes>) {
    let lifted = extract(headers);
    #[cfg(feature = "std")]
    {
        let (trace_id, span_id, flags) = lifted
            .traceparent
            .as_deref()
            .and_then(crate::id::parse_traceparent)
            .unwrap_or_else(|| {
                (
                    crate::id::TraceId::generate(),
                    crate::id::SpanId::generate(),
                    crate::id::TraceFlags::SAMPLED,
                )
            });
        let restamped = crate::id::format_traceparent(&trace_id, &span_id, flags);
        (Some(Bytes::copy_from_slice(&restamped)), lifted.baggage)
    }
    #[cfg(not(feature = "std"))]
    {
        (lifted.traceparent, lifted.baggage)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const VALID_TP: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn extract_picks_up_valid_headers() {
        let headers = HeaderList::from_pairs([
            (TRACEPARENT, VALID_TP),
            (BAGGAGE, "userId=alice,region=us-east"),
            ("content-type", "application/json"),
        ]);
        let propagation = extract(&headers);
        assert_eq!(
            propagation.traceparent.as_deref(),
            Some(VALID_TP.as_bytes())
        );
        assert_eq!(
            propagation.baggage.as_deref(),
            Some(b"userId=alice,region=us-east".as_slice())
        );
    }

    #[test]
    fn extract_drops_malformed_traceparent() {
        let headers = HeaderList::from_pairs([(TRACEPARENT, "not-a-traceparent")]);
        assert!(extract(&headers).traceparent.is_none());
    }

    #[test]
    fn extract_drops_baggage_with_injection_bytes() {
        let headers = HeaderList::from_pairs([(BAGGAGE, "k=v\r\nx-evil: 1")]);
        assert!(extract(&headers).baggage.is_none());
    }

    #[test]
    fn extract_drops_baggage_without_member() {
        let headers = HeaderList::from_pairs([(BAGGAGE, "no-equals-sign")]);
        assert!(extract(&headers).baggage.is_none());
    }

    #[test]
    fn inject_writes_both_headers() {
        let propagation = Propagation {
            traceparent: Some(Bytes::from_static(VALID_TP.as_bytes())),
            baggage: Some(Bytes::from_static(b"k=v")),
        };
        let mut headers = HeaderList::new();
        inject(&propagation, &mut headers);
        assert_eq!(headers.get_str(TRACEPARENT), Some(VALID_TP));
        assert_eq!(headers.get_str(BAGGAGE), Some("k=v"));
    }

    #[test]
    fn inject_does_not_clobber_caller_value() {
        let propagation = Propagation {
            traceparent: Some(Bytes::from_static(VALID_TP.as_bytes())),
            baggage: None,
        };
        let mut headers = HeaderList::from_pairs([(TRACEPARENT, "caller-owned")]);
        inject(&propagation, &mut headers);
        assert_eq!(headers.get_str(TRACEPARENT), Some("caller-owned"));
    }

    #[test]
    fn extract_then_inject_round_trips_the_trace() {
        let inbound = HeaderList::from_pairs([(TRACEPARENT, VALID_TP), (BAGGAGE, "k=v")]);
        let propagation = extract(&inbound);
        let mut outbound = HeaderList::new();
        inject(&propagation, &mut outbound);
        assert_eq!(outbound.get_str(TRACEPARENT), Some(VALID_TP));
        assert_eq!(outbound.get_str(BAGGAGE), Some("k=v"));
    }

    #[test]
    fn empty_propagation_injects_nothing() {
        let mut headers = HeaderList::new();
        inject(&Propagation::default(), &mut headers);
        assert!(headers.is_empty());
        assert!(Propagation::default().is_empty());
    }

    #[test]
    fn establish_trace_context_preserves_trace_and_span_and_carries_baggage() {
        let inbound_tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let headers = HeaderList::from_pairs([
            (TRACEPARENT, inbound_tp),
            (BAGGAGE, "userId=alice"),
            ("content-type", "application/json"),
        ]);
        let (trace_id, baggage) = establish_trace_context(&headers);
        let outbound = trace_id.expect("trace context established");
        let (out_trace, out_span, _) =
            crate::id::parse_traceparent(&outbound).expect("restamped traceparent is valid");
        let (in_trace, in_span, _) =
            crate::id::parse_traceparent(inbound_tp.as_bytes()).expect("inbound is valid");
        assert_eq!(out_trace, in_trace, "trace id is preserved across the hop");
        assert_eq!(
            out_span, in_span,
            "the inbound span id is preserved, not discarded -- a span opened with \
             parent = traceparent() must record the real inbound span as its parent"
        );
        assert_eq!(baggage.as_deref(), Some(b"userId=alice".as_slice()));
    }

    #[test]
    fn establish_trace_context_originates_trace_when_absent() {
        let headers = HeaderList::from_pairs([("content-type", "application/json")]);
        let (trace_id, _) = establish_trace_context(&headers);
        let originated = trace_id.expect("trace originated");
        assert!(crate::id::parse_traceparent(&originated).is_some());
    }
}
