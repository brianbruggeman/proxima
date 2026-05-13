//! Error -> HTTP status/body rendering shared by the h1, h2, and h3
//! server drivers.
//!
//! h1 (`http1::serve`) writes these straight onto the wire. h2 and h3
//! wrap the same status + body into a `Response<Bytes>` and push it
//! through their own head/body emission paths. One mapping, reused —
//! a `filter`'s `RejectMode::Drop` rejection (`ProximaError::Forbidden`)
//! renders identically as a 403 no matter which protocol served it.

use bytes::Bytes;
use proxima_core::ProximaError;

/// HTTP status code surfaced when a pipe errors. Maps semantic error
/// variants onto the closest HTTP shape; cache-style "no data here"
/// becomes 404 rather than 502.
pub(crate) fn http_status_for(error: &ProximaError) -> u16 {
    match error {
        ProximaError::NoData
        | ProximaError::NotFound(_)
        | ProximaError::NotFoundKind(_)
        | ProximaError::ReplayMiss { .. } => 404,
        ProximaError::Body(_)
        | ProximaError::BodyKind(_)
        | ProximaError::Decode(_)
        | ProximaError::DecodeKind(_) => 400,
        ProximaError::Forbidden(_) => 403,
        ProximaError::RateLimited => 429,
        ProximaError::Timeout(_) => 504,
        ProximaError::Config(_)
        | ProximaError::ConfigKind(_)
        | ProximaError::Registry(_)
        | ProximaError::RegistryKind(_)
        | ProximaError::Record(_)
        | ProximaError::RecordKind(_) => 500,
        ProximaError::Upstream(_)
        | ProximaError::UpstreamKind(_)
        | ProximaError::Io(_)
        | ProximaError::Encode(_)
        | ProximaError::EncodeKind(_)
        | ProximaError::RetriesExhausted { .. } => 502,
    }
}

/// The response body for an errored request. `Forbidden` is a deliberate
/// refusal (e.g. a `filter` predicate's `RejectMode::Drop`) whose payload
/// IS the body, verbatim — everything else keeps the `proxima error:
/// {error}` rendering.
pub(crate) fn error_response_body(error: &ProximaError) -> Bytes {
    match error {
        ProximaError::Forbidden(payload) => Bytes::from(payload.clone()),
        other => Bytes::from(format!("proxima error: {other}")),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_maps_to_403_with_verbatim_payload() {
        let error = ProximaError::Forbidden("blocked by filter".into());
        assert_eq!(http_status_for(&error), 403);
        assert_eq!(error_response_body(&error).as_ref(), b"blocked by filter");
    }

    #[test]
    fn upstream_maps_to_502_with_wrapped_message() {
        let error = ProximaError::Upstream("boom".into());
        assert_eq!(http_status_for(&error), 502);
        assert_eq!(
            error_response_body(&error).as_ref(),
            b"proxima error: upstream error: boom"
        );
    }
}
