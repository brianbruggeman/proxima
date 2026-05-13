use bytes::Bytes;

/// Vendor-defined tracestate blob (W3C tracestate header value).
///
/// `None` = absent/empty, which covers the common case with zero allocation.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct TraceState(pub Option<Bytes>);

impl TraceState {
    pub const fn empty() -> Self {
        Self(None)
    }

    pub fn from_bytes(bytes: Bytes) -> Self {
        if bytes.is_empty() {
            Self(None)
        } else {
            Self(Some(bytes))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }
}
