use bytes::Bytes;

// HeaderName's own enum + inherent methods are core-tier (only bytes::Bytes,
// which compiles without alloc); only the IntoHeaderBytes bridge below needs
// header_list, which is alloc-gated.
#[cfg(feature = "alloc")]
use crate::pipe::header_list::IntoHeaderBytes;

/// An HTTP header name. A name we already know is a unit variant that resolves
/// to a `'static` canonical (lowercase) byte string — zero allocation, integer
/// compare. An unknown name carries its bytes in [`HeaderName::Custom`].
///
/// `HeaderName` implements [`IntoHeaderBytes`], so `headers.insert(HeaderName::
/// ContentLength, ..)` stores `Bytes::from_static(b"content-length")` with no
/// copy — the outbound construction win — while `HeaderList` keeps its `Bytes`
/// storage and case-insensitive `&str` lookups unchanged. HTTP names are
/// case-insensitive, so `from_bytes` matches any casing and canonicalizes to
/// the lowercase static form (what `get`'s `eq_ignore_ascii_case` already finds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeaderName {
    Host,
    ContentType,
    ContentLength,
    ContentEncoding,
    TransferEncoding,
    Connection,
    Accept,
    AcceptEncoding,
    Authorization,
    UserAgent,
    CacheControl,
    Location,
    Date,
    Server,
    Cookie,
    SetCookie,
    Traceparent,
    Tracestate,
    Baggage,
    /// A non-standard header name, holding its bytes.
    Custom(Bytes),
}

impl HeaderName {
    /// The known names paired with their canonical lowercase wire bytes. The
    /// single source of truth for `from_bytes` / `as_bytes`.
    const KNOWN: &'static [(HeaderName, &'static [u8])] = &[
        (HeaderName::Host, b"host"),
        (HeaderName::ContentType, b"content-type"),
        (HeaderName::ContentLength, b"content-length"),
        (HeaderName::ContentEncoding, b"content-encoding"),
        (HeaderName::TransferEncoding, b"transfer-encoding"),
        (HeaderName::Connection, b"connection"),
        (HeaderName::Accept, b"accept"),
        (HeaderName::AcceptEncoding, b"accept-encoding"),
        (HeaderName::Authorization, b"authorization"),
        (HeaderName::UserAgent, b"user-agent"),
        (HeaderName::CacheControl, b"cache-control"),
        (HeaderName::Location, b"location"),
        (HeaderName::Date, b"date"),
        (HeaderName::Server, b"server"),
        (HeaderName::Cookie, b"cookie"),
        (HeaderName::SetCookie, b"set-cookie"),
        (HeaderName::Traceparent, b"traceparent"),
        (HeaderName::Tracestate, b"tracestate"),
        (HeaderName::Baggage, b"baggage"),
    ];

    /// Classify raw bytes case-insensitively. A known name returns a unit
    /// variant (no allocation); an unknown name copies into `Custom`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        for (name, canonical) in Self::KNOWN {
            if bytes.eq_ignore_ascii_case(canonical) {
                return name.clone();
            }
        }
        Self::Custom(Bytes::copy_from_slice(bytes))
    }

    /// Take ownership of wire `Bytes`: a known name drops the bytes and returns
    /// a unit variant; an unknown name keeps the bytes **zero-copy** in `Custom`.
    #[must_use]
    pub fn from_wire(bytes: Bytes) -> Self {
        for (name, canonical) in Self::KNOWN {
            if bytes.as_ref().eq_ignore_ascii_case(canonical) {
                return name.clone();
            }
        }
        Self::Custom(bytes)
    }

    /// The canonical lowercase wire bytes. Known names return a `'static`
    /// slice; `Custom` returns its carried bytes. No allocation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        if let Self::Custom(bytes) = self {
            return bytes.as_ref();
        }
        for (name, canonical) in Self::KNOWN {
            if name == self {
                return canonical;
            }
        }
        // unreachable: every non-Custom variant is in KNOWN.
        b""
    }

    /// The name as `Bytes` for storage/wire. Known names are
    /// `Bytes::from_static` (zero allocation); `Custom` clones its `Bytes`
    /// (a refcount bump).
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        if let Self::Custom(bytes) = self {
            return Bytes::clone(bytes);
        }
        for (name, canonical) in Self::KNOWN {
            if name == self {
                return Bytes::from_static(canonical);
            }
        }
        // unreachable: every non-Custom variant is in KNOWN.
        Bytes::new()
    }

    /// The name as `&str` (always valid UTF-8 for known names).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_bytes()).ok()
    }
}

#[cfg(feature = "alloc")]
impl IntoHeaderBytes for HeaderName {
    fn into_header_bytes(self) -> Bytes {
        self.to_bytes()
    }
}

impl From<&[u8]> for HeaderName {
    fn from(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<&str> for HeaderName {
    fn from(value: &str) -> Self {
        Self::from_bytes(value.as_bytes())
    }
}

impl From<Bytes> for HeaderName {
    fn from(bytes: Bytes) -> Self {
        Self::from_wire(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_match_case_insensitively_to_unit_variants() {
        assert_eq!(
            HeaderName::from_bytes(b"Content-Length"),
            HeaderName::ContentLength
        );
        assert_eq!(HeaderName::from("CONTENT-TYPE"), HeaderName::ContentType);
        assert_eq!(HeaderName::from_bytes(b"host"), HeaderName::Host);
    }

    #[test]
    fn known_name_to_bytes_is_static_lowercase_no_alloc() {
        // a known name yields the canonical lowercase static form.
        assert_eq!(
            HeaderName::ContentLength.to_bytes(),
            Bytes::from_static(b"content-length")
        );
        assert_eq!(HeaderName::ContentLength.as_bytes(), b"content-length");
    }

    #[test]
    fn from_wire_keeps_unknown_name_zero_copy() {
        let wire = Bytes::from_static(b"x-custom-thing");
        let name = HeaderName::from_wire(Bytes::clone(&wire));
        match name {
            HeaderName::Custom(bytes) => assert_eq!(bytes.as_ptr(), wire.as_ptr()),
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[test]
    fn into_header_bytes_uses_static_for_known() {
        let bytes = HeaderName::ContentLength.into_header_bytes();
        assert_eq!(bytes, Bytes::from_static(b"content-length"));
    }
}
