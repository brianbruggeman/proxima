use bytes::Bytes;

/// HTTP request method (RFC 9110 §9). Standard methods are unit variants — an
/// integer-discriminant compare, no allocation. A non-standard method carries
/// its wire bytes in [`Method::Other`], so inbound parsing of an unknown method
/// never copies beyond the wire buffer it already holds (use [`Method::from_wire`]).
///
/// Construction is via `From<&str>` / `From<&[u8]>` / `From<Bytes>`, so builder
/// sites that wrote `.method("GET")` keep working — a standard method parses to
/// a unit variant with zero heap, replacing the prior `Bytes::copy_from_slice`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Method {
    #[default]
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
    Connect,
    Trace,
    /// A non-standard method, holding its wire bytes.
    Other(Bytes),
}

impl Method {
    /// Parse from raw bytes. Methods are case-sensitive uppercase tokens
    /// (RFC 9110 §9.1), so the match is exact. Standard methods return a unit
    /// variant without allocating; an unknown method copies into `Other`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        match bytes {
            b"GET" => Self::Get,
            b"POST" => Self::Post,
            b"PUT" => Self::Put,
            b"PATCH" => Self::Patch,
            b"DELETE" => Self::Delete,
            b"HEAD" => Self::Head,
            b"OPTIONS" => Self::Options,
            b"CONNECT" => Self::Connect,
            b"TRACE" => Self::Trace,
            other => Self::Other(Bytes::copy_from_slice(other)),
        }
    }

    /// Take ownership of wire `Bytes`: a standard method drops the bytes and
    /// returns a unit variant; an unknown method keeps the bytes **zero-copy**
    /// in `Other`. This is the inbound-decode boundary — no allocation beyond
    /// the wire `Bytes` the caller already owns.
    #[must_use]
    pub fn from_wire(bytes: Bytes) -> Self {
        match bytes.as_ref() {
            b"GET" => Self::Get,
            b"POST" => Self::Post,
            b"PUT" => Self::Put,
            b"PATCH" => Self::Patch,
            b"DELETE" => Self::Delete,
            b"HEAD" => Self::Head,
            b"OPTIONS" => Self::Options,
            b"CONNECT" => Self::Connect,
            b"TRACE" => Self::Trace,
            _ => Self::Other(bytes),
        }
    }

    /// The method's wire bytes. Standard methods return a `'static` slice; an
    /// `Other` returns its carried bytes. No allocation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Get => b"GET",
            Self::Post => b"POST",
            Self::Put => b"PUT",
            Self::Patch => b"PATCH",
            Self::Delete => b"DELETE",
            Self::Head => b"HEAD",
            Self::Options => b"OPTIONS",
            Self::Connect => b"CONNECT",
            Self::Trace => b"TRACE",
            Self::Other(bytes) => bytes.as_ref(),
        }
    }

    /// The method as `Bytes` for wire serialization. Standard methods are
    /// `Bytes::from_static` (zero allocation); `Other` clones its `Bytes`
    /// (a refcount bump, no copy).
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        match self {
            Self::Get => Bytes::from_static(b"GET"),
            Self::Post => Bytes::from_static(b"POST"),
            Self::Put => Bytes::from_static(b"PUT"),
            Self::Patch => Bytes::from_static(b"PATCH"),
            Self::Delete => Bytes::from_static(b"DELETE"),
            Self::Head => Bytes::from_static(b"HEAD"),
            Self::Options => Bytes::from_static(b"OPTIONS"),
            Self::Connect => Bytes::from_static(b"CONNECT"),
            Self::Trace => Bytes::from_static(b"TRACE"),
            Self::Other(bytes) => Bytes::clone(bytes),
        }
    }

    /// The method as `&str` when its bytes are valid UTF-8 (always true for
    /// standard methods).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_bytes()).ok()
    }

    /// Whether this is a safe + idempotent read method (GET/HEAD).
    #[must_use]
    pub fn is_read(&self) -> bool {
        matches!(self, Self::Get | Self::Head)
    }
}

impl From<&[u8]> for Method {
    fn from(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<&str> for Method {
    fn from(value: &str) -> Self {
        Self::from_bytes(value.as_bytes())
    }
}

impl From<Bytes> for Method {
    fn from(bytes: Bytes) -> Self {
        Self::from_wire(bytes)
    }
}

impl From<&Bytes> for Method {
    fn from(bytes: &Bytes) -> Self {
        Self::from_bytes(bytes.as_ref())
    }
}

impl PartialEq<[u8]> for Method {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }
}

impl PartialEq<&[u8]> for Method {
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_bytes() == *other
    }
}

impl PartialEq<str> for Method {
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<&str> for Method {
    fn eq(&self, other: &&str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_methods_parse_to_unit_variants() {
        assert_eq!(Method::from_bytes(b"GET"), Method::Get);
        assert_eq!(Method::from(&b"POST"[..]), Method::Post);
        assert_eq!(Method::from("DELETE"), Method::Delete);
    }

    #[test]
    fn from_wire_keeps_unknown_method_zero_copy() {
        let wire = Bytes::from_static(b"PURGE");
        let method = Method::from_wire(Bytes::clone(&wire));
        // the Other arm holds the SAME allocation (ptr-equal), not a copy.
        match method {
            Method::Other(bytes) => assert_eq!(bytes.as_ptr(), wire.as_ptr()),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn from_wire_standard_drops_bytes_to_unit() {
        assert_eq!(Method::from_wire(Bytes::from_static(b"PUT")), Method::Put);
    }

    #[test]
    fn round_trips_through_wire_bytes() {
        for raw in [
            &b"GET"[..],
            b"POST",
            b"PUT",
            b"PATCH",
            b"DELETE",
            b"HEAD",
            b"OPTIONS",
            b"CONNECT",
            b"TRACE",
            b"PURGE",
        ] {
            let method = Method::from_bytes(raw);
            assert_eq!(method.as_bytes(), raw);
            assert_eq!(method.to_bytes().as_ref(), raw);
        }
    }

    #[test]
    fn standard_to_bytes_is_static_no_alloc() {
        // from_static carries no heap pointer the allocator owns; sanity that
        // the standard path never touches Other.
        assert!(matches!(Method::Get.to_bytes(), bytes if bytes.as_ref() == b"GET"));
    }
}
