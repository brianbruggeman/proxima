//! QPACK static table per [RFC 9204 Appendix A].
//!
//! Const array of 99 `(name, value)` byte pairs. Look up by index
//! (decoder) or by (name, value) match (encoder hint).
//!
//! [RFC 9204 Appendix A]: https://www.rfc-editor.org/rfc/rfc9204#appendix-A
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). Pure const data; lookup is
//! O(1) by index + O(N=99) by linear scan for the encoder hint. The
//! linear scan is bounded by the table size (99) — no asymptotic
//! concern.

/// One static-table entry. Both name + value are byte slices since
/// HTTP header names + values may contain non-UTF-8 bytes.
pub struct StaticEntry {
    pub name: &'static [u8],
    pub value: &'static [u8],
}

/// RFC 9204 Appendix A static table — indices 0..=98. Total 99 entries.
pub const STATIC_TABLE: &[StaticEntry] = &[
    StaticEntry {
        name: b":authority",
        value: b"",
    },
    StaticEntry {
        name: b":path",
        value: b"/",
    },
    StaticEntry {
        name: b"age",
        value: b"0",
    },
    StaticEntry {
        name: b"content-disposition",
        value: b"",
    },
    StaticEntry {
        name: b"content-length",
        value: b"0",
    },
    StaticEntry {
        name: b"cookie",
        value: b"",
    },
    StaticEntry {
        name: b"date",
        value: b"",
    },
    StaticEntry {
        name: b"etag",
        value: b"",
    },
    StaticEntry {
        name: b"if-modified-since",
        value: b"",
    },
    StaticEntry {
        name: b"if-none-match",
        value: b"",
    },
    StaticEntry {
        name: b"last-modified",
        value: b"",
    },
    StaticEntry {
        name: b"link",
        value: b"",
    },
    StaticEntry {
        name: b"location",
        value: b"",
    },
    StaticEntry {
        name: b"referer",
        value: b"",
    },
    StaticEntry {
        name: b"set-cookie",
        value: b"",
    },
    StaticEntry {
        name: b":method",
        value: b"CONNECT",
    },
    StaticEntry {
        name: b":method",
        value: b"DELETE",
    },
    StaticEntry {
        name: b":method",
        value: b"GET",
    },
    StaticEntry {
        name: b":method",
        value: b"HEAD",
    },
    StaticEntry {
        name: b":method",
        value: b"OPTIONS",
    },
    StaticEntry {
        name: b":method",
        value: b"POST",
    },
    StaticEntry {
        name: b":method",
        value: b"PUT",
    },
    StaticEntry {
        name: b":scheme",
        value: b"http",
    },
    StaticEntry {
        name: b":scheme",
        value: b"https",
    },
    StaticEntry {
        name: b":status",
        value: b"103",
    },
    StaticEntry {
        name: b":status",
        value: b"200",
    },
    StaticEntry {
        name: b":status",
        value: b"304",
    },
    StaticEntry {
        name: b":status",
        value: b"404",
    },
    StaticEntry {
        name: b":status",
        value: b"503",
    },
    StaticEntry {
        name: b"accept",
        value: b"*/*",
    },
    StaticEntry {
        name: b"accept",
        value: b"application/dns-message",
    },
    StaticEntry {
        name: b"accept-encoding",
        value: b"gzip, deflate, br",
    },
    StaticEntry {
        name: b"accept-ranges",
        value: b"bytes",
    },
    StaticEntry {
        name: b"access-control-allow-headers",
        value: b"cache-control",
    },
    StaticEntry {
        name: b"access-control-allow-headers",
        value: b"content-type",
    },
    StaticEntry {
        name: b"access-control-allow-origin",
        value: b"*",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"max-age=0",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"max-age=2592000",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"max-age=604800",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"no-cache",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"no-store",
    },
    StaticEntry {
        name: b"cache-control",
        value: b"public, max-age=31536000",
    },
    StaticEntry {
        name: b"content-encoding",
        value: b"br",
    },
    StaticEntry {
        name: b"content-encoding",
        value: b"gzip",
    },
    StaticEntry {
        name: b"content-type",
        value: b"application/dns-message",
    },
    StaticEntry {
        name: b"content-type",
        value: b"application/javascript",
    },
    StaticEntry {
        name: b"content-type",
        value: b"application/json",
    },
    StaticEntry {
        name: b"content-type",
        value: b"application/x-www-form-urlencoded",
    },
    StaticEntry {
        name: b"content-type",
        value: b"image/gif",
    },
    StaticEntry {
        name: b"content-type",
        value: b"image/jpeg",
    },
    StaticEntry {
        name: b"content-type",
        value: b"image/png",
    },
    StaticEntry {
        name: b"content-type",
        value: b"text/css",
    },
    StaticEntry {
        name: b"content-type",
        value: b"text/html; charset=utf-8",
    },
    StaticEntry {
        name: b"content-type",
        value: b"text/plain",
    },
    StaticEntry {
        name: b"content-type",
        value: b"text/plain;charset=utf-8",
    },
    StaticEntry {
        name: b"range",
        value: b"bytes=0-",
    },
    StaticEntry {
        name: b"strict-transport-security",
        value: b"max-age=31536000",
    },
    StaticEntry {
        name: b"strict-transport-security",
        value: b"max-age=31536000; includesubdomains",
    },
    StaticEntry {
        name: b"strict-transport-security",
        value: b"max-age=31536000; includesubdomains; preload",
    },
    StaticEntry {
        name: b"vary",
        value: b"accept-encoding",
    },
    StaticEntry {
        name: b"vary",
        value: b"origin",
    },
    StaticEntry {
        name: b"x-content-type-options",
        value: b"nosniff",
    },
    StaticEntry {
        name: b"x-xss-protection",
        value: b"1; mode=block",
    },
    StaticEntry {
        name: b":status",
        value: b"100",
    },
    StaticEntry {
        name: b":status",
        value: b"204",
    },
    StaticEntry {
        name: b":status",
        value: b"206",
    },
    StaticEntry {
        name: b":status",
        value: b"302",
    },
    StaticEntry {
        name: b":status",
        value: b"400",
    },
    StaticEntry {
        name: b":status",
        value: b"403",
    },
    StaticEntry {
        name: b":status",
        value: b"421",
    },
    StaticEntry {
        name: b":status",
        value: b"425",
    },
    StaticEntry {
        name: b":status",
        value: b"500",
    },
    StaticEntry {
        name: b"accept-language",
        value: b"",
    },
    StaticEntry {
        name: b"access-control-allow-credentials",
        value: b"FALSE",
    },
    StaticEntry {
        name: b"access-control-allow-credentials",
        value: b"TRUE",
    },
    StaticEntry {
        name: b"access-control-allow-headers",
        value: b"*",
    },
    StaticEntry {
        name: b"access-control-allow-methods",
        value: b"get",
    },
    StaticEntry {
        name: b"access-control-allow-methods",
        value: b"get, post, options",
    },
    StaticEntry {
        name: b"access-control-allow-methods",
        value: b"options",
    },
    StaticEntry {
        name: b"access-control-expose-headers",
        value: b"content-length",
    },
    StaticEntry {
        name: b"access-control-request-headers",
        value: b"content-type",
    },
    StaticEntry {
        name: b"access-control-request-method",
        value: b"get",
    },
    StaticEntry {
        name: b"access-control-request-method",
        value: b"post",
    },
    StaticEntry {
        name: b"alt-svc",
        value: b"clear",
    },
    StaticEntry {
        name: b"authorization",
        value: b"",
    },
    StaticEntry {
        name: b"content-security-policy",
        value: b"script-src 'none'; object-src 'none'; base-uri 'none'",
    },
    StaticEntry {
        name: b"early-data",
        value: b"1",
    },
    StaticEntry {
        name: b"expect-ct",
        value: b"",
    },
    StaticEntry {
        name: b"forwarded",
        value: b"",
    },
    StaticEntry {
        name: b"if-range",
        value: b"",
    },
    StaticEntry {
        name: b"origin",
        value: b"",
    },
    StaticEntry {
        name: b"purpose",
        value: b"prefetch",
    },
    StaticEntry {
        name: b"server",
        value: b"",
    },
    StaticEntry {
        name: b"timing-allow-origin",
        value: b"*",
    },
    StaticEntry {
        name: b"upgrade-insecure-requests",
        value: b"1",
    },
    StaticEntry {
        name: b"user-agent",
        value: b"",
    },
    StaticEntry {
        name: b"x-forwarded-for",
        value: b"",
    },
    StaticEntry {
        name: b"x-frame-options",
        value: b"deny",
    },
    StaticEntry {
        name: b"x-frame-options",
        value: b"sameorigin",
    },
];

/// Look up entry by index (0..=98). Returns `None` past the table end.
#[must_use]
pub fn get(index: usize) -> Option<&'static StaticEntry> {
    STATIC_TABLE.get(index)
}

/// Linear-scan for an entry exactly matching `(name, value)`. Returns
/// the first matching index, or `None`.
#[must_use]
pub fn find_exact(name: &[u8], value: &[u8]) -> Option<usize> {
    for (index, entry) in STATIC_TABLE.iter().enumerate() {
        if entry.name == name && entry.value == value {
            return Some(index);
        }
    }
    None
}

/// Linear-scan for the first entry matching `name` (any value).
/// Returns the first matching index. Used by the encoder when there's
/// a name-but-not-value hit (still saves the name bytes).
#[must_use]
pub fn find_name(name: &[u8]) -> Option<usize> {
    for (index, entry) in STATIC_TABLE.iter().enumerate() {
        if entry.name == name {
            return Some(index);
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn table_has_exactly_99_entries() {
        assert_eq!(STATIC_TABLE.len(), 99);
    }

    #[test]
    fn get_returns_known_entries() {
        let entry = get(0).expect("entry 0");
        assert_eq!(entry.name, b":authority");
        assert_eq!(entry.value, b"");

        let entry = get(17).expect("entry 17");
        assert_eq!(entry.name, b":method");
        assert_eq!(entry.value, b"GET");

        let entry = get(25).expect("entry 25");
        assert_eq!(entry.name, b":status");
        assert_eq!(entry.value, b"200");
    }

    #[test]
    fn get_returns_none_past_end() {
        assert!(get(99).is_none());
        assert!(get(1000).is_none());
    }

    #[test]
    fn find_exact_returns_first_matching_index() {
        assert_eq!(find_exact(b":method", b"GET"), Some(17));
        assert_eq!(find_exact(b":scheme", b"https"), Some(23));
        assert_eq!(find_exact(b":status", b"200"), Some(25));
        assert_eq!(find_exact(b":status", b"100"), Some(63));
    }

    #[test]
    fn find_exact_returns_none_for_unknown() {
        assert_eq!(find_exact(b":method", b"BREW"), None);
    }

    #[test]
    fn find_name_returns_first_match_regardless_of_value() {
        // :status first appears at 24 (":status: 103").
        assert_eq!(find_name(b":status"), Some(24));
        // user-agent has empty value at index 95.
        assert_eq!(find_name(b"user-agent"), Some(95));
    }
}
