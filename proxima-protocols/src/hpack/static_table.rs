//! HPACK static table (RFC 7541 Appendix A).
//!
//! 61 predefined `(name, value)` entries that every HPACK
//! encoder/decoder agrees on out-of-band. Indices 1..=61.
//! The decoder uses [`entry`] to map an integer back to the pair;
//! the encoder uses [`lookup`] to map a `(name, value)` pair to
//! its index (if any), reporting whether the value also matched.
//!
//! Algorithm: single `match` on the lowercased name byte slice. The
//! compiler emits a length-dispatched comparison tree (effectively
//! a jump table on length, then a string compare per arm). Same
//! shape h2 uses but without the `HeaderName` indirection. Average
//! cost ≈ one length-dispatched compare + an `eq` against the
//! candidate name; ~1-3 ns on Apple M-class.
//!
//! ### Considered and rejected: PHF
//!
//! A `phf::Map` variant was prototyped and benched alongside this
//! one (see commit history). For these 53 short keys (average ~15
//! bytes), SipHash-based PHF costs 11-20 ns regardless of input —
//! 5-12× slower than the length-dispatched match. PHF would only
//! win if keys were either (a) much longer (~100+ bytes, where
//! strcmp dominates) or (b) much more numerous (~1000+, where
//! length-dispatch buckets blow up). Neither applies here.
//!
//! ## Zero-copy
//!
//! Both the static table data and the lookup output are `'static`
//! byte slices — no allocation, no copies.

/// (name, value) pairs for indices 1..=61. Index 0 is invalid and
/// is included as a zero-length pair so callers can index by 1.
#[rustfmt::skip]
pub static STATIC_TABLE: [(&[u8], &[u8]); 62] = [
    (b"",                              b""),                // index 0 (unused)
    (b":authority",                    b""),                // 1
    (b":method",                       b"GET"),             // 2
    (b":method",                       b"POST"),            // 3
    (b":path",                         b"/"),               // 4
    (b":path",                         b"/index.html"),     // 5
    (b":scheme",                       b"http"),            // 6
    (b":scheme",                       b"https"),           // 7
    (b":status",                       b"200"),             // 8
    (b":status",                       b"204"),             // 9
    (b":status",                       b"206"),             // 10
    (b":status",                       b"304"),             // 11
    (b":status",                       b"400"),             // 12
    (b":status",                       b"404"),             // 13
    (b":status",                       b"500"),             // 14
    (b"accept-charset",                b""),                // 15
    (b"accept-encoding",               b"gzip, deflate"),   // 16
    (b"accept-language",               b""),                // 17
    (b"accept-ranges",                 b""),                // 18
    (b"accept",                        b""),                // 19
    (b"access-control-allow-origin",   b""),                // 20
    (b"age",                           b""),                // 21
    (b"allow",                         b""),                // 22
    (b"authorization",                 b""),                // 23
    (b"cache-control",                 b""),                // 24
    (b"content-disposition",           b""),                // 25
    (b"content-encoding",              b""),                // 26
    (b"content-language",              b""),                // 27
    (b"content-length",                b""),                // 28
    (b"content-location",              b""),                // 29
    (b"content-range",                 b""),                // 30
    (b"content-type",                  b""),                // 31
    (b"cookie",                        b""),                // 32
    (b"date",                          b""),                // 33
    (b"etag",                          b""),                // 34
    (b"expect",                        b""),                // 35
    (b"expires",                       b""),                // 36
    (b"from",                          b""),                // 37
    (b"host",                          b""),                // 38
    (b"if-match",                      b""),                // 39
    (b"if-modified-since",             b""),                // 40
    (b"if-none-match",                 b""),                // 41
    (b"if-range",                      b""),                // 42
    (b"if-unmodified-since",           b""),                // 43
    (b"last-modified",                 b""),                // 44
    (b"link",                          b""),                // 45
    (b"location",                      b""),                // 46
    (b"max-forwards",                  b""),                // 47
    (b"proxy-authenticate",            b""),                // 48
    (b"proxy-authorization",           b""),                // 49
    (b"range",                         b""),                // 50
    (b"referer",                       b""),                // 51
    (b"refresh",                       b""),                // 52
    (b"retry-after",                   b""),                // 53
    (b"server",                        b""),                // 54
    (b"set-cookie",                    b""),                // 55
    (b"strict-transport-security",     b""),                // 56
    (b"transfer-encoding",             b""),                // 57
    (b"user-agent",                    b""),                // 58
    (b"vary",                          b""),                // 59
    (b"via",                           b""),                // 60
    (b"www-authenticate",              b""),                // 61
];

/// Forward lookup: index → entry. Returns `None` if `index == 0` or
/// `index > 61`.
#[inline]
#[must_use]
pub fn entry(index: u8) -> Option<&'static (&'static [u8], &'static [u8])> {
    if index == 0 || index as usize >= STATIC_TABLE.len() {
        return None;
    }
    Some(&STATIC_TABLE[index as usize])
}

/// Reverse lookup for the encoder: given a `(name, value)` pair,
/// find its static-table index. Returns `Some((index, value_matched))`
/// if the name is in the table — `value_matched` is `true` iff the
/// value also matched a predefined entry. Returns `None` if the
/// name isn't in the table at all (encoder must fall back to a
/// literal-with-new-name representation).
///
/// Single `match` on the name byte slice; compiler dispatches by
/// length, then string-compares. No allocation, no hashing.
#[inline]
#[must_use]
pub fn lookup(name: &[u8], value: &[u8]) -> Option<(u8, bool)> {
    match name {
        b":authority" => Some((1, false)),
        b":method" => match value {
            b"GET" => Some((2, true)),
            b"POST" => Some((3, true)),
            _ => Some((2, false)),
        },
        b":path" => match value {
            b"/" => Some((4, true)),
            b"/index.html" => Some((5, true)),
            _ => Some((4, false)),
        },
        b":scheme" => match value {
            b"http" => Some((6, true)),
            b"https" => Some((7, true)),
            _ => Some((6, false)),
        },
        b":status" => match value {
            b"200" => Some((8, true)),
            b"204" => Some((9, true)),
            b"206" => Some((10, true)),
            b"304" => Some((11, true)),
            b"400" => Some((12, true)),
            b"404" => Some((13, true)),
            b"500" => Some((14, true)),
            _ => Some((8, false)),
        },
        b"accept-charset" => Some((15, false)),
        b"accept-encoding" => {
            if value == b"gzip, deflate" {
                Some((16, true))
            } else {
                Some((16, false))
            }
        }
        b"accept-language" => Some((17, false)),
        b"accept-ranges" => Some((18, false)),
        b"accept" => Some((19, false)),
        b"access-control-allow-origin" => Some((20, false)),
        b"age" => Some((21, false)),
        b"allow" => Some((22, false)),
        b"authorization" => Some((23, false)),
        b"cache-control" => Some((24, false)),
        b"content-disposition" => Some((25, false)),
        b"content-encoding" => Some((26, false)),
        b"content-language" => Some((27, false)),
        b"content-length" => Some((28, false)),
        b"content-location" => Some((29, false)),
        b"content-range" => Some((30, false)),
        b"content-type" => Some((31, false)),
        b"cookie" => Some((32, false)),
        b"date" => Some((33, false)),
        b"etag" => Some((34, false)),
        b"expect" => Some((35, false)),
        b"expires" => Some((36, false)),
        b"from" => Some((37, false)),
        b"host" => Some((38, false)),
        b"if-match" => Some((39, false)),
        b"if-modified-since" => Some((40, false)),
        b"if-none-match" => Some((41, false)),
        b"if-range" => Some((42, false)),
        b"if-unmodified-since" => Some((43, false)),
        b"last-modified" => Some((44, false)),
        b"link" => Some((45, false)),
        b"location" => Some((46, false)),
        b"max-forwards" => Some((47, false)),
        b"proxy-authenticate" => Some((48, false)),
        b"proxy-authorization" => Some((49, false)),
        b"range" => Some((50, false)),
        b"referer" => Some((51, false)),
        b"refresh" => Some((52, false)),
        b"retry-after" => Some((53, false)),
        b"server" => Some((54, false)),
        b"set-cookie" => Some((55, false)),
        b"strict-transport-security" => Some((56, false)),
        b"transfer-encoding" => Some((57, false)),
        b"user-agent" => Some((58, false)),
        b"vary" => Some((59, false)),
        b"via" => Some((60, false)),
        b"www-authenticate" => Some((61, false)),
        _ => None,
    }
}

/// Name-only lookup: skip the value check, return the first matching
/// index for the name. Used when the encoder has decided to emit a
/// literal with an indexed name (value is sent as a literal).
#[inline]
#[must_use]
pub fn lookup_name(name: &[u8]) -> Option<u8> {
    lookup(name, b"").map(|(index, _)| index)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn entry_round_trips_every_index() {
        for index in 1..=61_u8 {
            let entry = entry(index).expect("static entry");
            let (name, value) = *entry;
            let (idx, matched) = lookup(name, value).expect("lookup hit");
            assert_eq!(idx, index, "index round-trip");
            assert!(matched || value.is_empty(), "value match for non-empty");
        }
    }

    #[test]
    fn entry_zero_is_invalid() {
        assert!(entry(0).is_none());
        assert!(entry(62).is_none());
        assert!(entry(255).is_none());
    }

    #[test]
    fn pseudo_method_get() {
        assert_eq!(lookup(b":method", b"GET"), Some((2, true)));
        assert_eq!(lookup(b":method", b"POST"), Some((3, true)));
        assert_eq!(lookup(b":method", b"DELETE"), Some((2, false)));
    }

    #[test]
    fn pseudo_path() {
        assert_eq!(lookup(b":path", b"/"), Some((4, true)));
        assert_eq!(lookup(b":path", b"/index.html"), Some((5, true)));
        assert_eq!(lookup(b":path", b"/api/v1"), Some((4, false)));
    }

    #[test]
    fn pseudo_status_codes() {
        for (status, idx) in [
            (b"200" as &[u8], 8),
            (b"204", 9),
            (b"206", 10),
            (b"304", 11),
            (b"400", 12),
            (b"404", 13),
            (b"500", 14),
        ] {
            assert_eq!(lookup(b":status", status), Some((idx, true)));
        }
        assert_eq!(lookup(b":status", b"418"), Some((8, false)));
    }

    #[test]
    fn accept_encoding_value_match() {
        assert_eq!(
            lookup(b"accept-encoding", b"gzip, deflate"),
            Some((16, true))
        );
        assert_eq!(lookup(b"accept-encoding", b"br"), Some((16, false)));
    }

    #[test]
    fn unknown_name_returns_none() {
        assert_eq!(lookup(b"x-custom-header", b"foo"), None);
        assert_eq!(lookup(b"", b""), None);
    }

    #[test]
    fn name_only_lookup_skips_value() {
        assert_eq!(lookup_name(b":method"), Some(2));
        assert_eq!(lookup_name(b"content-type"), Some(31));
        assert_eq!(lookup_name(b"x-not-real"), None);
    }
}
