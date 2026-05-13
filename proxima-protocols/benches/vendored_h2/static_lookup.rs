// Vendored from h2-0.4.14 src/hpack/table.rs::index_static (MIT).
// Adapted to take `&[u8]` slices directly instead of going through
// `http::HeaderName` + h2's `Header` enum — that wrapper is what a
// caller would have to construct from wire bytes, so stripping it
// gives a fair head-to-head against proxima's byte-slice match.
// The match arm structure + dispatch logic is preserved verbatim.

#![allow(dead_code)]

pub fn index_static(name: &[u8], value: &[u8]) -> Option<(usize, bool)> {
    match name {
        b":authority" => Some((1, false)),
        b":method" => match value {
            b"GET" => Some((2, true)),
            b"POST" => Some((3, true)),
            _ => Some((2, false)),
        },
        b":scheme" => match value {
            b"http" => Some((6, true)),
            b"https" => Some((7, true)),
            _ => Some((6, false)),
        },
        b":path" => match value {
            b"/" => Some((4, true)),
            b"/index.html" => Some((5, true)),
            _ => Some((4, false)),
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
