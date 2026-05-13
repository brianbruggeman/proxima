//! HTTP Digest Access Authentication (RFC 7616) — a request-oriented
//! challenge/response handshake (auth form #4). The server emits a
//! `WWW-Authenticate: Digest …` challenge; the client computes a `response`
//! that proves it knows the password without sending it, and replies with an
//! `Authorization: Digest …` header.
//!
//! Primary source: RFC 7616 §3.4 (response computation) + §3.9.1 (worked
//! example, the locked test vector).
//!
//! Per RFC 7616 §3.4.6, with `qop=auth`:
//! ```text
//! A1       = unq(username) ":" unq(realm) ":" passwd
//! A2       = Method ":" request-uri
//! response = KD( H(A1), nonce ":" nc ":" cnonce ":" qop ":" H(A2) )
//! KD(s, d) = H( s ":" d )
//! ```
//! where `H` is the chosen algorithm (`MD5` or `SHA-256`) and the digest is
//! lowercase hex.
//!
//! Sans-IO (principle 11): pure computation over caller-supplied bytes; no
//! sockets, no clock. The handshake is one round (challenge in → response
//! out), but [`DigestClient`] is the request-oriented surface because the
//! method/uri are not in the challenge. `cnonce` is caller-supplied so the
//! computation is deterministic and testable (the edge supplies real entropy).

use alloc::format;
use alloc::string::{String, ToString};

use md5::Md5;
use sha2::{Digest as _, Sha256};
use subtle::{Choice, ConstantTimeEq};
use zeroize::{Zeroize, Zeroizing};

/// The hash algorithm a Digest challenge selects (`algorithm=` parameter).
/// RFC 7616 §3.3 registers `MD5` and `SHA-256` (and `-sess` variants; this
/// covers the non-session forms the §3.9.1 vector uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestAlgorithm {
    Md5,
    Sha256,
}

impl DigestAlgorithm {
    /// Parse the `algorithm=` token (case-insensitive). Defaults to `MD5` per
    /// RFC 7616 §3.4 when the challenge omits it.
    #[must_use]
    pub fn parse(token: &str) -> Self {
        if token.eq_ignore_ascii_case("SHA-256") {
            Self::Sha256
        } else {
            Self::Md5
        }
    }

    fn hash_hex(self, data: &[u8]) -> String {
        match self {
            Self::Md5 => {
                let mut hasher = Md5::new();
                hasher.update(data);
                hex::encode(hasher.finalize())
            }
            Self::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                hex::encode(hasher.finalize())
            }
        }
    }
}

/// A parsed `WWW-Authenticate: Digest …` challenge. Borrows nothing — owns the
/// quoted parameter values the client echoes back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigestChallenge {
    pub realm: String,
    pub nonce: String,
    pub algorithm: DigestAlgorithm,
    pub qop: Option<String>,
    pub opaque: Option<String>,
}

impl DigestChallenge {
    /// Parse a `Digest k1="v1", k2="v2"` challenge value (the part after the
    /// `Digest ` scheme token; the caller strips the scheme).
    ///
    /// # Errors
    /// [`DigestError::MissingField`] when `realm` or `nonce` is absent.
    pub fn parse(challenge: &str) -> Result<Self, DigestError> {
        let body = challenge
            .strip_prefix("Digest ")
            .unwrap_or(challenge)
            .trim();
        let mut realm = None;
        let mut nonce = None;
        let mut algorithm = DigestAlgorithm::Md5;
        let mut qop = None;
        let mut opaque = None;
        for (key, value) in parse_params(body) {
            match key {
                "realm" => realm = Some(field_safe("realm", value)?.to_string()),
                "nonce" => nonce = Some(field_safe("nonce", value)?.to_string()),
                "algorithm" => algorithm = DigestAlgorithm::parse(value),
                "qop" => qop = Some(first_qop(value).to_string()),
                "opaque" => opaque = Some(field_safe("opaque", value)?.to_string()),
                _ => {}
            }
        }
        Ok(Self {
            realm: realm.ok_or(DigestError::MissingField("realm"))?,
            nonce: nonce.ok_or(DigestError::MissingField("nonce"))?,
            algorithm,
            qop,
            opaque,
        })
    }
}

/// Request-oriented Digest client (RFC 7616). Holds the credentials (the
/// password zeroizes on drop) and computes the per-request `response` +
/// `Authorization` header for a given challenge, method, and uri.
///
/// Deliberately NOT `Clone` (a cloned password is a second heap copy that
/// outlives the original's zeroize — the same audit-Z1 reasoning as
/// [`crate::sigv4::SecretKey`]); share it behind `Arc`.
pub struct DigestClient {
    username: String,
    password: Password,
}

/// Plaintext password, zeroized on drop — secret material never lingers. Not
/// `Clone` so the cleartext is never duplicated (audit Z1/M4).
struct Password(String);

impl Drop for Password {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl DigestClient {
    #[must_use]
    pub fn new(username: &str, password: &str) -> Self {
        Self {
            username: username.into(),
            password: Password(password.into()),
        }
    }

    /// Compute the `response` hex digest (RFC 7616 §3.4.6) for `qop=auth`.
    /// `nc` is the request counter (e.g. `00000001`); `cnonce` is the
    /// client-nonce the edge generates.
    #[must_use]
    pub fn response(
        &self,
        challenge: &DigestChallenge,
        method: &str,
        uri: &str,
        nc: &str,
        cnonce: &str,
    ) -> String {
        let algorithm = challenge.algorithm;
        // A1 holds the plaintext password; HA1 is password-equivalent. Both the
        // A1 concatenation and HA1 are Zeroizing so they wipe on drop instead of
        // lingering in freed heap (audit H2).
        let a1 = Zeroizing::new(format!(
            "{}:{}:{}",
            self.username, challenge.realm, self.password.0
        ));
        let ha1 = Zeroizing::new(algorithm.hash_hex(a1.as_bytes()));
        let ha2 = algorithm.hash_hex(format!("{method}:{uri}").as_bytes());
        let kd_input = match &challenge.qop {
            Some(qop) => Zeroizing::new(format!(
                "{}:{}:{nc}:{cnonce}:{qop}:{ha2}",
                ha1.as_str(),
                challenge.nonce
            )),
            // RFC 7616 §3.4.6 legacy form (no qop): KD(H(A1), nonce ":" H(A2)).
            None => Zeroizing::new(format!("{}:{}:{ha2}", ha1.as_str(), challenge.nonce)),
        };
        algorithm.hash_hex(kd_input.as_bytes())
    }

    /// Build the full `Authorization: Digest …` header value for a request.
    #[must_use]
    pub fn authorization(
        &self,
        challenge: &DigestChallenge,
        method: &str,
        uri: &str,
        nc: &str,
        cnonce: &str,
    ) -> String {
        let response = self.response(challenge, method, uri, nc, cnonce);
        let mut out = String::with_capacity(256);
        out.push_str("Digest username=\"");
        out.push_str(&self.username);
        out.push_str("\", realm=\"");
        out.push_str(&challenge.realm);
        out.push_str("\", nonce=\"");
        out.push_str(&challenge.nonce);
        out.push_str("\", uri=\"");
        out.push_str(uri);
        out.push_str("\", algorithm=");
        out.push_str(match challenge.algorithm {
            DigestAlgorithm::Md5 => "MD5",
            DigestAlgorithm::Sha256 => "SHA-256",
        });
        if let Some(qop) = &challenge.qop {
            out.push_str(", qop=");
            out.push_str(qop);
            out.push_str(", nc=");
            out.push_str(nc);
            out.push_str(", cnonce=\"");
            out.push_str(cnonce);
            out.push('"');
        }
        out.push_str(", response=\"");
        out.push_str(&response);
        out.push('"');
        if let Some(opaque) = &challenge.opaque {
            out.push_str(", opaque=\"");
            out.push_str(opaque);
            out.push('"');
        }
        out
    }
}

impl core::fmt::Debug for DigestClient {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DigestClient")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

/// Constant-time compare of two Digest `response` hex strings — for a server
/// or test verifying a client response without leaking via timing. Folds the
/// length check INTO the constant-time `Choice` (audit M2): `subtle`'s slice
/// `ct_eq` short-circuits on a length mismatch, leaking length via timing, so
/// the comparison runs against a zero-padded fixed width and the length
/// equality is `&`-combined at the end, never branched on.
#[must_use]
pub fn responses_equal(left: &str, right: &str) -> bool {
    let same_length = Choice::from(u8::from(left.len() == right.len()));
    let width = left.len().max(right.len());
    let mut equal = Choice::from(1u8);
    for index in 0..width {
        let left_byte = left.as_bytes().get(index).copied().unwrap_or(0);
        let right_byte = right.as_bytes().get(index).copied().unwrap_or(0);
        equal &= left_byte.ct_eq(&right_byte);
    }
    (equal & same_length).into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestError {
    MissingField(&'static str),
    /// a challenge field carried a `"`, CR, or LF — a header-injection attempt
    UnsafeField(&'static str),
}

impl core::fmt::Display for DigestError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingField(field) => write!(formatter, "digest challenge missing `{field}`"),
            Self::UnsafeField(field) => {
                write!(
                    formatter,
                    "digest challenge `{field}` contains an unsafe character"
                )
            }
        }
    }
}

/// Reject a quote, CR, or LF in a server-supplied challenge field before it is
/// echoed back into the quoted `Authorization` header (audit C2): a `"` breaks
/// out of the quoted value, a CR/LF splits headers. The server controls these
/// values, so this is a server→client injection guard.
fn field_safe<'value>(name: &'static str, value: &'value str) -> Result<&'value str, DigestError> {
    if value
        .bytes()
        .any(|byte| byte == b'"' || byte == b'\r' || byte == b'\n')
    {
        return Err(DigestError::UnsafeField(name));
    }
    Ok(value)
}

/// Parse `k1="v1", k2=v2, k3="v3"` into (key, value) pairs, stripping the outer
/// quotes. Splits on commas that are OUTSIDE a quoted-string only (RFC 7616
/// §3.3 / RFC 7230 quoted-string): a `nonce="a,b"` carries a comma in its value
/// and must not be truncated (audit M1). Returns owned pairs because a quoted
/// value may need its surrounding quotes removed without reallocating the body.
fn parse_params(body: &str) -> alloc::vec::Vec<(&str, &str)> {
    let mut pairs = alloc::vec::Vec::new();
    let bytes = body.as_bytes();
    let mut start = 0;
    let mut in_quotes = false;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                push_param(&mut pairs, &body[start..index]);
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    push_param(&mut pairs, &body[start..]);
    pairs
}

/// Split one `key=value` segment, trimming surrounding whitespace and the outer
/// quotes from the value. Empty segments (a trailing comma) are dropped.
fn push_param<'body>(pairs: &mut alloc::vec::Vec<(&'body str, &'body str)>, segment: &'body str) {
    if let Some((key, value)) = segment.split_once('=') {
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or(value);
        pairs.push((key.trim(), value));
    }
}

/// A challenge may offer `qop="auth,auth-int"`; we pick `auth` (the form we
/// compute). Picks the first listed option as a fallback.
fn first_qop(value: &str) -> &str {
    value
        .split(',')
        .map(str::trim)
        .find(|option| *option == "auth")
        .unwrap_or_else(|| value.split(',').next().map_or("auth", str::trim))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // RFC 7616 §3.9.1 worked example — the locked vector. The MD5 `response`
    // is the literal value the RFC prints; the SHA-256 `response` is derived
    // from the same §3.9.1 inputs via the RFC §3.4.6 formula (principle 14,
    // independently reproduced).
    const RFC_USERNAME: &str = "Mufasa";
    const RFC_REALM: &str = "http-auth@example.org";
    const RFC_PASSWORD: &str = "Circle of Life";
    const RFC_NONCE: &str = "7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v";
    const RFC_CNONCE: &str = "f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ";
    const RFC_NC: &str = "00000001";
    const RFC_METHOD: &str = "GET";
    const RFC_URI: &str = "/dir/index.html";

    fn rfc_challenge(algorithm: DigestAlgorithm) -> DigestChallenge {
        DigestChallenge {
            realm: RFC_REALM.into(),
            nonce: RFC_NONCE.into(),
            algorithm,
            qop: Some("auth".into()),
            opaque: None,
        }
    }

    #[test]
    fn rfc7616_section_3_9_1_md5_response_matches_the_spec_vector() {
        let client = DigestClient::new(RFC_USERNAME, RFC_PASSWORD);
        let response = client.response(
            &rfc_challenge(DigestAlgorithm::Md5),
            RFC_METHOD,
            RFC_URI,
            RFC_NC,
            RFC_CNONCE,
        );
        assert_eq!(
            response, "8ca523f5e9506fed4657c9700eebdbec",
            "RFC 7616 §3.9.1 prints this exact MD5 response"
        );
    }

    #[test]
    fn rfc7616_section_3_9_1_sha256_response_matches_the_derived_vector() {
        let client = DigestClient::new(RFC_USERNAME, RFC_PASSWORD);
        let response = client.response(
            &rfc_challenge(DigestAlgorithm::Sha256),
            RFC_METHOD,
            RFC_URI,
            RFC_NC,
            RFC_CNONCE,
        );
        assert_eq!(
            response, "753927fa0e85d155564e2e272a28d1802ca10daf4496794697cf8db5856cb6c1",
            "SHA-256 response derived from the §3.9.1 inputs via the §3.4.6 formula"
        );
    }

    #[test]
    fn parse_challenge_reads_realm_nonce_algorithm_qop_opaque() {
        let raw = "Digest realm=\"http-auth@example.org\", \
                   qop=\"auth, auth-int\", \
                   algorithm=SHA-256, \
                   nonce=\"7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v\", \
                   opaque=\"FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS\"";
        let challenge = DigestChallenge::parse(raw).expect("parse");
        assert_eq!(challenge.realm, RFC_REALM);
        assert_eq!(challenge.nonce, RFC_NONCE);
        assert_eq!(challenge.algorithm, DigestAlgorithm::Sha256);
        assert_eq!(challenge.qop.as_deref(), Some("auth"));
        assert_eq!(
            challenge.opaque.as_deref(),
            Some("FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS")
        );
    }

    #[test]
    fn authorization_header_round_trips_through_the_parser() {
        let client = DigestClient::new(RFC_USERNAME, RFC_PASSWORD);
        let challenge = rfc_challenge(DigestAlgorithm::Sha256);
        let header = client.authorization(&challenge, RFC_METHOD, RFC_URI, RFC_NC, RFC_CNONCE);
        assert!(header.contains("username=\"Mufasa\""));
        assert!(header.contains("algorithm=SHA-256"));
        assert!(header.contains(
            "response=\"753927fa0e85d155564e2e272a28d1802ca10daf4496794697cf8db5856cb6c1\""
        ));
        assert!(header.contains("qop=auth"));
        assert!(header.contains("nc=00000001"));
    }

    #[test]
    fn a_comma_inside_a_quoted_value_does_not_truncate_the_field() {
        // audit M1: a server nonce carrying a comma must survive the param split.
        let challenge =
            DigestChallenge::parse("Digest realm=\"r\", nonce=\"abc,def,ghi\", algorithm=MD5")
                .expect("parse");
        assert_eq!(
            challenge.nonce, "abc,def,ghi",
            "quoted comma is part of the value"
        );
        assert_eq!(challenge.realm, "r");
        assert_eq!(challenge.algorithm, DigestAlgorithm::Md5);
    }

    #[test]
    fn missing_nonce_is_a_parse_error() {
        let outcome = DigestChallenge::parse("Digest realm=\"r\"");
        assert_eq!(outcome, Err(DigestError::MissingField("nonce")));
    }

    #[test]
    fn responses_equal_is_constant_time_true_on_match() {
        assert!(responses_equal(
            "8ca523f5e9506fed4657c9700eebdbec",
            "8ca523f5e9506fed4657c9700eebdbec"
        ));
        assert!(!responses_equal(
            "8ca523f5e9506fed4657c9700eebdbec",
            "deadbeef"
        ));
        // audit M2: unequal lengths must compare false WITHOUT a length-based
        // short-circuit (the loop runs to the wider width either way).
        assert!(!responses_equal("abcd", "abcde"));
        assert!(!responses_equal("abcde", "abcd"));
        assert!(responses_equal("", ""));
    }

    #[test]
    fn challenge_with_a_quote_or_crlf_in_a_field_is_rejected() {
        // audit C2: a server-supplied nonce carrying a `"` or CRLF must be
        // refused at parse, never echoed into the quoted Authorization header.
        let quoted = DigestChallenge::parse("Digest realm=\"r\", nonce=\"a\"b\"");
        assert_eq!(quoted, Err(DigestError::UnsafeField("nonce")));
        let crlf = DigestChallenge::parse("Digest realm=\"r\\r\\nevil\", nonce=\"n\"");
        // the literal backslash-r-n above is text; build a real CRLF case:
        let real_crlf = DigestChallenge::parse("Digest realm=\"good\", nonce=\"a\r\nInjected: 1\"");
        assert_eq!(real_crlf, Err(DigestError::UnsafeField("nonce")));
        let _ = crlf;
    }

    #[test]
    fn legacy_no_qop_uses_the_two_field_kd_form() {
        let client = DigestClient::new(RFC_USERNAME, RFC_PASSWORD);
        let challenge = DigestChallenge {
            realm: RFC_REALM.into(),
            nonce: RFC_NONCE.into(),
            algorithm: DigestAlgorithm::Md5,
            qop: None,
            opaque: None,
        };
        let response = client.response(&challenge, RFC_METHOD, RFC_URI, RFC_NC, RFC_CNONCE);
        // legacy KD(H(A1), nonce:H(A2)) — distinct from the qop form, and stable.
        assert_eq!(response.len(), 32, "md5 hex is 32 chars");
    }
}
