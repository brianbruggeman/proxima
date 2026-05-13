//! AWS Signature Version 4 request signer (auth form #5) — RFC-equivalent to
//! the AWS `SigV4` specification. The attach edge *computes* an `Authorization`
//! value from the canonical request + the derived signing key, rather than
//! attaching a static credential.
//!
//! Primary source: AWS "Create a signed AWS API request" + the public
//! `aws-sig-v4-test-suite`. The four steps the spec names:
//! 1. canonical request = `METHOD\nURI\nQUERY\nHEADERS\n\nSIGNED\nHASH`
//! 2. string-to-sign = `AWS4-HMAC-SHA256\nDATE\nSCOPE\nHEX(SHA256(canonical))`
//! 3. signing key = chained `HMAC-SHA256` over date/region/service/`aws4_request`
//! 4. signature = `HEX(HMAC-SHA256(signing_key, string_to_sign))`
//!
//! Sans-IO (principle 11): no clock read — the caller stamps the request date
//! into the `x-amz-date` header it passes in; this module never touches the
//! wire. The secret key zeroizes on drop (`SecretKey`).

use alloc::string::String;
use alloc::vec::Vec;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

type HmacSha256 = Hmac<Sha256>;

/// A secret access key held only as long as needed, zeroized on drop
/// (principle: secret material never lingers in freed heap). Wraps the raw
/// `AWS4` + secret bytes used as the first HMAC key in the chain.
///
/// Deliberately NOT `Clone`: a cloned secret is a second heap copy that
/// outlives the original's zeroize, doubling the exposure window (audit Z1).
/// Share a `SigV4Signer` behind `Arc` rather than cloning it.
pub struct SecretKey(Vec<u8>);

impl SecretKey {
    /// Build from a secret access key string. Prepends the literal `AWS4`
    /// prefix the spec's first derivation step requires.
    #[must_use]
    pub fn new(secret_access_key: &str) -> Self {
        let mut buffer = Vec::with_capacity(4 + secret_access_key.len());
        buffer.extend_from_slice(b"AWS4");
        buffer.extend_from_slice(secret_access_key.as_bytes());
        Self(buffer)
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("SecretKey(<redacted>)")
    }
}

/// One header to include in the signature — name + value. The caller owns
/// these; the signer lowercases the name and trims the value per spec.
#[derive(Clone, Debug)]
pub struct SignedHeader {
    pub name: String,
    pub value: String,
}

/// AWS `SigV4` signer. Holds the access-key id (public, travels in the
/// `Credential=` field) + the secret-derived key material + the credential
/// scope (region/service). Computes the full `Authorization` header via
/// [`SigV4Signer::authorization`] — the request-signing surface (auth form #5):
/// the credential is *computed per request* from the canonical request + the
/// derived key, not attached statically. Not `Clone` (the secret must not be
/// copied — audit Z1); share behind `Arc`.
#[derive(Debug)]
pub struct SigV4Signer {
    access_key_id: String,
    secret: SecretKey,
    region: String,
    service: String,
}

impl SigV4Signer {
    /// Build a signer for `region`/`service` (e.g. `us-east-1`/`s3`). The
    /// secret zeroizes on drop; the access-key id is public.
    #[must_use]
    pub fn new(access_key_id: &str, secret_access_key: &str, region: &str, service: &str) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret: SecretKey::new(secret_access_key),
            region: region.into(),
            service: service.into(),
        }
    }

    /// Sign a request described by its parts, producing the full
    /// `Authorization` header value (`AWS4-HMAC-SHA256 Credential=…,
    /// SignedHeaders=…, Signature=…`).
    ///
    /// `amz_date` is the `x-amz-date` value (`YYYYMMDDTHHMMSSZ`); the caller
    /// must also include it among `headers`. `payload` is the raw body bytes.
    #[must_use]
    pub fn authorization(
        &self,
        method: &str,
        canonical_uri: &str,
        canonical_query: &str,
        headers: &[SignedHeader],
        payload: &[u8],
        amz_date: &str,
    ) -> String {
        let mut sorted = headers.to_vec();
        for header in &mut sorted {
            header.name = header.name.to_ascii_lowercase();
        }
        sorted.sort_by(|left, right| left.name.cmp(&right.name));

        let signed_headers = signed_header_list(&sorted);
        let encoded_uri = uri_encode_path(canonical_uri);
        let canonical_request = canonical_request(
            method,
            &encoded_uri,
            canonical_query,
            &sorted,
            &signed_headers,
            payload,
        );

        let date_stamp = &amz_date[..8.min(amz_date.len())];
        let scope = scope(date_stamp, &self.region, &self.service);
        let string_to_sign = string_to_sign(amz_date, &scope, &canonical_request);

        let signing_key = self.signing_key(date_stamp);
        let signature = hex::encode(hmac(&signing_key, string_to_sign.as_bytes()).as_slice());

        let mut out = String::with_capacity(160);
        out.push_str("AWS4-HMAC-SHA256 Credential=");
        out.push_str(&self.access_key_id);
        out.push('/');
        out.push_str(&scope);
        out.push_str(", SignedHeaders=");
        out.push_str(&signed_headers);
        out.push_str(", Signature=");
        out.push_str(&signature);
        out
    }

    /// The chained `HMAC-SHA256` signing-key derivation (step 3). Every
    /// intermediate is `Zeroizing` so the secret-derived key material is wiped
    /// as each step's `Vec` drops, not left in freed heap (audit H1).
    fn signing_key(&self, date_stamp: &str) -> Zeroizing<Vec<u8>> {
        let date_key = hmac(&self.secret.0, date_stamp.as_bytes());
        let region_key = hmac(date_key.as_slice(), self.region.as_bytes());
        let service_key = hmac(region_key.as_slice(), self.service.as_bytes());
        hmac(service_key.as_slice(), b"aws4_request")
    }
}

/// `Hex(SHA256(bytes))` — lowercase hex of the SHA-256 digest.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// HMAC-SHA256, returning `Zeroizing` so secret-derived output wipes on drop
/// (audit H1).
fn hmac(key: &[u8], data: &[u8]) -> Zeroizing<Vec<u8>> {
    // HMAC accepts a key of any length (it hashes over-long keys, pads short
    // ones), so `new_from_slice` is infallible here; the `Err` arm is
    // unreachable, so return an empty digest rather than panic on it.
    let Ok(mut mac) = <HmacSha256 as Mac>::new_from_slice(key) else {
        return Zeroizing::new(Vec::new());
    };
    mac.update(data);
    Zeroizing::new(mac.finalize().into_bytes().to_vec())
}

/// `host;x-amz-date` — sorted lowercase names, semicolon-joined.
fn signed_header_list(sorted: &[SignedHeader]) -> String {
    let mut out = String::new();
    for (index, header) in sorted.iter().enumerate() {
        if index > 0 {
            out.push(';');
        }
        out.push_str(&header.name);
    }
    out
}

/// Step 1: the canonical request. Note the spec's structure ends with a blank
/// line after the canonical headers block (each header line already carries a
/// trailing `\n`), then the signed-header list, then the payload hash.
fn canonical_request(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    sorted: &[SignedHeader],
    signed_headers: &str,
    payload: &[u8],
) -> String {
    let mut out = String::new();
    out.push_str(method);
    out.push('\n');
    out.push_str(canonical_uri);
    out.push('\n');
    out.push_str(canonical_query);
    out.push('\n');
    for header in sorted {
        out.push_str(&header.name);
        out.push(':');
        out.push_str(&canonical_header_value(&header.value));
        out.push('\n');
    }
    out.push('\n');
    out.push_str(signed_headers);
    out.push('\n');
    out.push_str(&sha256_hex(payload));
    out
}

/// Canonicalize a header value per the AWS SigV4 rules: trim ends, then
/// collapse every run of whitespace (spaces, tabs, AND newlines) to a single
/// space. Collapsing newlines also defends against canonical-request injection
/// — an attacker-supplied `\n` cannot terminate the header line early (audit
/// C1), because it becomes one space inside the value.
fn canonical_header_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut in_whitespace = false;
    for character in value.trim().chars() {
        if character.is_whitespace() {
            in_whitespace = true;
        } else {
            if in_whitespace {
                out.push(' ');
                in_whitespace = false;
            }
            out.push(character);
        }
    }
    out
}

/// SigV4 canonical-URI encoding (spec step 1, "create the canonical URI"):
/// URI-encode each path segment, preserving `/` as the segment separator and
/// the RFC 3986 unreserved set (`A-Z a-z 0-9 - . _ ~`); every other byte becomes
/// `%XX` uppercase hex. Without this, a path with a space or non-ASCII byte is
/// signed verbatim while the wire sends the encoded form, so the signature never
/// matches what the server canonicalizes (audit H3). The `aws-sig-v4-test-suite`
/// `get-space` vector (`/example space/` → `/example%20space/`) pins this.
///
/// This is the non-S3 "encode once" form: a segment is treated as a literal, so
/// `%` itself is escaped. The common `/` path is unchanged, so the get-vanilla
/// vector is byte-identical.
fn uri_encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for (index, segment) in path.split('/').enumerate() {
        if index > 0 {
            out.push('/');
        }
        for byte in segment.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                    out.push(byte as char);
                }
                other => {
                    out.push('%');
                    out.push_str(&hex::encode_upper([other]));
                }
            }
        }
    }
    out
}

/// `YYYYMMDD/region/service/aws4_request`.
fn scope(date_stamp: &str, region: &str, service: &str) -> String {
    let mut out = String::with_capacity(date_stamp.len() + region.len() + service.len() + 14);
    out.push_str(date_stamp);
    out.push('/');
    out.push_str(region);
    out.push('/');
    out.push_str(service);
    out.push_str("/aws4_request");
    out
}

/// Step 2: the string-to-sign.
fn string_to_sign(amz_date: &str, scope: &str, canonical_request: &str) -> String {
    let mut out = String::new();
    out.push_str("AWS4-HMAC-SHA256\n");
    out.push_str(amz_date);
    out.push('\n');
    out.push_str(scope);
    out.push('\n');
    out.push_str(&sha256_hex(canonical_request.as_bytes()));
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The `aws-sig-v4-test-suite` `get-vanilla` case, bit-exact. Inputs +
    /// expected signature independently derived from the AWS `SigV4` spec via a
    /// reference HMAC/SHA-256 implementation (principle 14): the published
    /// suite's `get-vanilla` signature is this exact value.
    /// Source: AWS `SigV4` spec + aws-sig-v4-test-suite get-vanilla.
    fn get_vanilla_signer() -> SigV4Signer {
        SigV4Signer::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "service",
        )
    }

    #[test]
    fn aws_sig_v4_test_suite_get_vanilla_canonical_request_is_bit_exact() {
        let headers = vec![
            SignedHeader {
                name: "host".into(),
                value: "example.amazonaws.com".into(),
            },
            SignedHeader {
                name: "x-amz-date".into(),
                value: "20150830T123600Z".into(),
            },
        ];
        let mut sorted = headers;
        sorted.sort_by(|left, right| left.name.cmp(&right.name));
        let signed = signed_header_list(&sorted);
        let canonical = canonical_request("GET", "/", "", &sorted, &signed, b"");
        assert_eq!(
            canonical,
            "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
             host;x-amz-date\n\
             e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "canonical request must match the SigV4 get-vanilla vector byte-for-byte"
        );
    }

    #[test]
    fn header_value_newlines_collapse_to_a_space_no_canonical_injection() {
        // audit C1: a header value carrying an embedded CRLF + phantom line must
        // NOT terminate the canonical header line early; it collapses to one
        // space, so no injected `evil:1` line appears in the canonical request.
        let headers = vec![
            SignedHeader {
                name: "host".into(),
                value: "h.example".into(),
            },
            SignedHeader {
                name: "x-injected".into(),
                value: "a\r\nevil:1".into(),
            },
        ];
        let mut sorted = headers;
        sorted.sort_by(|left, right| left.name.cmp(&right.name));
        let signed = signed_header_list(&sorted);
        let canonical = canonical_request("GET", "/", "", &sorted, &signed, b"");
        assert!(
            canonical.contains("x-injected:a evil:1\n"),
            "embedded CRLF must collapse to a single space, not split the line: {canonical}"
        );
        assert!(
            !canonical.contains("\nevil:1"),
            "no phantom `evil:1` header line may appear in the canonical request"
        );
    }

    #[test]
    fn aws_sig_v4_test_suite_get_vanilla_string_to_sign_is_bit_exact() {
        let scope = scope("20150830", "us-east-1", "service");
        let canonical = "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
                         host;x-amz-date\n\
                         e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let sts = string_to_sign("20150830T123600Z", &scope, canonical);
        assert_eq!(
            sts,
            "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/service/aws4_request\n\
             bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63",
            "string-to-sign must match the SigV4 get-vanilla vector"
        );
    }

    #[test]
    fn aws_sig_v4_test_suite_get_vanilla_signature_is_bit_exact() {
        let signer = get_vanilla_signer();
        let headers = vec![
            SignedHeader {
                name: "host".into(),
                value: "example.amazonaws.com".into(),
            },
            SignedHeader {
                name: "x-amz-date".into(),
                value: "20150830T123600Z".into(),
            },
        ];
        let authorization = signer.authorization("GET", "/", "", &headers, b"", "20150830T123600Z");
        assert!(
            authorization.ends_with(
                "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
            ),
            "signature must equal the published get-vanilla value; got `{authorization}`"
        );
    }

    #[test]
    fn authorization_carries_credential_scope_and_signed_headers() {
        let signer = get_vanilla_signer();
        let headers = vec![
            SignedHeader {
                name: "host".into(),
                value: "example.amazonaws.com".into(),
            },
            SignedHeader {
                name: "x-amz-date".into(),
                value: "20150830T123600Z".into(),
            },
        ];
        let authorization = signer.authorization("GET", "/", "", &headers, b"", "20150830T123600Z");
        assert!(authorization.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request"
        ));
        assert!(authorization.contains(", SignedHeaders=host;x-amz-date, "));
    }

    #[test]
    fn aws_sig_v4_test_suite_get_space_path_is_uri_encoded_before_signing() {
        // audit H3 / principle 14: the `get-space` suite case signs the path
        // `/example space/`, whose canonical URI is `/example%20space/`. The
        // published signature is this exact value.
        let signer = get_vanilla_signer();
        let headers = vec![
            SignedHeader {
                name: "host".into(),
                value: "example.amazonaws.com".into(),
            },
            SignedHeader {
                name: "x-amz-date".into(),
                value: "20150830T123600Z".into(),
            },
        ];
        let authorization = signer.authorization(
            "GET",
            "/example space/",
            "",
            &headers,
            b"",
            "20150830T123600Z",
        );
        assert!(
            authorization.ends_with(
                "Signature=652487583200325589f1fba4c7e578f72c47cb61beeca81406b39ddec1366741"
            ),
            "space in the path must encode to %20 before signing; got `{authorization}`"
        );
    }

    #[test]
    fn uri_encode_path_preserves_slash_and_unreserved_escapes_the_rest() {
        assert_eq!(uri_encode_path("/"), "/");
        assert_eq!(uri_encode_path("/a/b"), "/a/b");
        assert_eq!(uri_encode_path("/example space/"), "/example%20space/");
        assert_eq!(uri_encode_path("/keep-._~"), "/keep-._~");
    }

    #[test]
    fn header_names_are_lowercased_and_sorted_before_signing() {
        let signer = get_vanilla_signer();
        // pass the host header in mixed case + out of order; the signature must
        // still equal the canonical (lowercased, sorted) get-vanilla value.
        let headers = vec![
            SignedHeader {
                name: "X-Amz-Date".into(),
                value: "20150830T123600Z".into(),
            },
            SignedHeader {
                name: "Host".into(),
                value: "example.amazonaws.com".into(),
            },
        ];
        let authorization = signer.authorization("GET", "/", "", &headers, b"", "20150830T123600Z");
        assert!(authorization.ends_with(
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        ));
    }
}
