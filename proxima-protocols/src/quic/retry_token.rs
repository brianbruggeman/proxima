//! Server-private address-validation tokens per [RFC 9000 §8.1.3] / §8.1.4.
//!
//! The token format is **server-private** — the RFC explicitly says
//! "there is no need for a single well-defined format because the
//! server that generates the token also consumes it." The discipline
//! is therefore not format-parity with any incumbent (each impl
//! cooks its own) but instead:
//!
//! 1. AEAD-sealed with a long-lived rotating server secret.
//! 2. Length-prefix every variable field; reject any extra trailing
//!    bytes after the parsed end.
//! 3. Bind client address byte-for-byte inside the sealed plaintext.
//! 4. Embed `(version, kind, issued_at)` INSIDE the sealed plaintext
//!    so a tamper either breaks the AEAD tag or is caught by the
//!    plaintext-field check on verify.
//!
//! [RFC 9000 §8.1.3]: https://www.rfc-editor.org/rfc/rfc9000#section-8.1.3
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). State is a `[u8; 32]` server
//! secret. Issue + verify operate on caller-owned `&mut [u8]` buffers.

use rand_core::{CryptoRng, Rng};

use crate::quic::crypto::aead::{self, AeadError, CHACHA20_POLY1305_KEY_LEN, NONCE_LEN, TAG_LEN};
use crate::quic::sized;

/// Maximum length of the on-wire token body (server-private). Sourced
/// from `proxima-quic-proto.toml [retry_token].max_token_len`.
pub const MAX_TOKEN_LEN: usize = sized::RETRY_TOKEN_MAX_TOKEN_LEN;

/// Maximum bytes of opaque client-address material the token binds.
/// Sourced from `proxima-quic-proto.toml [retry_token].max_client_addr_len`.
pub const MAX_CLIENT_ADDR_LEN: usize = sized::RETRY_TOKEN_MAX_CLIENT_ADDR_LEN;

/// Maximum Connection ID byte length embedded in a token. RFC 9000
/// §5.1.1 caps QUIC v1 CIDs at 20 bytes; we mirror that.
pub const MAX_CID_LEN: usize = sized::RETRY_TOKEN_MAX_CID_LEN;

/// Format version for our private token shape. Bumped if plaintext
/// layout changes; old-version tokens fail verify.
pub const TOKEN_VERSION: u8 = 0x01;

/// Differentiates Retry tokens (RFC 9000 §8.1.2 — bound to a single
/// connection attempt) from NEW_TOKEN tokens (§8.1.3 — usable across
/// connection attempts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenKind {
    /// Issued in a Retry packet — RFC 9000 §8.1.2. Includes both
    /// `original_destination_connection_id` and `retry_source_connection_id`.
    Retry,
    /// Issued in a NEW_TOKEN frame — RFC 9000 §8.1.3. Carries only
    /// the original DCID (no retry SCID); usable on a future connection.
    NewToken,
}

impl TokenKind {
    const fn as_byte(self) -> u8 {
        match self {
            Self::Retry => 0x00,
            Self::NewToken => 0x01,
        }
    }

    const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Retry),
            0x01 => Some(Self::NewToken),
            _ => None,
        }
    }
}

/// Errors from [`RetryTokenIssuer::issue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IssueError {
    /// Caller passed a `client_addr` longer than [`MAX_CLIENT_ADDR_LEN`].
    ClientAddressTooLong,
    /// Caller passed an `odcid` (or non-`None` `retry_scid`) longer
    /// than [`MAX_CID_LEN`].
    ConnectionIdTooLong,
    /// `kind = Retry` requires a `retry_scid`; `kind = NewToken`
    /// forbids one.
    KindMismatch,
    /// Caller-provided output buffer too small for the encoded token.
    OutputBufferTooSmall { needed: usize },
    /// AEAD failure (extremely unlikely outside of programmer error).
    Aead,
}

impl From<AeadError> for IssueError {
    fn from(_: AeadError) -> Self {
        Self::Aead
    }
}

/// Errors from [`RetryTokenVerifier::verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyError {
    /// Token shorter than nonce+tag floor.
    TooShort,
    /// AEAD authentication failed — secret mismatch, tampering, or
    /// wrong nonce.
    AuthenticationFailed,
    /// Decoded plaintext has the wrong version byte (token issued
    /// under a different format).
    WrongVersion { observed: u8 },
    /// Decoded `kind` doesn't match what the caller asked for.
    WrongKind {
        observed: TokenKind,
        expected: TokenKind,
    },
    /// Bound client address differs from what the caller supplied.
    AddressMismatch,
    /// Token's `issued_at` is in the past beyond `max_age`.
    Expired,
    /// Token's `issued_at` is in the future (caller's clock issue, or
    /// the token was clearly fabricated).
    NotYetValid,
    /// Decoded plaintext is malformed (bad length-prefix, trailing
    /// bytes, etc.) — defensive case; reachable only if the AEAD pass
    /// succeeds on garbage.
    MalformedPlaintext,
}

impl From<AeadError> for VerifyError {
    fn from(_: AeadError) -> Self {
        Self::AuthenticationFailed
    }
}

/// Parsed contents of a verified token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedToken<'a> {
    /// `issued_at` value the server stamped in at issue time (server-private epoch).
    pub issued_at: u64,
    /// Original destination CID the client used in its first Initial.
    pub odcid: &'a [u8],
    /// Retry source CID — `Some(_)` iff `kind == Retry`.
    pub retry_scid: Option<&'a [u8]>,
}

/// Server-side token issuer/verifier. Holds the long-lived AEAD key.
///
/// Rotation strategy is the caller's responsibility: hold N
/// [`RetryTokenIssuer`]s (current + previous), verify against each,
/// drop the oldest when the rotation window passes.
#[derive(Debug, Clone)]
pub struct RetryTokenIssuer {
    secret: [u8; CHACHA20_POLY1305_KEY_LEN],
}

impl RetryTokenIssuer {
    /// Construct from a 32-byte secret. Use a CSPRNG (e.g. an external
    /// crypto crate's `Rng`) to generate the secret at server boot.
    #[must_use]
    pub const fn new(secret: [u8; CHACHA20_POLY1305_KEY_LEN]) -> Self {
        Self { secret }
    }

    /// Encode + AEAD-seal a token. Writes the encoded token into
    /// `out` and returns the number of bytes written.
    ///
    /// `out.len()` must be at least [`MAX_TOKEN_LEN`] (or the actual
    /// length for the field sizes supplied); if too small, returns
    /// [`IssueError::OutputBufferTooSmall`].
    ///
    /// # Errors
    ///
    /// See [`IssueError`].
    // every parameter is an independent input (kind, addr, two CIDs,
    // timestamp, rng, out); grouping into a builder struct adds
    // indirection with no clarity gain on a tier-3 primitive.
    #[allow(clippy::too_many_arguments)]
    pub fn issue<R: CryptoRng + Rng>(
        &self,
        kind: TokenKind,
        client_addr: &[u8],
        odcid: &[u8],
        retry_scid: Option<&[u8]>,
        issued_at: u64,
        rng: &mut R,
        out: &mut [u8],
    ) -> Result<usize, IssueError> {
        if client_addr.len() > MAX_CLIENT_ADDR_LEN {
            return Err(IssueError::ClientAddressTooLong);
        }
        if odcid.len() > MAX_CID_LEN {
            return Err(IssueError::ConnectionIdTooLong);
        }
        match (kind, retry_scid) {
            (TokenKind::Retry, Some(scid)) if scid.len() <= MAX_CID_LEN => {}
            (TokenKind::Retry, Some(_)) => return Err(IssueError::ConnectionIdTooLong),
            (TokenKind::Retry, None) | (TokenKind::NewToken, Some(_)) => {
                return Err(IssueError::KindMismatch);
            }
            (TokenKind::NewToken, None) => {}
        }

        let rscid_bytes: &[u8] = retry_scid.unwrap_or(&[]);
        let plaintext_len = plaintext_len(client_addr.len(), odcid.len(), rscid_bytes.len());
        let total_len = NONCE_LEN + plaintext_len + TAG_LEN;
        if out.len() < total_len {
            return Err(IssueError::OutputBufferTooSmall { needed: total_len });
        }

        let (nonce_buf, rest) = out.split_at_mut(NONCE_LEN);
        rng.fill_bytes(nonce_buf);
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(nonce_buf);

        let (body, tag_buf) = rest.split_at_mut(plaintext_len);
        encode_plaintext(body, kind, issued_at, client_addr, odcid, rscid_bytes);

        let mut tag = [0u8; TAG_LEN];
        aead::chacha20_poly1305_encrypt(&self.secret, &nonce, &[], body, &mut tag)?;
        tag_buf[..TAG_LEN].copy_from_slice(&tag);

        Ok(total_len)
    }

    /// Decode + AEAD-verify + field-validate a token. On success,
    /// returns a [`VerifiedToken`] borrowing into the caller's
    /// `scratch` slice (which is overwritten with decrypted plaintext).
    ///
    /// `token` is the on-wire bytes. `scratch` must be at least
    /// `token.len() - NONCE_LEN - TAG_LEN` bytes.
    ///
    /// `now` and `max_age` are in the same caller-chosen units as
    /// `issued_at` at issue time. The verifier rejects `now < issued_at`
    /// (NotYetValid) and `now - issued_at > max_age` (Expired).
    ///
    /// # Errors
    ///
    /// See [`VerifyError`].
    pub fn verify<'a>(
        &self,
        expected_kind: TokenKind,
        expected_client_addr: &[u8],
        token: &[u8],
        now: u64,
        max_age: u64,
        scratch: &'a mut [u8],
    ) -> Result<VerifiedToken<'a>, VerifyError> {
        if token.len() < NONCE_LEN + TAG_LEN {
            return Err(VerifyError::TooShort);
        }
        let plaintext_len = token.len() - NONCE_LEN - TAG_LEN;
        if scratch.len() < plaintext_len {
            return Err(VerifyError::MalformedPlaintext);
        }
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&token[..NONCE_LEN]);
        let body = &token[NONCE_LEN..NONCE_LEN + plaintext_len];
        let mut tag = [0u8; TAG_LEN];
        tag.copy_from_slice(&token[NONCE_LEN + plaintext_len..]);

        let plaintext = &mut scratch[..plaintext_len];
        plaintext.copy_from_slice(body);
        aead::chacha20_poly1305_decrypt(&self.secret, &nonce, &[], plaintext, &tag)?;

        let parsed = decode_plaintext(plaintext)?;
        if parsed.version != TOKEN_VERSION {
            return Err(VerifyError::WrongVersion {
                observed: parsed.version,
            });
        }
        if parsed.kind != expected_kind {
            return Err(VerifyError::WrongKind {
                observed: parsed.kind,
                expected: expected_kind,
            });
        }
        if parsed.client_addr != expected_client_addr {
            return Err(VerifyError::AddressMismatch);
        }
        if now < parsed.issued_at {
            return Err(VerifyError::NotYetValid);
        }
        if now - parsed.issued_at > max_age {
            return Err(VerifyError::Expired);
        }

        Ok(VerifiedToken {
            issued_at: parsed.issued_at,
            odcid: parsed.odcid,
            retry_scid: parsed.retry_scid,
        })
    }
}

const fn plaintext_len(client_addr_len: usize, odcid_len: usize, rscid_len: usize) -> usize {
    // version(1) + kind(1) + issued_at(8) + addr_len(1) + addr +
    // odcid_len(1) + odcid + rscid_len(1) + rscid
    1 + 1 + 8 + 1 + client_addr_len + 1 + odcid_len + 1 + rscid_len
}

fn encode_plaintext(
    out: &mut [u8],
    kind: TokenKind,
    issued_at: u64,
    client_addr: &[u8],
    odcid: &[u8],
    rscid: &[u8],
) {
    let mut cursor = 0;
    out[cursor] = TOKEN_VERSION;
    cursor += 1;
    out[cursor] = kind.as_byte();
    cursor += 1;
    out[cursor..cursor + 8].copy_from_slice(&issued_at.to_le_bytes());
    cursor += 8;
    out[cursor] = u8::try_from(client_addr.len()).unwrap_or(u8::MAX);
    cursor += 1;
    out[cursor..cursor + client_addr.len()].copy_from_slice(client_addr);
    cursor += client_addr.len();
    out[cursor] = u8::try_from(odcid.len()).unwrap_or(u8::MAX);
    cursor += 1;
    out[cursor..cursor + odcid.len()].copy_from_slice(odcid);
    cursor += odcid.len();
    out[cursor] = u8::try_from(rscid.len()).unwrap_or(u8::MAX);
    cursor += 1;
    out[cursor..cursor + rscid.len()].copy_from_slice(rscid);
}

struct ParsedPlaintext<'a> {
    version: u8,
    kind: TokenKind,
    issued_at: u64,
    client_addr: &'a [u8],
    odcid: &'a [u8],
    retry_scid: Option<&'a [u8]>,
}

fn decode_plaintext(buf: &[u8]) -> Result<ParsedPlaintext<'_>, VerifyError> {
    let mut cursor = 0;
    let version = take_u8(buf, &mut cursor)?;
    let kind_byte = take_u8(buf, &mut cursor)?;
    let kind = TokenKind::from_byte(kind_byte).ok_or(VerifyError::MalformedPlaintext)?;
    let issued_at_bytes = take_slice(buf, &mut cursor, 8)?;
    let mut issued_at_arr = [0u8; 8];
    issued_at_arr.copy_from_slice(issued_at_bytes);
    let issued_at = u64::from_le_bytes(issued_at_arr);

    let addr_len = take_u8(buf, &mut cursor)? as usize;
    if addr_len > MAX_CLIENT_ADDR_LEN {
        return Err(VerifyError::MalformedPlaintext);
    }
    let client_addr = take_slice(buf, &mut cursor, addr_len)?;

    let odcid_len = take_u8(buf, &mut cursor)? as usize;
    if odcid_len > MAX_CID_LEN {
        return Err(VerifyError::MalformedPlaintext);
    }
    let odcid = take_slice(buf, &mut cursor, odcid_len)?;

    let rscid_len = take_u8(buf, &mut cursor)? as usize;
    if rscid_len > MAX_CID_LEN {
        return Err(VerifyError::MalformedPlaintext);
    }
    let retry_scid = if rscid_len == 0 {
        None
    } else {
        Some(take_slice(buf, &mut cursor, rscid_len)?)
    };

    if cursor != buf.len() {
        return Err(VerifyError::MalformedPlaintext);
    }
    match (kind, retry_scid) {
        (TokenKind::Retry, Some(_)) | (TokenKind::NewToken, None) => {}
        _ => return Err(VerifyError::MalformedPlaintext),
    }

    Ok(ParsedPlaintext {
        version,
        kind,
        issued_at,
        client_addr,
        odcid,
        retry_scid,
    })
}

fn take_u8(buf: &[u8], cursor: &mut usize) -> Result<u8, VerifyError> {
    if *cursor >= buf.len() {
        return Err(VerifyError::MalformedPlaintext);
    }
    let byte = buf[*cursor];
    *cursor += 1;
    Ok(byte)
}

fn take_slice<'a>(buf: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], VerifyError> {
    if buf.len() < *cursor + len {
        return Err(VerifyError::MalformedPlaintext);
    }
    let slice = &buf[*cursor..*cursor + len];
    *cursor += len;
    Ok(slice)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    const SECRET: [u8; CHACHA20_POLY1305_KEY_LEN] = [0x42; CHACHA20_POLY1305_KEY_LEN];
    const CLIENT_ADDR: [u8; 6] = [127, 0, 0, 1, 0x40, 0x1F];
    const ODCID: [u8; 8] = [0x83, 0x94, 0xC8, 0xF0, 0x3E, 0x51, 0x57, 0x08];
    const RSCID: [u8; 8] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11];
    const ISSUED_AT: u64 = 1_700_000_000_000_000;
    const MAX_AGE: u64 = 30_000_000;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(0xC0FF_EE99_DEAD_BEEF)
    }

    fn issue_retry(issuer: &RetryTokenIssuer, out: &mut [u8]) -> usize {
        issuer
            .issue(
                TokenKind::Retry,
                &CLIENT_ADDR,
                &ODCID,
                Some(&RSCID),
                ISSUED_AT,
                &mut rng(),
                out,
            )
            .expect("issue ok")
    }

    #[test]
    fn retry_token_round_trip() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let verified = issuer
            .verify(
                TokenKind::Retry,
                &CLIENT_ADDR,
                &buf[..len],
                ISSUED_AT + 1_000,
                MAX_AGE,
                &mut scratch,
            )
            .expect("verify ok");
        assert_eq!(verified.issued_at, ISSUED_AT);
        assert_eq!(verified.odcid, &ODCID);
        assert_eq!(verified.retry_scid, Some(&RSCID[..]));
    }

    #[test]
    fn verify_rejects_expired_token() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer.verify(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &buf[..len],
            ISSUED_AT + MAX_AGE + 1,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::Expired));
    }

    #[test]
    fn verify_rejects_not_yet_valid() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer.verify(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &buf[..len],
            ISSUED_AT - 1,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::NotYetValid));
    }

    #[test]
    fn verify_rejects_address_mismatch() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let wrong_addr = [192, 168, 1, 1, 0x00, 0x50];
        let result = issuer.verify(
            TokenKind::Retry,
            &wrong_addr,
            &buf[..len],
            ISSUED_AT + 1_000,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::AddressMismatch));
    }

    #[test]
    fn verify_rejects_kind_flip() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer.verify(
            TokenKind::NewToken,
            &CLIENT_ADDR,
            &buf[..len],
            ISSUED_AT + 1_000,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(
            result,
            Err(VerifyError::WrongKind {
                observed: TokenKind::Retry,
                expected: TokenKind::NewToken,
            })
        );
    }

    #[test]
    fn verify_rejects_tampered_byte() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer, &mut buf);
        // Flip a byte in the ciphertext.
        buf[NONCE_LEN + 4] ^= 0x01;
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer.verify(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &buf[..len],
            ISSUED_AT + 1_000,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::AuthenticationFailed));
    }

    #[test]
    fn verify_rejects_wrong_secret() {
        let issuer_alice = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issue_retry(&issuer_alice, &mut buf);
        let issuer_bob = RetryTokenIssuer::new([0x43; CHACHA20_POLY1305_KEY_LEN]);
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer_bob.verify(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &buf[..len],
            ISSUED_AT + 1_000,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::AuthenticationFailed));
    }

    #[test]
    fn verify_rejects_token_too_short() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let short = [0u8; NONCE_LEN + TAG_LEN - 1];
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let result = issuer.verify(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &short,
            ISSUED_AT + 1_000,
            MAX_AGE,
            &mut scratch,
        );
        assert_eq!(result, Err(VerifyError::TooShort));
    }

    #[test]
    fn issue_rejects_client_addr_too_long() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let too_long = [0u8; MAX_CLIENT_ADDR_LEN + 1];
        let result = issuer.issue(
            TokenKind::Retry,
            &too_long,
            &ODCID,
            Some(&RSCID),
            ISSUED_AT,
            &mut rng(),
            &mut buf,
        );
        assert_eq!(result, Err(IssueError::ClientAddressTooLong));
    }

    #[test]
    fn issue_rejects_odcid_too_long() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let too_long = [0u8; MAX_CID_LEN + 1];
        let result = issuer.issue(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &too_long,
            Some(&RSCID),
            ISSUED_AT,
            &mut rng(),
            &mut buf,
        );
        assert_eq!(result, Err(IssueError::ConnectionIdTooLong));
    }

    #[test]
    fn issue_rejects_retry_without_scid() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let result = issuer.issue(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &ODCID,
            None,
            ISSUED_AT,
            &mut rng(),
            &mut buf,
        );
        assert_eq!(result, Err(IssueError::KindMismatch));
    }

    #[test]
    fn issue_rejects_new_token_with_scid() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let result = issuer.issue(
            TokenKind::NewToken,
            &CLIENT_ADDR,
            &ODCID,
            Some(&RSCID),
            ISSUED_AT,
            &mut rng(),
            &mut buf,
        );
        assert_eq!(result, Err(IssueError::KindMismatch));
    }

    #[test]
    fn issue_rejects_undersized_output() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; 4]; // way too small
        let result = issuer.issue(
            TokenKind::Retry,
            &CLIENT_ADDR,
            &ODCID,
            Some(&RSCID),
            ISSUED_AT,
            &mut rng(),
            &mut buf,
        );
        assert!(matches!(
            result,
            Err(IssueError::OutputBufferTooSmall { .. })
        ));
    }

    #[test]
    fn new_token_round_trip_carries_no_rscid() {
        let issuer = RetryTokenIssuer::new(SECRET);
        let mut buf = [0u8; MAX_TOKEN_LEN];
        let len = issuer
            .issue(
                TokenKind::NewToken,
                &CLIENT_ADDR,
                &ODCID,
                None,
                ISSUED_AT,
                &mut rng(),
                &mut buf,
            )
            .expect("issue ok");
        let mut scratch = [0u8; MAX_TOKEN_LEN];
        let verified = issuer
            .verify(
                TokenKind::NewToken,
                &CLIENT_ADDR,
                &buf[..len],
                ISSUED_AT + 1,
                MAX_AGE,
                &mut scratch,
            )
            .expect("verify ok");
        assert_eq!(verified.odcid, &ODCID);
        assert!(verified.retry_scid.is_none());
    }
}
