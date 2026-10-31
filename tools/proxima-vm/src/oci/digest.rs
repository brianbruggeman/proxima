use std::fmt;

use sha2::{Digest as _, Sha256};

use crate::oci::error::OciError;

const ALGORITHM: &str = "sha256:";
const HEX_LENGTH: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Digest(String);

impl Digest {
    pub fn parse(text: &str) -> Result<Self, OciError> {
        let hex = text.strip_prefix(ALGORITHM).ok_or_else(|| invalid(text, "only sha256 digests are supported"))?;
        match (hex.len() == HEX_LENGTH, hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))) {
            (true, true) => Ok(Self(text.to_string())),
            _ => Err(invalid(text, "expected 64 lowercase hex characters after `sha256:`")),
        }
    }

    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let hash = Sha256::digest(bytes);
        let hex: String = hash.iter().map(|byte| format!("{byte:02x}")).collect();
        Self(format!("{ALGORITHM}{hex}"))
    }

    pub fn verify(&self, what: &str, bytes: &[u8]) -> Result<(), OciError> {
        let actual = Self::of(bytes);
        if &actual == self {
            return Ok(());
        }
        Err(OciError::DigestMismatch { what: what.to_string(), expected: self.0.clone(), actual: actual.0 })
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn hex(&self) -> &str {
        &self.0[ALGORITHM.len()..]
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

fn invalid(text: &str, reason: &str) -> OciError {
    OciError::Digest { digest: text.to_string(), reason: reason.to_string() }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const EMPTY_SHA256: &str = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn the_empty_input_has_the_published_sha256() {
        assert_eq!(Digest::of(b"").as_str(), EMPTY_SHA256);
    }

    #[test]
    fn verify_reports_both_expected_and_actual_on_mismatch() {
        let expected = Digest::parse(EMPTY_SHA256).expect("well-formed digest");

        let error = expected.verify("layer", b"not empty").expect_err("different bytes");

        let message = error.to_string();
        assert!(message.contains(&format!("expected {EMPTY_SHA256}")), "{message}");
        assert!(message.contains(&format!("actual {}", Digest::of(b"not empty"))), "{message}");
    }

    #[test]
    fn verify_accepts_matching_bytes() {
        Digest::parse(EMPTY_SHA256).expect("well-formed digest").verify("layer", b"").expect("empty matches");
    }

    #[proxima::test]
    #[case::other_algorithm("sha512:abcd")]
    #[case::short_hex("sha256:abcd")]
    #[case::uppercase("sha256:E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")]
    #[case::no_algorithm("e3b0")]
    async fn parse_rejects_malformed_digests(#[case] text: &str) {
        let error = Digest::parse(text).expect_err("malformed digest");

        assert!(error.to_string().contains(text), "{error}");
    }
}
