//! Retry-packet integrity tag per [RFC 9001 §5.8].
//!
//! Computes / verifies the AEAD-GCM tag over the "pseudo-Retry packet"
//! using the canonical key + IV published in RFC 9001 §5.8 (QUIC v1).
//! The key + IV are NOT secrets — they protect path integrity (the
//! computing party must have observed the original Initial's DCID),
//! not server authenticity (TLS does that).
//!
//! [RFC 9001 §5.8]: https://www.rfc-editor.org/rfc/rfc9001#section-5.8
//!
//! # Tier
//!
//! Tier-3 (no_std + no_alloc). Pure functions over caller-owned
//! slices + a stack-only AEAD invocation.
//!
//! # Security review (per workspace principle 13)
//!
//! See [`docs/proxima-quic/c19-retry-tokens-design.md`] for the
//! composition-flaw scan. The constant-time tag compare is the
//! single security-critical invariant — implemented via byte-wise
//! XOR-or + zero check that the optimiser cannot short-circuit.
//!
//! [`docs/proxima-quic/c19-retry-tokens-design.md`]: ../../../docs/proxima-quic/c19-retry-tokens-design.md

use arrayvec::ArrayVec;

use super::aead::{self, AES_128_GCM_KEY_LEN, NONCE_LEN, TAG_LEN};

/// QUIC v1 retry integrity key per RFC 9001 §5.8.
pub const RETRY_KEY_V1: [u8; AES_128_GCM_KEY_LEN] = [
    0xbe, 0x0c, 0x69, 0x0b, 0x9f, 0x66, 0x57, 0x5a, 0x1d, 0x76, 0x6b, 0x54, 0xe3, 0x68, 0xc8, 0x4e,
];

/// QUIC v1 retry integrity IV per RFC 9001 §5.8.
pub const RETRY_IV_V1: [u8; NONCE_LEN] = [
    0x46, 0x15, 0x99, 0xd3, 0x5d, 0x63, 0x2b, 0xf2, 0x23, 0x98, 0x25, 0xbb,
];

/// Maximum bytes a pseudo-Retry input can occupy inline. Retry
/// packets are bounded by the initial-MTU (1200 B) plus the
/// length-prefixed original DCID (≤21 B) — 1500 B comfortably
/// covers the worst case.
pub const MAX_PSEUDO_RETRY_BYTES: usize = 1500;

/// Failure modes from [`compute_retry_tag`] + [`verify_retry_tag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryIntegrityError {
    /// `original_dcid.len()` exceeded the protocol max (20).
    OriginalDcidTooLong,
    /// `retry_packet_without_tag` + length prefix overran
    /// [`MAX_PSEUDO_RETRY_BYTES`].
    PseudoRetryTooLong,
    /// Underlying AEAD encrypt failed.
    AeadFailed,
    /// Tag did not match the expected value (constant-time compare).
    TagMismatch,
}

impl From<aead::AeadError> for RetryIntegrityError {
    fn from(_: aead::AeadError) -> Self {
        Self::AeadFailed
    }
}

/// Build the pseudo-Retry packet per RFC 9001 §5.8:
///
/// ```text
/// pseudo_retry =
///   original_destination_cid_length (1 byte)
///   || original_destination_cid
///   || retry_packet_without_integrity_tag
/// ```
///
/// Returns the constructed bytes inline.
fn build_pseudo_retry(
    original_dcid: &[u8],
    retry_packet_without_tag: &[u8],
) -> Result<ArrayVec<u8, MAX_PSEUDO_RETRY_BYTES>, RetryIntegrityError> {
    if original_dcid.len() > 20 {
        return Err(RetryIntegrityError::OriginalDcidTooLong);
    }
    let total = 1 + original_dcid.len() + retry_packet_without_tag.len();
    if total > MAX_PSEUDO_RETRY_BYTES {
        return Err(RetryIntegrityError::PseudoRetryTooLong);
    }
    let mut buffer: ArrayVec<u8, MAX_PSEUDO_RETRY_BYTES> = ArrayVec::new();
    let _ = buffer.try_push(original_dcid.len() as u8);
    let _ = buffer.try_extend_from_slice(original_dcid);
    let _ = buffer.try_extend_from_slice(retry_packet_without_tag);
    Ok(buffer)
}

/// Compute the RFC 9001 §5.8 retry integrity tag for the given
/// Retry packet body (without its own tag) bound to the original
/// DCID.
///
/// # Errors
///
/// See [`RetryIntegrityError`].
pub fn compute_retry_tag(
    original_dcid: &[u8],
    retry_packet_without_tag: &[u8],
) -> Result<[u8; TAG_LEN], RetryIntegrityError> {
    let pseudo = build_pseudo_retry(original_dcid, retry_packet_without_tag)?;
    let mut empty_plaintext: [u8; 0] = [];
    let mut tag = [0u8; TAG_LEN];
    aead::aes_128_gcm_encrypt(
        &RETRY_KEY_V1,
        &RETRY_IV_V1,
        &pseudo,
        &mut empty_plaintext,
        &mut tag,
    )?;
    Ok(tag)
}

/// Verify the RFC 9001 §5.8 retry integrity tag in constant time.
///
/// Returns `Ok(())` if `received_tag` matches the computed tag for
/// the given inputs; [`RetryIntegrityError::TagMismatch`] otherwise.
///
/// # Errors
///
/// See [`RetryIntegrityError`].
pub fn verify_retry_tag(
    original_dcid: &[u8],
    retry_packet_without_tag: &[u8],
    received_tag: &[u8; TAG_LEN],
) -> Result<(), RetryIntegrityError> {
    let expected = compute_retry_tag(original_dcid, retry_packet_without_tag)?;
    if constant_time_eq(&expected, received_tag) {
        Ok(())
    } else {
        Err(RetryIntegrityError::TagMismatch)
    }
}

/// Constant-time byte-array equality. Returns `true` iff every byte
/// matches. The compiler cannot short-circuit on the bitwise OR
/// accumulator.
#[inline]
fn constant_time_eq(left: &[u8; TAG_LEN], right: &[u8; TAG_LEN]) -> bool {
    let mut acc: u8 = 0;
    for index in 0..TAG_LEN {
        acc |= left[index] ^ right[index];
    }
    acc == 0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// RFC 9001 §A.4 canonical vector: original DCID +
    /// pre-tag Retry bytes + expected tag.
    const APPENDIX_A4_ORIGINAL_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    /// Pre-tag Retry packet: long-header byte + version + DCID(0) +
    /// SCID(8) + retry token. Reconstructed per RFC 9001 §A.4.
    /// Total = 1 (first byte) + 4 (version) + 1 (DCID len) + 0 (DCID)
    ///       + 1 (SCID len) + 8 (SCID) + 5 ("token") = 20.
    const APPENDIX_A4_RETRY_WITHOUT_TAG: [u8; 20] = [
        0xff, 0x00, 0x00, 0x00, 0x01, // type byte + version v1
        0x00, // dcid length (0)
        0x08, // scid length (8)
        0xf0, 0x67, 0xa5, 0x50, 0x2a, 0x42, 0x62, 0xb5, // scid
        0x74, 0x6f, 0x6b, 0x65, 0x6e, // "token"
    ];

    /// Expected integrity tag from RFC 9001 §A.4. The full §A.4 wire
    /// bytes per the published RFC are
    /// `ff000000010008f067a5502a4262b5746f6b656e04a265ba2eff4d829058fb3f0f2496ba`
    /// — the trailing 16 bytes are this tag.
    const APPENDIX_A4_EXPECTED_TAG: [u8; TAG_LEN] = [
        0x04, 0xa2, 0x65, 0xba, 0x2e, 0xff, 0x4d, 0x82, 0x90, 0x58, 0xfb, 0x3f, 0x0f, 0x24, 0x96,
        0xba,
    ];

    #[test]
    fn compute_and_verify_succeed_on_arbitrary_inputs() {
        // The §A.4 expected tag was hand-derived for a specific
        // pre-tag retry bytes layout — we don't rely on its
        // exact match here; only on compute → verify round trip.
        let original_dcid = [0xaa, 0xbb, 0xcc, 0xdd];
        let retry_bytes = [0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xde, 0xad];
        let tag = compute_retry_tag(&original_dcid, &retry_bytes).expect("compute");
        verify_retry_tag(&original_dcid, &retry_bytes, &tag).expect("verify");
    }

    #[test]
    fn verify_rejects_tampered_tag() {
        let original_dcid = [0xaa, 0xbb, 0xcc, 0xdd];
        let retry_bytes = [0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xde, 0xad];
        let mut tag = compute_retry_tag(&original_dcid, &retry_bytes).expect("compute");
        tag[0] ^= 0x01;
        let result = verify_retry_tag(&original_dcid, &retry_bytes, &tag);
        assert!(matches!(result, Err(RetryIntegrityError::TagMismatch)));
    }

    #[test]
    fn verify_rejects_tampered_retry_bytes() {
        let original_dcid = [0xaa, 0xbb, 0xcc, 0xdd];
        let mut retry_bytes = [0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xde, 0xad];
        let tag = compute_retry_tag(&original_dcid, &retry_bytes).expect("compute");
        retry_bytes[5] ^= 0x01;
        let result = verify_retry_tag(&original_dcid, &retry_bytes, &tag);
        assert!(matches!(result, Err(RetryIntegrityError::TagMismatch)));
    }

    #[test]
    fn verify_rejects_wrong_original_dcid() {
        let original_dcid = [0xaa, 0xbb, 0xcc, 0xdd];
        let retry_bytes = [0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xde, 0xad];
        let tag = compute_retry_tag(&original_dcid, &retry_bytes).expect("compute");
        let wrong_dcid = [0xaa, 0xbb, 0xcc, 0xde];
        let result = verify_retry_tag(&wrong_dcid, &retry_bytes, &tag);
        assert!(matches!(result, Err(RetryIntegrityError::TagMismatch)));
    }

    /// Direct aes-gcm invocation matching RFC 9001 §A.4 — uses the
    /// same `crate::quic::crypto::aead::aes_128_gcm_encrypt` wrapper that
    /// passes NIST AES-128-GCM Test Case 4 verbatim. Isolates whether
    /// the discrepancy is in pseudo-Retry construction (eliminated by
    /// the existing tamper-rejection tests + this test's verbatim AAD)
    /// or somewhere else entirely.
    #[test]
    fn rfc_9001_appendix_a4_direct_aes_gcm_invocation() {
        // AAD verbatim from RFC §A.4 (29 bytes):
        // 088394c8f03e515708ff000000010008f067a5502a4262b5746f6b656e
        let aad: [u8; 29] = [
            0x08, 0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08, // ODCID len + ODCID
            0xff, 0x00, 0x00, 0x00, 0x01, // first byte + version
            0x00, // DCID len
            0x08, // SCID len
            0xf0, 0x67, 0xa5, 0x50, 0x2a, 0x42, 0x62, 0xb5, // SCID
            0x74, 0x6f, 0x6b, 0x65, 0x6e, // token
        ];
        let mut empty: [u8; 0] = [];
        let mut tag = [0u8; TAG_LEN];
        aead::aes_128_gcm_encrypt(&RETRY_KEY_V1, &RETRY_IV_V1, &aad, &mut empty, &mut tag)
            .expect("aead ok");
        assert_eq!(
            tag, APPENDIX_A4_EXPECTED_TAG,
            "direct aes-gcm with §A.4 AAD must match expected tag",
        );
    }

    #[test]
    fn rfc_9001_appendix_a4_canonical_vector_via_wrapper() {
        let tag = compute_retry_tag(&APPENDIX_A4_ORIGINAL_DCID, &APPENDIX_A4_RETRY_WITHOUT_TAG)
            .expect("compute");
        assert_eq!(
            tag, APPENDIX_A4_EXPECTED_TAG,
            "RFC 9001 §A.4 canonical retry-integrity tag mismatch via wrapper",
        );
    }

    #[test]
    fn original_dcid_above_20_bytes_is_rejected() {
        let dcid = [0u8; 21];
        let result = compute_retry_tag(&dcid, &[]);
        assert!(matches!(
            result,
            Err(RetryIntegrityError::OriginalDcidTooLong)
        ));
    }

    #[test]
    fn constant_time_eq_returns_true_for_equal_tags() {
        let left = [0xaa; TAG_LEN];
        let right = [0xaa; TAG_LEN];
        assert!(constant_time_eq(&left, &right));
    }

    #[test]
    fn constant_time_eq_returns_false_for_one_bit_diff() {
        let left = [0xaa; TAG_LEN];
        let mut right = [0xaa; TAG_LEN];
        right[15] ^= 0x01;
        assert!(!constant_time_eq(&left, &right));
    }
}
